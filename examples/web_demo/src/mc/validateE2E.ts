/**
 * Stage 2 + Stage 3 integration test. Runs the full pipeline minus the
 * worker boundary (Web Workers need Vite's bundler at runtime, so this
 * harness exercises the orchestrator's pure-TS path):
 *
 *     buildSpikeTrains  →  routeSynapseEvents  →  runMcHH
 *
 * Invocation:
 *     cd examples/web_demo
 *     npx tsx src/mc/validateE2E.ts
 *
 * Verifies:
 *   1. Spike trains for a sustained-active outer neuron land in the
 *      expected rate range under Poisson sampling.
 *   2. With non-zero outer W, synapse events make it onto the post-
 *      neuron's dendritic compartments.
 *   3. The bio sim integrates with synaptic + axial coupling without
 *      diverging, and produces at least one AIS spike when the input
 *      is strong enough.
 */

import { buildSpikeTrains } from "./spikeTrains";
import { routeSynapseEvents } from "./synapseRouter";
import { runMcHH } from "./hhSolver";
import type { SolverMorphology } from "./morphology";
import type { Payload } from "../payload";

function checkApprox(name: string, ok: boolean, detail: string): void {
  const tag = ok ? "PASS" : "FAIL";
  console.log(`[${tag}] ${name}: ${detail}`);
  if (!ok) process.exitCode = 1;
}

// ----- Minimal two-neuron payload (pre → post) -----
// `pre` is sustained-active (rate=0.8), `post` is silent. Outer weight
// matrix has a single non-zero entry: W[post, pre] = 1.0.
const N_FRAMES = 100;
const DURATION_S = 0.5;
const times: number[] = [];
const rates: number[][] = [];
for (let f = 0; f < N_FRAMES; f++) {
  times.push((f / (N_FRAMES - 1)) * DURATION_S);
  rates.push([0.8, 0.0]); // [pre, post]
}
const payload: Payload = {
  metadata: {
    schema_version: 4,
    dataset_id: "synthetic",
    scenario_id: "e2e_test",
    scenario_label: "e2e test",
    description: "",
    hyperparams: {},
    dataset_version: "synthetic:0",
    n_neurons: 2,
    n_frames: N_FRAMES,
    dt_sim: DURATION_S / N_FRAMES,
    duration: DURATION_S,
  },
  bbox: {
    min: [0, 0, 0],
    max: [0, 0, 0],
    center: [0, 0, 0],
    scale: 1,
  },
  neurons: [
    { id: 1, cell_type: "pre", hemisphere: null, angle: null, soma: null, segments: [], radii: [] },
    { id: 2, cell_type: "post", hemisphere: null, angle: null, soma: null, segments: [], radii: [] },
  ],
  times,
  rates,
  model: {
    weights: [
      [0, 0],   // pre is post-of-nothing
      [1.0, 0], // W[post=1][pre=0] = 1
    ],
    tau: [0.02, 0.02],
    bias: [0, 0],
    global_gain: 1.0,
  },
};

// ----- Solver morphology for the post-neuron: three-compartment "Y" -----
// soma + thin AIS + thin dendrite. Stiff geometry — axial RC < 1 μs at
// some edges. The Hines implicit step handles it at dt = 25 μs; the old
// forward-Euler implementation diverged here unconditionally.
const postMorph: SolverMorphology = {
  nCompartments: 3,
  compartments: [
    // 0: soma (parent: -1)
    { anchorNode: 0, members: [0], lengthNm: 10000, radiusNm: 5000, parent: -1 },
    // 1: AIS (child of soma) — narrow + short, stress the cable solve.
    { anchorNode: 1, members: [1], lengthNm: 5000, radiusNm: 500, parent: 0 },
    // 2: dendrite (child of soma) — long + thin.
    { anchorNode: 2, members: [2], lengthNm: 50000, radiusNm: 800, parent: 0 },
  ],
  nodeToCompartment: new Int32Array([0, 1, 2]),
  somaCompartment: 0,
  aisCompartment: 1,
};

// ----- 1. Spike-train sampling -----
const bundle = buildSpikeTrains(payload, { peakHz: 50, seed: 7 });
{
  const preSpikes = bundle[0];
  const postSpikes = bundle[1];
  // pre is at rate 0.8 × 50 Hz = 40 Hz for 0.5s ≈ 20 spikes (Poisson, allow wide window)
  checkApprox(
    "pre-neuron Poisson sampling lands in expected range",
    preSpikes.length >= 8 && preSpikes.length <= 35,
    `${preSpikes.length} spikes in 0.5s (target ≈ 20)`,
  );
  checkApprox(
    "silent post-neuron produces no Poisson spikes",
    postSpikes.length === 0,
    `${postSpikes.length} spikes`,
  );
}

// ----- 2. Synapse routing -----
const events = routeSynapseEvents(payload, /* post */ 1, postMorph, bundle, {
  gPerWeightNs: 5.0,
});
{
  const onDendrite = events.filter((e) => e.compartment === 2).length;
  const onSomaOrAis = events.filter(
    (e) => e.compartment === 0 || e.compartment === 1,
  ).length;
  checkApprox(
    "all synapse events land on dendritic (non-soma/non-AIS) compartments",
    events.length > 0 && onSomaOrAis === 0,
    `${events.length} events; on dendrite=${onDendrite}, on soma/AIS=${onSomaOrAis}`,
  );
  // The synapse is excitatory (W > 0) so the kick should be at e_rev = 0 mV.
  const wrongRev = events.filter((e) => Math.abs(e.eRevMv - 0) > 1e-6).length;
  checkApprox(
    "positive outer W → excitatory reversal (E_rev = 0 mV)",
    wrongRev === 0,
    `${events.length - wrongRev}/${events.length} events at E_rev=0`,
  );
}

// ----- 3. Solver consumes events without diverging -----
{
  const result = runMcHH(postMorph, events, {
    durationMs: DURATION_S * 1000,
    dtMs: 0.025,
    frameStride: 40,
  });
  const finalV = result.voltageTrace[result.voltageTrace.length - 1];
  const finite = Number.isFinite(finalV);
  checkApprox(
    "solver integrates with synaptic input, no NaN / Inf",
    finite,
    `final V at soma=${finalV.toFixed(1)} mV, AIS spikes=${result.aisSpikesMs.length}`,
  );
  // With 20 excitatory inputs at 5 nS each (g_peak), AIS should reach
  // threshold at least a couple of times unless the geometry damps it
  // entirely.
  checkApprox(
    "synaptic input produces at least one AIS spike",
    result.aisSpikesMs.length >= 1,
    `${result.aisSpikesMs.length} spikes at ${result.aisSpikesMs.slice(0, 5).map((t) => t.toFixed(1)).join(", ")} ms`,
  );
}
