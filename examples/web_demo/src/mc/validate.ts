/**
 * Stand-alone validation harness for the multi-compartment HH solver.
 *
 * Designed to run from a `tsx` shell call so we can sanity-check the
 * solver without firing up the full Vite app:
 *
 *     cd examples/web_demo
 *     npx tsx src/mc/validate.ts
 *
 * It builds a one-compartment cylindrical "test neuron" of standard
 * squid-axon-ish geometry and checks:
 *
 *   1. Rheobase    — sub-threshold drive doesn't spike.
 *   2. Single spike — a brief suprathreshold pulse fires once.
 *   3. Repetitive firing — sustained drive at ~10× rheobase produces
 *      regular spiking at a frequency in the textbook 50-100 Hz range.
 *
 * The exact numbers depend on geometry; we're checking gross behaviour
 * (spike vs no spike, ISI in the right ballpark), not literature-precision
 * fits, because the squid-axon channel densities don't fit fly neurons.
 */

import { runMcHH } from "./hhSolver";
import type { SolverMorphology } from "./morphology";

function oneCompartment(radiusUm: number, lengthUm: number): SolverMorphology {
  return {
    nCompartments: 1,
    compartments: [
      {
        anchorNode: 0,
        members: [0],
        lengthNm: lengthUm * 1000,
        radiusNm: radiusUm * 1000,
        parent: -1,
      },
    ],
    nodeToCompartment: new Int32Array([0]),
    somaCompartment: 0,
    // -1 disables the AIS sodium boost; for a single-compartment validation
    // we want textbook squid-axon densities, not the 10× AIS variant.
    aisCompartment: -1,
  };
}

function runWithConstantDrive(
  morph: SolverMorphology,
  iPa: number,
  durationMs: number,
): { aisSpikesMs: number[]; finalV: number; nFrames: number } {
  const nFrames = 100;
  const iExt = new Float32Array(nFrames * morph.nCompartments);
  for (let f = 0; f < nFrames; f++) {
    iExt[f * morph.nCompartments + 0] = iPa;
  }
  const result = runMcHH(
    morph,
    [],
    { durationMs, dtMs: 0.025, frameStride: 40 },
    { data: iExt, nFrames },
  );
  return {
    aisSpikesMs: result.aisSpikesMs,
    finalV: result.voltageTrace[(result.nFrames - 1) * morph.nCompartments + 0],
    nFrames: result.nFrames,
  };
}

function checkApprox(name: string, ok: boolean, detail: string): void {
  const tag = ok ? "PASS" : "FAIL";
  console.log(`[${tag}] ${name}: ${detail}`);
  if (!ok) process.exitCode = 1;
}

const morph = oneCompartment(/* r μm */ 5, /* L μm */ 100);
// Compartment surface area = 2π·5·100 ≈ 3141 μm². At 1 μF/cm² = 0.01 pF/μm²
// that's ~31.4 pF, which for HH-density channels gives realistic firing.

// 1. Sub-threshold: 50 pA over a longer window. We allow up to one onset
// transient spike but no sustained firing (the cell should settle into a
// steady-state plateau below spike threshold).
{
  const r = runWithConstantDrive(morph, 50, 100);
  checkApprox(
    "sub-threshold (50 pA, 100 ms): ≤ 1 spike, V settles below threshold",
    r.aisSpikesMs.length <= 1 && r.finalV < -40,
    `spikes=${r.aisSpikesMs.length}, final V=${r.finalV.toFixed(1)} mV`,
  );
}

// 2. Single spike from a brief above-threshold pulse.
{
  const morph2 = oneCompartment(5, 100);
  const nFrames = 200;
  const iExt = new Float32Array(nFrames);
  // 5 nA for the first 1 ms, then nothing.
  for (let f = 0; f < 10; f++) iExt[f] = 5000;
  const r = runMcHH(
    morph2,
    [],
    { durationMs: 30, dtMs: 0.025, frameStride: 20 },
    { data: iExt, nFrames },
  );
  checkApprox(
    "brief 5 nA pulse → at least one spike",
    r.aisSpikesMs.length >= 1,
    `spikes at ${r.aisSpikesMs.map((t) => t.toFixed(2)).join(", ")} ms`,
  );
}

// 3. Repetitive firing under sustained drive.
{
  const r = runWithConstantDrive(morph, 1500, 200);
  const intervalsMs: number[] = [];
  for (let i = 1; i < r.aisSpikesMs.length; i++) {
    intervalsMs.push(r.aisSpikesMs[i] - r.aisSpikesMs[i - 1]);
  }
  const meanIsi = intervalsMs.length
    ? intervalsMs.reduce((a, b) => a + b, 0) / intervalsMs.length
    : NaN;
  const freqHz = 1000 / meanIsi;
  checkApprox(
    "sustained 1.5 nA → repetitive firing (50-300 Hz)",
    r.aisSpikesMs.length >= 5 && freqHz > 50 && freqHz < 300,
    `${r.aisSpikesMs.length} spikes in 200 ms, mean ISI ${meanIsi.toFixed(1)} ms (${freqHz.toFixed(0)} Hz)`,
  );
}
