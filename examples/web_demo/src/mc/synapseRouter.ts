/**
 * Synapse routing — translates the outer-sim's per-neuron activity into
 * post-compartment input for the bio sim. Two routing modes:
 *
 *   • `buildConductanceSchedule` (C+D design) — for inputs from neurons
 *     OUTSIDE the bio neighborhood. Drives each post-compartment with a
 *     continuous conductance proportional to the outer rate. No discrete
 *     EPSPs; collapses the timescale mismatch on the input side. This is
 *     what `runBioSession` uses.
 *   • `routeSynapseEvents` (legacy) — Poisson-sampled discrete spike
 *     events. Kept for the validateE2E test and as the future hook for
 *     within-neighborhood spike coupling (a presynaptic neuron whose
 *     own AIS spikes are being resolved should drive its post via
 *     discrete events, not via the smoothed outer rate).
 *
 * Anatomically-accurate routing would put each synapse at the compartment
 * nearest to its neuPrint xyz coordinate. We don't have those coordinates
 * in the payload yet (deferred to Stage 0 follow-up), so this v1 router
 * distributes synapses uniformly across the post-neuron's dendritic
 * compartments — every compartment except the soma and the AIS.
 *
 * For each presynaptic neuron `pre` with non-zero weight onto the target:
 *   - sample `pre`'s spike train
 *   - for each presynaptic spike, deposit a conductance kick at a
 *     deterministically-chosen dendritic compartment
 *   - the kick's peak amplitude is proportional to |W[target, pre]|, the
 *     same scalar the outer point-neuron model uses
 *
 * Sign convention: the kick's reversal potential is 0 mV for excitatory
 * NTs (AMPA-like) and −75 mV for inhibitory (GABA-like). We infer the
 * polarity from the sign of the outer model's `W[target, pre]` so the
 * router doesn't need to look up per-neuron NT separately.
 */

import type { Payload } from "../payload";
import type { SolverMorphology } from "./morphology";
import type { SynEvent } from "./hhSolver";
import type { SpikeTrainBundle } from "./spikeTrains";

export interface SynapseRoutingConfig {
  /** Per-unit-weight peak conductance, in nS. Outer-model weights are
   *  dimensionless log1p-derived scalars; this is the scaling that turns
   *  them into a biophysically-meaningful conductance kick. */
  gPerWeightNs?: number;
  /** Excitatory reversal potential (mV). AMPA-like. */
  eExcMv?: number;
  /** Inhibitory reversal potential (mV). GABA-like. */
  eInhMv?: number;
  /** PRNG seed (unused for now — compartment assignment is deterministic
   *  via round-robin hashing — but reserved for the future per-synapse
   *  randomised placement). */
  seed?: number;
  /** Outer rate of 1.0 maps to this many "equivalent Hz" of presynaptic
   *  drive. Used by the smooth-conductance path to scale outer rates
   *  into a steady conductance: g_steady ≈ rate · peakHz · area_per_event.
   *  Lowering this damps the bio activity; raising it drives faster
   *  spiking. Default 50. */
  peakHz?: number;
}

/**
 * Per-compartment continuous conductance schedule for one target neuron.
 *
 * The smooth-conductance design (C+D in the design doc) skips Poisson
 * sampling. Each presynaptic neuron's outer rate `r(t)` drives a steady
 * conductance at its post-compartment proportional to `|W[post, pre]| ·
 * r(t)`. Spikes only happen when the integrated drive crosses the bio
 * cell's threshold — they're not pre-imposed.
 *
 * Layout (row-major):
 *   gSyn[frame * nCompartments + compartment]
 *   gSynRev[frame * nCompartments + compartment]   (= Σ g_i · E_rev_i)
 *
 * The solver samples these at every integration step using zero-order
 * hold (frame stride is ~1 ms, integration dt is ~25 μs).
 */
export interface ConductanceSchedule {
  gSyn: Float32Array;
  gSynRev: Float32Array;
  nFrames: number;
  nCompartments: number;
  /** Bio-sim time between frames in ms. */
  dtMs: number;
}

/**
 * Discrete synapse between two neurons that are both being simulated in
 * the bio neighborhood. When the pre target's AIS spikes, the solver
 * injects a conductance kick at the post target's compartment. This is
 * the within-neighborhood spike-coupling path: smooth conductance is
 * already covering everything OUTSIDE the neighborhood, so these edges
 * only describe couplings *between* targets.
 */
export interface IntraEdge {
  /** Index into the targets array. */
  preTargetIdx: number;
  /** Index into the targets array. */
  postTargetIdx: number;
  /** Solver-compartment index on the post target. */
  postCompartment: number;
  /** Peak conductance kick on spike, nS. */
  gPeakNs: number;
  /** Reversal potential of this synapse, mV. */
  eRevMv: number;
}

/**
 * Build a continuous conductance schedule for one target neuron.
 *
 * Walks every presynaptic neuron `pre` with non-zero `W[post, pre]`, picks
 * one post-compartment per pre (round-robin over the dendrite — same
 * approach as the event-based router; both will switch to per-synapse
 * coordinates once those make it into the payload), and writes a per-bio-
 * frame conductance trace into `gSyn` / `gSynRev`. Sign follows the outer
 * model's `W`: positive entries route to `eExcMv`, negative to `eInhMv`.
 *
 * `anchorOuterMs` is the outer-scenario time at which the bio window
 * begins. Bio frame f corresponds to outer time `anchorOuterMs + f·dtMs`,
 * which we look up via linear interpolation in `payload.rates / times`.
 */
export function buildConductanceSchedule(
  payload: Payload,
  postNeuronIdx: number,
  postMorph: SolverMorphology,
  anchorOuterMs: number,
  bioDurationMs: number,
  bioFrameStrideMs: number = 1,
  config: SynapseRoutingConfig = {},
  /** Outer-payload neuron indices to OMIT from the smooth conductance.
   *  Used by the coupled-bio path: any neuron also being simulated in
   *  the bio neighborhood drives its post via discrete intra-edges
   *  instead, so it must be excluded here to avoid double-counting. */
  excludeFromSmooth: ReadonlySet<number> | null = null,
): ConductanceSchedule {
  const W = payload.model?.weights;
  const nComp = postMorph.nCompartments;
  const nFrames = Math.max(1, Math.ceil(bioDurationMs / bioFrameStrideMs));
  const empty: ConductanceSchedule = {
    gSyn: new Float32Array(nFrames * nComp),
    gSynRev: new Float32Array(nFrames * nComp),
    nFrames,
    nCompartments: nComp,
    dtMs: bioFrameStrideMs,
  };
  if (!W) return empty;

  const gPerWeight = config.gPerWeightNs ?? 0.15;
  const eExc = config.eExcMv ?? 0;
  const eInh = config.eInhMv ?? -75;
  const peakHz = config.peakHz ?? 50;
  // Outer `rate · peakHz` is the equivalent Hz of presynaptic activity.
  // We treat each "spike" as an exponential conductance of unit area
  // gPerWeight·|W|·τ; the time-average over a window many τ wide is
  // therefore (rate·peakHz/1000ms) · gPerWeight·|W|·τ. We fold the
  // τ factor in here so the gain has the same feel as the event-based
  // router. τ = 5 ms (synaptic decay default).
  const synTauMs = 5.0;
  const rateToG = (peakHz / 1000) * synTauMs * gPerWeight;

  // Dendritic targets — every compartment except the soma and the AIS.
  const dendritic: number[] = [];
  for (let i = 0; i < nComp; i++) {
    if (i === postMorph.somaCompartment) continue;
    if (i === postMorph.aisCompartment) continue;
    dendritic.push(i);
  }
  if (dendritic.length === 0) {
    for (let i = 0; i < nComp; i++) dendritic.push(i);
  }

  // Active pre's: those with non-zero W onto this post, with their
  // assigned dendritic compartment + sign baked in.
  type PreEntry = { pre: number; gainNs: number; eRevMv: number; compartment: number };
  const active: PreEntry[] = [];
  const NPop = payload.metadata.n_neurons;
  let compCursor = 0;
  for (let pre = 0; pre < NPop; pre++) {
    if (pre === postNeuronIdx) continue;
    if (excludeFromSmooth && excludeFromSmooth.has(pre)) continue;
    const w = W[postNeuronIdx][pre];
    if (Math.abs(w) < 1e-6) continue;
    active.push({
      pre,
      gainNs: Math.abs(w) * rateToG,
      eRevMv: w >= 0 ? eExc : eInh,
      compartment: dendritic[compCursor % dendritic.length],
    });
    compCursor++;
  }
  if (active.length === 0) return empty;

  // Index the outer times once. They're already in seconds.
  const tOuter = payload.times;
  const nOuter = tOuter.length;
  const rates = payload.rates;
  // Linear-interp helper that writes one row into `out` (length NPop)
  // from the rate trace at `tSec`.
  const sampleRow = (tSec: number, out: Float32Array) => {
    if (tSec <= tOuter[0]) {
      const r = rates[0];
      for (let i = 0; i < NPop; i++) out[i] = r[i];
      return;
    }
    if (tSec >= tOuter[nOuter - 1]) {
      const r = rates[nOuter - 1];
      for (let i = 0; i < NPop; i++) out[i] = r[i];
      return;
    }
    let lo = 0;
    let hi = nOuter - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (tOuter[mid] <= tSec) lo = mid;
      else hi = mid;
    }
    const u = (tSec - tOuter[lo]) / (tOuter[hi] - tOuter[lo]);
    const rA = rates[lo];
    const rB = rates[hi];
    for (let i = 0; i < NPop; i++) out[i] = rA[i] + u * (rB[i] - rA[i]);
  };

  const gSyn = empty.gSyn;
  const gSynRev = empty.gSynRev;
  const rateRow = new Float32Array(NPop);
  for (let f = 0; f < nFrames; f++) {
    const tBioMs = f * bioFrameStrideMs;
    const tOuterSec = (anchorOuterMs + tBioMs) / 1000;
    sampleRow(tOuterSec, rateRow);
    const off = f * nComp;
    for (const ap of active) {
      const r = Math.max(0, rateRow[ap.pre]);
      if (r <= 0) continue;
      const g = ap.gainNs * r;
      gSyn[off + ap.compartment] += g;
      gSynRev[off + ap.compartment] += g * ap.eRevMv;
    }
  }

  return { gSyn, gSynRev, nFrames, nCompartments: nComp, dtMs: bioFrameStrideMs };
}

/**
 * Build the within-neighborhood discrete-spike edges. For every pair of
 * targets (pre, post) with non-zero outer weight, one edge that fires
 * when pre's AIS spikes and deposits a conductance kick on the chosen
 * post-compartment. Compartment assignment is round-robin over the post
 * target's dendrites — same heuristic as the smooth schedule.
 *
 * `gPerWeightNs` here uses the *event* scaling (default 0.6 nS) rather
 * than the smooth scaling (default 0.15) — discrete kicks contribute
 * area under the conductance curve, not standing conductance.
 */
export function buildIntraEdges(
  payload: Payload,
  targetNeuronIndices: ReadonlyArray<number>,
  morphsByTarget: ReadonlyArray<SolverMorphology>,
  config: SynapseRoutingConfig = {},
): IntraEdge[] {
  const W = payload.model?.weights;
  if (!W) return [];
  const gPerWeight = config.gPerWeightNs ?? 0.6;
  const eExc = config.eExcMv ?? 0;
  const eInh = config.eInhMv ?? -75;

  // Precompute per-post-target dendrite lists + a per-target round-robin
  // cursor (so multiple pre's onto the same post land on different
  // compartments rather than stacking all on the first dendrite).
  const dendriticByTarget: number[][] = morphsByTarget.map((m) => {
    const out: number[] = [];
    for (let i = 0; i < m.nCompartments; i++) {
      if (i === m.somaCompartment) continue;
      if (i === m.aisCompartment) continue;
      out.push(i);
    }
    if (out.length === 0) for (let i = 0; i < m.nCompartments; i++) out.push(i);
    return out;
  });
  const cursorByTarget = new Int32Array(morphsByTarget.length);

  const edges: IntraEdge[] = [];
  for (let postIdx = 0; postIdx < targetNeuronIndices.length; postIdx++) {
    const postNeuron = targetNeuronIndices[postIdx];
    for (let preIdx = 0; preIdx < targetNeuronIndices.length; preIdx++) {
      if (preIdx === postIdx) continue;
      const preNeuron = targetNeuronIndices[preIdx];
      const w = W[postNeuron][preNeuron];
      if (Math.abs(w) < 1e-6) continue;
      const dendrites = dendriticByTarget[postIdx];
      const compartment = dendrites[cursorByTarget[postIdx] % dendrites.length];
      cursorByTarget[postIdx]++;
      edges.push({
        preTargetIdx: preIdx,
        postTargetIdx: postIdx,
        postCompartment: compartment,
        gPeakNs: Math.abs(w) * gPerWeight,
        eRevMv: w >= 0 ? eExc : eInh,
      });
    }
  }
  return edges;
}

/**
 * Build the SynEvent list for one target neuron given the bundle of
 * presynaptic spike trains.
 *
 * `postNeuronIdx` is the target neuron's index in the outer payload.
 * Returned events are ready to pass straight to `runMcHH`.
 */
export function routeSynapseEvents(
  payload: Payload,
  postNeuronIdx: number,
  postMorph: SolverMorphology,
  presynapticSpikes: SpikeTrainBundle,
  config: SynapseRoutingConfig = {},
): SynEvent[] {
  const W = payload.model?.weights;
  if (!W) return [];

  const gPerWeight = config.gPerWeightNs ?? 0.6;
  const eExc = config.eExcMv ?? 0;
  const eInh = config.eInhMv ?? -75;

  // Dendritic compartments = everything except the soma and the AIS.
  // For the rare case the morphology has < 3 compartments (single-comp
  // validation, or a stub circuit), fall back to all compartments.
  const dendritic: number[] = [];
  for (let i = 0; i < postMorph.nCompartments; i++) {
    if (i === postMorph.somaCompartment) continue;
    if (i === postMorph.aisCompartment) continue;
    dendritic.push(i);
  }
  if (dendritic.length === 0) {
    for (let i = 0; i < postMorph.nCompartments; i++) dendritic.push(i);
  }

  const events: SynEvent[] = [];
  const N = payload.metadata.n_neurons;
  // Round-robin compartment cursor per presynaptic neuron so each spike
  // train lands on a consistent compartment (not all stacked on the
  // first one).
  let compCursor = 0;
  for (let pre = 0; pre < N; pre++) {
    if (pre === postNeuronIdx) continue;
    const w = W[postNeuronIdx][pre];
    if (Math.abs(w) < 1e-6) continue;
    const peakG = Math.abs(w) * gPerWeight;
    const eRev = w >= 0 ? eExc : eInh;
    const spikes = presynapticSpikes[pre];
    if (!spikes || spikes.length === 0) continue;
    const compartment = dendritic[compCursor % dendritic.length];
    compCursor++;
    for (let s = 0; s < spikes.length; s++) {
      events.push({
        compartment,
        timeMs: spikes[s],
        gPeakNs: peakG,
        eRevMv: eRev,
      });
    }
  }

  // Events must be time-sorted; runMcHH does its own sort but emitting
  // them sorted here saves a pass.
  events.sort((a, b) => a.timeMs - b.timeMs);
  return events;
}
