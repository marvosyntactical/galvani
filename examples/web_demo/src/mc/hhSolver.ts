/**
 * Multi-compartment Hodgkin-Huxley solver, pure TypeScript.
 *
 * Forward Euler over a soma-rooted compartment tree. State per compartment:
 * voltage `v` (mV) and three gating variables `m`, `h`, `n`. Axial coupling
 * between a compartment and its parent is computed as
 *
 *     g_axial = π · r̄² / (R_a · L̄)
 *
 * where r̄ and L̄ are the means of the parent's and child's radii / lengths.
 * Inputs into the cable equation: per-compartment external current
 * (`iExtPa`, in pA) plus synaptic conductance events that decay
 * exponentially. The AIS compartment gets a 10× sodium / potassium boost
 * so spikes initiate where they should rather than wherever the soma's
 * capacitance happens to make them.
 *
 * Units throughout:
 *   - voltage:           mV
 *   - time:              ms
 *   - current:           pA
 *   - capacitance:       pF
 *   - conductance:       nS
 *   - length, radius:    μm
 *
 * These are consistent: pA / pF = mV/ms, nS · mV = pA, μm² · μm·nS·μm⁻³ = ...
 * (everything cancels).
 *
 * Stability: at dt = 25 μs and compartment lengths ≥ 1 μm this is stable
 * for the squid-axon channel densities below. For finer compartments the
 * cable time constant collapses; the `morphology.ts` aggregator coarsens
 * the SWC tree to a solver-safe scale before this code ever runs.
 */

import {
  AIS_BOOST,
  CM_UF_PER_CM2,
  E_K_MV,
  E_L_MV,
  E_NA_MV,
  G_K_DEFAULT_MS_PER_CM2,
  G_L_DEFAULT_MS_PER_CM2,
  G_NA_DEFAULT_MS_PER_CM2,
  H_REST,
  M_REST,
  N_REST,
  V_REST_MV,
  alphaH,
  alphaM,
  alphaN,
  betaH,
  betaM,
  betaN,
} from "./hhChannels";
import type { SolverMorphology } from "./morphology";

/** A synaptic event arriving at a specific compartment at a specific time. */
export interface SynEvent {
  /** Solver-compartment index. */
  compartment: number;
  /** Arrival time in ms. */
  timeMs: number;
  /** Peak conductance kick in nS. */
  gPeakNs: number;
  /** Reversal potential in mV (0 for AMPA-like, -75 for GABA-like). */
  eRevMv: number;
}

/** Pre-computed continuous conductance schedule. Stride dtMs per frame
 *  in bio time; row-major `(nFrames, nCompartments)`. The solver samples
 *  this each step using zero-order hold so the slow input from the
 *  outside-of-neighborhood drive doesn't require per-step recomputation
 *  of rates. */
export interface GScheduleInput {
  gSyn: Float32Array;
  gSynRev: Float32Array;
  nFrames: number;
  nCompartments: number;
  dtMs: number;
}

/** Within-neighborhood discrete edge — see synapseRouter.buildIntraEdges. */
export interface CoupledIntraEdge {
  preTargetIdx: number;
  postTargetIdx: number;
  postCompartment: number;
  gPeakNs: number;
  eRevMv: number;
}

/** One target neuron passed to the coupled solver. */
export interface CoupledTarget {
  morph: SolverMorphology;
  gSchedule: GScheduleInput | null;
}

/** Per-target result returned by `runCoupledMcHH`. */
export interface CoupledResult {
  results: HhResult[];
}

export interface HhRunConfig {
  /** Total simulation duration in ms. */
  durationMs: number;
  /** Integration step size in ms. Forward Euler tolerates ~25 μs (= 0.025). */
  dtMs: number;
  /** How many integration steps per recorded output frame. */
  frameStride: number;
  /** Synaptic conductance decay τ in ms (single-exponential). */
  synTauDecayMs?: number;
  /** Spike-detection threshold at the AIS, mV. Default −20. */
  spikeThresholdMv?: number;
  /** Minimum interspike interval at the AIS, ms. Default 1.5. */
  minIsiMs?: number;
}

export interface HhResult {
  /** voltageTrace[frame * nCompartments + comp] in mV. */
  voltageTrace: Float32Array;
  /** Number of recorded output frames. */
  nFrames: number;
  /** Number of compartments (== voltageTrace.length / nFrames). */
  nCompartments: number;
  /** Frame timestamps in ms. */
  frameTimesMs: Float32Array;
  /** AIS spike times in ms. */
  aisSpikesMs: number[];
  /** Compartment index used for AIS spike detection. */
  aisCompartment: number;
}

interface ProgressReporter {
  /** Called at most every 100 ms with a fraction in [0, 1]. */
  (fraction: number): void;
}

/**
 * Run the multi-compartment HH simulation.
 *
 * `iExtPa` is optional per-compartment current injection in pA, shape
 * (nCompartments, nInputFrames). The input is held constant between frames
 * (zero-order hold) when nInputFrames < total integration steps. Pass
 * `null` if you only want synaptic input.
 */
export function runMcHH(
  morph: SolverMorphology,
  synEvents: SynEvent[],
  config: HhRunConfig,
  iExtPa: { data: Float32Array; nFrames: number } | null = null,
  onProgress: ProgressReporter | null = null,
  gSchedule: GScheduleInput | null = null,
): HhResult {
  const N = morph.nCompartments;
  if (N === 0) {
    return {
      voltageTrace: new Float32Array(0),
      nFrames: 0,
      nCompartments: 0,
      frameTimesMs: new Float32Array(0),
      aisSpikesMs: [],
      aisCompartment: 0,
    };
  }

  const dt = config.dtMs;
  const totalSteps = Math.ceil(config.durationMs / dt);
  const stride = Math.max(1, config.frameStride);
  const nFrames = Math.floor(totalSteps / stride) + 1;
  const synTauDecay = config.synTauDecayMs ?? 5.0;
  const decayPerStep = Math.exp(-dt / synTauDecay);
  const spikeThreshold = config.spikeThresholdMv ?? -20;
  const minIsiMs = config.minIsiMs ?? 1.5;

  // ----- Geometry → per-compartment capacitance + ion-density scaling.
  // We work in (pA, pF, nS, mV, ms, μm). See module docstring.
  const radiusUm = new Float32Array(N);
  const lengthUm = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    radiusUm[i] = morph.compartments[i].radiusNm / 1000;
    lengthUm[i] = morph.compartments[i].lengthNm / 1000;
  }

  // Surface area (μm²) = 2π r L (cylinder side).
  const surfaceUm2 = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    surfaceUm2[i] = 2 * Math.PI * radiusUm[i] * lengthUm[i];
  }

  // 1 mS/cm² = 0.01 nS/μm². So total compartment conductance in nS is:
  //   gMax = (g_density_mS_per_cm2) · 0.01 · surfaceUm2
  // and capacitance in pF is C = 0.01 · surfaceUm2 (since 1 μF/cm² = 0.01 pF/μm²).
  const gNaMax = new Float32Array(N);
  const gKMax = new Float32Array(N);
  const gL = new Float32Array(N);
  const cap = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const isAis = i === morph.aisCompartment;
    const naDens = G_NA_DEFAULT_MS_PER_CM2 * (isAis ? AIS_BOOST : 1);
    const kDens = G_K_DEFAULT_MS_PER_CM2 * (isAis ? AIS_BOOST : 1);
    gNaMax[i] = naDens * 0.01 * surfaceUm2[i];
    gKMax[i] = kDens * 0.01 * surfaceUm2[i];
    gL[i] = G_L_DEFAULT_MS_PER_CM2 * 0.01 * surfaceUm2[i];
    cap[i] = CM_UF_PER_CM2 * 0.01 * surfaceUm2[i];
  }

  // Axial conductance to parent (nS). R_a = 150 Ω·cm = 1.5e6 Ω·μm. Then
  //   g_axial = (π · r̄²) / (R_a · L̄) S = π · r̄² / (1.5e6 · L̄) S
  //   = (π · r̄² / L̄) / 1.5 [μS] = (π · r̄² / L̄) / 1.5 · 1000 [nS]
  // i.e. ≈ 2094 · r̄² / L̄  (r̄, L̄ in μm, result in nS).
  // For a stub at the soma we cap to the smaller compartment's geometry.
  const gAxToParent = new Float32Array(N);
  const RA_FACTOR_NS = (1000 * Math.PI) / 1.5; // ≈ 2094.4
  for (let i = 0; i < N; i++) {
    const p = morph.compartments[i].parent;
    if (p < 0) continue;
    const rMean = 0.5 * (radiusUm[i] + radiusUm[p]);
    const lMean = 0.5 * (lengthUm[i] + lengthUm[p]);
    gAxToParent[i] = (RA_FACTOR_NS * rMean * rMean) / Math.max(lMean, 1e-3);
  }

  // Sum of axial conductances at each compartment (own parent edge + every
  // child edge). Goes on the diagonal of the Hines matrix; precomputed
  // once since geometry doesn't change during the run.
  const axialSum = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const p = morph.compartments[i].parent;
    if (p < 0) continue;
    axialSum[i] += gAxToParent[i];
    axialSum[p] += gAxToParent[i];
  }

  // ----- State.
  const v = new Float32Array(N).fill(V_REST_MV);
  const m = new Float32Array(N).fill(M_REST);
  const h = new Float32Array(N).fill(H_REST);
  const n = new Float32Array(N).fill(N_REST);
  // Hines scratch buffers: diagonal of the implicit-axial matrix and the
  // right-hand-side vector. Both get rebuilt every timestep — `diag` is
  // not the same as `axialSum` because the leak and synapse conductances
  // (both implicit) also contribute, and elimination mutates it in
  // place. `rhs` holds C·V^n/dt plus the explicit ionic + bias terms.
  const diag = new Float32Array(N);
  const rhs = new Float32Array(N);
  // Synaptic conductance bookkeeping: one (gSyn, eRev * gSyn) accumulator
  // per compartment. We sum-of-exponentials by collapsing into two
  // running totals — exact when all events share τ_decay, which they do.
  const gSyn = new Float32Array(N);          // total active conductance (nS)
  const gSynRev = new Float32Array(N);       // Σ g_i · E_rev_i  (nS·mV)

  // Precompute parents as a plain array for tight loops below.
  const parents = new Int32Array(N);
  for (let i = 0; i < N; i++) parents[i] = morph.compartments[i].parent;

  // Sort events by time so we can sweep them in step order.
  const events = [...synEvents].sort((a, b) => a.timeMs - b.timeMs);
  let nextEventIdx = 0;

  // Output buffers.
  const voltageTrace = new Float32Array(nFrames * N);
  const frameTimesMs = new Float32Array(nFrames);
  const aisSpikesMs: number[] = [];
  // `aisCompartment < 0` is the "no AIS boost" sentinel (used for
  // single-compartment validation runs). Spike detection still has to
  // read from a real compartment — fall back to the soma so the detector
  // doesn't try to dereference v[-1] and silently miss every spike.
  const aisIdx = morph.aisCompartment >= 0
    ? morph.aisCompartment
    : morph.somaCompartment;
  let lastSpikeMs = -Infinity;
  let prevAisV = V_REST_MV;

  let nextFrame = 0;
  let nextProgress = performance.now();
  const reportEveryMs = 100;

  // ----- Integration loop.
  for (let step = 0; step <= totalSteps; step++) {
    const tMs = step * dt;

    // Inject any synaptic events that came due in [t, t+dt).
    while (
      nextEventIdx < events.length &&
      events[nextEventIdx].timeMs < tMs + dt
    ) {
      const e = events[nextEventIdx];
      gSyn[e.compartment] += e.gPeakNs;
      gSynRev[e.compartment] += e.gPeakNs * e.eRevMv;
      nextEventIdx++;
    }

    // Periodic frame snapshot.
    if (step % stride === 0 && nextFrame < nFrames) {
      const off = nextFrame * N;
      voltageTrace.set(v, off);
      frameTimesMs[nextFrame] = tMs;
      nextFrame++;
      if (onProgress && performance.now() > nextProgress) {
        onProgress(step / totalSteps);
        nextProgress = performance.now() + reportEveryMs;
      }
    }

    // External current at this step (zero-order hold from input frames).
    let iInputFrame = -1;
    if (iExtPa !== null) {
      iInputFrame = Math.min(
        iExtPa.nFrames - 1,
        Math.floor((step / totalSteps) * iExtPa.nFrames),
      );
    }
    // Scheduled (smooth) conductance — current bio-frame in the schedule.
    let schedFrame = -1;
    if (gSchedule !== null) {
      schedFrame = Math.min(
        gSchedule.nFrames - 1,
        Math.floor(tMs / gSchedule.dtMs),
      );
    }

    // AIS spike detection (rising-edge crossing).
    const aisV = v[aisIdx];
    if (
      aisV >= spikeThreshold &&
      prevAisV < spikeThreshold &&
      tMs - lastSpikeMs > minIsiMs
    ) {
      aisSpikesMs.push(tMs);
      lastSpikeMs = tMs;
    }
    prevAisV = aisV;

    // ----- One implicit step (Hines 1984).
    //
    // Backward Euler on the stiff terms: axial coupling, leak, synaptic
    // conductance. Forward Euler on the rest: Na, K, gating. This is the
    // standard NEURON discretisation. Stiff piece becomes a sparse linear
    // system whose matrix has the structure of the compartment tree:
    //
    //     diag[i] · V_i^{n+1}  −  Σ_{j ∈ neighbours(i)} g_ij · V_j^{n+1}
    //                                                = rhs[i]
    //
    // For a soma-rooted tree (parents[i] < i for all i > 0, which our
    // BFS morphology builder guarantees), the off-diagonal pattern is
    // banded: each row has exactly one off-diagonal entry to its parent
    // plus one per child. Hines's trick is to eliminate children before
    // parents in a single O(N) pass; the algebra reduces to two scalar
    // operations per edge.
    const dtRecip = 1.0 / dt;
    for (let i = 0; i < N; i++) {
      const vi = v[i];
      // Channel currents (explicit at V^n, gates at their current value).
      const iNa = gNaMax[i] * m[i] * m[i] * m[i] * h[i] * (vi - E_NA_MV);
      const iK = gKMax[i] * n[i] * n[i] * n[i] * n[i] * (vi - E_K_MV);

      let iExt = 0;
      if (iExtPa !== null && iInputFrame >= 0) {
        iExt = iExtPa.data[iInputFrame * N + i];
      }

      // Leak and synaptic conductance go onto the diagonal; their
      // reversal-potential contribution goes to the RHS. The scheduled
      // (smooth) drive — from neurons OUTSIDE the bio neighborhood — is
      // summed into the same accumulator. See MC.md / C+D design.
      let gSchedI = 0;
      let gSchedRevI = 0;
      if (gSchedule !== null && schedFrame >= 0) {
        const so = schedFrame * gSchedule.nCompartments + i;
        gSchedI = gSchedule.gSyn[so];
        gSchedRevI = gSchedule.gSynRev[so];
      }
      diag[i] = cap[i] * dtRecip + axialSum[i] + gL[i] + gSyn[i] + gSchedI;
      rhs[i] =
        cap[i] * dtRecip * vi -
        iNa -
        iK +
        gL[i] * E_L_MV +
        gSynRev[i] +
        gSchedRevI +
        iExt;

      // Gating Euler step — uses the same V^n. Order-independent because
      // we read `v[i]` here (snapshot, not yet overwritten by back-sub).
      const aM = alphaM(vi);
      const bM = betaM(vi);
      const aH = alphaH(vi);
      const bH = betaH(vi);
      const aN = alphaN(vi);
      const bN = betaN(vi);
      m[i] += dt * (aM * (1 - m[i]) - bM * m[i]);
      h[i] += dt * (aH * (1 - h[i]) - bH * h[i]);
      n[i] += dt * (aN * (1 - n[i]) - bN * n[i]);
      if (m[i] < 0) m[i] = 0; else if (m[i] > 1) m[i] = 1;
      if (h[i] < 0) h[i] = 0; else if (h[i] > 1) h[i] = 1;
      if (n[i] < 0) n[i] = 0; else if (n[i] > 1) n[i] = 1;
    }

    // Forward elimination: walk leaves-to-root (descendants have higher
    // indices than ancestors in BFS order). Each step removes a child
    // from its parent's row, modifying diag[p] and rhs[p].
    for (let i = N - 1; i >= 1; i--) {
      const p = parents[i];
      if (p < 0) continue;
      const g = gAxToParent[i];
      const ratio = g / diag[i];
      diag[p] -= g * ratio;
      rhs[p] += ratio * rhs[i];
    }

    // Back-substitution: root first, then walk back out to leaves. Roots
    // get the trivial scalar solve; everything else uses its parent's
    // (already-solved) voltage.
    for (let i = 0; i < N; i++) {
      const p = parents[i];
      if (p < 0) {
        v[i] = rhs[i] / diag[i];
      } else {
        v[i] = (rhs[i] + gAxToParent[i] * v[p]) / diag[i];
      }
    }

    // Synaptic conductance decay.
    for (let i = 0; i < N; i++) {
      gSyn[i] *= decayPerStep;
      gSynRev[i] *= decayPerStep;
    }
  }

  if (onProgress) onProgress(1.0);

  return {
    voltageTrace,
    nFrames,
    nCompartments: N,
    frameTimesMs,
    aisSpikesMs,
    aisCompartment: aisIdx,
  };
}

/**
 * Coupled multi-neuron Hodgkin-Huxley solver — single-pass within-
 * neighborhood spike coupling.
 *
 * Why one function vs. spawning N workers in parallel: a worker can't
 * read another worker's spike events at integration cadence (every 25 μs)
 * because the `postMessage` round-trip is two orders of magnitude
 * slower. So if we want pre-target A's AIS spike to drive post-target
 * B's dendrite *at the same biological instant* — which is the whole
 * point of within-neighborhood coupling — we put A and B in the same
 * solver loop.
 *
 * Each step proceeds in this order:
 *
 *   1. For every target, read V^n at its AIS, compare with prev V^n.
 *      A rising-edge crossing → record an AIS spike for that target.
 *      For each `IntraEdge` whose `preTargetIdx` just spiked, add a
 *      conductance kick into the post target's per-compartment
 *      `gSynEvt` / `gSynRevEvt` accumulator.
 *   2. For every target, build its Hines diag/rhs from V^n + gating at
 *      V^n + smooth schedule contribution + event accumulator. Solve.
 *   3. For every target, decay the event accumulator by exp(−dt/τ_syn).
 *
 * Spike detection happens *before* any target's integration step, so
 * the kicks land in `gSynEvt` of every relevant post target before any
 * of them solve — V^n is consistent across the neighborhood at the
 * spike-detection boundary, the kicks affect every post target's
 * integration this step, and no target sees a half-updated mix of
 * V^n + V^{n+1} from its neighbors.
 */
export function runCoupledMcHH(
  targets: CoupledTarget[],
  intraEdges: CoupledIntraEdge[],
  config: HhRunConfig,
  onProgress: ((fraction: number) => void) | null = null,
): CoupledResult {
  const nTargets = targets.length;
  if (nTargets === 0) {
    return { results: [] };
  }

  const dt = config.dtMs;
  const totalSteps = Math.ceil(config.durationMs / dt);
  const stride = Math.max(1, config.frameStride);
  const nFrames = Math.floor(totalSteps / stride) + 1;
  const synTauDecay = config.synTauDecayMs ?? 5.0;
  const decayPerStep = Math.exp(-dt / synTauDecay);
  const spikeThreshold = config.spikeThresholdMv ?? -20;
  const minIsiMs = config.minIsiMs ?? 1.5;

  // Per-target state. Same machinery as `runMcHH`, replicated N times.
  type State = {
    morph: SolverMorphology;
    N: number;
    v: Float32Array;
    m: Float32Array;
    h: Float32Array;
    n: Float32Array;
    gSynEvt: Float32Array;
    gSynRevEvt: Float32Array;
    diag: Float32Array;
    rhs: Float32Array;
    cap: Float32Array;
    gNaMax: Float32Array;
    gKMax: Float32Array;
    gL: Float32Array;
    axialSum: Float32Array;
    gAxToParent: Float32Array;
    parents: Int32Array;
    aisIdx: number;
    schedule: GScheduleInput | null;
    voltageTrace: Float32Array;
    frameTimesMs: Float32Array;
    aisSpikesMs: number[];
    lastSpikeMs: number;
    prevAisV: number;
    nextFrame: number;
  };

  const RA_FACTOR_NS = (1000 * Math.PI) / 1.5;

  const states: State[] = targets.map((t) => {
    const morph = t.morph;
    const N = morph.nCompartments;
    const radiusUm = new Float32Array(N);
    const lengthUm = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      radiusUm[i] = morph.compartments[i].radiusNm / 1000;
      lengthUm[i] = morph.compartments[i].lengthNm / 1000;
    }
    const surfaceUm2 = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      surfaceUm2[i] = 2 * Math.PI * radiusUm[i] * lengthUm[i];
    }
    const gNaMax = new Float32Array(N);
    const gKMax = new Float32Array(N);
    const gL = new Float32Array(N);
    const cap = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const isAis = i === morph.aisCompartment;
      const naDens = G_NA_DEFAULT_MS_PER_CM2 * (isAis ? AIS_BOOST : 1);
      const kDens = G_K_DEFAULT_MS_PER_CM2 * (isAis ? AIS_BOOST : 1);
      gNaMax[i] = naDens * 0.01 * surfaceUm2[i];
      gKMax[i] = kDens * 0.01 * surfaceUm2[i];
      gL[i] = G_L_DEFAULT_MS_PER_CM2 * 0.01 * surfaceUm2[i];
      cap[i] = CM_UF_PER_CM2 * 0.01 * surfaceUm2[i];
    }
    const gAxToParent = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const p = morph.compartments[i].parent;
      if (p < 0) continue;
      const rMean = 0.5 * (radiusUm[i] + radiusUm[p]);
      const lMean = 0.5 * (lengthUm[i] + lengthUm[p]);
      gAxToParent[i] = (RA_FACTOR_NS * rMean * rMean) / Math.max(lMean, 1e-3);
    }
    const axialSum = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const p = morph.compartments[i].parent;
      if (p < 0) continue;
      axialSum[i] += gAxToParent[i];
      axialSum[p] += gAxToParent[i];
    }
    const parents = new Int32Array(N);
    for (let i = 0; i < N; i++) parents[i] = morph.compartments[i].parent;
    const aisIdx = morph.aisCompartment >= 0 ? morph.aisCompartment : morph.somaCompartment;
    return {
      morph,
      N,
      v: new Float32Array(N).fill(V_REST_MV),
      m: new Float32Array(N).fill(M_REST),
      h: new Float32Array(N).fill(H_REST),
      n: new Float32Array(N).fill(N_REST),
      gSynEvt: new Float32Array(N),
      gSynRevEvt: new Float32Array(N),
      diag: new Float32Array(N),
      rhs: new Float32Array(N),
      cap,
      gNaMax,
      gKMax,
      gL,
      axialSum,
      gAxToParent,
      parents,
      aisIdx,
      schedule: t.gSchedule,
      voltageTrace: new Float32Array(nFrames * N),
      frameTimesMs: new Float32Array(nFrames),
      aisSpikesMs: [] as number[],
      lastSpikeMs: -Infinity,
      prevAisV: V_REST_MV,
      nextFrame: 0,
    };
  });

  // Group intra-edges by pre target so the spike-detection step can
  // look up affected post targets in O(degree) rather than scanning all
  // edges.
  const edgesByPre: CoupledIntraEdge[][] = Array.from({ length: nTargets }, () => []);
  for (const e of intraEdges) edgesByPre[e.preTargetIdx].push(e);

  let nextProgress = performance.now();
  const reportEveryMs = 100;
  const dtRecip = 1.0 / dt;

  for (let step = 0; step <= totalSteps; step++) {
    const tMs = step * dt;

    // ---- 1. Spike detection (V^n) across every target. Each detected
    //         spike deposits kicks into post targets' event accumulators
    //         BEFORE any integration step starts, so V^n consistency
    //         holds for the kicks too.
    for (let t = 0; t < nTargets; t++) {
      const s = states[t];
      const aisV = s.v[s.aisIdx];
      if (
        aisV >= spikeThreshold &&
        s.prevAisV < spikeThreshold &&
        tMs - s.lastSpikeMs > minIsiMs
      ) {
        s.aisSpikesMs.push(tMs);
        s.lastSpikeMs = tMs;
        const outs = edgesByPre[t];
        for (let e = 0; e < outs.length; e++) {
          const edge = outs[e];
          const post = states[edge.postTargetIdx];
          post.gSynEvt[edge.postCompartment] += edge.gPeakNs;
          post.gSynRevEvt[edge.postCompartment] += edge.gPeakNs * edge.eRevMv;
        }
      }
      s.prevAisV = aisV;
    }

    // ---- 2. Frame snapshot per target.
    if (step % stride === 0) {
      for (let t = 0; t < nTargets; t++) {
        const s = states[t];
        if (s.nextFrame < nFrames) {
          s.voltageTrace.set(s.v, s.nextFrame * s.N);
          s.frameTimesMs[s.nextFrame] = tMs;
          s.nextFrame++;
        }
      }
      if (onProgress && performance.now() > nextProgress) {
        onProgress(step / totalSteps);
        nextProgress = performance.now() + reportEveryMs;
      }
    }

    // ---- 3. Per-target Hines integration. Each target's system is
    //         independent (block-diagonal across the multi-neuron
    //         linear system, since coupling enters through the
    //         conductance accumulators only, not via axial currents).
    for (let t = 0; t < nTargets; t++) {
      const s = states[t];
      const N = s.N;
      const v = s.v;
      const m = s.m;
      const h = s.h;
      const nGate = s.n;
      const diag = s.diag;
      const rhs = s.rhs;
      const cap = s.cap;
      const gNaMax = s.gNaMax;
      const gKMax = s.gKMax;
      const gL = s.gL;
      const axialSum = s.axialSum;
      const gAxToParent = s.gAxToParent;
      const parents = s.parents;
      const gSynEvt = s.gSynEvt;
      const gSynRevEvt = s.gSynRevEvt;
      const sched = s.schedule;
      let schedFrame = -1;
      if (sched !== null) {
        schedFrame = Math.min(sched.nFrames - 1, Math.floor(tMs / sched.dtMs));
      }

      for (let i = 0; i < N; i++) {
        const vi = v[i];
        const iNa = gNaMax[i] * m[i] * m[i] * m[i] * h[i] * (vi - E_NA_MV);
        const iK = gKMax[i] * nGate[i] * nGate[i] * nGate[i] * nGate[i] * (vi - E_K_MV);
        let gSchedI = 0;
        let gSchedRevI = 0;
        if (sched !== null && schedFrame >= 0) {
          const so = schedFrame * sched.nCompartments + i;
          gSchedI = sched.gSyn[so];
          gSchedRevI = sched.gSynRev[so];
        }
        diag[i] =
          cap[i] * dtRecip + axialSum[i] + gL[i] + gSynEvt[i] + gSchedI;
        rhs[i] =
          cap[i] * dtRecip * vi -
          iNa -
          iK +
          gL[i] * E_L_MV +
          gSynRevEvt[i] +
          gSchedRevI;

        const aM = alphaM(vi);
        const bM = betaM(vi);
        const aH = alphaH(vi);
        const bH = betaH(vi);
        const aN = alphaN(vi);
        const bN = betaN(vi);
        m[i] += dt * (aM * (1 - m[i]) - bM * m[i]);
        h[i] += dt * (aH * (1 - h[i]) - bH * h[i]);
        nGate[i] += dt * (aN * (1 - nGate[i]) - bN * nGate[i]);
        if (m[i] < 0) m[i] = 0; else if (m[i] > 1) m[i] = 1;
        if (h[i] < 0) h[i] = 0; else if (h[i] > 1) h[i] = 1;
        if (nGate[i] < 0) nGate[i] = 0; else if (nGate[i] > 1) nGate[i] = 1;
      }

      // Forward elimination (leaves → root).
      for (let i = N - 1; i >= 1; i--) {
        const p = parents[i];
        if (p < 0) continue;
        const g = gAxToParent[i];
        const ratio = g / diag[i];
        diag[p] -= g * ratio;
        rhs[p] += ratio * rhs[i];
      }
      // Back-substitution.
      for (let i = 0; i < N; i++) {
        const p = parents[i];
        if (p < 0) {
          v[i] = rhs[i] / diag[i];
        } else {
          v[i] = (rhs[i] + gAxToParent[i] * v[p]) / diag[i];
        }
      }
    }

    // ---- 4. Decay the event accumulators on every target. (The
    //         schedule contribution doesn't decay — it's sampled fresh
    //         each step.)
    for (let t = 0; t < nTargets; t++) {
      const s = states[t];
      const N = s.N;
      const gSynEvt = s.gSynEvt;
      const gSynRevEvt = s.gSynRevEvt;
      for (let i = 0; i < N; i++) {
        gSynEvt[i] *= decayPerStep;
        gSynRevEvt[i] *= decayPerStep;
      }
    }
  }

  if (onProgress) onProgress(1.0);

  return {
    results: states.map((s) => ({
      voltageTrace: s.voltageTrace,
      nFrames: s.nextFrame,
      nCompartments: s.N,
      frameTimesMs: s.frameTimesMs,
      aisSpikesMs: s.aisSpikesMs,
      aisCompartment: s.aisIdx,
    })),
  };
}
