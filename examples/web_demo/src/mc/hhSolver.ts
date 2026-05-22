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
