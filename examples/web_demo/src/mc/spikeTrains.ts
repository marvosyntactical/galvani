/**
 * Convert an outer-sim payload's per-frame activity into per-neuron spike
 * trains, ready to be routed onto post-synaptic compartments.
 *
 * The four outer models we ship — rate, LIF, AdEx, HH — all bake their
 * output into `payload.rates` as a (T, N) float array. For the spiking
 * backends this is a smoothed spike-density estimate (events / window);
 * for the rate model it's the rate variable directly, already in [0, 1].
 *
 * Either way we treat `rates[t, neuron]` as a proxy for instantaneous
 * firing rate and sample spikes via an inhomogeneous Poisson process at
 * a fine dt (~0.5 ms by default). The sampling is seeded so re-runs are
 * deterministic — a property the SNV detail view relies on for caching.
 *
 * The conversion is unitful: outer rates are dimensionless [0, 1]; we
 * multiply by `peakHz` (default 50 Hz) to get instantaneous Hz. The
 * scaling factor is arbitrary and exposed as a config, because the outer
 * sim doesn't define a "1 Hz" anchor anywhere.
 */

import type { Payload } from "../payload";

export interface SpikeTrainConfig {
  /** Outer-sim "rate = 1.0" maps to this many Hz. Default 50. */
  peakHz?: number;
  /** Sampling step for the Poisson process, in ms. Smaller = more
   *  events but also more accurate (events per dt should be << 1). */
  dtMs?: number;
  /** PRNG seed. Same seed → same spike trains, frame-by-frame. */
  seed?: number;
}

/**
 * Spike times in ms, per neuron index (length = payload.metadata.n_neurons).
 * Element `i` is the sorted array of spike times for neuron i.
 */
export type SpikeTrainBundle = Float32Array[];

/** xorshift32 — small, deterministic, plenty good enough for Poisson. */
function makeRng(seed: number): () => number {
  let state = seed | 0 || 0x12345678;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    // Convert to a uniform [0, 1).
    return ((state >>> 0) % 0xffffffff) / 0xffffffff;
  };
}

/** Linear interpolation across the rate trace at a given simulated time. */
function rateAt(rates: number[][], times: number[], tMs: number): number[] {
  const tSec = tMs / 1000;
  if (tSec <= times[0]) return rates[0];
  if (tSec >= times[times.length - 1]) return rates[rates.length - 1];
  // Binary search for the right frame.
  let lo = 0;
  let hi = times.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= tSec) lo = mid;
    else hi = mid;
  }
  const f = (tSec - times[lo]) / (times[hi] - times[lo]);
  const left = rates[lo];
  const right = rates[hi];
  const out = new Array<number>(left.length);
  for (let i = 0; i < left.length; i++) {
    out[i] = left[i] + f * (right[i] - left[i]);
  }
  return out;
}

/**
 * Build a spike-train bundle for the whole population covered by the
 * payload. Each neuron's train is independently sampled.
 *
 * Pass `restrictTo` to only generate trains for a subset of neuron indices
 * — useful when the bio neighborhood only needs ~20 cells out of 130.
 */
export function buildSpikeTrains(
  payload: Payload,
  config: SpikeTrainConfig = {},
  restrictTo?: ReadonlyArray<number>,
): SpikeTrainBundle {
  const peakHz = config.peakHz ?? 50;
  const dtMs = config.dtMs ?? 0.5;
  const seed = config.seed ?? 1;
  const rng = makeRng(seed);

  const N = payload.metadata.n_neurons;
  const durationMs = payload.metadata.duration * 1000;
  const nSteps = Math.ceil(durationMs / dtMs);
  const wanted = new Set<number>(
    restrictTo ?? Array.from({ length: N }, (_, i) => i),
  );

  // Spike events accumulate into per-neuron arrays.
  const spikesPerNeuron: number[][] = Array.from({ length: N }, () => []);

  const lastSampleTimeMs = new Array<number>(N).fill(-Infinity);
  const refractoryMs = 2.0; // ad-hoc post-Poisson refractoriness

  for (let step = 0; step < nSteps; step++) {
    const tMs = step * dtMs;
    const r = rateAt(payload.rates, payload.times, tMs);
    for (const i of wanted) {
      const rateHz = Math.max(0, r[i] * peakHz);
      // P(spike in dt) = 1 - exp(-rate · dt). For rate · dt << 1 this is
      // approximately rate · dt; we use the closed form for safety.
      const p = 1 - Math.exp(-(rateHz / 1000) * dtMs);
      if (rng() < p && tMs - lastSampleTimeMs[i] >= refractoryMs) {
        spikesPerNeuron[i].push(tMs);
        lastSampleTimeMs[i] = tMs;
      }
    }
  }

  const bundle: SpikeTrainBundle = new Array(N);
  for (let i = 0; i < N; i++) {
    bundle[i] = new Float32Array(spikesPerNeuron[i]);
  }
  return bundle;
}

/** Returns the synapse-count weight from pre → post under the outer
 *  model's `W` matrix. We use `Math.abs(W[post, pre])` as the per-spike
 *  conductance magnitude; the sign comes from the neurotransmitter. */
export function preToPostWeight(
  payload: Payload,
  postNeuronIdx: number,
  preNeuronIdx: number,
): number {
  const W = payload.model?.weights;
  if (!W) return 0;
  return Math.abs(W[postNeuronIdx][preNeuronIdx]);
}
