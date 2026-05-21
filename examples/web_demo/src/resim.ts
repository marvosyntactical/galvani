/**
 * In-browser rate-model resimulation.
 *
 * Mirrors `galvani.model.rate.simulate`: forward Euler integration of
 *   tau_i * dr_i/dt = -r_i + phi(global_gain * W @ r + I + b)
 *
 * Uses the model and stim arrays already baked into the payload.
 * Designed to finish in under a second for 130-neuron HD ring at
 * dt = 1 ms, ~100 output frames.
 */

import type { Payload } from "./payload";

type Activation = (x: number) => number;

const ACTIVATIONS: Record<string, Activation> = {
  tanh: Math.tanh,
  relu: (x: number) => (x > 0 ? x : 0),
  LIF: Math.tanh, // fallback for LIF scenarios when the user drags the slider
};

export interface ResimResult {
  /** (n_frames, n_neurons) */
  rates: number[][];
  /** (n_frames,) */
  times: number[];
}

export function canResim(p: Payload): boolean {
  return Boolean(
    (p as Payload & { model?: unknown }).model &&
      p.stim_signal &&
      p.metadata.hyperparams.activation,
  );
}

/**
 * Re-run the rate model on the same stim with a different `global_gain`.
 *
 * The baked stim_signal is downsampled (~120 frames over 4 s = 33 ms per
 * frame). For a faithful resim we interpolate the stim to the integration
 * dt and downsample the output back to the original n_frames.
 */
export function resimulateWithGain(
  payload: Payload,
  newGain: number,
  dtIntegrate: number = 1e-3,
): ResimResult {
  const model = (payload as Payload & {
    model?: { weights: number[][]; tau: number[]; bias: number[]; global_gain: number };
  }).model;
  if (!model) throw new Error("Payload has no model; cannot re-simulate.");
  if (!payload.stim_signal) throw new Error("Payload has no stim_signal.");

  const N = payload.metadata.n_neurons;
  const T = payload.metadata.duration;
  const nIntSteps = Math.max(1, Math.round(T / dtIntegrate));
  const nOutFrames = payload.metadata.n_frames;

  const W = model.weights; // (N, N)
  const tau = model.tau;
  const bias = model.bias;
  const activation =
    ACTIVATIONS[payload.metadata.hyperparams.activation as string] ?? Math.tanh;

  // Frame schedule that produces nOutFrames evenly across [0, T].
  const outIndices = new Set<number>();
  for (let f = 0; f < nOutFrames; f++) {
    outIndices.add(Math.round((f * (nIntSteps - 1)) / Math.max(1, nOutFrames - 1)));
  }

  // Pre-compute per-frame stim signal interpolated to integration steps.
  const stimSignal = payload.stim_signal; // (n_out, N)
  const nStimFrames = stimSignal.length;

  function stimAtStep(step: number): number[] {
    const t = (step / Math.max(1, nIntSteps - 1)) * (nStimFrames - 1);
    const i0 = Math.floor(t);
    const i1 = Math.min(nStimFrames - 1, i0 + 1);
    const a = t - i0;
    const out = new Array(N);
    const row0 = stimSignal[i0];
    const row1 = stimSignal[i1];
    for (let i = 0; i < N; i++) {
      out[i] = row0[i] * (1 - a) + row1[i] * a;
    }
    return out;
  }

  const rates = new Float64Array(N);
  const drive = new Float64Array(N);
  const outRates: number[][] = [];
  const outTimes: number[] = [];

  for (let step = 0; step < nIntSteps; step++) {
    // drive = newGain * W @ rates + stim + bias
    const stim = stimAtStep(step);
    for (let i = 0; i < N; i++) {
      let acc = 0;
      const Wi = W[i];
      for (let j = 0; j < N; j++) acc += Wi[j] * rates[j];
      drive[i] = newGain * acc + stim[i] + bias[i];
    }
    // Forward Euler: r += dt * (-r + phi(drive)) / tau
    for (let i = 0; i < N; i++) {
      rates[i] += (dtIntegrate * (-rates[i] + activation(drive[i]))) / tau[i];
    }
    if (outIndices.has(step)) {
      outRates.push(Array.from(rates));
      outTimes.push((step / Math.max(1, nIntSteps - 1)) * T);
    }
  }

  // Make sure we hit nOutFrames exactly.
  while (outRates.length < nOutFrames) {
    outRates.push(Array.from(rates));
    outTimes.push(T);
  }
  outRates.length = nOutFrames;
  outTimes.length = nOutFrames;

  return { rates: outRates, times: outTimes };
}
