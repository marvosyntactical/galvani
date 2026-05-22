"""Adaptive exponential integrate-and-fire (AdEx) backend.

A step up from plain LIF: the membrane voltage `v` is augmented with an
adaptation current `w` that builds up with spiking and lets the cell
exhibit spike-frequency adaptation, bursting, and other realistic firing
patterns. Two ODEs per neuron, still very cheap.

    C * dv/dt = -gL*(v - v_rest) + gL*delta_T*exp((v - v_T)/delta_T)
                - w + I_syn + I_ext
    tau_w * dw/dt = a*(v - v_rest) - w

When v >= v_threshold:  v := v_reset, w := w + b   (refractory)

Reference: Brette & Gerstner (2005), J Neurophysiol.

Defaults (cortical pyramidal cell)::

    C            = 200 pF (we use unitless / scaled)
    gL           = 10 nS
    v_rest       = -70 mV  (we use 0 in arbitrary units)
    v_threshold  = -50 mV  (we use 1.0)
    v_reset      = -55 mV  (we use 0.0)
    v_T          = -50 mV  (soft threshold)
    delta_T      = 2 mV
    tau_w        = 144 ms
    a            = 4 nS     (subthreshold coupling -- 0 turns off adaptation)
    b            = 80.5 pA  (spike-triggered jump -- the bursting knob)

We use arbitrary normalised units so the values look like (1.0, 0.0, 0.5)
not (-50, -70, 4). The dynamics are qualitatively equivalent. For
brian2-level realism, use the [brian] extra and pass real units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from galvani.model.spec import ModelSpec

Stimulus = Callable[[float], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class AdExResult:
    """Output of `simulate_adex`."""

    times: NDArray[np.float64]
    voltages: NDArray[np.float64]
    adaptations: NDArray[np.float64]
    rates: NDArray[np.float64]
    spike_trains: list[list[float]]
    spec: ModelSpec
    dt: float

    @property
    def neuron_ids(self) -> tuple[int, ...]:
        return self.spec.neuron_ids


def simulate_adex(
    spec: ModelSpec,
    duration: float,
    stimulus: Stimulus | None = None,
    *,
    v_rest: float = 0.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
    v_t: float = 0.85,
    delta_t: float = 0.08,
    tau_w: float = 0.15,
    a: float = 0.04,
    b: float = 0.20,
    t_refractory: float = 0.002,
    syn_tau: float = 0.005,
    noise_sigma: float = 0.0,
    rng_seed: int = 0,
    dt: float = 2e-4,
) -> AdExResult:
    """Forward-Euler AdEx integration.

    `noise_sigma > 0` adds Gaussian white noise to v, turning the model
    into the *stochastic AdEx*. Useful for showing spike-time variability.
    """
    n = spec.n_neurons
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if duration < 0:
        raise ValueError("duration must be non-negative.")
    if dt > syn_tau:
        raise ValueError(f"dt={dt} > syn_tau={syn_tau}; synaptic filter unstable.")

    rng = np.random.default_rng(rng_seed)
    n_steps = int(np.ceil(duration / dt)) + 1
    times = np.arange(n_steps, dtype=np.float64) * dt
    voltages = np.full((n_steps, n), v_rest, dtype=np.float64)
    adaptations = np.zeros((n_steps, n), dtype=np.float64)
    rates = np.zeros((n_steps, n), dtype=np.float64)
    spikes: list[list[float]] = [[] for _ in range(n)]

    weights = spec.global_gain * spec.weights
    tau = spec.tau
    bias = spec.bias

    v = voltages[0].copy()
    w = np.zeros(n, dtype=np.float64)
    syn = np.zeros(n, dtype=np.float64)
    last_spike = np.full(n, -np.inf, dtype=np.float64)
    last_isi = np.full(n, np.inf, dtype=np.float64)

    stim = stimulus or (lambda _t: np.zeros(n, dtype=np.float64))

    sqrt_dt = float(np.sqrt(dt))

    for step in range(1, n_steps):
        t = times[step - 1]
        syn += -dt * syn / syn_tau
        not_ref = (t - last_spike) >= t_refractory
        drive = weights @ syn + bias + stim(t)

        # AdEx voltage equation. tau plays the role of C/gL here in arbitrary
        # units so the same ModelSpec serves all simulators consistently.
        exp_term = delta_t * np.exp(np.clip((v - v_t) / delta_t, -50.0, 50.0))
        dv = (-(v - v_rest) + exp_term - w + drive) / tau
        dw = (a * (v - v_rest) - w) / tau_w

        if noise_sigma > 0:
            dv = dv + noise_sigma * rng.standard_normal(n) * sqrt_dt / dt

        v[not_ref] = v[not_ref] + dt * dv[not_ref]
        w = w + dt * dw

        spiked = v >= v_threshold
        if spiked.any():
            for j in np.flatnonzero(spiked):
                jt = float(times[step])
                if last_spike[j] > -np.inf:
                    last_isi[j] = jt - last_spike[j]
                last_spike[j] = jt
                spikes[j].append(jt)
            v[spiked] = v_reset
            w[spiked] = w[spiked] + b
            syn[spiked] += 1.0

        voltages[step] = v
        adaptations[step] = w
        with np.errstate(divide="ignore"):
            r_est = np.where(np.isfinite(last_isi), 1.0 / last_isi, 0.0)
        decay = np.exp(-(t - last_spike) / max(syn_tau * 4, 0.02))
        rates[step] = r_est * decay

    return AdExResult(
        times=times,
        voltages=voltages,
        adaptations=adaptations,
        rates=rates,
        spike_trains=spikes,
        spec=spec,
        dt=dt,
    )


__all__ = ["AdExResult", "simulate_adex"]
