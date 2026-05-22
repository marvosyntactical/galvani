"""Single-compartment Hodgkin-Huxley simulator.

Full HH dynamics: membrane voltage `v` plus three gating variables
`m, h, n` for sodium activation, sodium inactivation, and potassium
activation. Four ODEs per neuron, ~3-5x slower than LIF/AdEx.

    C * dv/dt = -(g_Na * m^3 * h * (v - E_Na) +
                  g_K * n^4 * (v - E_K) +
                  g_L * (v - E_L)) + I_syn + I_ext
    dm/dt = alpha_m(v) * (1-m) - beta_m(v) * m       (likewise h, n)

This is the textbook model from Hodgkin & Huxley (1952). Defaults are
the squid giant axon parameters they fit, scaled to dimensionless units
for compatibility with the Galvani ModelSpec convention.

For multi-compartmental work, use the [brian] extra and Brian2's NeuronGroup
with HH equations. This module is the lightweight in-pipeline option.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from galvani.model.spec import ModelSpec

Stimulus = Callable[[float], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class HHResult:
    """Output of `simulate_hh`."""

    times: NDArray[np.float64]
    voltages: NDArray[np.float64]
    rates: NDArray[np.float64]
    spike_trains: list[list[float]]
    spec: ModelSpec
    dt: float

    @property
    def neuron_ids(self) -> tuple[int, ...]:
        return self.spec.neuron_ids


# Original Hodgkin-Huxley squid axon parameters (mV, mS/cm^2):
#   E_Na = 50, E_K = -77, E_L = -54.4
#   g_Na = 120, g_K = 36, g_L = 0.3
# We re-scale into arbitrary units centred on 0 so the same ModelSpec
# scale that runs the rate/LIF/AdEx backends also runs HH. Spike
# threshold ~ +0.4 in these units. dt at 0.05 ms is comfortable.


def _ab_m(v: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vv = (v - 0.0) * 25.0  # rescale to mV-ish range
    am = 0.1 * (25 - vv) / (np.exp((25 - vv) / 10) - 1 + 1e-9)
    bm = 4.0 * np.exp(-vv / 18)
    return am, bm


def _ab_h(v: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vv = (v - 0.0) * 25.0
    ah = 0.07 * np.exp(-vv / 20)
    bh = 1.0 / (np.exp((30 - vv) / 10) + 1)
    return ah, bh


def _ab_n(v: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vv = (v - 0.0) * 25.0
    an = 0.01 * (10 - vv) / (np.exp((10 - vv) / 10) - 1 + 1e-9)
    bn = 0.125 * np.exp(-vv / 80)
    return an, bn


def simulate_hh(
    spec: ModelSpec,
    duration: float,
    stimulus: Stimulus | None = None,
    *,
    g_na: float = 120.0,
    g_k: float = 36.0,
    g_l: float = 0.3,
    e_na: float = 2.0,
    e_k: float = -3.1,
    e_l: float = -0.55,
    spike_threshold: float = 0.4,
    syn_tau: float = 0.005,
    noise_sigma: float = 0.0,
    rng_seed: int = 0,
    dt: float = 5e-5,
) -> HHResult:
    """Forward-Euler HH integration.

    `noise_sigma > 0` adds Gaussian noise to v (stochastic HH).
    `dt` must be quite small (~50 μs) for HH stability. With 130 neurons
    and 1 s duration that's 20000 steps and ~30M flops -- still fast in
    NumPy.
    """
    n = spec.n_neurons
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if duration < 0:
        raise ValueError("duration must be non-negative.")

    rng = np.random.default_rng(rng_seed)
    n_steps = int(np.ceil(duration / dt)) + 1
    times = np.arange(n_steps, dtype=np.float64) * dt
    voltages = np.zeros((n_steps, n), dtype=np.float64)
    rates = np.zeros((n_steps, n), dtype=np.float64)
    spikes: list[list[float]] = [[] for _ in range(n)]

    weights = spec.global_gain * spec.weights
    bias = spec.bias

    v = voltages[0].copy()
    # Initialise gating to steady state at v=0.
    am, bm = _ab_m(v)
    ah, bh = _ab_h(v)
    an, bn = _ab_n(v)
    m = am / (am + bm)
    h = ah / (ah + bh)
    nn = an / (an + bn)

    syn = np.zeros(n, dtype=np.float64)
    last_spike = np.full(n, -np.inf, dtype=np.float64)
    last_isi = np.full(n, np.inf, dtype=np.float64)
    above = np.zeros(n, dtype=bool)

    stim = stimulus or (lambda _t: np.zeros(n, dtype=np.float64))
    sqrt_dt = float(np.sqrt(dt))

    for step in range(1, n_steps):
        t = times[step - 1]
        syn += -dt * syn / syn_tau
        drive = weights @ syn + bias + stim(t)

        am, bm = _ab_m(v)
        ah, bh = _ab_h(v)
        an, bn = _ab_n(v)
        m = m + dt * (am * (1 - m) - bm * m)
        h = h + dt * (ah * (1 - h) - bh * h)
        nn = nn + dt * (an * (1 - nn) - bn * nn)

        i_na = g_na * m**3 * h * (v - e_na)
        i_k = g_k * nn**4 * (v - e_k)
        i_l = g_l * (v - e_l)
        dv = -(i_na + i_k + i_l) + drive
        if noise_sigma > 0:
            dv = dv + noise_sigma * rng.standard_normal(n) * sqrt_dt / dt
        v = v + dt * dv

        # Spike detection on threshold crossing (rising edge).
        new_above = v >= spike_threshold
        spiked = new_above & ~above
        above = new_above
        if spiked.any():
            for j in np.flatnonzero(spiked):
                jt = float(times[step])
                if last_spike[j] > -np.inf:
                    last_isi[j] = jt - last_spike[j]
                last_spike[j] = jt
                spikes[j].append(jt)
                syn[j] += 1.0

        voltages[step] = v
        with np.errstate(divide="ignore"):
            r_est = np.where(np.isfinite(last_isi), 1.0 / last_isi, 0.0)
        decay = np.exp(-(t - last_spike) / max(syn_tau * 4, 0.02))
        rates[step] = r_est * decay

    return HHResult(
        times=times,
        voltages=voltages,
        rates=rates,
        spike_trains=spikes,
        spec=spec,
        dt=dt,
    )


__all__ = ["HHResult", "simulate_hh"]
