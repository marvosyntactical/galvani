"""Leaky integrate-and-fire (LIF) biophysical simulator backend.

A minimal-dependency biophysical alternative to `model.rate`. Same `ModelSpec`
input -- `weights`, `tau`, `bias`, `global_gain` -- now interpreted at the
membrane-voltage level. Each neuron has a voltage `v` that integrates input:

    tau_i * dv_i/dt = -(v_i - v_rest) + R * (sum_j W_ij * f_j(t) + I_i(t) + b_i)

where `f_j(t)` is the post-synaptic conductance traced by an exponential
filter on neuron j's recent spikes. When `v_i` crosses `v_threshold`, neuron
i emits a spike, `v_i` is reset to `v_reset`, and i enters a refractory
period of `t_refractory` during which it doesn't integrate.

Defaults (typical fly central-complex values):
  v_rest        = 0.0  (arbitrary units)
  v_threshold   = 1.0
  v_reset       = 0.0
  t_refractory  = 2 ms
  syn_tau       = 5 ms  (exponential synaptic filter)
  R             = 1.0  (membrane resistance)

This is the simplest spiking model with the right qualitative behaviour
(spikes, refractoriness, leaky integration). Hodgkin-Huxley would add ion
channels and is left to a v2 follow-up.

This is implemented in pure NumPy (no Brian2 dependency) so it works in CI
without extra installs. For high-fidelity work, the `[brian]` extra still
exists; this is the lightweight option.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from galvani.model.spec import ModelSpec

Stimulus = Callable[[float], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class LIFResult:
    """Output of `simulate_lif`.

    `voltages`: shape (T, N), membrane voltages over time.
    `spike_trains`: list of length N, each entry a sorted list of spike times.
    `rates`: shape (T, N), instantaneous estimated rate (1 / inter-spike-interval
    or 0 between spikes). For viz parity with the rate-model `Result`.
    `times`: shape (T,).
    """

    times: NDArray[np.float64]
    voltages: NDArray[np.float64]
    rates: NDArray[np.float64]
    spike_trains: list[list[float]]
    spec: ModelSpec
    dt: float

    @property
    def neuron_ids(self) -> tuple[int, ...]:
        return self.spec.neuron_ids


def simulate_lif(
    spec: ModelSpec,
    duration: float,
    stimulus: Stimulus | None = None,
    *,
    v_rest: float = 0.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
    t_refractory: float = 0.002,
    syn_tau: float = 0.005,
    R: float = 1.0,
    dt: float = 5e-4,
) -> LIFResult:
    """Forward-Euler LIF integration.

    Stable when `dt < min(tau, syn_tau, t_refractory)`. With the v1 defaults
    that means `dt < 2 ms` -- 0.5 ms is comfortable.
    """
    n = spec.n_neurons
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if duration < 0:
        raise ValueError("duration must be non-negative.")
    if dt > float(spec.tau.min()):
        raise ValueError(f"dt={dt} > min tau={float(spec.tau.min())}; integration unstable.")
    if dt > syn_tau:
        raise ValueError(f"dt={dt} > syn_tau={syn_tau}; synaptic filter unstable.")

    n_steps = int(np.ceil(duration / dt)) + 1
    times = np.arange(n_steps, dtype=np.float64) * dt
    voltages = np.full((n_steps, n), v_rest, dtype=np.float64)
    rates = np.zeros((n_steps, n), dtype=np.float64)
    spikes: list[list[float]] = [[] for _ in range(n)]

    weights = spec.global_gain * spec.weights
    tau = spec.tau
    bias = spec.bias

    v = voltages[0].copy()
    syn = np.zeros(n, dtype=np.float64)  # synaptic conductance trace
    last_spike = np.full(n, -np.inf, dtype=np.float64)

    stim = stimulus or (lambda _t: np.zeros(n, dtype=np.float64))
    last_isi = np.full(n, np.inf, dtype=np.float64)

    for step in range(1, n_steps):
        t = times[step - 1]
        # Synaptic conductance decays exponentially between spikes.
        syn += -dt * syn / syn_tau
        # Refractory mask.
        not_ref = (t - last_spike) >= t_refractory
        drive = weights @ syn + bias + stim(t)
        dv = (-(v - v_rest) + R * drive) / tau
        v[not_ref] += dt * dv[not_ref]

        # Spike detection.
        spiked = v >= v_threshold
        if spiked.any():
            for j in np.flatnonzero(spiked):
                jt = float(times[step])
                if last_spike[j] > -np.inf:
                    last_isi[j] = jt - last_spike[j]
                last_spike[j] = jt
                spikes[j].append(jt)
            v[spiked] = v_reset
            syn[spiked] += 1.0  # add unit conductance kick from this spike

        voltages[step] = v
        # Instantaneous rate estimate: 1 / last ISI (Hz), 0 if never spiked.
        with np.errstate(divide="ignore"):
            r_est = np.where(np.isfinite(last_isi), 1.0 / last_isi, 0.0)
        # Decay rate estimate after refractory window so it returns to 0 if
        # the neuron stops firing.
        decay = np.exp(-(t - last_spike) / max(syn_tau * 4, 0.02))
        rates[step] = r_est * decay

    return LIFResult(
        times=times,
        voltages=voltages,
        rates=rates,
        spike_trains=spikes,
        spec=spec,
        dt=dt,
    )


__all__ = ["LIFResult", "simulate_lif"]
