"""NumPy rate-model simulator (default backend).

Solves:

    tau_i * dr_i/dt = -r_i + phi(sum_j W_ij r_j + I_i(t) + b_i)

with `W = global_gain * weights`, using explicit forward Euler at a fixed
`dt = 0.5 ms`. Fast for HD-ring-sized networks (~100 neurons) on a laptop;
no compilation overhead.

The Brian2 backend (Phase 4 extension, opt-in) is deferred: per
DESIGN_DECISIONS, NumPy is what users will actually run in a notebook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from galvani.model.spec import ModelSpec

Stimulus = Callable[[float], NDArray[np.float64]]
"""A stimulus is a function `t -> I_per_neuron` (length N, units of input)."""

Activation = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""An activation phi: vectorized NumPy callable. Defaults to `relu`."""


def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard rectified linear activation. The HD-ring literature uses this
    (or tanh) on rate units; relu is the simpler choice and easier to reason
    about during validation."""
    return np.maximum(x, 0.0)


def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bounded saturating activation. Useful when relu lets activity blow up."""
    return np.tanh(x)


@dataclass(frozen=True, slots=True)
class Result:
    """Output of `simulate`.

    `rates` is `(T, N)`: rows are timesteps, columns are neurons (canonical
    ordering matching `spec.neuron_ids`).
    """

    times: NDArray[np.float64]
    rates: NDArray[np.float64]
    spec: ModelSpec
    dt: float

    @property
    def neuron_ids(self) -> tuple[int, ...]:
        return self.spec.neuron_ids


def zero_stimulus(n: int) -> Stimulus:
    """Stimulus that is zero everywhere. Useful for free-evolution / persistence
    tests after an initial transient."""
    zeros = np.zeros(n, dtype=np.float64)
    return lambda _t: zeros


def simulate(
    spec: ModelSpec,
    duration: float,
    stimulus: Stimulus | None = None,
    initial_rates: NDArray[np.float64] | None = None,
    activation: Activation = relu,
    dt: float = 5e-4,
) -> Result:
    """Forward-Euler integration of the rate model.

    Args:
        spec: ModelSpec with weights/tau/bias/global_gain.
        duration: total simulated time, seconds.
        stimulus: callable `t -> I (N,)`. If None, zero input.
        initial_rates: starting rates of shape `(N,)`. Defaults to zeros.
        activation: nonlinearity. Defaults to `relu`.
        dt: integrator step in seconds. Default 0.5 ms. Must be smaller than
            the smallest tau for stability.
    """
    n = spec.n_neurons
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if duration < 0:
        raise ValueError("duration must be non-negative.")
    if dt > float(spec.tau.min()):
        raise ValueError(
            f"dt={dt} is larger than min tau={float(spec.tau.min())}; Euler step will be unstable."
        )

    n_steps = int(np.ceil(duration / dt)) + 1
    times = np.arange(n_steps, dtype=np.float64) * dt
    rates = np.zeros((n_steps, n), dtype=np.float64)

    if initial_rates is not None:
        if initial_rates.shape != (n,):
            raise ValueError(f"initial_rates must have shape ({n},); got {initial_rates.shape}")
        rates[0] = initial_rates

    stim = stimulus or zero_stimulus(n)
    weights = spec.global_gain * spec.weights
    tau = spec.tau
    bias = spec.bias

    r = rates[0].copy()
    for step in range(1, n_steps):
        t = times[step - 1]
        drive = weights @ r + stim(t) + bias
        dr = (-r + activation(drive)) / tau
        r = r + dt * dr
        rates[step] = r

    return Result(times=times, rates=rates, spec=spec, dt=dt)


__all__ = [
    "Activation",
    "Result",
    "Stimulus",
    "relu",
    "simulate",
    "tanh",
    "zero_stimulus",
]
