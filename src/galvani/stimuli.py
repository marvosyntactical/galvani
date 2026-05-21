"""Stimulus utilities for ring-attractor simulations.

A stimulus is just a callable `t -> I (N,)` consumed by `model.rate.simulate`.
These constructors return ready-to-pass callables; combining them is just
function composition by the user.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from galvani.model.rate import Stimulus


def _gaussian_on_circle(
    angles: NDArray[np.float64],
    center: float,
    width: float,
) -> NDArray[np.float64]:
    """A wrapped-Gaussian bump over the angular variable.

    Uses cos(angle - center) as the angular distance proxy. `width` is in
    radians and controls how sharp the bump is.
    """
    if width <= 0:
        raise ValueError("width must be positive.")
    diff = angles - center
    # Map angular distance to a scalar that peaks at center: 1 - cos(d) is
    # 0 at center and 2 at the antipode. Equivalent to a periodic Gaussian
    # to leading order, without needing to handle the wraparound.
    return np.exp(-(1.0 - np.cos(diff)) / (width * width))


def ring_stimulus(
    angles: NDArray[np.float64],
    *,
    center: float,
    width: float = 0.5,
    amplitude: float = 1.0,
) -> Stimulus:
    """Stationary Gaussian bump centered at `center` (radians).

    Args:
        angles: per-neuron angular positions of length N. NaN entries get
            zero stimulus (e.g. Delta7 in HD-ring layouts).
        center: bump center angle in radians.
        width: angular width in radians (standard-deviation-like).
        amplitude: peak input current.
    """
    valid = ~np.isnan(angles)
    pattern = np.zeros_like(angles)
    pattern[valid] = amplitude * _gaussian_on_circle(angles[valid], center, width)
    pattern_const = pattern.astype(np.float64)

    def _stim(_t: float) -> NDArray[np.float64]:
        return pattern_const

    return _stim


def rotating_stimulus(
    angles: NDArray[np.float64],
    *,
    omega: float,
    width: float = 0.5,
    amplitude: float = 1.0,
    initial_center: float = 0.0,
) -> Stimulus:
    """Bump whose center rotates at angular velocity `omega` (rad/s)."""
    valid_mask = ~np.isnan(angles)
    angles_valid = angles.astype(np.float64)
    n = angles.shape[0]

    def _stim(t: float) -> NDArray[np.float64]:
        center = initial_center + omega * t
        out = np.zeros(n, dtype=np.float64)
        out[valid_mask] = amplitude * _gaussian_on_circle(angles_valid[valid_mask], center, width)
        return out

    return _stim


def pulse_stimulus_for_ids(
    target_ids: Iterable[int],
    neuron_ids: Iterable[int],
    *,
    t0: float,
    duration: float,
    amplitude: float = 1.0,
) -> Stimulus:
    """A square-pulse input applied to a chosen subset of neurons.

    Used for the Kim et al. 2017 velocity-integration test: pulse all PEN_a
    cells -> bump moves one way; pulse all PEN_b cells -> bump moves the
    other.
    """
    if duration <= 0:
        raise ValueError("duration must be positive.")
    targets = set(int(i) for i in target_ids)
    ids = list(neuron_ids)
    mask = np.array([nid in targets for nid in ids], dtype=np.float64)
    on = amplitude * mask
    off = np.zeros_like(on)
    t1 = t0 + duration

    def _stim(t: float) -> NDArray[np.float64]:
        return on if t0 <= t < t1 else off

    return _stim


def sum_stimuli(*stims: Stimulus) -> Stimulus:
    """Pointwise sum of stimuli. Handy when overlaying e.g. a rotating bump
    plus a transient pulse."""

    def _stim(t: float) -> NDArray[np.float64]:
        out = stims[0](t).copy()
        for s in stims[1:]:
            out = out + s(t)
        return out

    return _stim


def population_vector(
    rates: NDArray[np.float64],
    angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the population-vector angle from per-neuron rates over time.

    Args:
        rates: shape (T, N) array of activity over time.
        angles: shape (N,) angular positions; NaN entries are excluded.

    Returns:
        Array of shape (T,) with the population-vector angle in (-pi, pi] at
        each timestep. Returns NaN where the resultant length is below 1e-6.
    """
    valid = ~np.isnan(angles)
    if not valid.any():
        return np.full(rates.shape[0], np.nan, dtype=np.float64)
    a = angles[valid]
    r = rates[:, valid]
    x = r @ np.cos(a)
    y = r @ np.sin(a)
    length = np.hypot(x, y)
    out = np.arctan2(y, x)
    out[length < 1e-6] = np.nan
    return out


__all__ = [
    "population_vector",
    "pulse_stimulus_for_ids",
    "ring_stimulus",
    "rotating_stimulus",
    "sum_stimuli",
]
