"""Tests for the NumPy rate-model simulator.

These are qualitative (matches the design-decision plan): the simulator
relaxes to a fixed point under a steady input, settles within a few tau, and
rejects unstable step sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from galvani.model.rate import Result, simulate, tanh, zero_stimulus
from galvani.model.spec import ModelSpec


def _isolated_neurons(n: int = 3, tau: float = 0.02) -> ModelSpec:
    """A network with no recurrent connectivity. Each neuron behaves as a
    leaky integrator and lets us assert against an analytical fixed point."""
    return ModelSpec(
        neuron_ids=tuple(range(n)),
        weights=np.zeros((n, n), dtype=np.float64),
        tau=np.full(n, tau, dtype=np.float64),
        bias=np.zeros(n, dtype=np.float64),
    )


def test_simulate_returns_correct_shape() -> None:
    spec = _isolated_neurons(n=4)
    duration = 0.05  # 50 ms
    dt = 5e-4
    result = simulate(spec, duration=duration, dt=dt)
    expected_steps = int(np.ceil(duration / dt)) + 1
    assert result.times.shape == (expected_steps,)
    assert result.rates.shape == (expected_steps, 4)
    assert isinstance(result, Result)


def test_isolated_neuron_relaxes_to_input_under_relu() -> None:
    # With W=0 and constant I > 0, the equation is tau*dr/dt = -r + relu(I) = -r + I.
    # The fixed point is r = I; we expect convergence in a few tau.
    spec = _isolated_neurons(n=2, tau=0.01)
    drive = np.array([0.7, 0.3], dtype=np.float64)

    def stim(_t: float) -> np.ndarray:
        return drive

    result = simulate(spec, duration=0.2, stimulus=stim, dt=5e-4)
    np.testing.assert_allclose(result.rates[-1], drive, atol=1e-3)


def test_relu_clamps_negative_drive_to_zero() -> None:
    spec = _isolated_neurons(n=1, tau=0.01)
    drive = np.array([-0.5], dtype=np.float64)

    def stim(_t: float) -> np.ndarray:
        return drive

    result = simulate(spec, duration=0.2, stimulus=stim, dt=5e-4)
    assert result.rates[-1, 0] == 0.0


def test_zero_stimulus_with_zero_initial_rates_stays_zero() -> None:
    spec = _isolated_neurons(n=3)
    result = simulate(spec, duration=0.05, stimulus=zero_stimulus(3))
    np.testing.assert_array_equal(result.rates, np.zeros_like(result.rates))


def test_initial_rates_decay_toward_zero_without_input() -> None:
    spec = _isolated_neurons(n=1, tau=0.01)
    initial = np.array([1.0], dtype=np.float64)
    result = simulate(spec, duration=0.1, initial_rates=initial, dt=5e-4)
    # After 10*tau the decayed value should be effectively 0.
    assert result.rates[-1, 0] < 1e-3


def test_tanh_activation_saturates_bounded() -> None:
    spec = _isolated_neurons(n=1, tau=0.01)

    def big_drive(_t: float) -> np.ndarray:
        return np.array([100.0], dtype=np.float64)

    result = simulate(spec, duration=0.2, stimulus=big_drive, activation=tanh, dt=5e-4)
    # tanh saturates at 1.
    assert 0.99 < result.rates[-1, 0] <= 1.0


def test_simulate_rejects_dt_larger_than_min_tau() -> None:
    spec = _isolated_neurons(n=1, tau=0.005)
    with pytest.raises(ValueError, match="Euler step"):
        simulate(spec, duration=0.05, dt=1e-2)


def test_simulate_rejects_negative_dt() -> None:
    spec = _isolated_neurons(n=1)
    with pytest.raises(ValueError, match="dt must be positive"):
        simulate(spec, duration=0.05, dt=-1e-4)


def test_simulate_rejects_negative_duration() -> None:
    spec = _isolated_neurons(n=1)
    with pytest.raises(ValueError, match="duration"):
        simulate(spec, duration=-0.05)


def test_initial_rates_shape_validated() -> None:
    spec = _isolated_neurons(n=2)
    with pytest.raises(ValueError, match="initial_rates"):
        simulate(spec, duration=0.05, initial_rates=np.zeros(3, dtype=np.float64))


def test_recurrent_excitation_amplifies_input() -> None:
    # Two mutually-exciting neurons. Recurrent gain should pull the steady
    # state above the open-loop value (input alone).
    spec = ModelSpec(
        neuron_ids=(0, 1),
        weights=np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
        tau=np.full(2, 0.01, dtype=np.float64),
        bias=np.zeros(2, dtype=np.float64),
    )
    drive = np.array([0.5, 0.5], dtype=np.float64)
    result = simulate(spec, duration=0.5, stimulus=lambda _t: drive, dt=5e-4)
    # Open-loop steady state would be r=I=0.5 each; recurrent excitation should
    # give r > 0.5. Closed-form: r = (I) / (1 - 0.4) ~= 0.833.
    np.testing.assert_allclose(result.rates[-1], np.full(2, 0.5 / 0.6), atol=5e-3)
