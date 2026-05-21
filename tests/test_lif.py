"""Tests for the LIF biophysical simulator."""

from __future__ import annotations

import numpy as np
import pytest

from galvani.model.lif import simulate_lif
from galvani.model.spec import ModelSpec


def _isolated(n: int = 3, tau: float = 0.02) -> ModelSpec:
    return ModelSpec(
        neuron_ids=tuple(range(n)),
        weights=np.zeros((n, n), dtype=np.float64),
        tau=np.full(n, tau, dtype=np.float64),
        bias=np.zeros(n, dtype=np.float64),
    )


def test_lif_returns_shapes() -> None:
    spec = _isolated(n=4)
    r = simulate_lif(spec, duration=0.05, dt=5e-4)
    assert r.voltages.shape == r.rates.shape
    assert r.voltages.shape[1] == 4
    assert len(r.spike_trains) == 4


def test_subthreshold_input_does_not_spike() -> None:
    spec = _isolated(n=1, tau=0.01)
    r = simulate_lif(
        spec,
        duration=0.2,
        stimulus=lambda _t: np.array([0.5]),
        dt=5e-4,
        v_threshold=1.0,
    )
    # With drive of 0.5 and threshold 1.0, the neuron should asymptote near 0.5
    # and never spike.
    assert len(r.spike_trains[0]) == 0
    assert 0.4 < r.voltages[-1, 0] < 0.6


def test_suprathreshold_input_spikes_periodically() -> None:
    spec = _isolated(n=1, tau=0.01)
    r = simulate_lif(
        spec,
        duration=0.5,
        stimulus=lambda _t: np.array([2.0]),
        dt=5e-4,
        v_threshold=1.0,
        t_refractory=0.002,
    )
    spikes = r.spike_trains[0]
    # With drive 2.0 and tau 10 ms, the asymptote is at 2.0 (well above
    # threshold of 1.0) so we expect regular spiking.
    assert len(spikes) >= 5
    # ISIs should be roughly constant after the first spike.
    if len(spikes) >= 3:
        isis = np.diff(spikes)
        assert isis.std() < 0.01


def test_refractory_period_caps_firing_rate() -> None:
    spec = _isolated(n=1, tau=0.005)
    r = simulate_lif(
        spec,
        duration=0.5,
        stimulus=lambda _t: np.array([100.0]),  # massive drive
        dt=1e-4,
        v_threshold=1.0,
        t_refractory=0.01,  # 10 ms refractory -> max 100 Hz
    )
    spikes = r.spike_trains[0]
    # 100 ms refractory -> at most ~50 spikes in 0.5 s.
    assert len(spikes) <= 55


def test_dt_must_be_smaller_than_min_tau() -> None:
    spec = _isolated(n=1, tau=0.001)
    with pytest.raises(ValueError, match="integration unstable"):
        simulate_lif(spec, duration=0.1, dt=5e-3)


def test_dt_must_be_smaller_than_syn_tau() -> None:
    spec = _isolated(n=1, tau=0.01)
    with pytest.raises(ValueError, match="synaptic filter"):
        simulate_lif(spec, duration=0.1, dt=8e-3, syn_tau=5e-3)
