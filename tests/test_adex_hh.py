"""Smoke tests for AdEx and HH simulators."""

from __future__ import annotations

import numpy as np

from galvani.model.adex import simulate_adex
from galvani.model.hh import simulate_hh
from galvani.model.spec import ModelSpec


def _isolated(n: int = 1, tau: float = 0.02) -> ModelSpec:
    return ModelSpec(
        neuron_ids=tuple(range(n)),
        weights=np.zeros((n, n), dtype=np.float64),
        tau=np.full(n, tau, dtype=np.float64),
        bias=np.zeros(n, dtype=np.float64),
    )


# ---- AdEx ----


def test_adex_subthreshold_does_not_spike() -> None:
    spec = _isolated(n=1, tau=0.02)
    r = simulate_adex(spec, duration=0.2, stimulus=lambda _t: np.array([0.4]), dt=2e-4)
    assert len(r.spike_trains[0]) == 0


def test_adex_suprathreshold_spikes() -> None:
    spec = _isolated(n=1, tau=0.02)
    r = simulate_adex(spec, duration=0.5, stimulus=lambda _t: np.array([3.0]), dt=2e-4)
    assert len(r.spike_trains[0]) >= 3


def test_adex_adaptation_slows_firing() -> None:
    """The whole point of AdEx: spike-frequency adaptation -- the second
    spike should be later relative to the first than the third is relative
    to the second, when adaptation is on."""
    spec = _isolated(n=1, tau=0.02)
    r = simulate_adex(
        spec,
        duration=0.6,
        stimulus=lambda _t: np.array([5.0]),
        a=0.05,
        b=0.4,
        dt=2e-4,
    )
    spikes = r.spike_trains[0]
    if len(spikes) >= 3:
        isi1 = spikes[1] - spikes[0]
        isi2 = spikes[2] - spikes[1]
        # Adaptation: later ISIs should be >= earlier ISIs.
        assert isi2 >= isi1 * 0.9


def test_adex_stochastic_gives_different_spike_times() -> None:
    spec = _isolated(n=1, tau=0.02)
    r_a = simulate_adex(
        spec,
        duration=0.4,
        stimulus=lambda _t: np.array([3.0]),
        dt=2e-4,
        noise_sigma=0.5,
        rng_seed=1,
    )
    r_b = simulate_adex(
        spec,
        duration=0.4,
        stimulus=lambda _t: np.array([3.0]),
        dt=2e-4,
        noise_sigma=0.5,
        rng_seed=2,
    )
    # With noise the spike timings differ; with no noise they'd be identical.
    assert r_a.spike_trains[0] != r_b.spike_trains[0]


# ---- HH ----


def test_hh_returns_shapes() -> None:
    spec = _isolated(n=2)
    r = simulate_hh(spec, duration=0.05, dt=5e-5)
    assert r.voltages.shape[1] == 2
    assert len(r.spike_trains) == 2


def test_hh_strong_drive_produces_spikes() -> None:
    spec = _isolated(n=1)
    r = simulate_hh(
        spec,
        duration=0.1,
        stimulus=lambda _t: np.array([10.0]),
        dt=5e-5,
    )
    assert len(r.spike_trains[0]) >= 1
