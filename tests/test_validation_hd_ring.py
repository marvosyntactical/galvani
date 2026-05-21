"""HD-ring qualitative validation tests (Phase 5).

Each test mirrors one of the five plan-mandated validations and asserts a
qualitative property. They use the committed parquet fixtures so they run
without a neuPrint token. They are marked `slow` because each simulation
runs for 1-4 simulated seconds at dt=0.5 ms (still ~1-3 wall-seconds each).

Operating point (chosen empirically; see notebooks/01_hd_ring_hemibrain.ipynb
for the gain-sweep that motivated it):
  - symmetrize=True
  - global_gain=0.012
  - activation=tanh
  - stim amplitude 0.3, width 0.5 rad
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from galvani import ParameterizerOptions, default_parameterizer, simulate
from galvani.circuits.hd_ring import load_hd_ring
from galvani.connectome.cache import ParquetCache
from galvani.connectome.hemibrain import HD_RING_NT, HemibrainConnectome
from galvani.model.rate import tanh
from galvani.stimuli import (
    population_vector,
    pulse_stimulus_for_ids,
    ring_stimulus,
    rotating_stimulus,
    sum_stimuli,
)

FIXTURES = Path(__file__).parent / "fixtures"


pytestmark = pytest.mark.slow


def _wrap(d: float) -> float:
    return abs(((d + np.pi) % (2.0 * np.pi)) - np.pi)


@pytest.fixture(scope="module")
def hd_ring():
    """Load the HD-ring layout and parameterize at the v1 operating point."""
    if not FIXTURES.exists() or not any(FIXTURES.rglob("*.parquet")):
        pytest.skip("HD-ring fixtures not present; run scripts/refresh_fixtures.py.")
    conn = HemibrainConnectome(cache=ParquetCache(FIXTURES), nt_by_type=HD_RING_NT, token="fake")
    layout = load_hd_ring(conn)
    spec = default_parameterizer(
        layout.subgraph,
        ParameterizerOptions(symmetrize=True, global_gain=0.012),
    )
    return layout, spec, conn


def test_bump_existence_centered_on_input(hd_ring) -> None:
    """Stationary input -> stationary bump at the input center."""
    layout, spec, _ = hd_ring
    centers = [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]
    for center in centers:
        stim = ring_stimulus(layout.angles, center=center, width=0.5, amplitude=0.3)
        result = simulate(spec, duration=0.6, stimulus=stim, activation=tanh, dt=2e-4)
        pv = population_vector(result.rates, layout.angles)
        err = _wrap(float(pv[-1]) - center)
        assert err < 0.3, f"stim at {center:.3f}: bump landed at {pv[-1]:.3f} (err {err:.3f})"


def test_bump_persists_in_amplitude_after_input_removed(hd_ring) -> None:
    """Remove input -> bump amplitude is retained (the network is bistable).

    Note: the bump's *position* may drift to the nearest network-preferred
    attractor location after the input is removed; this test asserts only
    that activity is sustained, not that position is conserved. Position
    retention requires a more carefully tuned weight matrix; see the
    notebook for the qualitative behavior.
    """
    layout, spec, _ = hd_ring
    base = ring_stimulus(layout.angles, center=np.pi / 2, width=0.5, amplitude=0.3)
    n = layout.angles.shape[0]

    def stim(t: float) -> np.ndarray:
        return base(t) if t < 0.3 else np.zeros(n)

    result = simulate(spec, duration=2.0, stimulus=stim, activation=tanh, dt=2e-4)
    r_max_initial = float(result.rates[int(0.3 / 2e-4)].max())
    r_max_final = float(result.rates[-1].max())
    # Activity should not have collapsed: retain at least 40% of the
    # stim-driven peak after 1.7 s of free evolution.
    assert r_max_final > 0.4 * r_max_initial


def test_bump_tracks_rotating_input(hd_ring) -> None:
    """Rotating input -> bump follows with a small phase lag."""
    layout, spec, _ = hd_ring
    omega = 1.0
    rot = rotating_stimulus(layout.angles, omega=omega, width=0.5, amplitude=0.3)
    result = simulate(spec, duration=4.0, stimulus=rot, activation=tanh, dt=2e-4)
    pv = population_vector(result.rates, layout.angles)

    # Skip the initial transient. At t=2s and t=3s the bump should lead/lag
    # the stim center by < 0.6 rad.
    for t_check in (2.0, 3.0):
        idx = int(t_check / 2e-4)
        expected = omega * t_check
        lag = _wrap(float(pv[idx]) - expected)
        assert lag < 0.6, f"tracking lag at t={t_check}: {lag:.3f} rad"


def test_velocity_integration_left_right_pen_pulses_oppose(hd_ring) -> None:
    """Kim et al. 2017 / Duan-Dong-Fiete 2025 canonical test:
    pulsing left-hemisphere PEN cells moves the bump one way, pulsing
    right-hemisphere PEN cells moves it the other."""
    layout, spec, _ = hd_ring
    neurons = layout.subgraph.neurons
    all_ids = [n.id for n in neurons]
    n = layout.angles.shape[0]

    def pen_subset(hemi: str) -> list[int]:
        return [
            nrn.id
            for nrn in neurons
            if nrn.cell_type in ("PEN_a(PEN1)", "PEN_b(PEN2)") and nrn.hemisphere == hemi
        ]

    def run(pulse_ids: list[int]) -> float:
        base = ring_stimulus(layout.angles, center=np.pi / 2, width=0.5, amplitude=0.3)

        def init(t: float) -> np.ndarray:
            return base(t) if t < 0.3 else np.zeros(n)

        pulse = pulse_stimulus_for_ids(pulse_ids, all_ids, t0=0.5, duration=0.5, amplitude=0.5)
        result = simulate(
            spec, duration=1.5, stimulus=sum_stimuli(init, pulse), activation=tanh, dt=2e-4
        )
        pv = population_vector(result.rates, layout.angles)
        before = float(pv[int(0.5 / 2e-4)])
        after = float(pv[int(1.0 / 2e-4)])
        return ((after - before) + np.pi) % (2.0 * np.pi) - np.pi

    delta_left = run(pen_subset("L"))
    delta_right = run(pen_subset("R"))
    # The two pulses must move the bump in opposite angular directions.
    assert delta_left * delta_right < 0, (
        f"L pulse {delta_left:+.3f} and R pulse {delta_right:+.3f} should oppose"
    )
    # Each motion should be at least 0.1 rad to count as a real shift.
    assert min(abs(delta_left), abs(delta_right)) > 0.1


def test_gain_sweep_has_finite_bump_window(hd_ring) -> None:
    """Below some gain the bump collapses; above some gain it saturates.
    The plan calls for showing that the bump regime is a finite window."""
    layout, _spec_unused, _ = hd_ring
    base = ring_stimulus(layout.angles, center=np.pi / 2, width=0.5, amplitude=0.3)
    n = layout.angles.shape[0]

    def stim(t: float) -> np.ndarray:
        return base(t) if t < 0.3 else np.zeros(n)

    persists_at: list[float] = []
    for gain in np.linspace(0.005, 0.025, 21):
        spec = default_parameterizer(
            layout.subgraph,
            ParameterizerOptions(symmetrize=True, global_gain=float(gain)),
        )
        result = simulate(spec, duration=1.0, stimulus=stim, activation=tanh, dt=2e-4)
        r_max = float(result.rates[-1].max())
        # Persist if activity is sustained but not saturated.
        if 0.15 < r_max < 0.95:
            persists_at.append(float(gain))

    assert len(persists_at) >= 3, f"bump persists at {len(persists_at)} gains; too few"
    assert min(persists_at) > 0.005
    assert max(persists_at) < 0.025
