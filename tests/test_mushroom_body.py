"""Mushroom body loader + sparse-coding validation (Phase 7).

Phase 7's purpose is to apply the pipeline to a non-HD-ring circuit and
surface whatever breaks. The MB is a sparse, feedforward + global-inhibition
network with ~2000 neurons. Tests here cover:

  - Unit-level: loader produces the right neuron set, APL is properly
    typed/NT-stamped, the subgraph has the right shape.
  - Qualitative (`slow`): with APL the network is sparse; without APL the
    network is dense. This is the canonical k-WTA test.

The fixture-based path reads from `tests/fixtures/`. Live tests are gated
by `NEUPRINT_TOKEN`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from galvani import ParameterizerOptions, default_parameterizer, simulate
from galvani.circuits.mushroom_body import (
    APL_BODY_ID,
    MB_KC_TYPES,
    MB_NT,
    load_mushroom_body,
)
from galvani.connectome.cache import ParquetCache
from galvani.connectome.hemibrain import HemibrainConnectome
from galvani.model.rate import relu

FIXTURES = Path(__file__).parent / "fixtures"


# -- fixture-based tests -----------------------------------------------------


@pytest.fixture(scope="module")
def mb_connectome() -> HemibrainConnectome:
    if not FIXTURES.exists() or not any(FIXTURES.rglob("*.parquet")):
        pytest.skip("Fixtures not present; run scripts/refresh_fixtures.py.")
    return HemibrainConnectome(cache=ParquetCache(FIXTURES), nt_by_type=MB_NT, token="fake")


def test_load_mushroom_body_returns_expected_population(
    mb_connectome: HemibrainConnectome,
) -> None:
    mb = load_mushroom_body(mb_connectome)
    # hemibrain:v1.2.1: 1923 KCs across the 10 subtypes + APL = 1924.
    assert len(mb.subgraph.neurons) == 1924
    assert mb.n_kc == 1923


def test_apl_is_patched_with_correct_type_and_nt(
    mb_connectome: HemibrainConnectome,
) -> None:
    """hemibrain leaves APL.type as NaN. The loader must patch it; without
    the patch, APL's GABA outgoing sign is lost and sparse coding breaks."""
    mb = load_mushroom_body(mb_connectome)
    apl = mb.subgraph.neurons[mb.apl_index]
    assert apl.id == APL_BODY_ID
    assert apl.cell_type == "APL"
    assert apl.nt == "gaba"


def test_kcs_have_acetylcholine_nt(mb_connectome: HemibrainConnectome) -> None:
    mb = load_mushroom_body(mb_connectome)
    kcs = [n for i, n in enumerate(mb.subgraph.neurons) if mb.kc_mask[i]]
    assert all(n.nt == "acetylcholine" for n in kcs)


def test_mb_subgraph_dataset_version_is_pinned(mb_connectome: HemibrainConnectome) -> None:
    mb = load_mushroom_body(mb_connectome)
    assert mb.subgraph.dataset_version == "hemibrain:v1.2.1"


def test_default_parameterizer_recognizes_apl_inhibition(
    mb_connectome: HemibrainConnectome,
) -> None:
    """APL's outgoing column in the weight matrix should be entirely
    non-positive (it's GABAergic and inhibits the KCs it projects to)."""
    mb = load_mushroom_body(mb_connectome)
    spec = default_parameterizer(mb.subgraph)
    apl_out = spec.weights[:, mb.apl_index]
    # APL doesn't autapse on itself, so its own diagonal is 0; everywhere
    # else where APL connects, the weight should be <= 0.
    assert (apl_out <= 0).all()
    assert apl_out.min() < -1.0, "APL should produce non-trivial inhibition"


def test_kc_count_matches_published_breakdown(
    mb_connectome: HemibrainConnectome,
) -> None:
    """Per-subtype counts in hemibrain:v1.2.1 documented for posterity."""
    mb = load_mushroom_body(mb_connectome)
    by_type: dict[str, int] = {}
    for i, n in enumerate(mb.subgraph.neurons):
        if mb.kc_mask[i]:
            by_type[n.cell_type] = by_type.get(n.cell_type, 0) + 1
    # Total KCs sums to 1923. Spot-check a couple of well-known subtypes.
    assert by_type["KCg-m"] == 590
    assert by_type["KCab-m"] == 354
    assert sum(by_type.values()) == 1923


# -- slow qualitative test ---------------------------------------------------


@pytest.mark.slow
def test_apl_produces_sparse_kc_activation(mb_connectome: HemibrainConnectome) -> None:
    """Drive 30% of KCs externally; APL feedback should drive activity into
    the 1-10% sparse-coding regime. Removing APL eliminates the inhibition
    and activity stays dense.

    This is the canonical functional signature of the mushroom body in the
    Honegger / Lin literature; it's what the connectome must reproduce for
    the pipeline to be useful here.
    """
    mb = load_mushroom_body(mb_connectome)
    spec_with_apl = default_parameterizer(mb.subgraph, ParameterizerOptions(global_gain=0.08))
    # Counterfactual: zero out APL's outgoing weights -> no inhibition.
    weights_no_apl = spec_with_apl.weights.copy()
    weights_no_apl[:, mb.apl_index] = 0.0
    spec_no_apl = type(spec_with_apl)(
        neuron_ids=spec_with_apl.neuron_ids,
        weights=weights_no_apl,
        tau=spec_with_apl.tau,
        bias=spec_with_apl.bias,
        global_gain=spec_with_apl.global_gain,
        dataset_version=spec_with_apl.dataset_version,
        defaults_used=dict(spec_with_apl.defaults_used),
        notes={"apl_ablated": True},
    )

    rng = np.random.default_rng(0)
    n_kc = mb.n_kc
    kc_indices = np.where(mb.kc_mask)[0]
    chosen = rng.choice(kc_indices, size=int(0.30 * n_kc), replace=False)
    pattern = np.zeros(spec_with_apl.n_neurons, dtype=np.float64)
    pattern[chosen] = 0.5

    def stim(_t: float) -> np.ndarray:
        return pattern

    def active_fraction(spec):
        r = simulate(spec, duration=0.2, stimulus=stim, activation=relu, dt=5e-4)
        kc_final = r.rates[-1, mb.kc_mask]
        if kc_final.max() <= 0:
            return 0.0
        active = int((kc_final > 0.01 * kc_final.max()).sum())
        return active / n_kc

    sparse = active_fraction(spec_with_apl)
    dense = active_fraction(spec_no_apl)
    # With APL, the network should land in roughly k-WTA territory.
    assert 0.005 < sparse < 0.15, f"with APL: {sparse:.3f} not in (0.5%, 15%)"
    # Without APL, the network should be at least 3x denser (effectively
    # everyone the input drove plus their downstream).
    assert dense > 3 * sparse, f"without APL: {dense:.3f} not >> with APL: {sparse:.3f}"


# -- live tests --------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("NEUPRINT_TOKEN"), reason="NEUPRINT_TOKEN not set")
def test_live_load_mushroom_body() -> None:
    conn = HemibrainConnectome(nt_by_type=MB_NT)
    mb = load_mushroom_body(conn)
    assert mb.n_kc > 1500  # robust to small dataset drift
    assert mb.subgraph.neurons[mb.apl_index].id == APL_BODY_ID


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("NEUPRINT_TOKEN"), reason="NEUPRINT_TOKEN not set")
def test_live_kc_types_constant_matches_dataset() -> None:
    """If hemibrain renames a KC subtype, the constant goes out of date."""
    conn = HemibrainConnectome(nt_by_type=MB_NT)
    for kc_type in MB_KC_TYPES:
        n = conn.query(type=kc_type)
        assert len(n) > 0, f"KC subtype {kc_type!r} returned 0 neurons"
