"""Tests for the HemibrainConnectome loader.

Three tiers:
  - Pure-function tests for the column mappers (no fixtures).
  - Fixture-based tests using the committed parquet snapshot of the HD-ring
    query under tests/fixtures/. These exercise the full query/subgraph path
    without touching the network.
  - Live tests marked `live`; skipped unless `NEUPRINT_TOKEN` is in the env.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from galvani import default_parameterizer
from galvani.connectome.cache import ParquetCache
from galvani.connectome.hemibrain import (
    HD_RING_NT,
    HemibrainConnectome,
    _parse_hemisphere,
    _soma_position,
)

FIXTURES = Path(__file__).parent / "fixtures"
HD_RING_TYPES = ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "Delta7")


# -- pure-function tests -----------------------------------------------------


def test_parse_hemisphere_handles_epg_pattern() -> None:
    assert _parse_hemisphere("EPG(PB08)_L3") == "L"
    assert _parse_hemisphere("EPG(PB08)_R4") == "R"


def test_parse_hemisphere_handles_delta7_pattern() -> None:
    # Delta7 instances end with a bare _L or _R (no trailing digits).
    assert _parse_hemisphere("Delta7(PB15)_L6R3_L") == "L"
    assert _parse_hemisphere("Delta7(PB15)_L4R5_R") == "R"


def test_parse_hemisphere_none_inputs() -> None:
    assert _parse_hemisphere(None) is None
    assert _parse_hemisphere("no_side_token") is None


def test_soma_position_extracts_xyz() -> None:
    assert _soma_position([1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)
    assert _soma_position((10, 20, 30)) == (10.0, 20.0, 30.0)


def test_soma_position_handles_missing() -> None:
    assert _soma_position(None) is None
    assert _soma_position([1.0, 2.0]) is None


# -- fixture-based tests -----------------------------------------------------


@pytest.fixture(scope="module")
def fixture_connectome() -> HemibrainConnectome:
    """A connectome backed by the committed parquet fixtures. No network."""
    if not FIXTURES.exists() or not any(FIXTURES.rglob("*.parquet")):
        pytest.skip("No fixtures committed; run scripts/refresh_fixtures.py.")
    return HemibrainConnectome(cache=ParquetCache(FIXTURES), nt_by_type=HD_RING_NT, token="fake")


def test_query_epg_returns_46_neurons(fixture_connectome: HemibrainConnectome) -> None:
    neurons = fixture_connectome.query(type="EPG")
    assert len(neurons) == 46


def test_query_hd_ring_returns_130_neurons(fixture_connectome: HemibrainConnectome) -> None:
    neurons = fixture_connectome.query(type=list(HD_RING_TYPES))
    assert len(neurons) == 130


def test_query_stamps_nt_from_lookup(fixture_connectome: HemibrainConnectome) -> None:
    neurons = fixture_connectome.query(type="EPG")
    assert all(n.nt == "acetylcholine" for n in neurons)


def test_query_extracts_hemisphere_for_epg(fixture_connectome: HemibrainConnectome) -> None:
    neurons = fixture_connectome.query(type="EPG")
    sides = {n.hemisphere for n in neurons}
    # EPG should split L/R roughly 23/23.
    assert "L" in sides and "R" in sides
    left = sum(1 for n in neurons if n.hemisphere == "L")
    right = sum(1 for n in neurons if n.hemisphere == "R")
    assert left + right == 46


def test_query_provides_soma_position_for_most_epg(
    fixture_connectome: HemibrainConnectome,
) -> None:
    # In hemibrain:v1.2.1, ~8 of 46 EPGs have no identified soma (only
    # neurites in the volume). The mappable majority should still come
    # through as three floats.
    neurons = fixture_connectome.query(type="EPG")
    with_soma = [n for n in neurons if n.soma_position is not None]
    assert len(with_soma) >= 30
    for n in with_soma:
        assert n.soma_position is not None
        assert len(n.soma_position) == 3
        assert all(isinstance(v, float) for v in n.soma_position)


def test_subgraph_dataset_version_is_pinned(fixture_connectome: HemibrainConnectome) -> None:
    # Use the full HD-ring neuron set: the committed adjacency fixture is
    # keyed on the sorted body-id hash of all 130 HD-ring neurons.
    neurons = fixture_connectome.query(type=list(HD_RING_TYPES))
    sg = fixture_connectome.subgraph(neurons)
    assert sg.dataset_version == "hemibrain:v1.2.1"


def test_full_hd_ring_pipeline_through_parameterizer(
    fixture_connectome: HemibrainConnectome,
) -> None:
    """Plan checkpoint: query -> subgraph -> default_parameterizer -> ModelSpec
    with sensible structure (no NaNs, weights signed by NT, ~130 neurons)."""
    neurons = fixture_connectome.query(type=list(HD_RING_TYPES))
    sg = fixture_connectome.subgraph(neurons)
    spec = default_parameterizer(sg)

    assert spec.n_neurons == 130
    assert not np.isnan(spec.weights).any()
    # NTs are EPG/PEN/PEG=ACh (+1) and Delta7=glutamate (-1), so the matrix
    # is genuinely signed -- not just zeros.
    assert spec.weights.min() < 0 and spec.weights.max() > 0
    # Every neuron has a valid tau.
    assert (spec.tau > 0).all()


def test_query_with_no_args_raises(fixture_connectome: HemibrainConnectome) -> None:
    with pytest.raises(ValueError, match="type"):
        fixture_connectome.query()


def test_subgraph_on_empty_neuron_list(fixture_connectome: HemibrainConnectome) -> None:
    sg = fixture_connectome.subgraph([])
    assert len(sg.neurons) == 0
    assert sg.pre_ids.size == 0


def test_query_ids_filter_after_type_fetch(fixture_connectome: HemibrainConnectome) -> None:
    all_epg = fixture_connectome.query(type="EPG")
    pick = [all_epg[0].id, all_epg[5].id]
    narrowed = fixture_connectome.query(type="EPG", ids=pick)
    assert {n.id for n in narrowed} == set(pick)


def test_subgraph_collapses_to_expected_hd_ring_synapse_total(
    fixture_connectome: HemibrainConnectome,
) -> None:
    """Snapshot-style assertion against the committed fixture. If neuPrint
    re-releases hemibrain:v1.2.1, this number changes and the fixture must be
    refreshed via scripts/refresh_fixtures.py."""
    neurons = fixture_connectome.query(type=list(HD_RING_TYPES))
    sg = fixture_connectome.subgraph(neurons)
    assert int(sg.counts.sum()) == 112705
    assert sg.counts.size == 8270


def test_missing_token_at_live_fetch_raises(tmp_path: Path) -> None:
    """A connectome instantiated without a token must still construct, but
    fail loudly when it tries to talk to neuPrint."""
    conn = HemibrainConnectome(
        cache=ParquetCache(tmp_path),
        token=None,
    )
    # Force-clear any env-var fallback for this assertion.
    conn._token = None  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="No neuPrint token"):
        conn.query(type="EPG")


def test_cache_key_collapses_versions_into_one_dir(tmp_path: Path) -> None:
    """Sanity: 'hemibrain:v1.2.1' yields a single version dir, not three."""
    conn = HemibrainConnectome(cache=ParquetCache(tmp_path), token="fake")
    df = pd.DataFrame({"bodyId": [1], "type": ["EPG"], "instance": ["EPG(PB08)_L1"]})
    conn._cache.store(conn._neuron_cache_key("EPG"), df)  # type: ignore[attr-defined]
    parquet_files = list(tmp_path.rglob("*.parquet"))
    assert len(parquet_files) == 1
    rel = parquet_files[0].relative_to(tmp_path)
    assert rel.parts[:2] == ("hemibrain", "v1_2_1")


# -- live tests --------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("NEUPRINT_TOKEN"), reason="NEUPRINT_TOKEN not set")
def test_live_query_epg_count_matches_plan() -> None:
    conn = HemibrainConnectome(nt_by_type=HD_RING_NT)
    neurons = conn.query(type="EPG")
    # Plan: ~46 EPGs in hemibrain:v1.2.1. Allow some drift but flag big swings.
    assert 40 <= len(neurons) <= 50


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("NEUPRINT_TOKEN"), reason="NEUPRINT_TOKEN not set")
def test_live_server_version_responds() -> None:
    conn = HemibrainConnectome()
    client = conn._ensure_client()  # type: ignore[attr-defined]
    version = client.fetch_version()
    assert isinstance(version, str) and len(version) > 0
