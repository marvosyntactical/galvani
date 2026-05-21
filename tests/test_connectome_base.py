"""Unit tests for the connectome interface dataclasses (Phase 1 API).

These pin down the invariants documented on `Neuron` and `Subgraph`. They run
without any backend (no neuPrint required).
"""

from __future__ import annotations

import numpy as np
import pytest

from galvani.connectome.base import Neuron, Subgraph


def _tiny_subgraph() -> Subgraph:
    a = Neuron(id=1, cell_type="EPG", hemisphere="L", nt="acetylcholine")
    b = Neuron(id=2, cell_type="PEN_a", hemisphere="R", nt="acetylcholine")
    c = Neuron(id=3, cell_type="Delta7", hemisphere="C", nt="glutamate")
    return Subgraph(
        neurons=(a, b, c),
        pre_ids=np.array([1, 2, 3], dtype=np.int64),
        post_ids=np.array([2, 3, 1], dtype=np.int64),
        counts=np.array([5, 7, 2], dtype=np.int32),
        nt_pre=("acetylcholine", "acetylcholine", "glutamate"),
        dataset_version="test:v0",
    )


def test_neuron_is_frozen() -> None:
    n = Neuron(id=42, cell_type="EPG")
    with pytest.raises(AttributeError):
        n.cell_type = "PEN_a"  # type: ignore[misc]


def test_subgraph_neuron_index_matches_neuron_order() -> None:
    sg = _tiny_subgraph()
    assert sg.neuron_index() == {1: 0, 2: 1, 3: 2}


def test_subgraph_rejects_unequal_parallel_arrays() -> None:
    a = Neuron(id=1, cell_type="EPG")
    with pytest.raises(ValueError, match="parallel arrays"):
        Subgraph(
            neurons=(a,),
            pre_ids=np.array([1, 1], dtype=np.int64),
            post_ids=np.array([1], dtype=np.int64),
            counts=np.array([1, 1], dtype=np.int32),
            nt_pre=("acetylcholine", "acetylcholine"),
            dataset_version="test:v0",
        )


def test_subgraph_rejects_negative_counts() -> None:
    a = Neuron(id=1, cell_type="EPG")
    with pytest.raises(ValueError, match="non-negative"):
        Subgraph(
            neurons=(a,),
            pre_ids=np.array([1], dtype=np.int64),
            post_ids=np.array([1], dtype=np.int64),
            counts=np.array([-1], dtype=np.int32),
            nt_pre=("acetylcholine",),
            dataset_version="test:v0",
        )


def test_subgraph_allows_empty_synapse_table() -> None:
    a = Neuron(id=1, cell_type="EPG")
    sg = Subgraph(
        neurons=(a,),
        pre_ids=np.array([], dtype=np.int64),
        post_ids=np.array([], dtype=np.int64),
        counts=np.array([], dtype=np.int32),
        nt_pre=(),
        dataset_version="test:v0",
    )
    assert sg.neuron_index() == {1: 0}
