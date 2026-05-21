"""End-to-end tests for `default_parameterizer`.

These act as the golden-snapshot tests for the parameterizer: a tiny fixed
subgraph runs through the pipeline and we assert the structural properties
of the resulting ModelSpec.
"""

from __future__ import annotations

import numpy as np

from galvani.connectome.base import Neuron, Subgraph
from galvani.parameterize import ParameterizerOptions, default_parameterizer
from galvani.parameterize.signs import fly_default


def _three_neuron_subgraph() -> Subgraph:
    # Two excitatory cholinergic neurons (ids 1, 2) and one inhibitory
    # glutamatergic neuron (id 3). Synapses cover all three sign combinations.
    exc_a = Neuron(id=1, cell_type="EPG", hemisphere="L", nt="acetylcholine")
    exc_b = Neuron(id=2, cell_type="PEN_a", hemisphere="R", nt="acetylcholine")
    inh = Neuron(id=3, cell_type="Delta7", hemisphere="C", nt="glutamate")
    return Subgraph(
        neurons=(exc_a, exc_b, inh),
        # pre -> post: 1->2 (exc), 3->1 (inh), 2->3 (exc), 1->2 (exc duplicate to test summing).
        pre_ids=np.array([1, 3, 2, 1], dtype=np.int64),
        post_ids=np.array([2, 1, 3, 2], dtype=np.int64),
        counts=np.array([10, 4, 7, 5], dtype=np.int32),
        nt_pre=("acetylcholine", "glutamate", "acetylcholine", "acetylcholine"),
        dataset_version="test:v0",
    )


def test_spec_shape_and_neuron_ids() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    assert spec.n_neurons == 3
    assert spec.neuron_ids == (1, 2, 3)
    assert spec.weights.shape == (3, 3)


def test_weights_are_signed_consistently_with_presynaptic_nt() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    # Column 0 == pre-neuron id 1 (cholinergic) -> outgoing weights should be >= 0.
    assert (spec.weights[:, 0] >= 0).all()
    # Column 1 == pre-neuron id 2 (cholinergic) -> outgoing weights should be >= 0.
    assert (spec.weights[:, 1] >= 0).all()
    # Column 2 == pre-neuron id 3 (glutamatergic, inhibitory) -> <= 0.
    assert (spec.weights[:, 2] <= 0).all()


def test_duplicate_synapse_rows_sum_into_one_matrix_entry() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    # pre=1 -> post=2 appears twice with counts 10 and 5. log1p heuristic
    # applies per-row, so we expect log(11) + log(6) at weights[1, 0].
    expected = np.log1p(10.0) + np.log1p(5.0)
    np.testing.assert_allclose(spec.weights[1, 0], expected)


def test_weight_index_convention_is_post_row_pre_column() -> None:
    # pre=3 -> post=1 with count 4 (glutamate -> inhibitory).
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    expected = -np.log1p(4.0)
    np.testing.assert_allclose(spec.weights[0, 2], expected)
    # Confirm there's no spurious entry at the transpose location.
    assert spec.weights[2, 0] >= 0


def test_diagonal_is_zero_when_no_autapses() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    np.testing.assert_array_equal(np.diag(spec.weights), np.zeros(3))


def test_defaults_used_records_heuristic_names() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    assert spec.defaults_used["weights"] == "log1p"
    assert spec.defaults_used["nt_to_sign"] == "fly_default"
    assert spec.defaults_used["tau"] == "by_type_v1"
    assert spec.defaults_used["symmetrize"] == "false"


def test_dataset_version_propagates_from_subgraph() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    assert spec.dataset_version == "test:v0"


def test_tau_is_positive_everywhere() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    assert (spec.tau > 0).all()


def test_symmetrize_option_produces_symmetric_matrix() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg, ParameterizerOptions(symmetrize=True))
    np.testing.assert_allclose(spec.weights, spec.weights.T)
    assert spec.defaults_used["symmetrize"] == "true"


def test_custom_weight_heuristic_is_recorded() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg, ParameterizerOptions(weight_heuristic="sqrt"))
    assert spec.defaults_used["weights"] == "sqrt"
    # sqrt(10) + sqrt(5) at the duplicated pre=1 -> post=2 pair.
    np.testing.assert_allclose(spec.weights[1, 0], np.sqrt(10.0) + np.sqrt(5.0))


def test_global_gain_passed_through() -> None:
    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg, ParameterizerOptions(global_gain=2.5))
    assert spec.global_gain == 2.5


def test_custom_nt_to_sign_takes_effect() -> None:
    sg = _three_neuron_subgraph()

    def all_excitatory(_nt: str | None) -> int:
        return +1

    spec = default_parameterizer(sg, ParameterizerOptions(nt_to_sign=all_excitatory))
    # With sign=+1 everywhere, no weight entry should be negative.
    assert (spec.weights >= 0).all()


def test_parameterizer_output_is_json_round_trippable() -> None:
    from galvani.model.spec import ModelSpec

    sg = _three_neuron_subgraph()
    spec = default_parameterizer(sg)
    restored = ModelSpec.from_json(spec.to_json())
    np.testing.assert_array_equal(restored.weights, spec.weights)
    assert restored.neuron_ids == spec.neuron_ids
    assert restored.defaults_used == spec.defaults_used


def test_empty_subgraph_produces_empty_spec() -> None:
    sg = Subgraph(
        neurons=(),
        pre_ids=np.array([], dtype=np.int64),
        post_ids=np.array([], dtype=np.int64),
        counts=np.array([], dtype=np.int32),
        nt_pre=(),
        dataset_version="test:v0",
    )
    spec = default_parameterizer(sg)
    assert spec.n_neurons == 0
    assert spec.weights.shape == (0, 0)


def test_fly_default_resolves_through_module() -> None:
    # Sanity check that the default callable identity is what we record.
    assert fly_default.__name__ == "fly_default"
