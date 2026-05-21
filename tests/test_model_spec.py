"""Unit tests for `ModelSpec`: shape invariants and JSON round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from galvani.model.spec import ModelSpec


def _tiny_spec() -> ModelSpec:
    return ModelSpec(
        neuron_ids=(10, 20, 30),
        weights=np.array(
            [[0.0, 0.5, -0.25], [0.5, 0.0, 0.1], [-0.25, 0.1, 0.0]],
            dtype=np.float64,
        ),
        tau=np.array([0.02, 0.02, 0.01], dtype=np.float64),
        bias=np.array([0.0, -0.1, 0.05], dtype=np.float64),
        global_gain=1.5,
        dataset_version="hemibrain:v1.2.1",
        defaults_used={"weights": "log1p", "nt_to_sign": "fly_default"},
        notes={"source": "unit-test"},
    )


def test_n_neurons_matches_ids() -> None:
    spec = _tiny_spec()
    assert spec.n_neurons == 3


def test_weights_shape_must_be_square_and_sized() -> None:
    with pytest.raises(ValueError, match="weights must be"):
        ModelSpec(
            neuron_ids=(1, 2),
            weights=np.zeros((3, 3), dtype=np.float64),
            tau=np.ones(2, dtype=np.float64),
            bias=np.zeros(2, dtype=np.float64),
        )


def test_tau_must_be_strictly_positive() -> None:
    with pytest.raises(ValueError, match="tau must be strictly positive"):
        ModelSpec(
            neuron_ids=(1, 2),
            weights=np.zeros((2, 2), dtype=np.float64),
            tau=np.array([0.01, 0.0], dtype=np.float64),
            bias=np.zeros(2, dtype=np.float64),
        )


def test_bias_shape_validated() -> None:
    with pytest.raises(ValueError, match="bias must be"):
        ModelSpec(
            neuron_ids=(1, 2),
            weights=np.zeros((2, 2), dtype=np.float64),
            tau=np.ones(2, dtype=np.float64),
            bias=np.zeros(3, dtype=np.float64),
        )


def test_json_round_trip_preserves_values() -> None:
    spec = _tiny_spec()
    restored = ModelSpec.from_json(spec.to_json())
    assert restored.neuron_ids == spec.neuron_ids
    np.testing.assert_array_equal(restored.weights, spec.weights)
    np.testing.assert_array_equal(restored.tau, spec.tau)
    np.testing.assert_array_equal(restored.bias, spec.bias)
    assert restored.global_gain == spec.global_gain
    assert restored.dataset_version == spec.dataset_version
    assert restored.defaults_used == spec.defaults_used
    assert restored.notes == spec.notes


def test_from_json_rejects_unknown_schema_version() -> None:
    payload = '{"_schema_version": 999}'
    with pytest.raises(ValueError, match="schema version"):
        ModelSpec.from_json(payload)
