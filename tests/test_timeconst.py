"""Unit tests for the cell-type -> tau defaults."""

from __future__ import annotations

from galvani.parameterize.timeconst import (
    DEFAULT_EXC_TAU,
    DEFAULT_INH_TAU,
    TYPE_TAU,
    by_type_v1,
)


def test_inhibitory_default_is_10ms() -> None:
    assert DEFAULT_INH_TAU == 0.010
    assert by_type_v1("UnknownInhibType", sign=-1) == 0.010


def test_excitatory_default_is_20ms() -> None:
    assert DEFAULT_EXC_TAU == 0.020
    assert by_type_v1("UnknownExcType", sign=+1) == 0.020


def test_zero_sign_falls_through_to_excitatory_default() -> None:
    assert by_type_v1("UnknownType", sign=0) == DEFAULT_EXC_TAU


def test_explicit_type_overrides_sign_based_default() -> None:
    for cell_type in TYPE_TAU:
        assert by_type_v1(cell_type, sign=-1) == TYPE_TAU[cell_type]
        assert by_type_v1(cell_type, sign=+1) == TYPE_TAU[cell_type]
