"""Unit tests for the NT -> sign mapping (fly convention)."""

from __future__ import annotations

from galvani.parameterize.signs import FLY_NT_SIGN, fly_default


def test_acetylcholine_is_excitatory() -> None:
    assert fly_default("acetylcholine") == +1


def test_gaba_is_inhibitory() -> None:
    assert fly_default("gaba") == -1


def test_glutamate_is_inhibitory_in_fly() -> None:
    # Mammalian convention would say +1 here; fly differs via GluClalpha.
    assert fly_default("glutamate") == -1


def test_monoamines_are_modulatory_zero() -> None:
    for nt in ("octopamine", "serotonin", "dopamine"):
        assert fly_default(nt) == 0


def test_none_and_unknown_default_to_zero() -> None:
    assert fly_default(None) == 0
    assert fly_default("histamine") == 0
    assert fly_default("") == 0


def test_case_insensitive() -> None:
    assert fly_default("Acetylcholine") == +1
    assert fly_default("GABA") == -1


def test_default_table_covers_documented_six_nts() -> None:
    expected = {"acetylcholine", "gaba", "glutamate", "octopamine", "serotonin", "dopamine"}
    assert set(FLY_NT_SIGN) == expected
