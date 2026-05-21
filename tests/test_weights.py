"""Unit tests for the synapse-count -> weight heuristics."""

from __future__ import annotations

import numpy as np
import pytest

from galvani.parameterize import weights as w


def test_log1p_matches_numpy_log1p() -> None:
    counts = np.array([0, 1, 10, 100], dtype=np.int32)
    np.testing.assert_allclose(w.log1p(counts), np.log1p(counts.astype(np.float64)))


def test_log1p_is_monotone() -> None:
    counts = np.array([1, 2, 100, 5000], dtype=np.int32)
    out = w.log1p(counts)
    assert np.all(np.diff(out) > 0)


def test_raw_returns_float_cast() -> None:
    counts = np.array([0, 3, 7], dtype=np.int32)
    out = w.raw(counts)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [0.0, 3.0, 7.0])


def test_sqrt_matches_numpy_sqrt() -> None:
    counts = np.array([0, 4, 9], dtype=np.int32)
    np.testing.assert_allclose(w.sqrt(counts), [0.0, 2.0, 3.0])


def test_rank_is_distribution_free_and_normalized() -> None:
    counts = np.array([5, 1, 100, 2], dtype=np.int32)
    out = w.rank(counts)
    # Values are rank/N; with N=4, sorted positions give {0.25, 0.5, 0.75, 1.0}.
    assert sorted(out.tolist()) == [0.25, 0.5, 0.75, 1.0]
    # Largest input maps to highest rank.
    assert out[np.argmax(counts)] == 1.0


def test_rank_handles_empty_input() -> None:
    out = w.rank(np.array([], dtype=np.int32))
    assert out.shape == (0,)


def test_get_returns_callable_for_known_name() -> None:
    fn = w.get("log1p")
    np.testing.assert_allclose(fn(np.array([0, 1], dtype=np.int32)), [0.0, np.log(2.0)])


def test_get_raises_on_unknown_name() -> None:
    with pytest.raises(KeyError):
        w.get("bogus")  # type: ignore[arg-type]
