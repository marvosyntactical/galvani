"""Tests for the parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from galvani.connectome.cache import ParquetCache, default_cache_dir


def test_default_cache_dir_exists() -> None:
    path = default_cache_dir()
    assert path.exists() and path.is_dir()


def test_store_and_load_round_trip(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    cache.store("ds.kind.foo", df)
    assert cache.has("ds.kind.foo")
    loaded = cache.load("ds.kind.foo")
    pd.testing.assert_frame_equal(loaded, df)


def test_key_path_splits_on_dots(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    cache.store("a.b.c", pd.DataFrame({"x": [1]}))
    expected = tmp_path / "a" / "b" / "c.parquet"
    assert expected.exists()


def test_load_missing_key_raises(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    with pytest.raises(KeyError):
        cache.load("nope.nada")


def test_drop_removes_entry(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    cache.store("a.b", pd.DataFrame({"x": [1]}))
    cache.drop("a.b")
    assert not cache.has("a.b")


def test_invalid_key_rejected(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    with pytest.raises(ValueError):
        cache.store("", pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError):
        cache.store("a..b", pd.DataFrame({"x": [1]}))
