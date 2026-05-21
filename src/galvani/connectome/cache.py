"""Local parquet cache for connectome queries.

We never put the cache inside the package directory (per plan: surprising
side-effects on `pip install`). `platformdirs.user_cache_dir` gives the right
location on every platform: `~/.cache/galvani/` on Linux,
`~/Library/Caches/galvani/` on macOS, `%LOCALAPPDATA%\\galvani` on Windows.

Cache entries are keyed by a dotted string the caller chooses (e.g.
`hemibrain.v1_2_1.neurons.EPG`). The caller is responsible for invalidation:
this module is intentionally dumb storage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from platformdirs import user_cache_dir

CACHE_APP = "galvani"


def default_cache_dir() -> Path:
    """Return the canonical galvani cache directory and ensure it exists."""
    path = Path(user_cache_dir(CACHE_APP))
    path.mkdir(parents=True, exist_ok=True)
    return path


class ParquetCache:
    """Tiny parquet-backed key-value cache for pandas DataFrames.

    The full filesystem path for key `a.b.c` is `<root>/a/b/c.parquet`. Keys
    are split on '.', so dots can't appear in path segments.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        parts = key.split(".")
        if not parts or any(not p for p in parts):
            raise ValueError(f"Invalid cache key: {key!r}")
        return self.root.joinpath(*parts[:-1], f"{parts[-1]}.parquet")

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> pd.DataFrame:
        path = self._path(key)
        if not path.exists():
            raise KeyError(key)
        return pd.read_parquet(path)

    def store(self, key: str, df: pd.DataFrame) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def drop(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()


__all__ = ["ParquetCache", "default_cache_dir"]
