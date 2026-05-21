"""Synapse count -> synaptic weight heuristics.

Synapse counts in connectomes are heavy-tailed (a few pairs share thousands of
synapses; the median pair shares one or two). `log1p` is the default because
it compresses that dynamic range into a magnitude consistent with what rate
models tolerate. Alternatives are exposed for users who want to challenge the
default.

These functions operate on the raw count array; sign assignment happens
separately (see `signs.py`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CountToWeight = Callable[[NDArray[np.int32 | np.int64]], NDArray[np.float64]]

WeightHeuristic = Literal["log1p", "sqrt", "raw", "rank"]


def log1p(counts: NDArray[np.int32 | np.int64]) -> NDArray[np.float64]:
    """`W = log(1 + count)`. Default. Compresses the heavy tail of counts."""
    return np.log1p(counts.astype(np.float64))


def sqrt(counts: NDArray[np.int32 | np.int64]) -> NDArray[np.float64]:
    """`W = sqrt(count)`. Milder compression than log1p."""
    return np.sqrt(counts.astype(np.float64))


def raw(counts: NDArray[np.int32 | np.int64]) -> NDArray[np.float64]:
    """`W = count`. No compression. Useful as a sanity check."""
    return counts.astype(np.float64)


def rank(counts: NDArray[np.int32 | np.int64]) -> NDArray[np.float64]:
    """`W = rank(count) / N`. Distribution-free; collapses absolute magnitude."""
    arr = counts.astype(np.float64)
    if arr.size == 0:
        return arr
    order = np.argsort(arr, kind="stable")
    ranks = np.empty_like(arr, dtype=np.float64)
    ranks[order] = np.arange(1, arr.size + 1, dtype=np.float64)
    return ranks / arr.size


HEURISTICS: dict[WeightHeuristic, CountToWeight] = {
    "log1p": log1p,
    "sqrt": sqrt,
    "raw": raw,
    "rank": rank,
}


def get(name: WeightHeuristic) -> CountToWeight:
    """Look up a heuristic by name. Raises `KeyError` on unknown names."""
    return HEURISTICS[name]
