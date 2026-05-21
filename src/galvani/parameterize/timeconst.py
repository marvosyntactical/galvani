"""Cell-type -> membrane time constant (tau) defaults.

Defaults: 20 ms for excitatory, 10 ms for inhibitory. The plan calls these out
explicitly because tau is the per-neuron quantity that sets the simulator's
relaxation timescale, and the difference between exc/inh tau is what makes a
ring attractor's bump shape behave.

Cell-type-specific overrides go in `TYPE_TAU`. The HD-ring entries below are
literature-anchored placeholders; users override per circuit when they have
better evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

Tau = float  # seconds

DEFAULT_EXC_TAU: Final[Tau] = 0.020  # 20 ms
DEFAULT_INH_TAU: Final[Tau] = 0.010  # 10 ms

TYPE_TAU: Final[dict[str, Tau]] = {
    # HD-ring placeholder defaults. Override per circuit when published values
    # differ; documented here so the user can see what they're inheriting.
    "EPG": 0.020,
    "PEN_a": 0.020,
    "PEN_b": 0.020,
    "Delta7": 0.020,
}
"""Per-cell-type tau overrides (seconds). Lookup falls back to the sign-based
default when a type is missing here."""


TypeToTau = Callable[[str, int], Tau]
"""Signature: `(cell_type, sign) -> tau_seconds`."""


def by_type_v1(cell_type: str, sign: int) -> Tau:
    """Default `cell_type -> tau` rule.

    Priority: explicit `TYPE_TAU` entry, then 20 ms (sign >= 0) / 10 ms
    (sign < 0). Unknown signs (0) fall through as excitatory; the actual sign
    of weights is sorted out by `signs.py`, not here.
    """
    if cell_type in TYPE_TAU:
        return TYPE_TAU[cell_type]
    return DEFAULT_INH_TAU if sign < 0 else DEFAULT_EXC_TAU
