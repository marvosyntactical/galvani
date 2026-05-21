"""Connectome interface and core data types.

The shape of `Neuron`, `Subgraph`, and `Connectome` is the v1 API contract.
Backends (hemibrain, FlyWire later) implement `Connectome`; the rest of the
library only depends on this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Hemisphere = Literal["L", "R", "C"]
"""Hemisphere label. 'C' = central/midline neuron with no clear side."""


@dataclass(frozen=True, slots=True)
class Neuron:
    """A single neuron in a connectome.

    Fields chosen to be the intersection of what hemibrain and FlyWire expose,
    so a backend swap doesn't break downstream code.
    """

    id: int
    """Backend-specific neuron id (e.g. hemibrain body_id, FlyWire root_id)."""

    cell_type: str
    """Cell type label, e.g. 'EPG', 'PEN_a'. Backend-specific taxonomy."""

    hemisphere: Hemisphere | None = None

    nt: str | None = None
    """Predicted neurotransmitter, lowercased: 'acetylcholine', 'gaba',
    'glutamate', 'octopamine', 'serotonin', 'dopamine', or None if unknown."""

    soma_position: tuple[float, float, float] | None = None
    """(x, y, z) in dataset-native units (nanometers for hemibrain)."""

    skeleton_path: str | None = None
    """Optional local filesystem path to a cached SWC skeleton."""


@dataclass(frozen=True, slots=True)
class Subgraph:
    """A connectivity subgraph: neurons plus their synapses.

    The synapse table is stored as four parallel arrays in COO (long) form
    rather than a DataFrame. Reasons:
      - Mypy-friendly under strict mode.
      - Cheap to slice and pass to numpy.
      - The DataFrame is a convenient *input* format but a poor *contract*.

    Invariants:
      - `len(pre_ids) == len(post_ids) == len(counts) == len(nt_pre)`.
      - Every id appearing in `pre_ids` or `post_ids` is present in
        `{n.id for n in neurons}`.
      - `counts` are non-negative integers.
    """

    neurons: tuple[Neuron, ...]
    pre_ids: NDArray[np.int64]
    post_ids: NDArray[np.int64]
    counts: NDArray[np.int32]
    nt_pre: tuple[str | None, ...]
    dataset_version: str
    """Pinned dataset version, e.g. 'hemibrain:v1.2.1'. Required for reproducibility."""

    extra: dict[str, object] = field(default_factory=dict)
    """Backend-specific extras (e.g. confidence scores). Do not rely on this."""

    def __post_init__(self) -> None:
        n = len(self.pre_ids)
        if not (len(self.post_ids) == n == len(self.counts) == len(self.nt_pre)):
            raise ValueError(
                f"Subgraph parallel arrays must have equal length; got "
                f"{len(self.pre_ids)}/{len(self.post_ids)}/{len(self.counts)}/{len(self.nt_pre)}"
            )
        if n > 0 and int(self.counts.min()) < 0:
            raise ValueError("Subgraph synapse counts must be non-negative.")

    def neuron_index(self) -> dict[int, int]:
        """Map neuron id → its position in the canonical neuron ordering.

        Defines the row/column ordering for the parameterizer's weight matrix.
        """
        return {n.id: i for i, n in enumerate(self.neurons)}


class Connectome(Protocol):
    """Abstract interface for connectome backends.

    The first concrete implementation is `HemibrainConnectome` (Phase 2);
    `FlyWireConnectome` follows in v1.5. Backends are expected to be cheap to
    instantiate (no upfront downloads); queries pull on demand and cache.
    """

    dataset_version: str
    """The pinned dataset version this connectome reads from."""

    def query(
        self,
        *,
        type: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
    ) -> tuple[Neuron, ...]:
        """Return neurons matching the query. One of `type` or `ids` is required."""
        ...

    def subgraph(self, neurons: Iterable[Neuron]) -> Subgraph:
        """Pull the induced subgraph (synapses among the given neurons)."""
        ...
