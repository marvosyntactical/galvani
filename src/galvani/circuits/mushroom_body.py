"""Mushroom body Kenyon cells + APL (Phase 7: second circuit).

Why this circuit. The plan calls for a non-HD-ring target that stresses the
pipeline's generality. The mushroom body is the canonical sparse-coding
network of the fly brain:

  - ~1900 Kenyon cells (KCs) across 10 subtypes, cholinergic, receive PN
    input (not modeled here -- represented by external drive on the KCs).
  - One APL (anterior paired lateral) neuron, GABAergic, provides global
    feedback inhibition: it pools activity across all KCs and inhibits them
    in turn. The result is k-WTA-like sparse coding (~5% of KCs active per
    odor).

This is "different topology" from HD ring on three axes simultaneously:

  1. Feedforward + global inhibition vs. dense recurrent connectivity.
  2. ~2000 neurons vs. 130.
  3. Sparse target activity (~5%) vs. broad bump (~25%).

Hemibrain gotchas the loader has to handle:

  - APL has no `type` field in hemibrain:v1.2.1 -- the `instance` is
    "APL_R" but `type` is NaN. The standard `query(type=...)` path can't
    find it. We pull it by body id and patch its `cell_type` and `nt` after
    query. The plan flagged that the second circuit would expose
    library-level gaps; this is the first one.

  - APL is a single neuron with thousands of synapses each way to the KC
    population. The synapse matrix is dense; the parameterizer's log1p
    weighting compresses the dynamic range as expected.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from galvani.connectome.base import Connectome, Neuron, Subgraph

MB_KC_TYPES: tuple[str, ...] = (
    "KCg-m",
    "KCab-m",
    "KCab-c",
    "KCab-s",
    "KCa'b'-ap2",
    "KCa'b'-m",
    "KCg-d",
    "KCa'b'-ap1",
    "KCab-p",
    "KCg-t",
)
"""All Kenyon-cell subtypes in hemibrain:v1.2.1."""

APL_BODY_ID: int = 425790257
"""The traced APL neuron in hemibrain:v1.2.1 (instance 'APL_R').

Other rows whose instance contains 'APL' are fragments or ambiguous; this
is the only one with a clean trace."""

MB_NT: dict[str, str] = {ct: "acetylcholine" for ct in MB_KC_TYPES} | {"APL": "gaba"}
"""NT lookup for the MB cell types. Pass to `HemibrainConnectome(nt_by_type=)`
so the parameterizer's sign assignment works."""


@dataclass(frozen=True, slots=True)
class MushroomBodyLayout:
    """Container for a loaded MB subgraph.

    `kc_mask[i]` is True if `subgraph.neurons[i]` is a Kenyon cell.
    `apl_index` is the integer position of APL in the canonical ordering
    (raises if APL was not loaded).
    """

    subgraph: Subgraph
    kc_mask: NDArray[np.bool_]
    apl_index: int

    @property
    def n_kc(self) -> int:
        return int(self.kc_mask.sum())

    @property
    def kc_ids(self) -> tuple[int, ...]:
        return tuple(int(n.id) for i, n in enumerate(self.subgraph.neurons) if self.kc_mask[i])


def _patch_apl(neurons: tuple[Neuron, ...]) -> tuple[Neuron, ...]:
    """Stamp APL's cell_type and NT explicitly.

    In hemibrain:v1.2.1, APL has no `type` field; `HemibrainConnectome`
    therefore brings it through with `cell_type='nan'` (string cast of
    pandas NaN) and `nt=None`. Without this fixup the parameterizer ignores
    APL's outgoing GABAergic sign and the sparse-coding feedback loop
    collapses.
    """
    return tuple(
        dataclasses.replace(n, cell_type="APL", nt="gaba") if n.id == APL_BODY_ID else n
        for n in neurons
    )


def load_mushroom_body(
    connectome: Connectome,
    *,
    kc_types: tuple[str, ...] = MB_KC_TYPES,
    apl_body_id: int = APL_BODY_ID,
) -> MushroomBodyLayout:
    """Pull KC + APL neurons + their synapses from a connectome backend.

    Args:
        connectome: any `Connectome` backend (HemibrainConnectome in v1).
        kc_types: KC subtypes to include. Defaults to all 10 in hemibrain.
        apl_body_id: APL's body id. Defaults to the hemibrain:v1.2.1 value.

    Returns:
        An `MushroomBodyLayout` with the subgraph, KC mask, and APL index.
    """
    kc_neurons = connectome.query(type=list(kc_types))
    apl_neurons = connectome.query(ids=[apl_body_id])
    neurons = _patch_apl(tuple(kc_neurons) + tuple(apl_neurons))

    subgraph = connectome.subgraph(neurons)
    # The subgraph object holds the original (un-patched) neuron tuple from
    # the backend's per-row mapping; rebuild it with the patched copies so
    # downstream code sees APL's correct cell_type/nt.
    subgraph = dataclasses.replace(subgraph, neurons=neurons)

    kc_mask = np.array([n.cell_type != "APL" for n in neurons], dtype=bool)
    apl_index_arr = np.where(~kc_mask)[0]
    if apl_index_arr.size == 0:
        raise RuntimeError(f"APL neuron (body_id={apl_body_id}) not present in the subgraph.")
    return MushroomBodyLayout(subgraph=subgraph, kc_mask=kc_mask, apl_index=int(apl_index_arr[0]))


__all__ = [
    "APL_BODY_ID",
    "MB_KC_TYPES",
    "MB_NT",
    "MushroomBodyLayout",
    "load_mushroom_body",
]
