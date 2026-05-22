"""H01-inspired cortical microcircuit (not the raw H01 EM data).

Background. The full H01 dataset (Shapson-Coe et al. 2024, Science) is a
~1 mm cubed sample of human temporal cortex, ~50k cells, ~130M synapses.
The raw data is hosted on Google Cloud in Neuroglancer-precomputed format
at `gs://h01-release/data/20210601/`. Pulling neurons and synapses out of
that into Galvani's `Subgraph` shape needs `cloud-volume` plus parsing of
the synapse-prediction annotation layers -- a substantial integration that
deserves its own engineering pass.

What this module ships in the meantime. A 60-neuron microcircuit with:
  - cell types drawn from the H01 paper (cortical layer 2/3 / 4 / 5
    pyramidal cells, layer 4 spiny stellate cells, three interneuron
    classes: PV-basket, SST-Martinotti, VIP-bipolar)
  - cell counts per type roughly proportional to the published H01
    statistics (excitatory ~80%, inhibitory ~20%)
  - connectivity densities per (pre, post) type pair from the H01 paper
    Figure 5 (intra-layer recurrence dense; cross-layer pyramidal pathway
    is L4 -> L2/3 -> L5; inhibitory interneurons heavily target nearby
    pyramidal cells)
  - cell positions placed by cortical layer (y-coordinate proxies depth)
  - mammalian NT signs: glutamate is EXCITATORY (opposite of fly)

This is honest: it's a representative *microcircuit*, not a literal patch
of EM. The dynamics it produces are the dynamics of a small mammalian
cortical canonical microcircuit (Douglas & Martin 2004), not specifically
the cells in H01's volume. Switching to actual H01 cells is a v1.5 follow-up
that needs `cloud-volume` plumbing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from galvani.connectome.base import Neuron, Subgraph


@dataclass(frozen=True, slots=True)
class _CellTypeSpec:
    """One H01-inspired cell type."""

    name: str
    count: int
    layer: int  # 23, 4, 5  (used as y-position)
    is_excitatory: bool
    nt: str


# Counts loosely proportional to H01 paper's per-type statistics, scaled
# down to ~60 cells total. The E:I ratio is ~4:1.
H01_TYPES: tuple[_CellTypeSpec, ...] = (
    _CellTypeSpec("L23_pyramidal", count=18, layer=23, is_excitatory=True, nt="glutamate"),
    _CellTypeSpec("L4_stellate", count=10, layer=4, is_excitatory=True, nt="glutamate"),
    _CellTypeSpec("L5_pyramidal", count=14, layer=5, is_excitatory=True, nt="glutamate"),
    _CellTypeSpec("PV_basket_L23", count=6, layer=23, is_excitatory=False, nt="gaba"),
    _CellTypeSpec("SST_Martinotti_L23", count=4, layer=23, is_excitatory=False, nt="gaba"),
    _CellTypeSpec("VIP_bipolar_L23", count=3, layer=23, is_excitatory=False, nt="gaba"),
    _CellTypeSpec("PV_basket_L5", count=5, layer=5, is_excitatory=False, nt="gaba"),
)

# (pre_type, post_type) -> approximate connection density (probability that
# a given pre cell projects to a given post cell). Numbers tuned from the
# H01 paper Fig 5 connectivity matrix + the Lefort 2009 cortical
# microcircuit. NOT a literal extraction.
_CONN_DENSITY: dict[tuple[str, str], float] = {
    # L4 sensory input pathway
    ("L4_stellate", "L4_stellate"): 0.20,
    ("L4_stellate", "L23_pyramidal"): 0.30,
    ("L4_stellate", "L5_pyramidal"): 0.10,
    # L2/3 recurrence + descending to L5
    ("L23_pyramidal", "L23_pyramidal"): 0.20,
    ("L23_pyramidal", "L5_pyramidal"): 0.25,
    ("L23_pyramidal", "L4_stellate"): 0.05,
    # L5 outputs (cortical output layer, also some recurrence)
    ("L5_pyramidal", "L5_pyramidal"): 0.15,
    ("L5_pyramidal", "L23_pyramidal"): 0.05,
    # Interneurons -> pyramidal (heavy inhibition)
    ("PV_basket_L23", "L23_pyramidal"): 0.50,
    ("PV_basket_L23", "L4_stellate"): 0.20,
    ("SST_Martinotti_L23", "L23_pyramidal"): 0.40,
    ("SST_Martinotti_L23", "L5_pyramidal"): 0.10,
    ("VIP_bipolar_L23", "SST_Martinotti_L23"): 0.40,  # disinhibition: VIP -> SST
    ("PV_basket_L5", "L5_pyramidal"): 0.50,
    # Excitatory -> interneuron drives the inhibitory loop
    ("L23_pyramidal", "PV_basket_L23"): 0.30,
    ("L23_pyramidal", "SST_Martinotti_L23"): 0.15,
    ("L4_stellate", "PV_basket_L23"): 0.20,
    ("L5_pyramidal", "PV_basket_L5"): 0.30,
    ("L5_pyramidal", "PV_basket_L23"): 0.05,
    # Mutual inhibition (sparse)
    ("PV_basket_L23", "PV_basket_L23"): 0.20,
    ("PV_basket_L5", "PV_basket_L5"): 0.20,
}


def synthetic_h01_microcircuit(
    *,
    seed: int = 0,
    weight_count_scale: float = 12.0,
    weight_count_sigma: float = 0.6,
) -> Subgraph:
    """Build the H01-inspired microcircuit Subgraph.

    `weight_count_scale` controls the per-edge mean synapse count (the
    parameterizer's log1p kicks in on top of this). Tuned so the resulting
    weight matrix has rate-model and biophysics-friendly magnitudes.
    """
    rng = np.random.default_rng(seed)

    # Allocate neuron ids by type, position them per cortical layer.
    neurons: list[Neuron] = []
    type_by_index: list[str] = []
    next_id = 1_000_000  # use a "human EM" id range disjoint from hemibrain
    layer_y = {23: 50.0, 4: 30.0, 5: 10.0}
    for spec in H01_TYPES:
        for _ in range(spec.count):
            # Spread cells within each layer in a 2D patch.
            x = float(-30.0 + 60.0 * rng.random())
            z = float(-30.0 + 60.0 * rng.random())
            y = layer_y[spec.layer] + float(2.0 * rng.standard_normal())
            neurons.append(
                Neuron(
                    id=next_id,
                    cell_type=spec.name,
                    hemisphere="C",
                    nt=spec.nt,
                    soma_position=(x, y, z),
                )
            )
            type_by_index.append(spec.name)
            next_id += 1

    # Build connectivity: for each (pre_type, post_type) pair, sample edges
    # at the configured density, weight by log-normal synapse count.
    pre_ids: list[int] = []
    post_ids: list[int] = []
    counts: list[int] = []
    nt_pre: list[str | None] = []
    type_to_neuron_idx: dict[str, list[int]] = {}
    for i, t in enumerate(type_by_index):
        type_to_neuron_idx.setdefault(t, []).append(i)

    for (pre_t, post_t), density in _CONN_DENSITY.items():
        pre_indices = type_to_neuron_idx.get(pre_t, [])
        post_indices = type_to_neuron_idx.get(post_t, [])
        if not pre_indices or not post_indices:
            continue
        for i in pre_indices:
            for j in post_indices:
                if i == j:
                    continue
                if rng.random() < density:
                    # Log-normal synapse count, mean ~ weight_count_scale
                    cnt = int(
                        max(
                            1,
                            np.round(
                                weight_count_scale
                                * np.exp(weight_count_sigma * rng.standard_normal())
                            ),
                        )
                    )
                    pre_ids.append(neurons[i].id)
                    post_ids.append(neurons[j].id)
                    counts.append(cnt)
                    nt_pre.append(neurons[i].nt)

    return Subgraph(
        neurons=tuple(neurons),
        pre_ids=np.array(pre_ids, dtype=np.int64),
        post_ids=np.array(post_ids, dtype=np.int64),
        counts=np.array(counts, dtype=np.int32),
        nt_pre=tuple(nt_pre),
        dataset_version="h01-inspired:v1",
    )


class H01StyleConnectome:
    """Connectome backend for the H01-inspired microcircuit.

    Implements the same `Connectome` protocol as `HemibrainConnectome` and
    `DTIConnectome` so the rest of the pipeline (Subgraph, Parameterizer,
    Simulator, viz) works unchanged.
    """

    def __init__(self, subgraph: Subgraph) -> None:
        self._subgraph = subgraph
        self.dataset_version = subgraph.dataset_version
        self._neurons = subgraph.neurons

    @classmethod
    def synthetic(cls, seed: int = 0) -> H01StyleConnectome:
        return cls(synthetic_h01_microcircuit(seed=seed))

    def query(
        self,
        *,
        type: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
    ) -> tuple[Neuron, ...]:
        if ids is not None:
            wanted_ids = {int(i) for i in ids}
            return tuple(n for n in self._neurons if n.id in wanted_ids)
        if type is not None:
            wanted_types: set[str] = {type} if isinstance(type, str) else set(type)
            return tuple(n for n in self._neurons if n.cell_type in wanted_types)
        return self._neurons

    def subgraph(self, neurons: Iterable[Neuron]) -> Subgraph:
        # The subgraph is fully synthesised at init time; just slice it
        # if a strict subset is requested.
        ids = {n.id for n in neurons}
        if ids == {n.id for n in self._neurons}:
            return self._subgraph
        keep = np.array(
            [
                int(p) in ids and int(q) in ids
                for p, q in zip(self._subgraph.pre_ids, self._subgraph.post_ids, strict=True)
            ],
            dtype=bool,
        )
        return Subgraph(
            neurons=tuple(n for n in self._neurons if n.id in ids),
            pre_ids=self._subgraph.pre_ids[keep],
            post_ids=self._subgraph.post_ids[keep],
            counts=self._subgraph.counts[keep],
            nt_pre=tuple(v for v, k in zip(self._subgraph.nt_pre, keep.tolist(), strict=True) if k),
            dataset_version=self._subgraph.dataset_version,
        )

    def fetch_skeleton(self, body_id: int) -> pd.DataFrame:
        """Tiny six-spoke star at the cell's soma position. The H01-style
        microcircuit doesn't carry real EM morphology -- this is enough
        for the existing skeleton-based viz to render cell positions."""
        neuron = next(n for n in self._neurons if n.id == int(body_id))
        if neuron.soma_position is None:
            raise ValueError(f"Neuron {body_id} has no soma_position")
        cx, cy, cz = neuron.soma_position
        r = 1.5
        rows = [
            (1, cx, cy, cz, 2.0, -1),
            (2, cx + r, cy, cz, 1.0, 1),
            (3, cx - r, cy, cz, 1.0, 1),
            (4, cx, cy + r, cz, 1.0, 1),
            (5, cx, cy - r, cz, 1.0, 1),
            (6, cx, cy, cz + r, 1.0, 1),
            (7, cx, cy, cz - r, 1.0, 1),
        ]
        return pd.DataFrame(rows, columns=["rowId", "x", "y", "z", "radius", "link"])


__all__ = ["H01_TYPES", "H01StyleConnectome", "synthetic_h01_microcircuit"]
