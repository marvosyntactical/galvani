"""DTI-tractography connectome backend (synthetic toy + loader API).

Connectomes from diffusion-MRI tractography are typically supplied as a
square connectivity matrix (region x region streamline counts) plus a
parcellation labelling each row/column with a brain region. Compared to
EM connectomes, the unit of analysis is a *region* (not a single neuron),
typical sizes are 30-500 regions, and there's no synaptic polarity --
all entries are non-negative streamline counts.

The Galvani pipeline accommodates this with no change to `Subgraph` or
`Simulator`: each region becomes a "neuron" with a population time
constant, a per-region position (centroid of the parcel), and no `nt`
label (signs come from a separate region-level rule, not from NT
prediction).

This module ships:
  - `synthetic_brain_subgraph(n_regions=...)` -- a toy DTI matrix with
    bilateral symmetry, log-normal weight distribution, and stronger
    intra-than-inter-hemispheric connectivity. Useful for demos and CI.
  - `DTIConnectome` -- a minimal `Connectome` implementation around a
    user-supplied (positions, label, connectivity) tuple. Lets a real
    DTI dataset (e.g. HCP MMP1, AAL, Schaefer-400) be plugged in by
    providing those arrays directly.

The natural follow-up is a `DTIConnectome.from_nifti(...)` constructor that
reads tractography matrices from standard formats (NIfTI / `.mat` /
`.csv`); leaving that for a real-dataset iteration so v1 has a fast,
zero-network-dependency demo.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from galvani.connectome.base import Hemisphere, Neuron, Subgraph

_AAL2_DATA_FILE = Path(__file__).with_name("_aal2_data.json")


def _load_aal2_data() -> tuple[list[str], list[tuple[float, float, float]]]:
    """Read AAL2 region labels + MNI centroids from the bundled JSON."""
    with _AAL2_DATA_FILE.open() as f:
        d = json.load(f)
    coords = [(float(x), float(y), float(z)) for x, y, z in d["coords"]]
    return list(d["labels"]), coords


@dataclass(frozen=True, slots=True)
class DTIRegion:
    """One DTI parcel."""

    id: int
    label: str
    hemisphere: str  # 'L' or 'R'
    centroid: tuple[float, float, float]


def synthetic_brain_subgraph(
    n_regions: int = 30,
    *,
    seed: int = 0,
    intra_hemisphere_boost: float = 3.0,
    homotopic_boost: float = 4.0,
    sparsity: float = 0.6,
    log_normal_sigma: float = 0.7,
) -> tuple[Subgraph, tuple[DTIRegion, ...]]:
    """Synthesise a toy DTI connectome with biologically suggestive structure.

    Regions live on two parallel circles (one per hemisphere) like a
    coarse cortical "ring of rings". Connectivity is:
      - dense intra-hemisphere (with distance falloff)
      - sparse contralateral, with a strong "homotopic" boost between a
        region and its mirror (canonical DTI motif: callosal connections
        most strongly link homotopic regions)
      - weights are log-normal (heavy-tailed, matches real tractography)

    Returns the `Subgraph` plus the per-region metadata so the demo can
    render parcel centroids in 3D.
    """
    if n_regions < 4 or n_regions % 2 != 0:
        raise ValueError("n_regions must be an even integer >= 4")
    rng = np.random.default_rng(seed)

    half = n_regions // 2
    # Position each L region on a circle in x>0, R region mirrored to x<0.
    thetas = np.linspace(0, 2 * np.pi, half, endpoint=False)
    radius = 20.0
    centers = []
    regions: list[DTIRegion] = []
    for hemi_sign, side in ((1.0, "R"), (-1.0, "L")):
        for k, theta in enumerate(thetas):
            x = hemi_sign * (radius + 4 * np.cos(theta))
            y = 8.0 * np.sin(theta)
            z = 2.0 * np.cos(2 * theta)
            centers.append((x, y, z))
            regions.append(
                DTIRegion(
                    id=len(regions) + 1,
                    label=f"{side}_region_{k:02d}",
                    hemisphere=side,
                    centroid=(float(x), float(y), float(z)),
                )
            )
    centers_arr = np.array(centers)

    # Build connectivity: prior strength = exp(-distance / characteristic
    # length), then mask by sparsity, then multiply by log-normal noise.
    char_len = 12.0
    dists = np.linalg.norm(centers_arr[:, None, :] - centers_arr[None, :, :], axis=-1)
    base_strength = np.exp(-dists / char_len)

    # Intra-hemisphere boost.
    hemis = np.array([r.hemisphere for r in regions])
    same_hemi = hemis[:, None] == hemis[None, :]
    base_strength = np.where(same_hemi, base_strength * intra_hemisphere_boost, base_strength)
    # Homotopic boost: L_region_k <-> R_region_k pair.
    homotopic = np.zeros_like(base_strength, dtype=bool)
    for k in range(half):
        # R_region_k is index k; L_region_k is index k + half.
        i, j = k, k + half
        homotopic[i, j] = True
        homotopic[j, i] = True
    base_strength = np.where(homotopic, base_strength * homotopic_boost, base_strength)
    np.fill_diagonal(base_strength, 0.0)  # no autapse

    # Sparsity: drop edges below a random threshold (per cell).
    keep_mask = rng.random(base_strength.shape) > sparsity
    base_strength *= keep_mask

    # Log-normal noise.
    log_noise = rng.normal(0.0, log_normal_sigma, size=base_strength.shape)
    counts = base_strength * np.exp(log_noise)
    counts = np.maximum(counts, 0.0)
    # Make symmetric (DTI streamline counts are direction-agnostic).
    counts = 0.5 * (counts + counts.T)
    # Quantise to integer "streamline counts" so the pipeline downstream
    # sees the same int dtype as a real DTI matrix.
    counts_int = np.round(counts * 50).astype(np.int32)

    # Build the Subgraph in COO form.
    pre_idx, post_idx = np.nonzero(counts_int)
    pre_ids = np.array([regions[i].id for i in pre_idx], dtype=np.int64)
    post_ids = np.array([regions[i].id for i in post_idx], dtype=np.int64)
    weights = counts_int[pre_idx, post_idx].astype(np.int32)

    neurons = tuple(
        Neuron(
            id=r.id,
            cell_type=f"cortex_{r.hemisphere}",
            hemisphere=cast(Hemisphere, r.hemisphere),
            nt="acetylcholine",  # placeholder; region-level NT is mixed
            soma_position=r.centroid,
        )
        for r in regions
    )
    nt_pre = tuple("acetylcholine" for _ in pre_ids)
    subgraph = Subgraph(
        neurons=neurons,
        pre_ids=pre_ids,
        post_ids=post_ids,
        counts=weights,
        nt_pre=nt_pre,
        dataset_version="synthetic-dti:v1",
    )
    return subgraph, tuple(regions)


class DTIConnectome:
    """Minimal `Connectome`-shaped adapter for an in-memory DTI matrix.

    Useful for two paths:
      - the synthetic demo here, where we generate everything at runtime;
      - real DTI loaders that read from disk and hand us the matrices.
    """

    def __init__(
        self,
        subgraph: Subgraph,
        regions: tuple[DTIRegion, ...],
        dataset_version: str = "synthetic-dti:v1",
    ) -> None:
        self._subgraph = subgraph
        self._regions = regions
        self.dataset_version = dataset_version
        self._by_id = {r.id: r for r in regions}

    @classmethod
    def synthetic(cls, n_regions: int = 30, seed: int = 0) -> DTIConnectome:
        sub, regions = synthetic_brain_subgraph(n_regions=n_regions, seed=seed)
        return cls(sub, regions, dataset_version="synthetic-dti:v1")

    @classmethod
    def from_mat(
        cls,
        cm_path: str | Path,
        *,
        labels: list[str],
        coords: list[tuple[float, float, float]],
        mat_key: str = "sc",
        dataset_version: str = "dti:v1",
    ) -> DTIConnectome:
        """Load a real DTI connectome from a MATLAB-format `.mat` file.

        Args:
            cm_path: path to a `.mat` containing a 2D streamline-count matrix.
            labels: per-region label strings, length N (matches matrix size).
            coords: per-region 3-tuple positions, length N.
            mat_key: which variable inside the .mat to use (default `sc` per
                the neurolib convention).
            dataset_version: pinned version string for reproducibility.

        Returns a `DTIConnectome` ready for `query()` and `subgraph()`.
        """
        import scipy.io as sio

        mat = sio.loadmat(str(cm_path))
        if mat_key not in mat:
            raise KeyError(
                f"{cm_path}: key {mat_key!r} not found "
                f"(available: {[k for k in mat if not k.startswith('_')]})"
            )
        matrix = np.asarray(mat[mat_key])
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            raise ValueError(f"Expected a square matrix, got shape {matrix.shape}")
        if len(labels) != n or len(coords) != n:
            raise ValueError(f"labels/coords length {len(labels)}/{len(coords)} != matrix size {n}")

        # Real DTI matrices are slightly asymmetric due to seeding direction.
        # Symmetrize for downstream use.
        matrix = 0.5 * (matrix + matrix.T)
        matrix = matrix.astype(np.int64)
        np.fill_diagonal(matrix, 0)

        # Hemisphere heuristic: trailing _L / _R in the label, fall back to
        # the sign of the x-centroid (positive = R in standard MNI).
        regions: list[DTIRegion] = []
        for i, (lbl, c) in enumerate(zip(labels, coords, strict=True)):
            if lbl.endswith("_L") or lbl.endswith("_l"):
                hemi = "L"
            elif lbl.endswith("_R") or lbl.endswith("_r"):
                hemi = "R"
            else:
                hemi = "R" if c[0] > 0 else "L"
            regions.append(
                DTIRegion(
                    id=i + 1,
                    label=lbl,
                    hemisphere=hemi,
                    centroid=(float(c[0]), float(c[1]), float(c[2])),
                )
            )

        pre_idx, post_idx = np.nonzero(matrix)
        pre_ids = np.array([regions[i].id for i in pre_idx], dtype=np.int64)
        post_ids = np.array([regions[i].id for i in post_idx], dtype=np.int64)
        weights = matrix[pre_idx, post_idx].astype(np.int32)

        neurons = tuple(
            Neuron(
                id=r.id,
                cell_type=f"cortex_{r.hemisphere}",
                hemisphere=cast(Hemisphere, r.hemisphere),
                nt="acetylcholine",
                soma_position=r.centroid,
            )
            for r in regions
        )
        nt_pre = tuple("acetylcholine" for _ in pre_ids)
        subgraph = Subgraph(
            neurons=neurons,
            pre_ids=pre_ids,
            post_ids=post_ids,
            counts=weights,
            nt_pre=nt_pre,
            dataset_version=dataset_version,
        )
        return cls(subgraph, tuple(regions), dataset_version=dataset_version)

    @classmethod
    def aal2_hcp_subject(
        cls,
        cm_path: str | Path,
    ) -> DTIConnectome:
        """Convenience: load a 94-region AAL2 DTI matrix using bundled
        region labels and MNI centroids. Used by the demo to ship a real
        human DTI connectome.
        """
        labels, coords = _load_aal2_data()
        return cls.from_mat(
            cm_path,
            labels=labels,
            coords=coords,
            mat_key="sc",
            dataset_version="hcp-aal2:v1",
        )

    def query(
        self,
        *,
        type: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
    ) -> tuple[Neuron, ...]:
        if ids is not None:
            wanted = set(int(i) for i in ids)
            return tuple(n for n in self._subgraph.neurons if n.id in wanted)
        if type is not None:
            wanted_types = {type} if isinstance(type, str) else set(type)
            return tuple(n for n in self._subgraph.neurons if n.cell_type in wanted_types)
        return self._subgraph.neurons

    def subgraph(self, neurons: Iterable[Neuron]) -> Subgraph:
        ids = {n.id for n in neurons}
        if ids == {n.id for n in self._subgraph.neurons}:
            return self._subgraph
        # Reduce to a sub-subgraph if a strict subset was requested.
        keep = np.array(
            [
                int(p) in ids and int(q) in ids
                for p, q in zip(self._subgraph.pre_ids, self._subgraph.post_ids, strict=True)
            ],
            dtype=bool,
        )
        return Subgraph(
            neurons=tuple(n for n in self._subgraph.neurons if n.id in ids),
            pre_ids=self._subgraph.pre_ids[keep],
            post_ids=self._subgraph.post_ids[keep],
            counts=self._subgraph.counts[keep],
            nt_pre=tuple(v for v, k in zip(self._subgraph.nt_pre, keep.tolist(), strict=True) if k),
            dataset_version=self._subgraph.dataset_version,
        )

    def fetch_skeleton(self, body_id: int) -> pd.DataFrame:
        """Return a minimal 'skeleton' for a region: a 6-point star around
        the centroid, so the skeleton-stride downsampler in viz.payload
        produces a tiny tree per region. Lets us reuse the existing viz
        pipeline unchanged."""
        r = self._by_id[int(body_id)]
        cx, cy, cz = r.centroid
        radius = 1.5
        # Root + six axial spokes
        rows = [
            (1, cx, cy, cz, 2.0, -1),
            (2, cx + radius, cy, cz, 1.0, 1),
            (3, cx - radius, cy, cz, 1.0, 1),
            (4, cx, cy + radius, cz, 1.0, 1),
            (5, cx, cy - radius, cz, 1.0, 1),
            (6, cx, cy, cz + radius, 1.0, 1),
            (7, cx, cy, cz - radius, 1.0, 1),
        ]
        return pd.DataFrame(rows, columns=["rowId", "x", "y", "z", "radius", "link"])


__all__ = ["DTIConnectome", "DTIRegion", "synthetic_brain_subgraph"]
