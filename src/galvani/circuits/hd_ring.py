"""Convenience constructors and angular layout for the Drosophila HD ring.

The HD ring is a recurrent network in the central complex that maintains a
heading-correlated activity bump. The canonical cell types in hemibrain are:

  - EPG: cholinergic, tile EB wedges and PB glomeruli (~46 in hemibrain:v1.2.1)
  - PEN_a(PEN1): cholinergic, velocity-driven asymmetric loop (~20)
  - PEN_b(PEN2): cholinergic, velocity-driven asymmetric loop (~22)
  - PEG: cholinergic, supports the loop (~18; not included in the minimal set)
  - Delta7: glutamatergic (inhibitory in fly via GluClalpha), broad PB-wide
    inhibition (~42)

For the v1 validation we use the minimal four-type set {EPG, PEN_a, PEN_b,
Delta7} -- 130 neurons total in hemibrain:v1.2.1. Including PEG and EL (Hulse
et al. 2021) is left to follow-up work.

Angular layout: we offer two methods.

  * **Spectral (default)**: derive ring coordinates from the symmetric weight
    matrix. The top two non-mean eigenvectors of `0.5*(W + W.T)` are the
    cos/sin ring modes; `arctan2` of them gives each neuron its empirical
    angular position. This works for any connectome that has ring topology
    -- no label parsing, no dataset-specific conventions. The Phase 5
    validation uses this.

  * **Instance-label**: parse the PB glomerulus index from hemibrain's
    `instance` field (e.g. 'EPG(PB08)_L3' -> glomerulus 3). This is fragile:
    the canonical Wolff & Rubin (2018) L<k> <-> R<9-k> EB-wedge mapping is
    not directly recoverable from the label alone, and connectome-derived
    bumps don't necessarily align with naive L<k>/R<k> -> angle conventions.
    Kept available for reference but not the default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from galvani.connectome.base import Connectome, Neuron, Subgraph
from galvani.connectome.hemibrain import HD_RING_NT

HD_RING_TYPES: tuple[str, ...] = ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "Delta7")
"""Minimal cell-type set used by the v1 HD-ring validation."""

N_PB_GLOMERULI_PER_SIDE = 8
"""Number of distinct angular positions per PB hemisphere."""

# Instance suffix for EPG / PEN cells: ..._L<k> or ..._R<k> at end of string.
_PB_GLOMERULUS_RX = re.compile(r"_([LR])(\d+)$")


def _pb_glomerulus(instance: str | None) -> int | None:
    """Extract the PB glomerulus index k from an EPG/PEN instance label."""
    if instance is None:
        return None
    match = _PB_GLOMERULUS_RX.search(instance)
    if match is None:
        return None
    return int(match.group(2))


def angle_of(neuron: Neuron) -> float | None:
    """Return the angular position of a neuron on the ring, or None if not
    defined for this cell type / instance.

    Angles are in [0, 2*pi); EPG/PEN cells map by PB glomerulus index modulo
    `N_PB_GLOMERULI_PER_SIDE`. Delta7 spans multiple glomeruli and has no
    single angle (returns None).
    """
    if neuron.cell_type not in ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG"):
        return None
    # The Neuron dataclass doesn't carry `instance`; we encode the PB
    # glomerulus into the neuron at load time via a separate helper.
    # Callers should use `assign_angles_from_instances` below instead of this
    # function when working with raw `Neuron` objects.
    return None


def assign_angles_from_instances(
    neurons: tuple[Neuron, ...],
    instances: dict[int, str],
) -> dict[int, float]:
    """Compute the angular position of each EPG/PEN neuron from its
    `instance` label.

    Args:
        neurons: tuple of `Neuron` objects (any cell types).
        instances: maps body id -> instance label (the loader has it; the
            `Neuron` dataclass does not carry it directly).

    Returns:
        A dict mapping body id -> angle in [0, 2*pi). Neurons without a
        derivable angle (Delta7, types unknown, instances without `_L<k>` /
        `_R<k>`) are absent from the result.
    """
    out: dict[int, float] = {}
    angular_types = {"EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG"}
    for n in neurons:
        if n.cell_type not in angular_types:
            continue
        inst = instances.get(n.id)
        k = _pb_glomerulus(inst)
        if k is None:
            continue
        k_mod = (k - 1) % N_PB_GLOMERULI_PER_SIDE
        out[n.id] = 2.0 * np.pi * k_mod / N_PB_GLOMERULI_PER_SIDE
    return out


@dataclass(frozen=True, slots=True)
class HDRingLayout:
    """Geometry of a loaded HD-ring subgraph.

    `angles[i]` is the angular position of `subgraph.neurons[i]` in
    [0, 2*pi); `has_angle` is a boolean mask matching the canonical neuron
    ordering (False where the layout is undefined).
    """

    subgraph: Subgraph
    angles: NDArray[np.float64]
    has_angle: NDArray[np.bool_]

    @property
    def neuron_ids(self) -> tuple[int, ...]:
        return tuple(n.id for n in self.subgraph.neurons)


def spectral_angles(subgraph: Subgraph, *, symmetrize: bool = True) -> NDArray[np.float64]:
    """Derive per-neuron ring angles from the symmetric weight matrix.

    Builds the unparameterized weight matrix from the subgraph (count + sign,
    optionally symmetrized), then takes the top two non-mean eigenvectors of
    its symmetric part as the cos/sin ring modes. Each neuron's angle is the
    `arctan2` of its projection on these two modes.

    This works whenever the connectome has ring topology -- the cos/sin pair
    will dominate the spectrum after the trivial mean mode. For non-ring
    networks the output is meaningless but harmless.

    Returns:
        An array of length `len(subgraph.neurons)`, values in [0, 2*pi). The
        absolute phase is arbitrary (only relative positions are meaningful).
    """
    from galvani.parameterize.defaults import ParameterizerOptions, default_parameterizer

    # Build a "structural" weight matrix without any global-gain rescaling
    # (gain just scales every eigenvalue uniformly and doesn't affect the
    # eigenvectors).
    spec = default_parameterizer(
        subgraph,
        ParameterizerOptions(symmetrize=symmetrize, global_gain=1.0),
    )
    w_sym = 0.5 * (spec.weights + spec.weights.T)
    _vals, vecs = np.linalg.eigh(w_sym)
    # Top eigenvalue is typically the "all-active" mean mode; the next two
    # are the ring (cos, sin) pair (degenerate in a perfectly symmetric
    # ring, near-degenerate on a real connectome).
    v_cos = vecs[:, -2]
    v_sin = vecs[:, -3]
    angles: NDArray[np.float64] = np.arctan2(v_sin, v_cos) % (2.0 * np.pi)
    return angles


def load_hd_ring(
    connectome: Connectome,
    *,
    types: tuple[str, ...] = HD_RING_TYPES,
    layout: str = "spectral",
    symmetrize: bool = True,
) -> HDRingLayout:
    """Pull the HD-ring subgraph from a connectome backend and attach
    per-neuron angular positions.

    Args:
        connectome: any `Connectome` backend.
        types: cell types to include in the HD-ring subgraph.
        layout: either `'spectral'` (default; derive angles from the
            symmetric weight matrix) or `'instance'` (parse PB glomerulus
            from hemibrain `instance` labels; HemibrainConnectome-only).
        symmetrize: only used for `layout='spectral'`; controls whether the
            weight matrix is symmetrized before the eigen-decomposition.

    Returns:
        An `HDRingLayout` carrying the subgraph and per-neuron angles.
    """
    from galvani.connectome.hemibrain import HemibrainConnectome

    flat_types = list(types)
    neurons = connectome.query(type=flat_types)
    subgraph = connectome.subgraph(neurons)

    if layout == "spectral":
        angles = spectral_angles(subgraph, symmetrize=symmetrize)
        has_angle = np.ones(len(subgraph.neurons), dtype=bool)
    elif layout == "instance":
        instances: dict[int, str] = {}
        if isinstance(connectome, HemibrainConnectome):
            for t in flat_types:
                df = connectome._fetch_neurons_df(type_label=t)
                for _, row in df.iterrows():
                    instances[int(row["bodyId"])] = str(row.get("instance", ""))
        angle_by_id = assign_angles_from_instances(neurons, instances)
        angles = np.array(
            [angle_by_id.get(n.id, np.nan) for n in subgraph.neurons], dtype=np.float64
        )
        has_angle = ~np.isnan(angles)
    else:
        raise ValueError(f"Unknown layout {layout!r}; choose 'spectral' or 'instance'.")
    return HDRingLayout(subgraph=subgraph, angles=angles, has_angle=has_angle)


__all__ = [
    "HD_RING_NT",
    "HD_RING_TYPES",
    "N_PB_GLOMERULI_PER_SIDE",
    "HDRingLayout",
    "angle_of",
    "assign_angles_from_instances",
    "load_hd_ring",
    "spectral_angles",
]
