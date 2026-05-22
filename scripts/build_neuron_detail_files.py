"""Bake per-neuron high-resolution JSON files for the demo's detail view.

Reads the per-neuron skeleton parquets we already cached under
`tests/fixtures/.../skeletons/` and writes one tiny JSON per neuron to
`examples/web_demo/public/neurons/{dataset_id}/{body_id}.json`. The frontend
only fetches one of these when the user clicks a neuron to enter detail
mode -- so the default page load is unaffected.

Per-neuron file size: ~30-100 KB depending on the cell. ~330 files total
across the HD ring + MB subset. Bundled into the static deploy.

Coordinates are normalised with the SAME bbox as the all-neuron payload of
the same dataset, so the focused neuron sits exactly where it was in the
full view.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from galvani.circuits.hd_ring import load_hd_ring
from galvani.circuits.mushroom_body import MB_NT, load_mushroom_body
from galvani.connectome.cache import ParquetCache
from galvani.connectome.hemibrain import HD_RING_NT, HemibrainConnectome


def _build_per_neuron(
    body_id: int,
    swc,
    bbox_center: tuple[float, float, float],
    bbox_scale: float,
    min_radius: float = 0.0,
) -> dict[str, Any]:
    """Convert one SWC DataFrame to a compact JSON dict.

    Format:
      positions: flat (x,y,z) array, length 3*N_nodes
      radii:     length N_nodes
      edges:     flat (parent_idx, child_idx) array, length 2*(N_nodes-1)
      parents:   length N_nodes, parent compartment idx (-1 for the soma /
                 disconnected roots). Forms the spanning tree needed by the
                 multi-compartment cable solver.
      compartment_length_nm: length N_nodes, Euclidean distance in raw nm
                 from each node to its parent (0 for roots). Needed for
                 axial conductance computation in the bio sim.
      soma_idx:  index of the inferred soma compartment (largest-radius
                 node in the skeleton — more reliable than "root" for
                 hemibrain SWCs, which often have disconnected branch
                 fragments with link = -1).
    """
    rows = swc.reset_index(drop=True)
    xs_nm = rows["x"].to_numpy(dtype=np.float64)
    ys_nm = rows["y"].to_numpy(dtype=np.float64)
    zs_nm = rows["z"].to_numpy(dtype=np.float64)
    radii_nm = rows["radius"].to_numpy(dtype=np.float64)
    parent_row = rows["link"].to_numpy(dtype=np.int64)

    rowid_to_orig_idx = dict(zip(rows["rowId"].astype(int), rows.index, strict=True))
    orig_parent_idx = np.array(
        [-1 if int(p) == -1 else rowid_to_orig_idx.get(int(p), -1) for p in parent_row],
        dtype=np.int64,
    )

    # Decide which original-index nodes survive the min_radius filter.
    if min_radius > 0:
        keep_mask = radii_nm >= min_radius
        keep_mask[0] = True  # always keep row 0 (typically the seed root)
    else:
        keep_mask = np.ones(len(rows), dtype=bool)

    # For each kept node, find its nearest kept ancestor under `orig_parent_idx`.
    # `new_idx[orig]` = its index in the kept list, or -1 if filtered out.
    new_idx = -np.ones(len(rows), dtype=np.int64)
    kept_orig = np.flatnonzero(keep_mask)
    new_idx[kept_orig] = np.arange(len(kept_orig))

    parents: list[int] = []
    parent_orig_for_kept: list[int] = []
    for orig in kept_orig:
        ancestor = int(orig_parent_idx[orig])
        # Walk up until we find a kept ancestor (or hit the root).
        steps = 0
        while ancestor != -1 and not keep_mask[ancestor] and steps < 1_000_000:
            ancestor = int(orig_parent_idx[ancestor])
            steps += 1
        parents.append(int(new_idx[ancestor]) if ancestor != -1 else -1)
        parent_orig_for_kept.append(ancestor)

    # Geometry: kept-node positions in raw nm (for compartment_length) and
    # in scaled demo space (for positions emitted to JSON).
    pts_nm = np.stack([xs_nm, ys_nm, zs_nm], axis=1)[kept_orig]
    radii_n = (radii_nm * bbox_scale)[kept_orig]
    cx, cy, cz = bbox_center
    pts_scaled = (pts_nm - np.array([cx, cy, cz])) * bbox_scale

    # Compartment lengths (Euclidean distance to parent in raw nm). Roots
    # take 0 — they're treated as the boundary of the cable for the solver.
    compartment_length_nm = np.zeros(len(kept_orig), dtype=np.float64)
    for i_kept, p_orig in enumerate(parent_orig_for_kept):
        if p_orig == -1:
            continue
        delta = pts_nm[i_kept] - np.stack(
            [xs_nm[p_orig], ys_nm[p_orig], zs_nm[p_orig]],
        )
        compartment_length_nm[i_kept] = float(np.linalg.norm(delta))

    # Soma = largest-radius node among the kept set. More reliable than
    # picking the SWC root, because hemibrain skeletons routinely ship with
    # disconnected branch fragments whose link is -1.
    soma_idx = int(np.argmax(radii_nm[kept_orig]))

    # Edge list (kept for the existing renderer).
    edges: list[int] = []
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            continue
        edges.append(parent_idx)
        edges.append(child_idx)

    positions = [round(v, 2) for v in pts_scaled.flatten().tolist()]
    radii_out = [round(v, 3) for v in radii_n.tolist()]
    comp_len_out = [round(v, 1) for v in compartment_length_nm.tolist()]

    return {
        "body_id": int(body_id),
        "n_nodes": len(kept_orig),
        "positions": positions,
        "radii": radii_out,
        "edges": edges,
        "parents": parents,
        "compartment_length_nm": comp_len_out,
        "soma_idx": soma_idx,
    }


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    fixtures = repo / "tests" / "fixtures"
    out_root = repo / "examples" / "web_demo" / "public" / "neurons"
    out_root.mkdir(parents=True, exist_ok=True)
    cache = ParquetCache(fixtures)

    # ----- HD ring -----
    conn = HemibrainConnectome(cache=cache, nt_by_type=HD_RING_NT)
    layout = load_hd_ring(conn)
    # Recompute the same bbox the main payload would use, so positions match.
    # We have to look up the bbox from the baked all-neuron payload to be
    # safe: load it and read .bbox.
    main_payload_path = repo / "examples" / "web_demo" / "public" / "hd_ring_tracking.json"
    with main_payload_path.open() as f:
        main_pl = json.load(f)
    bbox = main_pl["bbox"]
    center = tuple(bbox["center"])  # type: ignore[assignment]
    scale = float(bbox["scale"])
    print(f"\n[hd_ring] N={len(layout.subgraph.neurons)}  bbox center={center}  scale={scale:.5f}")

    hd_out = out_root / "hd_ring"
    hd_out.mkdir(parents=True, exist_ok=True)
    total = 0
    for neuron in layout.subgraph.neurons:
        swc = conn.fetch_skeleton(int(neuron.id))
        data = _build_per_neuron(int(neuron.id), swc, center, scale, min_radius=4.0)  # type: ignore[arg-type]
        path = hd_out / f"{int(neuron.id)}.json"
        with path.open("w") as f:
            json.dump(data, f, separators=(",", ":"))
        total += path.stat().st_size
    print(f"  wrote {len(layout.subgraph.neurons)} per-neuron files, total {total / 1024:.0f} KiB")

    # ----- Mushroom body subset -----
    # We need the same subset that the main MB payload uses (otherwise
    # neurons won't match between the overview and detail). Easiest: read
    # the IDs from the main MB payload.
    mb_payload_path = repo / "examples" / "web_demo" / "public" / "mushroom_body_with_apl.json"
    with mb_payload_path.open() as f:
        mb_pl = json.load(f)
    mb_bbox = mb_pl["bbox"]
    mb_center = tuple(mb_bbox["center"])  # type: ignore[assignment]
    mb_scale = float(mb_bbox["scale"])
    mb_ids = [n["id"] for n in mb_pl["neurons"]]
    print(f"\n[mushroom_body] N={len(mb_ids)}  bbox center={mb_center}  scale={mb_scale:.5f}")

    conn_mb = HemibrainConnectome(cache=cache, nt_by_type=MB_NT)
    # Make sure load_mushroom_body has populated the full skeleton cache.
    _ = load_mushroom_body(conn_mb)
    mb_out = out_root / "mushroom_body"
    mb_out.mkdir(parents=True, exist_ok=True)
    total = 0
    for body_id in mb_ids:
        try:
            swc = conn_mb.fetch_skeleton(int(body_id))
        except Exception as e:
            print(f"  skip {body_id}: {e}")
            continue
        data = _build_per_neuron(int(body_id), swc, mb_center, mb_scale, min_radius=4.0)  # type: ignore[arg-type]
        path = mb_out / f"{int(body_id)}.json"
        with path.open("w") as f:
            json.dump(data, f, separators=(",", ":"))
        total += path.stat().st_size
    print(f"  wrote {len(mb_ids)} per-neuron files, total {total / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
