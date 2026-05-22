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
    """
    rows = swc.reset_index(drop=True)
    xs = rows["x"].to_numpy(dtype=np.float64)
    ys = rows["y"].to_numpy(dtype=np.float64)
    zs = rows["z"].to_numpy(dtype=np.float64)
    radii = rows["radius"].to_numpy(dtype=np.float64)
    parent_row = rows["link"].to_numpy(dtype=np.int64)

    # Normalise to demo coords.
    cx, cy, cz = bbox_center
    pts = np.stack([xs, ys, zs], axis=1)
    pts = (pts - np.array([cx, cy, cz])) * bbox_scale
    radii_n = radii * bbox_scale

    # Optional: drop very fine processes for size.
    if min_radius > 0:
        keep = radii >= min_radius
        keep[0] = True
        idx_map = -np.ones(len(rows), dtype=np.int64)
        idx_map[keep] = np.arange(keep.sum())
        # For dropped nodes, parent becomes their nearest kept ancestor.
        rowid_to_idx = dict(zip(rows["rowId"].astype(int), rows.index, strict=True))
        new_parents = []
        kept_indices = np.flatnonzero(keep)
        for i in kept_indices:
            p = int(parent_row[i])
            steps = 0
            while p != -1 and steps < 100_000:
                pi = rowid_to_idx.get(p)
                if pi is None or pi >= len(keep):
                    p = -1
                    break
                if keep[pi]:
                    new_parents.append(int(idx_map[pi]))
                    break
                p = int(parent_row[pi])
                steps += 1
            else:
                new_parents.append(-1)
            if (
                p == -1
                and (
                    not new_parents
                    or (new_parents[-1] != -1 and len(new_parents) - 1 < len(kept_indices))
                )
                and len(new_parents) < kept_indices.tolist().index(i) + 1
            ):
                new_parents.append(-1)
        # Build outputs
        pts = pts[keep]
        radii_n = radii_n[keep]
    else:
        # Map each row's parent rowId to its index (or -1 for root).
        rowid_to_idx = dict(zip(rows["rowId"].astype(int), rows.index, strict=True))
        new_parents = [-1 if int(p) == -1 else rowid_to_idx.get(int(p), -1) for p in parent_row]

    # Build edge list: (parent_idx, child_idx) for every non-root node.
    edges: list[int] = []
    for child_idx, parent_idx in enumerate(new_parents):
        if parent_idx is None or parent_idx < 0:
            continue
        edges.append(int(parent_idx))
        edges.append(int(child_idx))

    # Round for size.
    positions = [round(v, 2) for v in pts.flatten().tolist()]
    radii_out = [round(v, 3) for v in radii_n.tolist()]

    return {
        "body_id": int(body_id),
        "n_nodes": len(pts),
        "positions": positions,
        "radii": radii_out,
        "edges": edges,
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
