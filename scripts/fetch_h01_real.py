"""Pull a small real-EM neighbourhood out of H01 and cache it for Galvani.

H01 (Shapson-Coe et al. 2024, Science) is ~1 mm³ of human temporal cortex
reconstructed at ~8×8×33 nm voxel resolution. The release at
`gs://h01-release/data/20210601/` is a Neuroglancer-format dataset; this
script reaches into:

  • `c3/segment_properties/`     — per-segment tags (layer, cell type)
  • `c3/skeletons/`              — pre-computed skeletons (sharded)
  • `c3/synapses/precomputed/`   — synapse annotation layer with bi-
                                    directional pre/post indexes
  • `c3/tables/somas.csv`        — soma coords per cell

The output is a small parquet bundle under
`tests/fixtures/h01_real/` that downstream `H01EmConnectome` can load
without ever talking to GCS again. Targets ~50 well-labelled neurons (a
mix of pyramidal classes across L2-5 + a smaller interneuron pool), pulls
their skeletons (downsampled later by the visualizer / solver), and
recovers the within-set (pre, post, count) connectivity by intersecting
the synapse index from both directions.

Run once:

    uv run python scripts/fetch_h01_real.py

It takes ~2-3 minutes on a residential connection. Cached output is
re-used by subsequent runs.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

H01_ROOT = "https://storage.googleapis.com/h01-release/data/20210601"


def _segment_properties() -> dict:
    """Fetch the per-segment property table (all 46 637 cells)."""
    r = requests.get(f"{H01_ROOT}/c3/segment_properties/info", timeout=60)
    r.raise_for_status()
    return r.json()


def _index_tags(props: dict) -> tuple[list[int], list[str], list[list[int]]]:
    """Pull (ids, tag-names, per-id tag-indices) from the segment-properties info."""
    inline = props["inline"]
    ids = [int(s) for s in inline["ids"]]
    tags_prop = next(p for p in inline["properties"] if p.get("type") == "tags")
    tag_names = tags_prop["tags"]
    values = tags_prop["values"]  # list of int lists
    return ids, tag_names, values


def _has_tag(tag_idxs: list[int], tag_names: list[str], wanted: str) -> bool:
    name_to_idx = {n: i for i, n in enumerate(tag_names)}
    if wanted not in name_to_idx:
        return False
    return name_to_idx[wanted] in tag_idxs


def _load_soma_table() -> pd.DataFrame:
    """Per-segment soma coords + cell-type + layer for every cell.
    Positions are in c3-voxel units (8×8×33 nm)."""
    url = f"{H01_ROOT}/c3/tables/somas.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    return df


# Voxel sides in nm — used to convert the soma table's x/y/z into
# the same nm-frame the skeletons live in.
VX_NM = np.array([8.0, 8.0, 33.0])

# Spatial cluster centre + half-extent (in nm). Picked to land in a
# region with all six cortical layers represented; coords from the H01
# Neuroglancer default view. ±100 µm in xy keeps the cluster small
# enough that random-pair connectivity is realistic.
# Centre B in the cluster-survey: lands across L4 and L5 with a small
# admixture of L3 and L1 — gives a four-layer cortical column slice. A
# ±100 µm window in xy + ±60 µm in z keeps the cell count manageable
# (~100-150) while still being dense enough that random pairs share
# synapses.
CLUSTER_CENTER_NM = np.array([1.5e6, 1.0e6, 90_000.0])
CLUSTER_HALF_NM = np.array([100_000.0, 100_000.0, 60_000.0])
# Hard cap on the number of targets even if the cluster has more.
MAX_TARGETS = 130


def _pick_spatial_cluster(soma_df: pd.DataFrame) -> tuple[list[int], dict[int, str], dict[int, str]]:
    """Take every well-typed neuron whose soma lands inside the cluster
    box. Returns (seg_ids, class_by_id, layer_by_id) — same shape as the
    old random picker, but the neurons are now within touching distance
    of each other, so pre/post pair connectivity is non-zero."""
    neurons = soma_df[
        soma_df["celltype"].isin(["PYRAMIDAL", "INTERNEURON", "SPINY_STELLATE"])
        & ~soma_df["layer"].isin(["White matter", "unclassified"])
    ].copy()
    nm = neurons[["x", "y", "z"]].to_numpy() * VX_NM
    in_box = np.all(np.abs(nm - CLUSTER_CENTER_NM) <= CLUSTER_HALF_NM, axis=1)
    sel = neurons[in_box].copy()
    print(f"  cluster center (nm): {CLUSTER_CENTER_NM.tolist()}")
    print(f"  cluster half-extent (nm): {CLUSTER_HALF_NM.tolist()}")
    print(f"  neurons in cluster: {len(sel)}")

    class_map = {
        "PYRAMIDAL": "pyramidal",
        "INTERNEURON": "interneuron",
        "SPINY_STELLATE": "spiny_stellate",
    }
    layer_map = {
        "Layer 1": "L1", "Layer 2": "L2", "Layer 3": "L3",
        "Layer 4": "L4", "Layer 5": "L5", "Layer 6": "L6",
    }
    target_ids: list[int] = []
    class_by_id: dict[int, str] = {}
    layer_by_id: dict[int, str] = {}
    for _, row in sel.iterrows():
        raw = row["c3_rep_strict"]
        if pd.isna(raw) or raw == 0:
            continue
        sid = int(raw)
        target_ids.append(sid)
        class_by_id[sid] = class_map[row["celltype"]]
        layer_by_id[sid] = layer_map.get(row["layer"], "unknown")
    # Dedupe just in case.
    seen = set()
    deduped = []
    for s in target_ids:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    # Cap so the fetch finishes in a few minutes.
    if len(deduped) > MAX_TARGETS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(deduped), size=MAX_TARGETS, replace=False)
        deduped = sorted([deduped[i] for i in idx])
        print(f"  capped at {MAX_TARGETS}")
    return deduped, class_by_id, layer_by_id


def _fetch_skeletons(seg_ids: list[int], cv_skel) -> dict[int, dict]:
    """One sharded lookup per id (the skeletons live in 2 GB shards but
    `cloudvolume` decompresses per-id and caches HTTP responses well).

    Returns vertices + edges + radii as a serialisable dict per id. Raw
    coordinates are in nm (cv.skeleton internally multiplies by mip-0
    resolution = 8×8×33 nm).
    """
    out: dict[int, dict] = {}
    for i, sid in enumerate(seg_ids):
        try:
            sk = cv_skel.get(sid)
        except Exception as e:
            print(f"  skip {sid}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        verts = np.asarray(sk.vertices, dtype=np.float32)
        edges = np.asarray(sk.edges, dtype=np.int32)
        radii = np.asarray(
            sk.radius if hasattr(sk, "radius") else sk.radii,
            dtype=np.float32,
        )
        out[sid] = {
            "vertices_nm": verts,  # (N, 3)
            "edges": edges,  # (M, 2)
            "radii_nm": radii,  # (N,)
        }
        if (i + 1) % 5 == 0:
            print(f"  fetched {i + 1}/{len(seg_ids)} skeletons")
    return out


def _fetch_synapses_for_set(
    target_ids: list[int],
    src,
) -> dict:
    """Pull both `pre_synaptic_cell` and `post_synaptic_cell` indexes for
    every target, then intersect to recover within-set connectivity.

    Each target gets one query per direction (~2 sharded lookups each).
    Total: 2N. Memory is bounded — each target has ~100-1000 synapses.

    Returns:
      synapses_by_pre[seg_id]  -> MultiLabelAnnotation
      synapses_by_post[seg_id] -> MultiLabelAnnotation
    """
    pre_results: dict[int, object] = {}
    post_results: dict[int, object] = {}
    for i, sid in enumerate(target_ids):
        try:
            pre_results[sid] = src.get_by_relationship("pre_synaptic_cell", sid)
        except Exception as e:
            print(f"  pre lookup failed for {sid}: {type(e).__name__}: {e}", file=sys.stderr)
            pre_results[sid] = None
        try:
            post_results[sid] = src.get_by_relationship("post_synaptic_cell", sid)
        except Exception as e:
            print(f"  post lookup failed for {sid}: {type(e).__name__}: {e}", file=sys.stderr)
            post_results[sid] = None
        if (i + 1) % 5 == 0:
            print(f"  pulled synapse indexes for {i + 1}/{len(target_ids)} cells")
    return {"by_pre": pre_results, "by_post": post_results}


def _build_connectivity(
    target_ids: list[int],
    syn: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Intersect synapse-id sets to get within-set (pre, post, count) rows.
    Synapse type (excitatory / inhibitory) is taken from whichever side
    has the record — both halves of the index point at the same record."""
    target_set = set(target_ids)

    # For each pre target: synapse_id -> (post_xyz_nm, syn_type_int)
    pre_lookup: dict[int, dict[int, tuple[np.ndarray, int]]] = {}
    for pre, ann in syn["by_pre"].items():
        if ann is None:
            continue
        ids = ann.ids
        # geometry rows: pre_xyz (cols 0-2), post_xyz (cols 3-5), in voxels.
        geom = ann.geometry
        types = ann.properties.get("type", np.zeros(len(ids), dtype=np.uint32))
        pre_lookup[pre] = {
            int(sid): (geom[i, 3:6].copy(), int(types[i]))
            for i, sid in enumerate(ids)
        }

    # For each post target: set of synapse IDs.
    post_lookup: dict[int, set[int]] = {}
    for post, ann in syn["by_post"].items():
        if ann is None:
            continue
        post_lookup[post] = set(int(sid) for sid in ann.ids)

    # Intersect: for each (pre, post) target pair, syn IDs in common.
    rows = []
    synapses = []
    for pre in target_ids:
        if pre not in pre_lookup:
            continue
        pre_syn = pre_lookup[pre]
        for post in target_ids:
            if post == pre or post not in post_lookup:
                continue
            shared = set(pre_syn.keys()) & post_lookup[post]
            if not shared:
                continue
            # Sign: type=1 inhibitory, type=2 excitatory. Use the majority.
            excitatory = sum(1 for sid in shared if pre_syn[sid][1] == 2)
            inhibitory = sum(1 for sid in shared if pre_syn[sid][1] == 1)
            count = len(shared)
            rows.append(
                {
                    "pre": pre,
                    "post": post,
                    "count": count,
                    "n_excitatory": excitatory,
                    "n_inhibitory": inhibitory,
                }
            )
            for sid in shared:
                post_xyz, syn_type = pre_syn[sid]
                synapses.append(
                    {
                        "syn_id": sid,
                        "pre": pre,
                        "post": post,
                        "post_x_vx": float(post_xyz[0]),
                        "post_y_vx": float(post_xyz[1]),
                        "post_z_vx": float(post_xyz[2]),
                        "type_code": syn_type,
                    }
                )
    return pd.DataFrame(rows), synapses


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "tests" / "fixtures" / "h01_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== H01 real-EM fetcher ===")
    print()
    print("[1/5] soma table → spatial cluster …", flush=True)
    soma_df = _load_soma_table()
    print(f"  somas.csv: {len(soma_df)} rows total")
    target_ids, class_by_id, layer_by_id = _pick_spatial_cluster(soma_df)
    print(f"  picked {len(target_ids)} target cells in cluster")
    cls_hist = Counter(class_by_id[s] for s in target_ids)
    lay_hist = Counter(layer_by_id[s] for s in target_ids)
    print(f"  class histogram: {dict(cls_hist)}")
    print(f"  layer histogram: {dict(lay_hist)}")

    print()
    print("[2/5] cloud-volume sources …", flush=True)
    from cloudvolume import CloudVolume
    from cloudvolume.datasource.precomputed.annotation import (
        PrecomputedAnnotationSource,
    )

    cv = CloudVolume(
        f"{H01_ROOT}/c3", progress=False, fill_missing=True, use_https=True
    )
    syn_src = PrecomputedAnnotationSource(
        f"{H01_ROOT}/c3/synapses/precomputed",
        use_https=True,
        progress=False,
    )
    print("  ok")

    print()
    print(f"[3/5] skeletons ({len(target_ids)} cells) …", flush=True)
    t0 = time.time()
    skels = _fetch_skeletons(target_ids, cv.skeleton)
    print(f"  fetched {len(skels)}/{len(target_ids)} in {time.time() - t0:.1f}s")

    # Drop targets we couldn't fetch a skeleton for (rare but possible).
    target_ids = [s for s in target_ids if s in skels]

    print()
    print(f"[4/5] synapse indexes (2 × {len(target_ids)} lookups) …", flush=True)
    t0 = time.time()
    syn = _fetch_synapses_for_set(target_ids, syn_src)
    print(f"  done in {time.time() - t0:.1f}s")

    print()
    print("[5/5] within-set connectivity intersection …", flush=True)
    conn_df, syn_records = _build_connectivity(target_ids, syn)
    print(f"  {len(conn_df)} non-zero (pre,post) pairs")
    if len(conn_df) > 0:
        print(f"  total synapses in set: {int(conn_df['count'].sum())}")
        print(f"  excitatory: {int(conn_df['n_excitatory'].sum())} · "
              f"inhibitory: {int(conn_df['n_inhibitory'].sum())}")

    # ---- write cache ----
    neurons_meta = pd.DataFrame(
        [
            {
                "seg_id": s,
                "broad_class": class_by_id[s],
                "layer": layer_by_id[s],
            }
            for s in target_ids
        ]
    )
    neurons_meta.to_parquet(out_dir / "neurons.parquet")
    conn_df.to_parquet(out_dir / "connectivity.parquet")
    pd.DataFrame(syn_records).to_parquet(out_dir / "synapses.parquet")
    with (out_dir / "skeletons.pkl").open("wb") as f:
        pickle.dump(skels, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(
            {
                "source": "H01 (Shapson-Coe et al. 2024), gs://h01-release/data/20210601/",
                "n_neurons": len(target_ids),
                "n_connections": len(conn_df),
                "voxel_resolution_nm": [8.0, 8.0, 33.0],
            },
            f,
            indent=2,
        )
    print()
    print(f"Wrote cache to {out_dir.relative_to(repo)}/ "
          f"({sum((out_dir / f).stat().st_size for f in ['neurons.parquet','connectivity.parquet','synapses.parquet','skeletons.pkl']) / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
