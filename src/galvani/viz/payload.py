"""Build static JSON payloads consumed by the web demo.

The web demo at `examples/web_demo/` loads exactly one file at a time. This
module produces those files from a simulation `Result`.

Schema (v2):

```
{
  "metadata": {
    "schema_version": 2,
    "dataset_id": "hd_ring" | "mushroom_body" | ...,
    "scenario_id": "tracking" | "persistence" | ...,
    "scenario_label": "Bump tracking",
    "description": "Human-readable explanation of what's being shown.",
    "hyperparams": {
      "global_gain": 0.012,
      "symmetrize": true,
      "activation": "tanh",
      "weight_heuristic": "log1p",
      "dt_sim": 0.0002,
      "stimulus": {"type": "rotating", "omega": 1.0, ...}
    },
    "dataset_version": "hemibrain:v1.2.1",
    "n_neurons": 130,
    "n_frames": 120,
    "duration": 4.0
  },
  "bbox": {"min": [...], "max": [...], "center": [...], "scale": 1.0},
  "neurons": [
    {
      "id": 12345,
      "cell_type": "EPG",
      "hemisphere": "L",
      "angle": 1.57,
      "soma": [x, y, z] | null,
      "segments": [ax,ay,az, bx,by,bz, ...],
      "radii": [r0, r1, ...]   // (n_segments,) -- midpoint radius per segment
    }
  ],
  "times": [...],
  "rates": [[...], ...]
}
```

Coordinates are translated and uniformly scaled so the network sits in a
~10-unit cube centered at the origin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from galvani.connectome.base import Subgraph
from galvani.model.rate import Result

SCHEMA_VERSION = 2
DEFAULT_TARGET_SCALE = 10.0


class SkeletonProvider(Protocol):
    """Anything that can return an SWC-style DataFrame for a body id."""

    def fetch_skeleton(self, body_id: int) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class _Segments:
    """Per-neuron downsampled skeleton segments.

    `flat`: line endpoints, (n_seg, 6) flattened to a list.
    `radii`: midpoint radius per segment, in raw SWC units (nm).
    """

    flat: list[float]
    radii: list[float]


def _downsample_skeleton(
    swc: pd.DataFrame,
    stride: int,
    min_radius: float = 0.0,
) -> _Segments:
    """Stride-based downsample of an SWC tree, preserving gross topology.

    Keep: root, branch points (>= 2 children, above radius threshold), and
    every Nth other node. Drop: pure leaves and unbranched mid-path nodes
    outside the stride beat. For each kept node, emit one segment to its
    nearest kept ancestor.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    rows = swc.reset_index(drop=True)
    by_row: dict[int, int] = dict(zip(rows["rowId"], rows.index, strict=True))

    parent_arr = rows["link"].to_numpy()
    child_count: dict[int, int] = {}
    for parent_id in parent_arr:
        if parent_id != -1:
            child_count[int(parent_id)] = child_count.get(int(parent_id), 0) + 1

    radii = rows["radius"].to_numpy(dtype=np.float32)

    keep = np.zeros(len(rows), dtype=bool)
    keep[0] = True
    for i in range(len(rows)):
        row_id = int(rows["rowId"].iloc[i])
        n_children = child_count.get(row_id, 0)
        is_branch = n_children >= 2
        is_stride = i % stride == 0
        passes_radius = float(radii[i]) >= min_radius
        if (is_branch or is_stride) and passes_radius:
            keep[i] = True

    xs = rows["x"].to_numpy(dtype=np.float32)
    ys = rows["y"].to_numpy(dtype=np.float32)
    zs = rows["z"].to_numpy(dtype=np.float32)

    flat: list[float] = []
    seg_radii: list[float] = []
    for i in range(len(rows)):
        if not keep[i] or i == 0:
            continue
        cursor_id = int(rows["link"].iloc[i])
        steps = 0
        while cursor_id != -1 and steps < 1_000_000:
            cursor_idx = by_row.get(cursor_id)
            if cursor_idx is None:
                break
            if keep[cursor_idx]:
                flat.extend(
                    (
                        float(xs[cursor_idx]),
                        float(ys[cursor_idx]),
                        float(zs[cursor_idx]),
                        float(xs[i]),
                        float(ys[i]),
                        float(zs[i]),
                    )
                )
                # midpoint radius
                seg_radii.append(0.5 * (float(radii[cursor_idx]) + float(radii[i])))
                break
            cursor_id = int(rows["link"].iloc[cursor_idx])
            steps += 1
    return _Segments(flat=flat, radii=seg_radii)


def _normalise_segments(
    per_neuron: list[_Segments],
    target_scale: float = DEFAULT_TARGET_SCALE,
) -> tuple[list[_Segments], dict[str, Any]]:
    """Translate and uniformly scale so the bounding box has the target span."""
    points: list[NDArray[np.float64]] = []
    for s in per_neuron:
        if not s.flat:
            continue
        arr = np.array(s.flat, dtype=np.float64).reshape(-1, 3)
        points.append(arr)
    if not points:
        return per_neuron, {"min": [0, 0, 0], "max": [0, 0, 0], "center": [0, 0, 0], "scale": 1.0}

    all_pts = np.concatenate(points, axis=0)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    center = 0.5 * (mn + mx)
    span = float((mx - mn).max())
    scale = target_scale / span if span > 0 else 1.0

    scaled: list[_Segments] = []
    for s in per_neuron:
        if not s.flat:
            scaled.append(s)
            continue
        arr = np.array(s.flat, dtype=np.float64).reshape(-1, 3)
        arr = (arr - center) * scale
        # Scale radii too so a unit radius in payload space still represents
        # the same fraction of the bounding box.
        scaled.append(
            _Segments(
                flat=arr.flatten().tolist(),
                radii=[r * scale for r in s.radii],
            )
        )
    return scaled, {
        "min": mn.tolist(),
        "max": mx.tolist(),
        "center": center.tolist(),
        "scale": scale,
    }


def _downsample_rates(
    times: NDArray[np.float64], rates: NDArray[np.float64], n_frames: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pick `n_frames` evenly-spaced indices from a (T, N) rate trace."""
    n_t = times.shape[0]
    if n_frames >= n_t:
        return times, rates
    idx = np.linspace(0, n_t - 1, n_frames).round().astype(np.int64)
    return times[idx], rates[idx]


def build_payload(
    subgraph: Subgraph,
    result: Result,
    skeleton_provider: SkeletonProvider,
    *,
    dataset_id: str,
    scenario_id: str,
    scenario_label: str,
    description: str,
    hyperparams: dict[str, Any],
    angles: NDArray[np.float64] | None = None,
    stim_fn: Any = None,
    spec: Any = None,
    stride: int = 40,
    min_radius: float = 8.0,
    n_frames: int = 120,
    target_scale: float = DEFAULT_TARGET_SCALE,
) -> dict[str, Any]:
    """Build the JSON-serialisable demo payload for one scenario.

    `dataset_id` / `scenario_id` are short slugs the manifest references.
    `description` is human-readable Markdown-lite shown in the UI infobox.
    `hyperparams` is a free-form dict echoed into metadata for the
    "Hyperparameters & assumptions" infobox.
    `stim_fn`: optional callable `(t: float) -> NDArray` returning the
    external input vector at simulated time t. If provided, the resulting
    per-frame stim is baked into the payload as `stim_signal` so the
    frontend can render the driving input alongside the activity.
    """
    n_neurons = len(subgraph.neurons)
    if angles is not None and len(angles) != n_neurons:
        raise ValueError(f"angles length {len(angles)} != number of neurons {n_neurons}")

    raw_segments = []
    for neuron in subgraph.neurons:
        swc = skeleton_provider.fetch_skeleton(int(neuron.id))
        raw_segments.append(_downsample_skeleton(swc, stride=stride, min_radius=min_radius))

    norm_segments, bbox = _normalise_segments(raw_segments, target_scale=target_scale)

    times, rates = _downsample_rates(result.times, result.rates, n_frames)
    rates_rounded = np.round(rates, 4).tolist()
    times_rounded = np.round(times, 6).tolist()

    stim_signal: list[list[float]] | None = None
    if stim_fn is not None:
        sampled = np.stack([np.asarray(stim_fn(float(t))) for t in times], axis=0)
        stim_signal = np.round(sampled, 3).tolist()

    neurons_payload = []
    for i, neuron in enumerate(subgraph.neurons):
        n_angle = float(angles[i]) if angles is not None and not np.isnan(angles[i]) else None
        soma = list(neuron.soma_position) if neuron.soma_position is not None else None
        if soma is not None:
            arr = (np.array(soma) - np.array(bbox["center"])) * bbox["scale"]
            soma = [float(x) for x in arr.tolist()]
        neurons_payload.append(
            {
                "id": int(neuron.id),
                "cell_type": neuron.cell_type,
                "hemisphere": neuron.hemisphere,
                "angle": n_angle,
                "soma": soma,
                "segments": [round(v, 2) for v in norm_segments[i].flat],
                "radii": [round(v, 3) for v in norm_segments[i].radii],
            }
        )

    payload: dict[str, Any] = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "scenario_id": scenario_id,
            "scenario_label": scenario_label,
            "description": description,
            "hyperparams": hyperparams,
            "dataset_version": subgraph.dataset_version,
            "n_neurons": n_neurons,
            "n_frames": len(times_rounded),
            "dt_sim": float(result.dt),
            "duration": float(result.times[-1]) if result.times.size else 0.0,
        },
        "bbox": bbox,
        "neurons": neurons_payload,
        "times": times_rounded,
        "rates": rates_rounded,
    }
    if stim_signal is not None:
        payload["stim_signal"] = stim_signal
    # Optional: include the model's weight matrix + tau + bias so the
    # frontend can re-run the rate sim live with a tweaked global gain.
    # We round to 3 decimals to keep size manageable -- ~70 KB for 130x130.
    if spec is not None:
        payload["model"] = {
            "weights": np.round(spec.weights, 3).tolist(),
            "tau": np.round(spec.tau, 4).tolist(),
            "bias": np.round(spec.bias, 4).tolist(),
            "global_gain": float(spec.global_gain),
        }
    return payload


# Backwards-compat alias for the old name kept for any imports we missed.
build_hd_ring_payload = build_payload


def write_payload(payload: dict[str, Any], path: Path | str) -> None:
    """Write a payload to disk as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
