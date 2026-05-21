"""ModelSpec: the executable representation of a parameterized circuit.

A `ModelSpec` is fully self-describing: every parameter, every choice, and the
identity of every default heuristic used to produce it. JSON round-trippable
for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A parameterized rate-model network ready for simulation.

    The recurrent dynamics are::

        tau_i * dr_i/dt = -r_i + phi(sum_j W_ij r_j + I_i(t) + b_i)

    where `W = global_gain * weights`. The activation `phi` is chosen at
    simulation time (the spec is activation-agnostic).

    Invariants:
      - `weights.shape == (N, N)` where `N = len(neuron_ids)`.
      - `tau.shape == bias.shape == (N,)`.
      - `tau > 0` everywhere.
      - `neuron_ids` defines the canonical ordering for rows/columns.
    """

    neuron_ids: tuple[int, ...]
    weights: NDArray[np.float64]
    tau: NDArray[np.float64]
    bias: NDArray[np.float64]
    global_gain: float = 1.0

    dataset_version: str = ""
    """Pinned source dataset version (e.g. 'hemibrain:v1.2.1')."""

    defaults_used: dict[str, str] = field(default_factory=dict)
    """Identifies the heuristics applied; e.g.
    `{"weights": "log1p", "nt_to_sign": "fly_default", "tau": "by_type_v1"}`."""

    notes: dict[str, Any] = field(default_factory=dict)
    """Free-form provenance bag. Don't put load-bearing semantics here."""

    def __post_init__(self) -> None:
        n = len(self.neuron_ids)
        if self.weights.shape != (n, n):
            raise ValueError(f"weights must be ({n}, {n}); got {self.weights.shape}")
        if self.tau.shape != (n,):
            raise ValueError(f"tau must be ({n},); got {self.tau.shape}")
        if self.bias.shape != (n,):
            raise ValueError(f"bias must be ({n},); got {self.bias.shape}")
        if n > 0 and bool(np.any(self.tau <= 0)):
            raise ValueError("All time constants tau must be strictly positive.")

    @property
    def n_neurons(self) -> int:
        return len(self.neuron_ids)

    def to_json(self) -> str:
        """Serialize to a JSON string. Round-trippable via `ModelSpec.from_json`."""
        payload: dict[str, Any] = {
            "neuron_ids": list(self.neuron_ids),
            "weights": self.weights.tolist(),
            "tau": self.tau.tolist(),
            "bias": self.bias.tolist(),
            "global_gain": self.global_gain,
            "dataset_version": self.dataset_version,
            "defaults_used": dict(self.defaults_used),
            "notes": dict(self.notes),
            "_schema_version": 1,
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, s: str) -> ModelSpec:
        """Inverse of `to_json`."""
        data = json.loads(s)
        if data.get("_schema_version") != 1:
            raise ValueError(f"Unknown ModelSpec schema version: {data.get('_schema_version')!r}")
        return cls(
            neuron_ids=tuple(int(x) for x in data["neuron_ids"]),
            weights=np.asarray(data["weights"], dtype=np.float64),
            tau=np.asarray(data["tau"], dtype=np.float64),
            bias=np.asarray(data["bias"], dtype=np.float64),
            global_gain=float(data["global_gain"]),
            dataset_version=str(data["dataset_version"]),
            defaults_used=dict(data["defaults_used"]),
            notes=dict(data["notes"]),
        )
