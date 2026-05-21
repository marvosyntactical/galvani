"""The default parameterizer: `Subgraph -> ModelSpec`.

This is where Galvani earns its keep. Every default below is a small modeling
decision; each is documented with the heuristic name recorded into
`ModelSpec.defaults_used` so the resulting model is self-describing.

The pipeline:
  1. Build a `(N, N)` weight matrix from the synapse table.
     - Magnitude := `count_to_weight(counts)` (default `log1p`).
     - Sign     := `nt_to_sign(nt_pre)` per row (presynaptic NT).
     - Multiple synapses between the same pair sum.
  2. Per-neuron tau from `(cell_type, presumed_sign)`.
  3. Per-neuron bias defaults to 0; exposed as an override hook.
  4. Global gain defaults to 1.0; the simulator-side scalar that users tune
     to put the network in the bump regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from galvani.connectome.base import Subgraph
from galvani.model.spec import ModelSpec
from galvani.parameterize import signs as sign_module
from galvani.parameterize import timeconst as tau_module
from galvani.parameterize import weights as weight_module
from galvani.parameterize.signs import NTToSign
from galvani.parameterize.timeconst import TypeToTau
from galvani.parameterize.weights import CountToWeight, WeightHeuristic


@dataclass(frozen=True, slots=True)
class ParameterizerOptions:
    """User-facing knobs on the default parameterization pipeline.

    Defaults match the v1 plan. Override any field to customize without
    rewriting the whole parameterizer.
    """

    weight_heuristic: WeightHeuristic = "log1p"
    nt_to_sign: NTToSign = sign_module.fly_default
    type_to_tau: TypeToTau = tau_module.by_type_v1
    global_gain: float = 1.0
    symmetrize: bool = False
    """If True, weights are averaged with their transpose. Useful for the HD
    ring where the connectome breaks an underlying left/right symmetry."""


def _presynaptic_sign_per_neuron(subgraph: Subgraph, nt_to_sign: NTToSign) -> dict[int, int]:
    """A neuron's outgoing sign is the sign of its predicted NT.

    We derive it from the `Neuron.nt` field rather than per-synapse `nt_pre`,
    because Dale's principle holds in practice for the fly NTs we care about
    in v1 (~85% of cases). When `nt_pre` disagrees with the neuron's NT, we
    log it in `notes['nt_conflicts']` but follow the per-synapse value.
    """
    return {n.id: nt_to_sign(n.nt) for n in subgraph.neurons}


def default_parameterizer(
    subgraph: Subgraph,
    options: ParameterizerOptions | None = None,
) -> ModelSpec:
    """Convert a `Subgraph` to a runnable `ModelSpec` using v1 defaults.

    All heuristics applied are recorded in `ModelSpec.defaults_used` so the
    spec is reproducible from its own attributes.
    """
    opts = options or ParameterizerOptions()
    n = len(subgraph.neurons)
    index = subgraph.neuron_index()

    weights = np.zeros((n, n), dtype=np.float64)

    if subgraph.pre_ids.size > 0:
        magnitudes = weight_module.get(opts.weight_heuristic)(subgraph.counts)
        signs = np.fromiter(
            (opts.nt_to_sign(nt) for nt in subgraph.nt_pre),
            dtype=np.int64,
            count=len(subgraph.nt_pre),
        )
        contributions = magnitudes * signs.astype(np.float64)

        # Convention: weights[i, j] is the weight from j (pre) to i (post).
        # The simulator computes W @ r, so post-neuron i receives sum over j.
        row_idx = np.fromiter(
            (index[int(i)] for i in subgraph.post_ids), dtype=np.int64, count=subgraph.post_ids.size
        )
        col_idx = np.fromiter(
            (index[int(j)] for j in subgraph.pre_ids), dtype=np.int64, count=subgraph.pre_ids.size
        )
        # Multiple entries between the same pair sum.
        np.add.at(weights, (row_idx, col_idx), contributions)

    if opts.symmetrize:
        weights = 0.5 * (weights + weights.T)

    # tau per neuron. The "sign" passed to type_to_tau here is the *outgoing*
    # sign of the neuron itself (Dale's principle) -- the post-synaptic side
    # of a neuron's incoming weights is what tau actually filters, but we
    # use the neuron's own sign as the type-level cue for inhibitory vs
    # excitatory time-constant defaults.
    presyn_sign = _presynaptic_sign_per_neuron(subgraph, opts.nt_to_sign)
    tau = np.array(
        [opts.type_to_tau(neuron.cell_type, presyn_sign[neuron.id]) for neuron in subgraph.neurons],
        dtype=np.float64,
    )

    bias = np.zeros(n, dtype=np.float64)

    return ModelSpec(
        neuron_ids=tuple(neuron.id for neuron in subgraph.neurons),
        weights=weights,
        tau=tau,
        bias=bias,
        global_gain=opts.global_gain,
        dataset_version=subgraph.dataset_version,
        defaults_used={
            "weights": opts.weight_heuristic,
            "nt_to_sign": getattr(opts.nt_to_sign, "__name__", "custom"),
            "tau": getattr(opts.type_to_tau, "__name__", "custom"),
            "symmetrize": "true" if opts.symmetrize else "false",
        },
        notes={"n_synapse_entries": int(subgraph.pre_ids.size)},
    )


__all__ = [
    "CountToWeight",
    "ParameterizerOptions",
    "default_parameterizer",
]
