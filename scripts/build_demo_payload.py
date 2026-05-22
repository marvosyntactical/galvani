"""Bake all web-demo payloads + manifest.

Writes one JSON per scenario plus `manifest.json` under
`examples/web_demo/public/`. The frontend reads the manifest to populate
the dataset / scenario selectors.

Usage:

    NEUPRINT_TOKEN=... uv run python scripts/build_demo_payload.py

Skeletons are cached under `tests/fixtures/` on first run so subsequent
runs are network-free.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from galvani import ParameterizerOptions, default_parameterizer, simulate
from galvani.circuits.hd_ring import load_hd_ring
from galvani.circuits.mushroom_body import MB_NT, load_mushroom_body
from galvani.connectome.base import Subgraph
from galvani.connectome.cache import ParquetCache
from galvani.connectome.dti import DTIConnectome
from galvani.connectome.h01 import H01StyleConnectome
from galvani.connectome.hemibrain import HD_RING_NT, HemibrainConnectome
from galvani.model.adex import simulate_adex
from galvani.model.hh import simulate_hh
from galvani.model.lif import simulate_lif
from galvani.model.rate import relu, tanh
from galvani.parameterize.signs import mammalian_default
from galvani.stimuli import (
    pulse_stimulus_for_ids,
    ring_stimulus,
    rotating_stimulus,
    sum_stimuli,
)
from galvani.viz import build_payload, write_payload


@dataclass(frozen=True, slots=True)
class ScenarioOutput:
    dataset_id: str
    scenario_id: str
    label: str
    file: str
    description: str
    size_kb: float
    model_id: str = "rate"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """A dataset == one connectome subgraph + its descriptive metadata."""

    id: str
    label: str
    summary: str
    biology: str
    scenarios: list[ScenarioSpec] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    id: str
    label: str
    description: str


# Map `hyperparams.activation` (set per scenario at bake time) to the
# canonical model id the frontend uses in the Model dropdown.
ACTIVATION_TO_MODEL_ID: dict[str, str] = {
    "tanh": "rate",
    "relu": "rate",
    "LIF": "lif",
    "AdEx": "adex",
    "HH": "hh",
}


def _scenario_filename(circuit_id: str, scenario_id: str, model_id: str) -> str:
    """Canonical file name for one (circuit, scenario, model) tuple."""
    if model_id == "rate":
        return f"{circuit_id}_{scenario_id}.json"
    return f"{circuit_id}_{scenario_id}_{model_id}.json"


def _model_id_of(payload: dict) -> str:
    act = payload.get("metadata", {}).get("hyperparams", {}).get("activation", "tanh")
    return ACTIVATION_TO_MODEL_ID.get(act, "rate")


# ---------------- HD ring ----------------------------------------------------


def hd_ring_scenarios(subgraph: Subgraph, angles, n_neurons: int, neuron_ids: list[int]):
    """Yield (ScenarioSpec, payload) tuples for the HD-ring dataset."""
    n = n_neurons
    # Phase 5 operating point (see DESIGN_DECISIONS).
    gain = 0.012
    sym = True
    opts = ParameterizerOptions(symmetrize=sym, global_gain=gain)
    spec = default_parameterizer(subgraph, opts)

    common_hp = {
        "global_gain": gain,
        "symmetrize": sym,
        "weight_heuristic": opts.weight_heuristic,
        "activation": "tanh",
        "dt_sim": 2e-4,
        "tau_default_ms": 20.0,
    }

    # 1) Bump tracking under a rotating stimulus
    omega = 1.0
    stim_rot = rotating_stimulus(angles, omega=omega, width=0.5, amplitude=0.3)
    result_rot = simulate(spec, duration=4.0, stimulus=stim_rot, activation=tanh, dt=2e-4)
    yield (
        ScenarioSpec(
            id="tracking",
            label="Bump tracking",
            description=(
                "A rotating Gaussian stimulus (omega = 1 rad/s) drives the EB ring; "
                "the bump should follow the stimulus with a small phase lag. "
                "This is the canonical Kim et al. (2017) test."
            ),
        ),
        build_payload(
            subgraph,
            result_rot,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="tracking",
            scenario_label="Bump tracking",
            description=(
                "Rotating Gaussian input (omega = 1 rad/s, width = 0.5 rad, "
                "amplitude = 0.3) drives the ring. The activity bump tracks the "
                "stimulus with a small phase lag (a fraction of a radian)."
            ),
            hyperparams={
                **common_hp,
                "stimulus": {
                    "type": "rotating",
                    "omega_rad_per_s": omega,
                    "width_rad": 0.5,
                    "amplitude": 0.3,
                },
            },
            angles=angles,
            stim_fn=stim_rot,
            stride=40,
            min_radius=8.0,
            n_frames=120,
        ),
    )

    # 2) Bump existence (stationary stim)
    stim_static = ring_stimulus(angles, center=np.pi / 2, width=0.5, amplitude=0.3)
    result_static = simulate(spec, duration=1.0, stimulus=stim_static, activation=tanh, dt=2e-4)
    yield (
        ScenarioSpec(
            id="existence",
            label="Bump existence",
            description=(
                "Stationary input centred at theta = pi/2. A localised bump should "
                "form at the input centre within ~100 ms."
            ),
        ),
        build_payload(
            subgraph,
            result_static,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="existence",
            scenario_label="Bump existence",
            description=(
                "Static Gaussian stimulus at theta = pi/2. The ring should form a "
                "localised bump at the input centre, demonstrating that the "
                "recurrent dynamics select a single heading."
            ),
            hyperparams={
                **common_hp,
                "stimulus": {
                    "type": "static_gaussian",
                    "center_rad": float(np.pi / 2),
                    "width_rad": 0.5,
                    "amplitude": 0.3,
                },
            },
            angles=angles,
            stim_fn=stim_static,
            stride=40,
            min_radius=8.0,
            n_frames=60,
        ),
    )

    # 3) Persistence: stim on then off
    base = ring_stimulus(angles, center=np.pi / 2, width=0.5, amplitude=0.3)

    def stim_persist(t: float) -> np.ndarray:
        return base(t) if t < 0.3 else np.zeros(n)

    result_persist = simulate(spec, duration=2.0, stimulus=stim_persist, activation=tanh, dt=2e-4)
    yield (
        ScenarioSpec(
            id="persistence",
            label="Bump persistence (stim then off)",
            description=(
                "Static stimulus for 300 ms then removed. Activity is sustained "
                "after the input goes away (the bump's amplitude persists, though "
                "its position drifts slightly toward a network attractor)."
            ),
        ),
        build_payload(
            subgraph,
            result_persist,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="persistence",
            scenario_label="Bump persistence",
            description=(
                "Gaussian input at theta = pi/2 for the first 300 ms, then off. "
                "The network sustains the bump after stim removal -- amplitude "
                "persistence is the canonical Kim 2017 test. Position drifts to "
                "the network's preferred attractor; see DESIGN_DECISIONS."
            ),
            hyperparams={
                **common_hp,
                "stimulus": {
                    "type": "gated_gaussian",
                    "center_rad": float(np.pi / 2),
                    "width_rad": 0.5,
                    "amplitude": 0.3,
                    "on_until_s": 0.3,
                },
            },
            angles=angles,
            stim_fn=stim_persist,
            stride=40,
            min_radius=8.0,
            n_frames=120,
        ),
    )

    # 4) Velocity integration: L-PEN vs R-PEN pulse
    neurons = subgraph.neurons
    pen_l = [
        n.id
        for n in neurons
        if n.cell_type in ("PEN_a(PEN1)", "PEN_b(PEN2)") and n.hemisphere == "L"
    ]
    pen_r = [
        n.id
        for n in neurons
        if n.cell_type in ("PEN_a(PEN1)", "PEN_b(PEN2)") and n.hemisphere == "R"
    ]
    n_total = len(neurons)

    def velocity_stim(target_ids: list[int]):
        b = ring_stimulus(angles, center=np.pi / 2, width=0.5, amplitude=0.3)

        def init(t: float, b=b, n_=n_total) -> np.ndarray:
            return b(t) if t < 0.3 else np.zeros(n_)

        pulse = pulse_stimulus_for_ids(target_ids, neuron_ids, t0=0.5, duration=0.5, amplitude=0.5)
        return sum_stimuli(init, pulse)

    stim_vel_l = velocity_stim(pen_l)
    result_vel_l = simulate(spec, duration=1.5, stimulus=stim_vel_l, activation=tanh, dt=2e-4)
    yield (
        ScenarioSpec(
            id="velocity_left",
            label="Velocity integration: left-PEN pulse",
            description=(
                "Establish a bump, then pulse left-hemisphere PEN cells. The bump "
                "shifts in one direction -- this is the asymmetric loop that lets "
                "the fly integrate angular velocity."
            ),
        ),
        build_payload(
            subgraph,
            result_vel_l,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="velocity_left",
            scenario_label="Velocity integration (L-PEN pulse)",
            description=(
                "Init bump at pi/2 (300 ms), then pulse the left-hemisphere PEN "
                "cells from t=0.5 to t=1.0 s. The bump rotates one way; pulsing "
                "the R-PEN cells (the other scenario) rotates it the other."
            ),
            hyperparams={
                **common_hp,
                "stimulus": {
                    "type": "velocity_pulse",
                    "init_center_rad": float(np.pi / 2),
                    "pulse_target": "PEN_L",
                    "pulse_t0_s": 0.5,
                    "pulse_duration_s": 0.5,
                    "pulse_amplitude": 0.5,
                },
            },
            angles=angles,
            stim_fn=stim_vel_l,
            stride=40,
            min_radius=8.0,
            n_frames=120,
        ),
    )

    stim_vel_r = velocity_stim(pen_r)
    result_vel_r = simulate(spec, duration=1.5, stimulus=stim_vel_r, activation=tanh, dt=2e-4)
    yield (
        ScenarioSpec(
            id="velocity_right",
            label="Velocity integration: right-PEN pulse",
            description=(
                "Same as the left-PEN scenario but pulsing right-hemisphere PEN "
                "cells. The bump should move in the opposite angular direction."
            ),
        ),
        build_payload(
            subgraph,
            result_vel_r,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="velocity_right",
            scenario_label="Velocity integration (R-PEN pulse)",
            description=(
                "Init bump at pi/2 (300 ms), then pulse the right-hemisphere PEN "
                "cells from t=0.5 to t=1.0 s. Bump rotates the opposite direction "
                "from the L-PEN scenario."
            ),
            hyperparams={
                **common_hp,
                "stimulus": {
                    "type": "velocity_pulse",
                    "init_center_rad": float(np.pi / 2),
                    "pulse_target": "PEN_R",
                    "pulse_t0_s": 0.5,
                    "pulse_duration_s": 0.5,
                    "pulse_amplitude": 0.5,
                },
            },
            angles=angles,
            stim_fn=stim_vel_r,
            stride=40,
            min_radius=8.0,
            n_frames=120,
        ),
    )

    # 5) Long story: short heading change → drift → counter-pulse → drift
    # Tells a story over ~6 seconds instead of a static rotation.
    n_neurons_local = len(neurons)
    bump_a = ring_stimulus(angles, center=0.0, width=0.5, amplitude=0.4)
    bump_b = ring_stimulus(angles, center=np.pi, width=0.5, amplitude=0.4)

    def stim_story(t: float, n=n_neurons_local) -> np.ndarray:
        # 0.0 - 0.4 s: drive at theta=0 (fly faces east)
        # 0.4 - 2.0 s: free evolution (no input)
        # 2.0 - 2.4 s: drive at theta=pi (fly turned around)
        # 2.4 - 4.0 s: free evolution
        # 4.0 - 5.0 s: pulse L-PEN to rotate the bump
        # 5.0 - 6.0 s: free evolution
        if t < 0.4:
            return bump_a(t)
        if 2.0 <= t < 2.4:
            return bump_b(t)
        if 4.0 <= t < 5.0:
            base = np.zeros(n)
            base[
                [
                    i
                    for i, nn in enumerate(neurons)
                    if nn.cell_type in ("PEN_a(PEN1)", "PEN_b(PEN2)") and nn.hemisphere == "L"
                ]
            ] = 0.5
            return base
        return np.zeros(n)

    result_story = simulate(spec, duration=6.0, stimulus=stim_story, activation=tanh, dt=2e-4)
    yield (
        ScenarioSpec(
            id="story",
            label="A short HD story (6 s)",
            description=(
                "6-second composite scenario: brief stim east, free drift, "
                "brief stim west, free drift, L-PEN pulse, free drift again. "
                "Lets you see how the ring transitions between attractor states "
                "and how PEN cells inject angular velocity."
            ),
        ),
        build_payload(
            subgraph,
            result_story,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="story",
            scenario_label="HD story · short stims + drift",
            description=(
                "A 6-second composite scenario combining everything: brief "
                "pulse at θ=0, free drift, brief pulse at θ=π (fly turned 180°), "
                "free drift, then a left-PEN pulse demonstrating velocity "
                "integration. Each phase lasts a few hundred ms; in between, "
                "the ring evolves freely so you can see how persistent it is "
                "and how it transitions between heading states."
            ),
            hyperparams={
                **common_hp,
                "stimulus": {
                    "type": "composite",
                    "phases": [
                        {"t": [0.0, 0.4], "kind": "bump", "center_rad": 0.0},
                        {"t": [0.4, 2.0], "kind": "free"},
                        {"t": [2.0, 2.4], "kind": "bump", "center_rad": float(np.pi)},
                        {"t": [2.4, 4.0], "kind": "free"},
                        {"t": [4.0, 5.0], "kind": "pulse", "target": "PEN_L"},
                        {"t": [5.0, 6.0], "kind": "free"},
                    ],
                },
            },
            angles=angles,
            stim_fn=stim_story,
            stride=40,
            min_radius=8.0,
            n_frames=180,
        ),
    )


# ---------------- Mushroom body (subset) -------------------------------------


def mushroom_body_scenarios(conn: HemibrainConnectome):
    """Yield (ScenarioSpec, payload) for a 200-neuron MB subset + APL.

    The full hemibrain MB has 1923 KCs -- too many skeletons for the browser
    payload budget. We sample ~20 KCs per subtype + APL = ~200 cells, which
    is enough to show the with/without-APL sparse-coding contrast.
    """
    mb_full = load_mushroom_body(conn)
    rng = np.random.default_rng(0)

    # Per-subtype subsample.
    per_subtype = 20
    by_type: dict[str, list[int]] = {}
    for i, neuron in enumerate(mb_full.subgraph.neurons):
        if not mb_full.kc_mask[i]:
            continue
        by_type.setdefault(neuron.cell_type, []).append(i)

    chosen_indices: list[int] = []
    for _kc_type, idxs in by_type.items():
        if len(idxs) <= per_subtype:
            chosen_indices.extend(idxs)
        else:
            chosen_indices.extend(rng.choice(idxs, size=per_subtype, replace=False).tolist())
    chosen_indices.sort()
    chosen_indices.append(mb_full.apl_index)

    # Build a subset subgraph from the chosen indices.
    full_neurons = mb_full.subgraph.neurons
    sub_neurons = tuple(full_neurons[i] for i in chosen_indices)
    sub_ids = {n.id for n in sub_neurons}
    full_pre = mb_full.subgraph.pre_ids
    full_post = mb_full.subgraph.post_ids
    keep_mask = np.array(
        [int(p) in sub_ids and int(q) in sub_ids for p, q in zip(full_pre, full_post, strict=True)],
        dtype=bool,
    )
    sub_pre = full_pre[keep_mask]
    sub_post = full_post[keep_mask]
    sub_counts = mb_full.subgraph.counts[keep_mask]
    sub_nt_pre = tuple(
        nt for nt, k in zip(mb_full.subgraph.nt_pre, keep_mask.tolist(), strict=True) if k
    )
    sub_subgraph = dataclasses.replace(
        mb_full.subgraph,
        neurons=sub_neurons,
        pre_ids=sub_pre,
        post_ids=sub_post,
        counts=sub_counts,
        nt_pre=sub_nt_pre,
    )

    # APL's index in the new ordering.
    apl_local_idx = next(i for i, n in enumerate(sub_neurons) if n.cell_type == "APL")

    # Hyperparams + stim
    gain = 0.08
    opts = ParameterizerOptions(symmetrize=False, global_gain=gain)
    spec_with = default_parameterizer(sub_subgraph, opts)
    # Ablation: zero APL's outgoing column.
    from galvani.model.spec import ModelSpec

    w_no_apl = spec_with.weights.copy()
    w_no_apl[:, apl_local_idx] = 0.0
    spec_without = ModelSpec(
        neuron_ids=spec_with.neuron_ids,
        weights=w_no_apl,
        tau=spec_with.tau,
        bias=spec_with.bias,
        global_gain=spec_with.global_gain,
        dataset_version=spec_with.dataset_version,
        defaults_used={**spec_with.defaults_used, "apl_ablated": "true"},
        notes={"apl_ablated": True},
    )

    rng2 = np.random.default_rng(7)
    kc_local_indices = np.array([i for i, n in enumerate(sub_neurons) if n.cell_type != "APL"])
    n_kc = len(kc_local_indices)
    chosen = rng2.choice(kc_local_indices, size=int(0.30 * n_kc), replace=False)
    pattern = np.zeros(spec_with.n_neurons, dtype=np.float64)
    pattern[chosen] = 0.5

    def stim(_t: float) -> np.ndarray:
        return pattern

    common_hp = {
        "global_gain": gain,
        "symmetrize": False,
        "weight_heuristic": "log1p",
        "activation": "relu",
        "dt_sim": 5e-4,
        "tau_default_ms": 20.0,
    }

    result_with = simulate(spec_with, duration=0.3, stimulus=stim, activation=relu, dt=5e-4)
    yield (
        ScenarioSpec(
            id="with_apl",
            label="Sparse coding (APL intact)",
            description=(
                "Drive 30% of Kenyon cells with constant input. APL pools KC "
                "activity and inhibits in return, producing k-WTA sparse coding: "
                "only a few percent of KCs remain active at steady state."
            ),
        ),
        build_payload(
            sub_subgraph,
            result_with,
            skeleton_provider=conn,
            dataset_id="mushroom_body",
            scenario_id="with_apl",
            scenario_label="Sparse coding (APL intact)",
            description=(
                "30% of KCs receive constant external input (amp=0.5). APL "
                "gathers KC drive and inhibits via GABA. Result: a few percent "
                "of KCs stay active -- the canonical mushroom-body sparse code."
            ),
            hyperparams={
                **common_hp,
                "n_kc_subset": int(n_kc),
                "kc_drive_fraction": 0.30,
                "stimulus": {"type": "constant_subset", "amplitude": 0.5},
            },
            stim_fn=stim,
            stride=30,
            min_radius=6.0,
            n_frames=60,
        ),
    )

    result_without = simulate(spec_without, duration=0.3, stimulus=stim, activation=relu, dt=5e-4)
    yield (
        ScenarioSpec(
            id="without_apl",
            label="APL ablated (no inhibition)",
            description=(
                "Same input drive as the previous scenario, but APL's outgoing "
                "synapses are silenced. Without global inhibition, KC activity "
                "stays dense -- demonstrating APL's role in sparse coding."
            ),
        ),
        build_payload(
            sub_subgraph,
            result_without,
            skeleton_provider=conn,
            dataset_id="mushroom_body",
            scenario_id="without_apl",
            scenario_label="APL ablated",
            description=(
                "APL's outgoing weights are zeroed in the ModelSpec. Identical "
                "external drive. KCs that were silent in the previous scenario "
                "now stay active, demonstrating APL's role in enforcing the "
                "sparse code."
            ),
            hyperparams={
                **common_hp,
                "apl_ablated": True,
                "n_kc_subset": int(n_kc),
                "kc_drive_fraction": 0.30,
                "stimulus": {"type": "constant_subset", "amplitude": 0.5},
            },
            stim_fn=stim,
            stride=30,
            min_radius=6.0,
            n_frames=60,
        ),
    )

    # Three-odor sequence: a different random ~30% of KCs fires at each
    # odor presentation, separated by quiet gaps so you can see the sparse
    # code re-form each time.
    rng3 = np.random.default_rng(13)
    odors = [rng3.choice(kc_local_indices, size=int(0.30 * n_kc), replace=False) for _ in range(3)]
    pattern_odors = []
    for chosen_idx in odors:
        p = np.zeros(spec_with.n_neurons, dtype=np.float64)
        p[chosen_idx] = 0.5
        pattern_odors.append(p)

    def stim_3odors(t: float) -> np.ndarray:
        # Three 0.15 s odor pulses at t = 0.05, 0.45, 0.85, gaps in between.
        if 0.05 <= t < 0.20:
            return pattern_odors[0]
        if 0.45 <= t < 0.60:
            return pattern_odors[1]
        if 0.85 <= t < 1.00:
            return pattern_odors[2]
        return np.zeros(spec_with.n_neurons, dtype=np.float64)

    result_3odors = simulate(
        spec_with, duration=1.2, stimulus=stim_3odors, activation=relu, dt=5e-4
    )
    yield (
        ScenarioSpec(
            id="three_odors",
            label="Three odor presentations",
            description=(
                "Three distinct 'odor' patterns presented in sequence (each "
                "activates a different random ~30% of KCs). Between odors, "
                "activity decays back via APL inhibition. Shows how each "
                "odor recruits a distinct sparse subset."
            ),
        ),
        build_payload(
            sub_subgraph,
            result_3odors,
            skeleton_provider=conn,
            dataset_id="mushroom_body",
            scenario_id="three_odors",
            scenario_label="Three odors in sequence",
            description=(
                "Three pulses (150 ms each, 250 ms apart). Each pulse drives "
                "a DIFFERENT random 30% of KCs -- think of it as three "
                "different odors hitting the antennal lobe. APL is intact "
                "so each odor settles into its own sparse subset, then quiets "
                "back down between odors. The set of active KCs changes "
                "between pulses -- that's the MB's odor identity code."
            ),
            hyperparams={
                **common_hp,
                "n_kc_subset": int(n_kc),
                "kc_drive_fraction": 0.30,
                "stimulus": {
                    "type": "three_odor_pulses",
                    "pulse_width_s": 0.15,
                    "amplitude": 0.5,
                },
            },
            stim_fn=stim_3odors,
            stride=30,
            min_radius=6.0,
            n_frames=120,
        ),
    )


# ---------------- LIF biophysical scenario -----------------------------------


def lif_hd_ring_scenarios(subgraph, angles, n_neurons, neuron_ids):
    """LIF, AdEx, and HH versions of the HD-ring tracking + velocity_left
    scenarios.

    Same circuit, same stimulus topology as the rate variants, different
    neuron model. Velocity_left is the default scenario on the demo, so
    we bake all four model variants for it.
    """
    gain = 0.04
    opts = ParameterizerOptions(symmetrize=True, global_gain=gain)
    spec = default_parameterizer(subgraph, opts)
    omega = 1.0
    n_total = n_neurons
    stim = rotating_stimulus(angles, omega=omega, width=0.5, amplitude=1.5)
    result = simulate_lif(spec, duration=2.0, stimulus=stim, dt=2e-4, v_threshold=0.5)
    yield (
        ScenarioSpec(
            id="tracking",
            label="LIF spikes: rotating stim",
            description=(
                "Same hemibrain HD ring, but simulated with a leaky integrate-"
                "and-fire model: neurons have voltages that spike when they "
                "cross threshold, are reset, and refract for 2 ms. Activity "
                "shown is instantaneous spike rate."
            ),
        ),
        build_payload(
            subgraph,
            result,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="tracking",
            scenario_label="LIF spiking · rotating stim",
            description=(
                "Biophysical LIF simulation of the HD ring under a rotating "
                "Gaussian stimulus. Each neuron integrates synaptic drive on "
                "a voltage trace; threshold crossings produce spikes (reset + "
                "2 ms refractory). The bump now consists of discrete spike "
                "events, not a continuous rate."
            ),
            hyperparams={
                "global_gain": gain,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "LIF",
                "dt_sim": 2e-4,
                "v_threshold": 0.5,
                "v_reset": 0.0,
                "t_refractory_s": 0.002,
                "syn_tau_s": 0.005,
                "stimulus": {
                    "type": "rotating",
                    "omega_rad_per_s": omega,
                    "width_rad": 0.5,
                    "amplitude": 1.5,
                },
            },
            angles=angles,
            stim_fn=stim,
            stride=40,
            min_radius=8.0,
            n_frames=120,
        ),
    )

    # AdEx version: same circuit + stimulus, adaptive-exponential IF cells
    result_adex = simulate_adex(
        spec,
        duration=2.0,
        stimulus=stim,
        dt=2e-4,
        v_threshold=0.6,
        a=0.04,
        b=0.2,
    )
    yield (
        ScenarioSpec(
            id="tracking",
            label="AdEx spikes (with adaptation)",
            description=(
                "Same HD ring + rotating stimulus, simulated with adaptive "
                "exponential integrate-and-fire (AdEx). Adds spike-frequency "
                "adaptation -- neurons fire faster at stim onset then slow "
                "down as adaptation current builds up."
            ),
        ),
        build_payload(
            subgraph,
            result_adex,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="tracking",
            scenario_label="AdEx spiking · rotating stim",
            description=(
                "Adaptive exponential integrate-and-fire (Brette & Gerstner "
                "2005). Two ODEs per neuron: voltage `v` plus adaptation "
                "current `w` that builds with each spike and decays slowly. "
                "Spike-frequency adaptation, bursting, and accommodation "
                "all emerge naturally. Compare to the LIF scenario to see "
                "what adaptation buys you."
            ),
            hyperparams={
                "global_gain": gain,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "AdEx",
                "dt_sim": 2e-4,
                "v_threshold": 0.6,
                "v_reset": 0.0,
                "t_refractory_s": 0.002,
                "tau_w_s": 0.15,
                "adaptation_a": 0.04,
                "adaptation_b": 0.2,
                "stimulus": {
                    "type": "rotating",
                    "omega_rad_per_s": omega,
                    "width_rad": 0.5,
                    "amplitude": 1.5,
                },
            },
            angles=angles,
            stim_fn=stim,
            stride=40,
            min_radius=8.0,
            n_frames=120,
        ),
    )

    # Hodgkin-Huxley version: full ion-channel dynamics
    stim_hh = rotating_stimulus(angles, omega=omega, width=0.5, amplitude=0.6)
    result_hh = simulate_hh(spec, duration=0.4, stimulus=stim_hh, dt=5e-5)
    yield (
        ScenarioSpec(
            id="tracking",
            label="Hodgkin-Huxley spikes (full biophysics)",
            description=(
                "Same HD ring + rotating stimulus, simulated with full "
                "Hodgkin-Huxley dynamics. Real ion channels (Na+/K+/leak) "
                "with voltage-gated kinetics. Much slower to simulate -- "
                "only 400 ms shown."
            ),
        ),
        build_payload(
            subgraph,
            result_hh,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="tracking",
            scenario_label="Hodgkin-Huxley · rotating stim",
            description=(
                "Single-compartment Hodgkin-Huxley (Hodgkin & Huxley 1952, "
                "Nobel 1963). Voltage v plus three gating variables m, h, n "
                "for Na+ activation/inactivation and K+ activation. Real "
                "action-potential shape emerges from the ion-channel "
                "dynamics rather than being prescribed by threshold + reset. "
                "Computationally heavy -- this scenario is 400 ms only."
            ),
            hyperparams={
                "global_gain": gain,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "HH",
                "dt_sim": 5e-5,
                "g_Na": 120.0,
                "g_K": 36.0,
                "g_L": 0.3,
                "stimulus": {
                    "type": "rotating",
                    "omega_rad_per_s": omega,
                    "width_rad": 0.5,
                    "amplitude": 0.6,
                },
            },
            angles=angles,
            stim_fn=stim_hh,
            stride=40,
            min_radius=8.0,
            n_frames=80,
        ),
    )
    # ---- Velocity-left variants (the demo's default scenario) ---------------
    # Bump-establish + left-PEN pulse, same stim shape as the rate variant.
    neurons_list = subgraph.neurons
    pen_l = [
        nn.id
        for nn in neurons_list
        if nn.cell_type in ("PEN_a(PEN1)", "PEN_b(PEN2)") and nn.hemisphere == "L"
    ]
    base_bump = ring_stimulus(angles, center=np.pi / 2, width=0.5, amplitude=1.5)

    def vel_left_init(t: float) -> np.ndarray:
        return base_bump(t) if t < 0.3 else np.zeros(n_total)

    pulse_vl = pulse_stimulus_for_ids(
        pen_l, neuron_ids, t0=0.5, duration=0.5, amplitude=1.0
    )
    stim_vl = sum_stimuli(vel_left_init, pulse_vl)

    result_vl_lif = simulate_lif(
        spec, duration=1.5, stimulus=stim_vl, dt=2e-4, v_threshold=0.5
    )
    yield (
        ScenarioSpec(
            id="velocity_left",
            label="LIF spikes · L-PEN velocity pulse",
            description=(
                "Establish a bump, then pulse left-PEN cells. Bump rotates "
                "via the asymmetric loop. Spiking (LIF) version."
            ),
        ),
        build_payload(
            subgraph,
            result_vl_lif,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="velocity_left",
            scenario_label="Velocity integration (L-PEN) · LIF",
            description=(
                "Init bump at pi/2 (300 ms) then pulse L-PEN from 0.5-1.0 s. "
                "Spiking LIF dynamics — bump rotates via the asymmetric loop."
            ),
            hyperparams={
                "global_gain": gain,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "LIF",
                "dt_sim": 2e-4,
                "v_threshold": 0.5,
                "v_reset": 0.0,
                "t_refractory_s": 0.002,
                "syn_tau_s": 0.005,
                "stimulus": {
                    "type": "velocity_pulse",
                    "init_center_rad": float(np.pi / 2),
                    "pulse_target": "PEN_L",
                    "pulse_t0_s": 0.5,
                    "pulse_duration_s": 0.5,
                    "pulse_amplitude": 1.0,
                },
            },
            angles=angles,
            stim_fn=stim_vl,
            stride=40,
            min_radius=8.0,
            n_frames=120,
        ),
    )

    result_vl_adex = simulate_adex(
        spec, duration=1.5, stimulus=stim_vl, dt=2e-4, v_threshold=0.6, a=0.04, b=0.2
    )
    yield (
        ScenarioSpec(
            id="velocity_left",
            label="AdEx spikes · L-PEN velocity pulse",
            description=(
                "Same as the LIF velocity_left scenario but with adaptive "
                "exponential IF dynamics (Brette & Gerstner 2005)."
            ),
        ),
        build_payload(
            subgraph,
            result_vl_adex,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="velocity_left",
            scenario_label="Velocity integration (L-PEN) · AdEx",
            description=(
                "Init bump then L-PEN pulse, adaptive-exponential IF cells. "
                "Spike-frequency adaptation reshapes the bump dynamics."
            ),
            hyperparams={
                "global_gain": gain,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "AdEx",
                "dt_sim": 2e-4,
                "v_threshold": 0.6,
                "v_reset": 0.0,
                "t_refractory_s": 0.002,
                "tau_w_s": 0.15,
                "adaptation_a": 0.04,
                "adaptation_b": 0.2,
                "stimulus": {
                    "type": "velocity_pulse",
                    "init_center_rad": float(np.pi / 2),
                    "pulse_target": "PEN_L",
                    "pulse_t0_s": 0.5,
                    "pulse_duration_s": 0.5,
                    "pulse_amplitude": 1.0,
                },
            },
            angles=angles,
            stim_fn=stim_vl,
            stride=40,
            min_radius=8.0,
            n_frames=120,
        ),
    )

    # HH is heavy — short window, lower-amplitude stim
    base_bump_hh = ring_stimulus(angles, center=np.pi / 2, width=0.5, amplitude=0.6)

    def vel_left_init_hh(t: float) -> np.ndarray:
        return base_bump_hh(t) if t < 0.1 else np.zeros(n_total)

    pulse_vl_hh = pulse_stimulus_for_ids(
        pen_l, neuron_ids, t0=0.15, duration=0.2, amplitude=0.4
    )
    stim_vl_hh = sum_stimuli(vel_left_init_hh, pulse_vl_hh)
    result_vl_hh = simulate_hh(spec, duration=0.4, stimulus=stim_vl_hh, dt=5e-5)
    yield (
        ScenarioSpec(
            id="velocity_left",
            label="HH spikes · L-PEN velocity pulse",
            description=(
                "Same as the LIF/AdEx velocity_left scenarios but with full "
                "Hodgkin-Huxley ion-channel dynamics. 400 ms only — HH is "
                "computationally heavy."
            ),
        ),
        build_payload(
            subgraph,
            result_vl_hh,
            skeleton_provider=_PROVIDER,
            spec=spec,
            dataset_id="hd_ring",
            scenario_id="velocity_left",
            scenario_label="Velocity integration (L-PEN) · HH",
            description=(
                "Init bump (100 ms) then L-PEN pulse, full Hodgkin-Huxley "
                "ion-channel dynamics. Real action-potential shape emerges "
                "from Na+/K+/leak conductances."
            ),
            hyperparams={
                "global_gain": gain,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "HH",
                "dt_sim": 5e-5,
                "g_Na": 120.0,
                "g_K": 36.0,
                "g_L": 0.3,
                "stimulus": {
                    "type": "velocity_pulse",
                    "init_center_rad": float(np.pi / 2),
                    "pulse_target": "PEN_L",
                    "pulse_t0_s": 0.15,
                    "pulse_duration_s": 0.2,
                    "pulse_amplitude": 0.4,
                },
            },
            angles=angles,
            stim_fn=stim_vl_hh,
            stride=40,
            min_radius=8.0,
            n_frames=80,
        ),
    )
    _ = n_total, neuron_ids  # silence unused (kept for signature parity)


# ---------------- DTI (real HCP/AAL2 data) -----------------------------------


def dti_real_scenarios(mat_path: Path):
    """Region-level dynamics on a REAL DTI matrix (HCP subject NAP_001,
    AAL2 parcellation, 94 cortical+subcortical regions)."""
    conn = DTIConnectome.aal2_hcp_subject(mat_path)
    neurons = conn.query()
    subgraph = conn.subgraph(neurons)
    n = len(neurons)
    # Region-level operating point: tau=100 ms, very small gain due to the
    # heavy log-normal range of real DTI streamline counts.
    opts = ParameterizerOptions(symmetrize=True, global_gain=0.0008, weight_heuristic="log1p")
    spec = default_parameterizer(subgraph, opts)
    import dataclasses as _dc

    spec = _dc.replace(spec, tau=np.full_like(spec.tau, 0.1))

    # Stimulus: drive visual cortex (occipital regions) first, watch
    # activity propagate into other lobes via real white-matter tracts.
    visual_substrs = ("Occipital", "Calcarine", "Cuneus", "Lingual", "Fusiform")
    visual_mask = np.array(
        [
            any(
                s in nn.cell_type or s in (str(nn.id) + "_" + (nn.hemisphere or ""))
                for s in visual_substrs
            )
            for nn in neurons
        ],
        dtype=bool,
    )
    # Fall back: drive any region whose label (we don't carry labels in
    # Neuron) — match by id mapping the AAL2 labels. We use the
    # connectome's regions directly.
    label_by_id = {r.id: r.label for r in conn._regions}  # type: ignore[attr-defined]
    visual_mask = np.array(
        [any(s in label_by_id[nn.id] for s in visual_substrs) for nn in neurons],
        dtype=bool,
    )
    pattern = np.zeros(n, dtype=np.float64)
    pattern[visual_mask] = 0.4
    print(f"  drive {int(visual_mask.sum())} visual regions")

    def stim(_t: float) -> np.ndarray:
        return pattern

    result = simulate(spec, duration=3.0, stimulus=stim, activation=tanh, dt=2e-3)
    yield (
        ScenarioSpec(
            id="visual_drive",
            label="Drive visual cortex → propagation",
            description=(
                "REAL human DTI connectome (94 AAL2 regions, HCP subject NAP_001 "
                "via the neurolib dataset). Drive the visual-cortex regions; "
                "activity propagates into the rest of the brain through real "
                "white-matter streamline counts."
            ),
        ),
        build_payload(
            subgraph,
            result,
            skeleton_provider=conn,
            spec=spec,
            dataset_id="dti_hcp",
            scenario_id="visual_drive",
            scenario_label="Visual-cortex drive → propagation",
            description=(
                "REAL 94-region human DTI connectome (Automated Anatomical "
                "Parcellation 2, Rolls et al. 2015) from one HCP subject's "
                "diffusion MRI scan, packaged via the neurolib project. "
                "Steady drive on the visual-cortex regions (Occipital/Calcarine/"
                "Cuneus/Lingual/Fusiform); activity propagates outwards through "
                "the real streamline-count matrix. Not synthetic this time."
            ),
            hyperparams={
                "global_gain": 0.0008,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "tanh",
                "dt_sim": 2e-3,
                "tau_default_ms": 100.0,
                "n_regions": n,
                "data_source": "neurolib gw/NAP_001 / AAL2 atlas",
                "stimulus": {
                    "type": "visual_cortex_drive",
                    "amplitude": 0.4,
                },
            },
            stim_fn=stim,
            stride=1,
            min_radius=0.0,
            n_frames=80,
            target_scale=20.0,
        ),
    )

    # Region-walk: drive Frontal → Motor → Visual in sequence, watch a wave
    # of activity sweep across the brain in real time.
    families = {
        "Frontal_Sup": np.array(
            [
                i
                for i, nn in enumerate(neurons)
                if "Frontal" in label_by_id[nn.id] and "Med" not in label_by_id[nn.id]
            ]
        ),
        "Motor": np.array(
            [
                i
                for i, nn in enumerate(neurons)
                if any(s in label_by_id[nn.id] for s in ("Precentral", "Supp_Motor", "Postcentral"))
            ]
        ),
        "Occipital": np.array(
            [
                i
                for i, nn in enumerate(neurons)
                if any(s in label_by_id[nn.id] for s in ("Occipital", "Calcarine", "Cuneus"))
            ]
        ),
    }
    print("  region-walk family sizes: " + ", ".join(f"{k}={len(v)}" for k, v in families.items()))

    def stim_walk(t: float) -> np.ndarray:
        # 0.0-1.0: Frontal, 1.5-2.5: Motor, 3.0-4.0: Occipital, then quiet.
        p = np.zeros(n, dtype=np.float64)
        if t < 1.0:
            p[families["Frontal_Sup"]] = 0.4
        elif 1.5 <= t < 2.5:
            p[families["Motor"]] = 0.4
        elif 3.0 <= t < 4.0:
            p[families["Occipital"]] = 0.4
        return p

    result_walk = simulate(spec, duration=5.0, stimulus=stim_walk, activation=tanh, dt=2e-3)
    yield (
        ScenarioSpec(
            id="region_walk",
            label="Frontal → Motor → Visual walk",
            description=(
                "Sequentially drive Frontal cortex (0-1 s), then primary "
                "Motor (1.5-2.5 s), then Visual (3-4 s). Lets you watch "
                "activity spread through the real white-matter network "
                "as the input target moves around the brain."
            ),
        ),
        build_payload(
            subgraph,
            result_walk,
            skeleton_provider=conn,
            spec=spec,
            dataset_id="dti_hcp",
            scenario_id="region_walk",
            scenario_label="Region-walk: Frontal → Motor → Visual",
            description=(
                "Five-second walk through three functional families. We drive "
                "frontal-cortex regions for 1 s, pause, drive motor cortex "
                "for 1 s, pause, drive visual cortex for 1 s. Activity rides "
                "the real streamline-count matrix from each driven cluster "
                "into the rest of the brain. Watch the propagation paths."
            ),
            hyperparams={
                "global_gain": 0.0008,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "tanh",
                "dt_sim": 2e-3,
                "tau_default_ms": 100.0,
                "n_regions": n,
                "data_source": "neurolib gw/NAP_001 / AAL2 atlas",
                "stimulus": {
                    "type": "region_walk",
                    "phases": [
                        {"t": [0.0, 1.0], "regions": "Frontal"},
                        {"t": [1.5, 2.5], "regions": "Motor"},
                        {"t": [3.0, 4.0], "regions": "Occipital"},
                    ],
                    "amplitude": 0.4,
                },
            },
            stim_fn=stim_walk,
            stride=1,
            min_radius=0.0,
            n_frames=120,
            target_scale=20.0,
        ),
    )


# ---------------- H01-inspired cortical microcircuit -------------------------


def h01_scenarios():
    """H01-inspired cortical microcircuit scenarios.

    Drives the L4 stellate ("sensory input") population, watches activity
    propagate through L4 -> L2/3 -> L5 -- the canonical cortical column
    pathway. Inhibition from PV/SST/VIP shapes the response.
    """
    conn = H01StyleConnectome.synthetic(seed=0)
    neurons = conn.query()
    subgraph = conn.subgraph(neurons)
    n = len(neurons)

    # Mammalian NT signs: glutamate is excitatory (opposite of fly).
    opts = ParameterizerOptions(
        symmetrize=False,
        global_gain=0.08,
        weight_heuristic="log1p",
        nt_to_sign=mammalian_default,
    )
    spec = default_parameterizer(subgraph, opts)

    common_hp = {
        "global_gain": 0.08,
        "symmetrize": False,
        "weight_heuristic": "log1p",
        "nt_to_sign": "mammalian_default",
        "activation": "tanh",
        "dt_sim": 5e-4,
        "tau_default_ms": 20.0,
    }

    # Scenario 1: brief L4 sensory pulse → L2/3 → L5 cascade.
    l4_indices = np.array([i for i, nn in enumerate(neurons) if nn.cell_type == "L4_stellate"])
    pat_l4 = np.zeros(n, dtype=np.float64)
    pat_l4[l4_indices] = 1.0

    def stim_cascade(t: float) -> np.ndarray:
        # 100 ms pulse, watch it propagate
        return pat_l4 if t < 0.1 else np.zeros(n)

    result_cascade = simulate(spec, duration=0.8, stimulus=stim_cascade, activation=tanh, dt=5e-4)
    yield (
        ScenarioSpec(
            id="sensory_pulse",
            label="L4 sensory pulse → cortical cascade",
            description=(
                "Brief input to L4 stellate cells (the canonical sensory "
                "input layer), watch activity propagate L4 → L2/3 → L5. "
                "PV/SST interneurons shape the response."
            ),
        ),
        build_payload(
            subgraph,
            result_cascade,
            skeleton_provider=conn,
            spec=spec,
            dataset_id="h01_style",
            scenario_id="sensory_pulse",
            scenario_label="Sensory pulse → cortical cascade",
            description=(
                "100 ms input to L4 spiny stellate cells (the cortical sensory "
                "input layer). The Douglas-Martin canonical microcircuit predicts "
                "activity flows L4 → L2/3 → L5; the L2/3 PV/SST interneurons "
                "shape the response. Watch which layers light up in sequence."
            ),
            hyperparams={
                **common_hp,
                "stimulus": {"type": "L4_pulse", "amplitude": 1.0, "duration_s": 0.1},
            },
            stim_fn=stim_cascade,
            stride=1,
            min_radius=0.0,
            n_frames=80,
            target_scale=15.0,
        ),
    )

    # Scenario 2: sustained drive → inhibition-stabilized network (ISN)
    def stim_sustained(t: float) -> np.ndarray:
        return pat_l4 * 0.5  # always on, half amplitude

    result_isn = simulate(spec, duration=1.0, stimulus=stim_sustained, activation=tanh, dt=5e-4)
    yield (
        ScenarioSpec(
            id="sustained_drive",
            label="Sustained drive → steady state",
            description=(
                "Constant input to L4. The network reaches an inhibition-"
                "stabilized steady state where excitation and inhibition "
                "balance each other -- the canonical cortical operating point."
            ),
        ),
        build_payload(
            subgraph,
            result_isn,
            skeleton_provider=conn,
            spec=spec,
            dataset_id="h01_style",
            scenario_id="sustained_drive",
            scenario_label="Sustained drive · ISN steady state",
            description=(
                "Constant drive on L4 stellates. The cortical microcircuit "
                "settles into an inhibition-stabilized state where every "
                "increase in excitatory activity is balanced by recruited "
                "inhibition from PV/SST cells. This balance is what real "
                "cortex spends most of its time in (Tsodyks 1997, Rubin et al. "
                "2015)."
            ),
            hyperparams={
                **common_hp,
                "stimulus": {"type": "L4_sustained", "amplitude": 0.5},
            },
            stim_fn=stim_sustained,
            stride=1,
            min_radius=0.0,
            n_frames=80,
            target_scale=15.0,
        ),
    )


# ---------------- Top-level driver -------------------------------------------


_PROVIDER: HemibrainConnectome | None = None


def main() -> None:
    global _PROVIDER
    repo = Path(__file__).resolve().parent.parent
    fixtures = repo / "tests" / "fixtures"
    out_dir = repo / "examples" / "web_demo" / "public"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = ParquetCache(fixtures)
    print(f"Building demo payloads into {out_dir.relative_to(repo)}")

    # ----- HD ring -----
    conn_hd = HemibrainConnectome(cache=cache, nt_by_type=HD_RING_NT)
    layout = load_hd_ring(conn_hd)
    print(f"\n[hd_ring] N={len(layout.subgraph.neurons)}")
    _PROVIDER = conn_hd

    hd_dataset = DatasetSpec(
        id="hd_ring",
        label="Fly HD ring (hemibrain)",
        summary=(
            "130 neurons in the *Drosophila* central complex that encode the "
            "fly's heading direction. Persistent bump that tracks turns."
        ),
        biology=(
            "The HD (head-direction) ring lives in the central complex (CX) of "
            "the fly's brain -- specifically the ellipsoid body (EB) and "
            "protocerebral bridge (PB). EPG neurons tile the EB and form a "
            "ring-attractor topology: a single localised bump of activity "
            "marks the fly's current heading. PEN cells inject angular-velocity "
            "input that shifts the bump when the fly turns. Delta7 neurons "
            "provide broad inhibition that stabilises the bump shape. The "
            "circuit was first characterised by Seelig & Jayaraman 2015 (Nature) "
            "and Kim et al. 2017 (Science). What you're seeing is rate-model "
            "activity (one number per neuron per timestep) on each neuron's "
            "EM-traced skeleton from the hemibrain dataset."
        ),
        scenarios=[],
    )
    hd_outputs: list[ScenarioOutput] = []
    neuron_ids = [n.id for n in layout.subgraph.neurons]
    for gen in (
        hd_ring_scenarios(layout.subgraph, layout.angles, len(layout.subgraph.neurons), neuron_ids),
        lif_hd_ring_scenarios(
            layout.subgraph, layout.angles, len(layout.subgraph.neurons), neuron_ids
        ),
    ):
        for scenario_spec, payload in gen:
            model_id = _model_id_of(payload)
            path = out_dir / _scenario_filename("hd_ring", scenario_spec.id, model_id)
            write_payload(payload, path)
            size_kb = path.stat().st_size / 1024
            hd_outputs.append(
                ScenarioOutput(
                    dataset_id="hd_ring",
                    scenario_id=scenario_spec.id,
                    label=scenario_spec.label,
                    file=path.name,
                    description=scenario_spec.description,
                    size_kb=size_kb,
                    model_id=model_id,
                )
            )
            print(f"  {path.name}: {size_kb:.0f} KiB · model={model_id} · {scenario_spec.label}")

    # ----- Mushroom body -----
    conn_mb = HemibrainConnectome(cache=cache, nt_by_type=MB_NT)
    _PROVIDER = conn_mb
    print("\n[mushroom_body] subset (200 cells)")

    mb_dataset = DatasetSpec(
        id="mushroom_body",
        label="Fly mushroom body (subset)",
        summary=(
            "~200 Kenyon cells + APL from the *Drosophila* mushroom body. "
            "Sparse coding via global inhibition."
        ),
        biology=(
            "The mushroom body (MB) is the fly's olfactory learning centre. "
            "Each odour activates a small fraction (~5%) of the ~1900 Kenyon "
            "cells (KCs) in hemibrain. This sparse code is enforced by APL "
            "(anterior paired lateral) -- a single GABAergic neuron that "
            "pools activity across all KCs and inhibits them in turn. The "
            "circuit is feedforward in the excitatory direction (projection "
            "neurons -> KCs -> mushroom-body output neurons), with APL "
            "providing the only recurrent loop. We show a random subsample of "
            "20 KCs per subtype (~200 KCs) plus APL, both with APL intact and "
            "with APL's outgoing synapses ablated, to show APL's role."
        ),
        scenarios=[],
    )
    mb_outputs: list[ScenarioOutput] = []
    for scenario_spec, payload in mushroom_body_scenarios(conn_mb):
        model_id = _model_id_of(payload)
        path = out_dir / _scenario_filename("mushroom_body", scenario_spec.id, model_id)
        write_payload(payload, path)
        size_kb = path.stat().st_size / 1024
        mb_outputs.append(
            ScenarioOutput(
                dataset_id="mushroom_body",
                scenario_id=scenario_spec.id,
                label=scenario_spec.label,
                file=path.name,
                description=scenario_spec.description,
                size_kb=size_kb,
                model_id=model_id,
            )
        )
        print(f"  {path.name}: {size_kb:.0f} KiB · model={model_id} · {scenario_spec.label}")

    # ----- H01-inspired cortical microcircuit -----
    print("\n[h01_style] H01-inspired cortical microcircuit (60 cells)")
    h01_dataset = DatasetSpec(
        id="h01_style",
        label="Human cortex microcircuit (H01-inspired)",
        summary=(
            "60-neuron mammalian cortical microcircuit. Cell types and "
            "connection densities drawn from the H01 human-cortex EM paper."
        ),
        biology=(
            "Inspired by Shapson-Coe et al. 2024 (Science) -- the H01 "
            "dataset is ~1 mm³ of EM-traced human temporal cortex, ~50k "
            "cells. We don't ship the raw H01 data (it needs cloud-volume "
            "and a serious processing pipeline). Instead this circuit is a "
            "canonical microcircuit (Douglas & Martin 2004): three "
            "excitatory pyramidal populations (L2/3, L4 stellate, L5) plus "
            "three inhibitory interneuron classes (PV-basket, SST-Martinotti, "
            "VIP-bipolar) with H01-paper-derived connection densities. "
            "Critical mammalian-vs-fly difference: glutamate is "
            "EXCITATORY here (AMPA / NMDA), not inhibitory like in fly. "
            "The parameterizer uses `mammalian_default` NT signs. Full "
            "cloud-volume integration to pull the literal H01 cells is "
            "in BACKLOG."
        ),
        scenarios=[],
    )
    h01_outputs: list[ScenarioOutput] = []
    for scenario_spec, payload in h01_scenarios():
        model_id = _model_id_of(payload)
        path = out_dir / _scenario_filename("h01", scenario_spec.id, model_id)
        write_payload(payload, path)
        size_kb = path.stat().st_size / 1024
        h01_outputs.append(
            ScenarioOutput(
                dataset_id="h01_style",
                scenario_id=scenario_spec.id,
                label=scenario_spec.label,
                file=path.name,
                description=scenario_spec.description,
                size_kb=size_kb,
                model_id=model_id,
            )
        )
        print(f"  {path.name}: {size_kb:.0f} KiB · model={model_id} · {scenario_spec.label}")

    # ----- Real DTI (HCP NAP_001 via neurolib, AAL2 parcellation) -----
    dti_real_path = repo / "tests" / "fixtures" / "dti_hcp" / "DTI_CM.mat"
    dti_real_dataset = DatasetSpec(
        id="dti_hcp",
        label="Human DTI brain · HCP NAP_001 (real)",
        summary=(
            "94-region human DTI connectome from a real diffusion-MRI scan "
            "(HCP subject NAP_001), parcelled with AAL2."
        ),
        biology=(
            "Real diffusion-MRI tractography from one Human Connectome Project "
            "subject. The 94 nodes are cortical and subcortical regions in the "
            "Automated Anatomical Parcellation 2 (AAL2, Rolls et al. 2015 "
            "NeuroImage); edge weights are streamline counts between parcels. "
            "Data shipped via the neurolib project (NAP_001 subject under "
            "neurolib's gw/ dataset). The same Galvani pipeline that runs the "
            "single-cell hemibrain HD ring runs on this region-level matrix "
            "with zero changes -- Subgraph -> Parameterizer -> Simulator is "
            "agnostic to whether each 'neuron' is one cell or one cortical "
            "parcel. Regions are rendered as small 6-spoked stars at their MNI "
            "centroids."
        ),
        scenarios=[],
    )
    dti_real_outputs: list[ScenarioOutput] = []
    if dti_real_path.exists():
        print("\n[dti_hcp] real HCP/AAL2 data")
        for scenario_spec, payload in dti_real_scenarios(dti_real_path):
            model_id = _model_id_of(payload)
            path = out_dir / _scenario_filename("dti_hcp", scenario_spec.id, model_id)
            write_payload(payload, path)
            size_kb = path.stat().st_size / 1024
            dti_real_outputs.append(
                ScenarioOutput(
                    dataset_id="dti_hcp",
                    scenario_id=scenario_spec.id,
                    label=scenario_spec.label,
                    file=path.name,
                    description=scenario_spec.description,
                    size_kb=size_kb,
                    model_id=model_id,
                )
            )
            print(f"  {path.name}: {size_kb:.0f} KiB · model={model_id} · {scenario_spec.label}")
    else:
        print(f"\n[dti_hcp] skipped: {dti_real_path} not found")

    # ----- Manifest -----
    # Group by (dataset_id, scenario_id), collect available models per scenario.
    def _group_by_scenario(
        outputs: list[ScenarioOutput], preferred_first: str | None = None
    ) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for o in outputs:
            entry = seen.setdefault(
                o.scenario_id,
                {
                    "id": o.scenario_id,
                    "label": o.label,
                    "description": o.description,
                    "models": {},
                },
            )
            entry["models"][o.model_id] = o.file
        items = list(seen.values())
        if preferred_first and any(e["id"] == preferred_first for e in items):
            items.sort(key=lambda e: 0 if e["id"] == preferred_first else 1)
        return items

    def _dataset_entry(
        ds: DatasetSpec,
        outputs: list[ScenarioOutput],
        preferred_first: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": ds.id,
            "label": ds.label,
            "summary": ds.summary,
            "biology": ds.biology,
            "scenarios": _group_by_scenario(outputs, preferred_first=preferred_first),
        }

    manifest: dict[str, Any] = {
        "schema_version": 3,
        "datasets": [
            _dataset_entry(hd_dataset, hd_outputs, preferred_first="velocity_left"),
            _dataset_entry(mb_dataset, mb_outputs),
            _dataset_entry(h01_dataset, h01_outputs),
            _dataset_entry(dti_real_dataset, dti_real_outputs),
        ],
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest: {manifest_path.relative_to(repo)}")

    total = sum(o.size_kb for o in hd_outputs + mb_outputs + h01_outputs + dti_real_outputs)
    print(f"total payload size: {total / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
