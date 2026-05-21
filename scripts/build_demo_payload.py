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
from galvani.connectome.hemibrain import HD_RING_NT, HemibrainConnectome
from galvani.model.lif import simulate_lif
from galvani.model.rate import relu, tanh
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


# ---------------- LIF biophysical scenario -----------------------------------


def lif_hd_ring_scenarios(subgraph, angles, n_neurons, neuron_ids):
    """One LIF (spiking) HD-ring scenario."""
    gain = 0.04  # LIF needs more drive than rate model
    opts = ParameterizerOptions(symmetrize=True, global_gain=gain)
    spec = default_parameterizer(subgraph, opts)
    omega = 1.0
    n_total = n_neurons
    stim = rotating_stimulus(angles, omega=omega, width=0.5, amplitude=1.5)
    result = simulate_lif(spec, duration=2.0, stimulus=stim, dt=2e-4, v_threshold=0.5)
    yield (
        ScenarioSpec(
            id="lif_tracking",
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
            scenario_id="lif_tracking",
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
    _ = n_total, neuron_ids  # silence unused (kept for signature parity)


# ---------------- DTI (synthetic brain) scenarios ----------------------------


def dti_scenarios():
    """Region-level rate dynamics on a synthetic DTI matrix."""
    conn = DTIConnectome.synthetic(n_regions=30, seed=0)
    neurons = conn.query()
    subgraph = conn.subgraph(neurons)
    # Region-level defaults: no NT signs (DTI is unsigned), tau = 100 ms,
    # symmetrize since DTI matrices are already symmetric.
    opts = ParameterizerOptions(symmetrize=True, global_gain=0.001, weight_heuristic="log1p")
    spec = default_parameterizer(subgraph, opts)
    # Override tau to a region-appropriate 100 ms (slower than single neurons).
    import dataclasses as _dc

    spec = _dc.replace(spec, tau=np.full_like(spec.tau, 0.1))

    n = len(neurons)
    # Stimulus: drive one hemisphere first, then switch.
    rng = np.random.default_rng(1)
    side_l = np.array([n_.hemisphere == "L" for n_ in neurons])
    side_r = ~side_l
    pattern_l = np.where(side_l, 0.5 + 0.1 * rng.normal(size=n), 0.0)
    pattern_r = np.where(side_r, 0.5 + 0.1 * rng.normal(size=n), 0.0)

    def stim(t):
        return pattern_l if t < 1.0 else pattern_r

    result = simulate(spec, duration=2.0, stimulus=stim, activation=tanh, dt=2e-3)
    yield (
        ScenarioSpec(
            id="hemispheric_drive",
            label="Hemispheric drive switch",
            description=(
                "30-region synthetic DTI brain. Drive the left hemisphere "
                "for 1 second, then switch to the right. Activity propagates "
                "across the inter-hemispheric connections."
            ),
        ),
        build_payload(
            subgraph,
            result,
            skeleton_provider=conn,
            spec=spec,
            dataset_id="dti_synthetic",
            scenario_id="hemispheric_drive",
            scenario_label="Hemispheric drive switch",
            description=(
                "Synthetic DTI connectome (30 regions, 15 per hemisphere) "
                "with biologically-plausible motifs: log-normal weight "
                "distribution, intra-hemisphere bias, homotopic-pair "
                "callosal boosts. Drive the L hemisphere for 1 s, then R. "
                "Watch activity propagate across the synthetic connectome."
            ),
            hyperparams={
                "global_gain": 0.001,
                "symmetrize": True,
                "weight_heuristic": "log1p",
                "activation": "tanh",
                "dt_sim": 2e-3,
                "tau_default_ms": 100.0,
                "n_regions": 30,
                "stimulus": {
                    "type": "hemispheric_switch",
                    "switch_at_s": 1.0,
                    "amplitude": 0.5,
                },
            },
            stim_fn=stim,
            stride=1,
            min_radius=0.0,
            n_frames=80,
            target_scale=20.0,
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
            path = out_dir / f"hd_ring_{scenario_spec.id}.json"
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
                )
            )
            print(f"  {path.name}: {size_kb:.0f} KiB ({scenario_spec.label})")

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
        path = out_dir / f"mushroom_body_{scenario_spec.id}.json"
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
            )
        )
        print(f"  {path.name}: {size_kb:.0f} KiB ({scenario_spec.label})")

    # ----- DTI synthetic -----
    print("\n[dti_synthetic]")
    dti_dataset = DatasetSpec(
        id="dti_synthetic",
        label="Synthetic DTI brain (30 regions)",
        summary=(
            "Region-level connectome via diffusion-MRI tractography. "
            "Each 'neuron' is one cortical parcel; weights are streamline counts."
        ),
        biology=(
            "Diffusion-tensor imaging (DTI) infers white-matter fiber tracts "
            "in living human / macaque brains by measuring water diffusion "
            "direction on MRI. The resulting 'tractography' yields a "
            "connectivity matrix between cortical parcels -- a connectome "
            "at *region* resolution, not single neurons. The same Galvani "
            "pipeline runs on it: Subgraph -> Parameterizer -> Simulator, "
            "unchanged. This synthetic example mimics canonical DTI motifs: "
            "log-normal streamline counts, dense intra-hemispheric "
            "connectivity, sparser inter-hemispheric connectivity with a "
            "strong 'homotopic' boost between mirror-image regions (the "
            "callosal connections in real brains). Region nodes are rendered "
            "as small 6-spoked stars at their centroids since DTI parcels "
            "don't have skeleton morphology like single neurons."
        ),
        scenarios=[],
    )
    dti_outputs: list[ScenarioOutput] = []
    for scenario_spec, payload in dti_scenarios():
        path = out_dir / f"dti_{scenario_spec.id}.json"
        write_payload(payload, path)
        size_kb = path.stat().st_size / 1024
        dti_outputs.append(
            ScenarioOutput(
                dataset_id="dti_synthetic",
                scenario_id=scenario_spec.id,
                label=scenario_spec.label,
                file=path.name,
                description=scenario_spec.description,
                size_kb=size_kb,
            )
        )
        print(f"  {path.name}: {size_kb:.0f} KiB ({scenario_spec.label})")

    # ----- Manifest -----
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "datasets": [
            {
                "id": hd_dataset.id,
                "label": hd_dataset.label,
                "summary": hd_dataset.summary,
                "biology": hd_dataset.biology,
                "scenarios": [
                    {
                        "id": o.scenario_id,
                        "label": o.label,
                        "file": o.file,
                        "description": o.description,
                    }
                    for o in hd_outputs
                ],
            },
            {
                "id": mb_dataset.id,
                "label": mb_dataset.label,
                "summary": mb_dataset.summary,
                "biology": mb_dataset.biology,
                "scenarios": [
                    {
                        "id": o.scenario_id,
                        "label": o.label,
                        "file": o.file,
                        "description": o.description,
                    }
                    for o in mb_outputs
                ],
            },
            {
                "id": dti_dataset.id,
                "label": dti_dataset.label,
                "summary": dti_dataset.summary,
                "biology": dti_dataset.biology,
                "scenarios": [
                    {
                        "id": o.scenario_id,
                        "label": o.label,
                        "file": o.file,
                        "description": o.description,
                    }
                    for o in dti_outputs
                ],
            },
        ],
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest: {manifest_path.relative_to(repo)}")

    total = sum(o.size_kb for o in hd_outputs + mb_outputs + dti_outputs)
    print(f"total payload size: {total / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
