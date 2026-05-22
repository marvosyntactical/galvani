# galvani

**A connectome → executable neural model pipeline.** Galvani takes a subgraph of a published connectome (fly hemibrain, human DTI, H01-style cortical microcircuits) and returns a runnable simulation — rate model, LIF, AdEx, or Hodgkin-Huxley — with biologically reasonable default parameters.

**Live demo:** [marvosyntactical.github.io/galvani](https://marvosyntactical.github.io/galvani)

Drag, zoom, click neurons. Switch between four levels of biophysical detail. The 3D morphology you see is real — electron-microscopy skeletons from the hemibrain dataset, or MNI centroids for the DTI scenes.

---

## What it does

There's a small but growing body of work that takes a connectome and builds a working neural model from it: Duan, Dong & Fiete (2025) for the fly head-direction ring, Lappalainen et al. (2024) for the optic lobe, Shiu et al. (2024) for the central complex, Pospisil et al. (2024) for the mushroom body. Every paper writes its own bespoke pipeline. None share code. The parameterization heuristics — synapse counts to weights, transmitter to sign, cell type to time constant — sit in supplementary methods sections.

Galvani is the pipeline pulled out of those papers and put behind one API.

```python
from galvani import HemibrainConnectome, default_parameterizer, simulate
from galvani.circuits import load_hd_ring
from galvani.stimuli import rotating_stimulus

conn = HemibrainConnectome()                       # neuPrint, with parquet cache
layout = load_hd_ring(conn)                        # EPG, PEN_a, PEN_b, Delta7
spec = default_parameterizer(layout.subgraph)      # ModelSpec dataclass

stim = rotating_stimulus(layout.angles, omega=1.0, width=0.5, amplitude=0.3)
result = simulate(spec, duration=4.0, stimulus=stim, dt=2e-4)
```

`result.rates` is a `(T, N)` array. `spec.weights` is the parameterized weight matrix you can write out as NumPy and feed to anyone else's simulator.

The same five lines also drive an LIF, AdEx, or HH simulation:

```python
from galvani.model.lif import simulate_lif
from galvani.model.adex import simulate_adex
from galvani.model.hh import simulate_hh

spikes = simulate_lif(spec, duration=2.0, stimulus=stim, v_threshold=0.5)
```

The point of the abstraction: `Subgraph` and `ParameterizerOptions` are fixed, the simulator backend varies. Same circuit, four levels of detail.

## What's in v1

### Connectome backends

| Backend | Source | Resolution | Notes |
|---|---|---|---|
| `HemibrainConnectome` | neuPrint hemibrain v1.2.1 | Single-cell EM | 130-neuron HD ring; 200-cell MB subset; per-neuron skeletons |
| `H01StyleConnectome` | Shapson-Coe et al. 2024, canonical microcircuit | Single-cell, 60 cells | L2/3, L4 stellate, L5, PV, SST, VIP. Real H01-via-cloud-volume is in the backlog |
| `DTIConnectome` | HCP NAP_001 via neurolib, AAL2 parcellation | Region-level, 94 nodes | Streamline counts from real diffusion-MRI; region centroids in MNI |

Adding a backend means implementing `BaseConnectome.query` and `subgraph` against your data source. The rest of the pipeline is unchanged.

### Models

| Backend | State | Notes |
|---|---|---|
| `model.rate` | scalar rate per neuron | `τ · dr/dt = -r + φ(W·r + I + b)`; default for population-level dynamics |
| `model.lif` | membrane voltage + spike | Threshold + reset + refractory; synaptic conductance trace τ_syn ≈ 5 ms |
| `model.adex` | voltage + adaptation `w` | Brette & Gerstner 2005; spike-frequency adaptation, bursting |
| `model.hh` | voltage + m/h/n gating | Hodgkin & Huxley 1952; Na⁺/K⁺/leak channels, real spike shape |

All four are pure NumPy. Brian2 is an opt-in extra for users who want the full SDE machinery.

### Parameterization

```python
ParameterizerOptions(
    weight_heuristic="log1p",       # synapse counts compress to weights
    nt_to_sign=fly_default,         # or mammalian_default for cortex
    symmetrize=True,                # ring-attractor circuits
    global_gain=0.012,              # set per circuit
)
```

The defaults are the ones from the HD-ring paper. The `weight_heuristic`, NT-to-sign mapping (`fly_default` treats glutamate as inhibitory, `mammalian_default` as excitatory), and per-cell-type tau lookup live in `parameterize/`. They're the actual research.

### Validation

The HD-ring circuit passes five qualitative tests from Kim et al. (2017) and Duan, Dong & Fiete (2025): bump existence, amplitude persistence, tracking under a rotating stimulus, velocity integration via L/R PEN pulses, and a finite gain window. They're pinned in `tests/test_validation_hd_ring.py` and run on every CI. Bump position drift without input is a known limitation (see `BACKLOG.md`).

The mushroom-body circuit demonstrates the canonical APL sparseness signature from Honegger et al. (2011) / Lin et al. (2014): drive 30 % of Kenyon cells, APL feedback knocks the population down to a few-percent sparse code. Counterfactually ablating APL's outgoing synapses eliminates the sparseness.

## Web demo

The browser app lives in `examples/web_demo/`. It's a Vite + React + react-three-fiber viewer over pre-baked simulation JSON (one file per circuit × scenario × model). The bake script (`scripts/build_demo_payload.py`) runs the Python pipeline and writes the JSON the viewer consumes.

```bash
NEUPRINT_TOKEN=... uv run python scripts/build_demo_payload.py
cd examples/web_demo && npm install && npm run dev
```

The viewer renders EM-traced skeletons (or DTI region stars) and animates per-neuron activity on top. Click a neuron to enter a detail mode with full SWC morphology, the model equations for the currently selected backend, and a top-k connectivity panel.

The deployed version at [marvosyntactical.github.io/galvani](https://marvosyntactical.github.io/galvani) is the same Vite build with the JSON payloads precomputed.

## For people learning the material

This section is for someone who has a working background in neuroscience or computational modeling and wants to figure out why connectome-constrained modeling is interesting right now.

The core idea: an EM connectome gives you the wiring diagram of a brain. A weight matrix lets you simulate dynamics. So in principle, the connectome should give you the dynamics — modulo turning synapse counts into synaptic weights, transmitter identities into signs, and cell types into time constants. The mapping from one to the other is where the modeling assumptions live, and where the literature disagrees.

The fly head-direction ring is the cleanest test case. EPG cells tile the ellipsoid body and form a ring-attractor topology. A single localized bump of activity marks the fly's current heading. PEN cells inject angular-velocity input that shifts the bump when the fly turns. Delta7 cells provide broad inhibition that stabilizes the bump shape. Kim et al. (2017) characterized the dynamics; Duan, Dong & Fiete (2025) showed you can recover the same dynamics from the hemibrain connectome with a fairly simple rate model. The Galvani HD-ring scenarios reproduce their qualitative findings.

If you want to understand the parameterization step, read `parameterize/weights.py`, `parameterize/signs.py`, and `parameterize/timeconst.py`. They're each under 100 lines. The `weight_heuristic="log1p"` choice — synapse counts get a `log(1 + x)` compression before being scaled by `global_gain` — is the single most consequential modeling decision in v1. See `DESIGN_DECISIONS.md` for the running log.

If you want to understand the models, the four files in `model/` correspond directly to the textbook equations. `rate.py` is one ODE per neuron. `lif.py` adds threshold + reset. `adex.py` adds the adaptation variable. `hh.py` adds the m/h/n gating variables and the Hodgkin-Huxley conductance equations.

If you want to dig into the connectome data: hemibrain via [`neuprint-python`](https://github.com/connectome-neuprint/neuprint-python) is the easiest entry point. FlyWire via [`fafbseg-py`](https://github.com/navis-org/fafbseg-py) and [`caveclient`](https://github.com/CAVEconnectome/CAVEclient) is next. MICrONS is also exposed through `caveclient`. The H01 human-cortex dataset is on Google Cloud and accessible via [`cloud-volume`](https://github.com/seung-lab/cloud-volume); the meshes need decimation through [`kimimaro`](https://github.com/seung-lab/kimimaro) before they're tractable. DTI tractography matrices ship with [`neurolib`](https://github.com/neurolib-dev/neurolib).

## Installation

```bash
# install uv: https://docs.astral.sh/uv/
uv sync --all-extras --group dev

uv run pytest
uv run ruff check .
uv run mypy src
```

The base install requires only NumPy, pandas, and pyarrow. neuPrint, Brian2, matplotlib, and plotly are optional extras (`uv sync --extra neuprint` etc.). A neuPrint API token is required for the first hemibrain query; subsequent queries hit a local parquet cache.

## Repository layout

```
src/galvani/
  connectome/     # backends: hemibrain (neuPrint), dti (neurolib), h01 (synthetic)
  parameterize/   # synapse counts → weights, NT → sign, type → tau — the research
  model/          # rate, lif, adex, hh simulators; ModelSpec dataclass
  circuits/       # convenience constructors (load_hd_ring, load_mushroom_body)
  stimuli.py      # rotating, ring, pulse, sum_stimuli, ...
  viz/            # payload builder for the web demo

examples/web_demo/   # Vite + R3F viewer; reads baked JSON payloads
scripts/build_demo_payload.py  # runs the full pipeline, writes payloads + manifest

tests/             # pytest, including validation tests pinned to literature
notebooks/         # validation notebooks (HD ring, mushroom body)
```

## Future directions

`BACKLOG.md` has the full deferred list. The big items:

- **Real H01 cells.** The current `H01StyleConnectome` is an H01-inspired canonical microcircuit (Douglas & Martin 2004 cell types, Shapson-Coe et al. 2024 densities). Pulling the literal 50k proofread cells via cloud-volume + decimating through kimimaro is ~2 days of integration work.
- **FlyWire / CAVE.** Second connectome backend. Will force a clean abstraction over neuPrint and CAVE conventions.
- **Eckstein NT loading.** hemibrain v1.2.1 predates Eckstein et al. (2024). Replace the hand-coded `HD_RING_NT` lookup with the Cell-2024 supplementary NT predictions joined on `bodyId`.
- **Live re-simulation in the browser.** WASM-compile the rate solver. Let users tweak gain / activation / time-constants and re-simulate at interactive rates.
- **Jaxley backend.** Differentiable simulation. Opens the door to parameter fitting against electrophysiology.
- **Interventions module.** Ablate, patch, scale weights. The mushroom-body APL ablation scenario hints at the right API shape.
- **Cell-type label translation.** hemibrain ↔ FlyWire have different naming conventions. A small translation table would let circuits cross datasets.
- **NeuroML / Brian2 export polish.** Round-trip `ModelSpec` to NeuroML so other simulators can pick it up.
- **Per-cell-type tau refinement.** The default 20 ms / 10 ms split is coarser than published values for the central complex. A literature-anchored table covering the HD ring and KC types is overdue.

## References

Connectome-constrained modeling:
- Duan, Dong & Fiete (2025) — fly HD ring from hemibrain, the closest analogue to what Galvani does
- Lappalainen et al. (2024) — *Connectome-constrained networks predict neural activity across the fly visual system*, Nature
- Shiu et al. (2024) — central-complex modeling from FlyWire
- Pospisil et al. (2024) — mushroom-body modeling

Connectomes:
- Scheffer et al. (2020) — hemibrain (Janelia), *eLife*
- Dorkenwald et al. (2024) — FlyWire whole-brain reconstruction, *Nature*
- MICrONS Consortium (2025) — mouse visual cortex EM + functional imaging, *Nature*
- Shapson-Coe et al. (2024) — H01 human cortex, *Science*
- Rolls, Joliot & Tzourio-Mazoyer (2015) — AAL2 atlas, *NeuroImage*

HD ring and mushroom body biology:
- Seelig & Jayaraman (2015) — first EB heading representation, *Nature*
- Kim et al. (2017) — ring-attractor dynamics, *Science*
- Honegger, Campbell & Turner (2011) — APL sparseness in mushroom body, *J. Neuroscience*
- Lin et al. (2014) — APL global inhibition, *Nature Neuroscience*

Models:
- Hodgkin & Huxley (1952) — ion channel kinetics, *J. Physiol.*
- Brette & Gerstner (2005) — AdEx, *J. Neurophysiology*
- Douglas & Martin (2004) — canonical cortical microcircuit, *Annu. Rev. Neurosci.*

Neurotransmitter predictions:
- Eckstein et al. (2024) — NT prediction across hemibrain, *Cell*

Libraries leaned on:
- [neuprint-python](https://github.com/connectome-neuprint/neuprint-python), [fafbseg-py](https://github.com/navis-org/fafbseg-py), [caveclient](https://github.com/CAVEconnectome/CAVEclient), [cloud-volume](https://github.com/seung-lab/cloud-volume), [kimimaro](https://github.com/seung-lab/kimimaro), [neurolib](https://github.com/neurolib-dev/neurolib), [Brian2](https://briansimulator.org/)

## License

MIT.

## Cite as

```bibtex
@software{koss_galvani_2026,
  author = {Koss, Marvin},
  title  = {Galvani: a connectome to executable neural model pipeline},
  year   = {2026},
  url    = {https://github.com/marvosyntactical/galvani},
  note   = {Live demo at https://marvosyntactical.github.io/galvani}
}
```
