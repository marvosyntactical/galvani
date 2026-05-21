# CircuitLab — Implementation Plan (v2)

*Reframed: the core contribution is the connectome-subgraph → executable-model pipeline. The HD ring becomes the first validation case. Interventions and visualization are downstream demonstrations that ride on top of the pipeline.*

*Replaces the v1 plan, which over-indexed on interventions and visual polish. The v1 framing was muddled between research infrastructure and educational toy; v2 commits to research infrastructure.*

---

## What we are building

**One sentence.** A Python library, with a small web demo, that takes any subgraph of a published connectome dataset and returns a runnable rate-model simulation with biologically reasonable default parameters — so that a researcher can go from "these neuron IDs" to "watch this bump form" in well under an hour.

**Why this rather than yet another circuit viewer.** Every recent paper in connectome-constrained modeling (Duan/Dong/Fiete 2025 for the HD ring; Lappalainen et al. 2024 for the fly visual system; Shiu et al. 2024 for sensorimotor processing; Pospisil et al. 2024 for the "effectome") writes its own bespoke pipeline from connectome → model. None share code. None expose the parameterization heuristics in a clean way. That is the structural gap in the field. A library that closes it is genuine infrastructure.

**Validation target.** Recover the Drosophila HD ring attractor from the hemibrain connectome, matching Duan/Dong/Fiete 2025 qualitatively. If our pipeline produces a network that exhibits a persistent bump that tracks rotating input, the pipeline works. If it doesn't, nothing else matters.

**Second target (proves generality, not optional).** Apply the same pipeline, unchanged, to a second circuit. Candidates: the mushroom-body input layer (Kenyon cells), or a small motion-detection subgraph from the optic lobe. The pipeline is "infrastructure" only if it works on more than one example.

**Non-goals.**

- We are not building a general-purpose brain simulator.
- We are not building a better Neuroglancer.
- We are not doing biophysical (Hodgkin-Huxley) modeling in v1. Rate models only.
- We are not building authentication, multi-user, or cloud deployment.
- We are not optimizing the visual aesthetic in v1. A working notebook beats a beautiful broken web app.

---

## Audience and what "success" means

The audience is **researchers doing connectome-constrained modeling** — small (~hundreds globally), but they cite tools they use. Success for v1 looks like:

- A Python package on PyPI, MIT or BSD licensed, with passing CI.
- A reproducible notebook that takes hemibrain IDs and produces the HD ring bump dynamics.
- A second notebook applying the same code to a second circuit.
- A short methods write-up suitable for a NeurIPS NeuroAI workshop or a Cosyne abstract.
- At least three people other than the author have run it without help.

Education, visual pop, and outreach are side effects, not goals. If the underlying library is good, they happen for free. If the underlying library is bad, no amount of polish saves it.

---

## Priorities (!!)

1. **!!! API design before implementation.** The single most important decision in the project is the shape of the connectome and model interfaces. Get this wrong and every downstream piece is harder. Spend the first two days sketching the API in a notebook, with no real code behind it. Iterate until it feels clean. Only then implement.

2. **!! Defaults are the actual research contribution.** The library is only useful if its defaults produce sensible models without expert tuning. The heuristics (synapse count → weight, neurotransmitter prediction → sign, cell type → time constant, etc.) are where the intellectual work lives. Document each one with the literature reference behind it.

3. **!! Validate against published results.** If the HD-ring output doesn't match the qualitative behavior in Duan/Dong/Fiete 2025 or Kim et al. 2017, the pipeline has a bug, not an interesting finding. Don't move on until the bump exists.

4. **! Two connectome sources before declaring v1 done.** hemibrain via neuPrint as the easy starter; FlyWire via CAVE as the second, harder one. Two sources forces a clean abstraction.

5. **! Two circuits before declaring v1 done.** HD ring as the first; a second circuit chosen near the end. Two circuits force the API to be actually general, not HD-ring-shaped.

6. **! Library first, web demo second.** A solid `pip install`-able library with notebooks is the real deliverable. The web UI is a demo to drive adoption. Inversion of v1's priorities.

7. **Scope discipline.** Interventions, FlyWire support, biophysical models, and pretty rendering all go in `BACKLOG.md` and stay there until the core is done. Re-read this list at the start of every work session.

---

## Architecture sketch

```
+--------------------------------------------------------------+
|                    circuitlab (Python library)               |
|                                                              |
|  +-------------+   +---------------+   +------------------+ |
|  | Connectome  |   | Parameterizer |   | Simulator        | |
|  | (interface) |-->| (defaults)    |-->| (Brian2 / NumPy) | |
|  +-------------+   +---------------+   +------------------+ |
|        ^                                                     |
|        | implements                                          |
|  +-----+--------+--------+                                   |
|  | Hemibrain    | FlyWire|     (more later)                  |
|  +--------------+--------+                                   |
|                                                              |
|  +----------------+    +---------------+                     |
|  | Interventions  |    | Visualization |                     |
|  | (downstream)   |    | (downstream)  |                     |
|  +----------------+    +---------------+                     |
+--------------------------------------------------------------+
                          |
                          | imports
                          v
+--------------------------------------------------------------+
|     Notebooks & web demo (separate repo or examples/ dir)    |
+--------------------------------------------------------------+
```

The library is the artifact. The notebooks demonstrate. The web demo is for adoption.

**Suggested package layout.**

```
circuitlab/
├── pyproject.toml
├── src/circuitlab/
│   ├── __init__.py
│   ├── connectome/
│   │   ├── base.py             # Abstract Connectome interface
│   │   ├── hemibrain.py        # neuPrint-backed implementation
│   │   ├── flywire.py          # CAVE-backed implementation (v1.5)
│   │   └── cache.py            # Local caching layer
│   ├── parameterize/
│   │   ├── defaults.py         # The default heuristics
│   │   ├── weights.py          # Synapse count -> weight
│   │   ├── signs.py            # NT prediction -> sign
│   │   └── timeconst.py        # Cell type -> tau
│   ├── model/
│   │   ├── spec.py             # Dataclass model representation
│   │   ├── rate.py             # Rate-model simulation
│   │   └── brian.py            # Brian2 export
│   ├── circuits/
│   │   └── hd_ring.py          # Convenience constructors
│   └── interventions/          # Downstream, v1.5+
├── tests/
├── notebooks/
│   ├── 01_hd_ring_hemibrain.ipynb
│   ├── 02_hd_ring_flywire.ipynb   # Later
│   └── 03_second_circuit.ipynb    # Later
└── examples/web_demo/             # Even later
```

---

## Phase 0 — Setup (½ day)

- Fresh Python 3.11+ environment. `uv` recommended for speed.
- Core dependencies: `neuprint-python`, `brian2`, `numpy`, `pandas`, `pyarrow`, `pytest`, `pytest-cov`, `ruff`, `mypy`.
- Initialize the repo with `pyproject.toml`, `ruff` config, GitHub Actions for CI (lint + test).
- Get a neuPrint auth token (free, requires Google sign-in at <https://neuprint.janelia.org>). Store outside the repo.
- Smoke test: `from neuprint import Client; c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1'); print(c.fetch_version())`.

**Checkpoint:** CI passes on an empty repo with a single placeholder test. neuPrint connection works.

---

## Phase 1 — API design (2 days, mostly thinking)

This is the most important phase. Do not skip it because it doesn't feel like progress.

Sketch — on paper or in a throwaway notebook — what calling code looks like. Aim for something like:

```python
from circuitlab import HemibrainConnectome, default_parameterizer, simulate

conn = HemibrainConnectome()                      # backed by neuPrint, caches locally
neurons = conn.query(type=['EPG', 'PEN_a', 'PEN_b', 'Delta7'])  # returns Neuron objects
subgraph = conn.subgraph(neurons)                 # weights, types, skeletons

model = default_parameterizer(subgraph)           # ModelSpec dataclass

results = simulate(model, duration='2*second', stimulus=ring_stimulus(angle=0.5))
results.plot_raster()
```

Things to nail down in the API:

- What is a `Neuron`? At minimum: ID, cell type, hemisphere, predicted NT, soma position, skeleton (optional). A dataclass.
- What is a `Subgraph`? A `Neuron` collection plus a sparse `(pre, post, count, NT_pre)` synapse table.
- What is a `ModelSpec`? A dataclass containing the weight matrix, per-neuron parameters, input/output specifications. Serializable to JSON for reproducibility.
- What is the `Parameterizer` interface? A callable that takes a `Subgraph` and returns a `ModelSpec`, with a `.defaults` attribute exposing every heuristic used.
- What is the `Simulator` interface? Takes a `ModelSpec`, a stimulus, and a duration, returns a `Result` object. Should be swappable (NumPy for fast iteration, Brian2 for correctness, Jaxley for later differentiable work).

**Gotcha:** the temptation to make everything pluggable will produce an abstract-factory-pattern monstrosity. Resist. Keep abstractions thin. One concrete `Connectome` subclass and one concrete `Parameterizer` exist at first.

**Checkpoint:** A `README.md` with a 20-line code snippet showing the intended usage. No real code under it yet, but the snippet feels right when read aloud. Share with one person and confirm it reads naturally.

---

## Phase 2 — hemibrain loader (2–3 days)

Concrete implementation of `Connectome` for hemibrain via neuPrint.

What it needs to do:

- Query neurons by cell type, with results cached locally as parquet.
- Fetch the synapse table for a set of neurons (pre × post matrix in long form).
- Pull skeletons (SWC) for visualization (cache locally as files).
- Pin to a specific dataset version (e.g., `hemibrain:v1.2.1`). Never silently follow the latest.

**Gotchas.**

- neuPrint's Python client returns pandas DataFrames with column conventions that are not stable across versions. Wrap them in your own typed objects ASAP.
- "Cell type" in hemibrain has multiple levels (instance, type, supertype). Be explicit about which one you query. For HD ring you want `type` (e.g., `EPG`, `PEN_a`).
- The neurotransmitter predictions in hemibrain come from Eckstein et al. 2024 and are stored per-neuron. They are ~85% accurate. Treat them as defaults that the user can override.
- Hemibrain only covers the central brain (no optic lobes, no VNC). If a circuit crosses that boundary, hemibrain alone isn't enough.

**Useful links:**

- neuPrint Python client: <https://connectome-neuprint.github.io/neuprint-python/docs/>
- Eckstein et al. 2024 (NT prediction): <https://www.cell.com/cell/fulltext/S0092-8674(24)00307-6>
- Janelia FlyEM hemibrain release notes: <https://www.janelia.org/project-team/flyem/hemibrain>

**Checkpoint:** `conn.query(type='EPG')` returns ~46 `Neuron` objects, each with a cell type, NT, position, and skeleton path. Plotting their soma positions in 3D produces the expected ring in the ellipsoid body.

---

## Phase 3 — Default parameterization (3–5 days, the actual research)

This is where the library earns its keep. Each default is a small modeling decision; document each one with a citation.

**Defaults to implement in v1:**

- **Synapse count → weight.** Recommended: `W_ij = log(1 + count_ij)`. Justification: synapse counts are heavy-tailed; log-compression matches the dynamic range expected of synaptic gains. Alternatives to expose: raw, sqrt, rank.
- **NT prediction → sign.** Cholinergic → +1; GABAergic and glutamatergic → −1 (in fly, glutamate is largely inhibitory via GluClα). Octopaminergic, serotonergic, dopaminergic → 0 in v1 (treated as modulatory, not in the recurrent dynamics). Expose a `nt_to_sign` callable that the user can override.
- **Cell type → time constant.** Default 20 ms for excitatory, 10 ms for inhibitory; literature-derived where possible. Per-cell-type override table.
- **Global gain.** Single scalar multiplier on the whole weight matrix. The model is sensitive to this; expose it explicitly rather than burying it.
- **Bias / threshold.** Per-cell-type bias, default 0. Often needed for stable bumps; expose for tuning.
- **Symmetrization (optional).** The HD ring has left/right symmetry that the connectome breaks. Duan/Dong/Fiete symmetrize synapse counts across hemispheres before modeling. Implement as opt-in.

**Gotcha:** every default is wrong somewhere. The right move is to document where it's likely to fail, not to find a default that's right everywhere. Make the heuristic legible so the user can override when they know better.

**Output:** a `ModelSpec` dataclass that is fully self-describing (every parameter, every choice, the version of every default function used) and JSON-serializable for reproducibility.

**Checkpoint:** `default_parameterizer(hd_ring_subgraph)` produces a `ModelSpec` whose attributes match expectations: ~130 neurons, weights with reasonable magnitudes, sensible signs, no NaNs.

---

## Phase 4 — Simulator (2–3 days)

A rate-model simulator that takes a `ModelSpec` and runs it.

$$\tau_i \frac{dr_i}{dt} = -r_i + \phi\left(\sum_j W_{ij} r_j + I_i(t) + b_i\right)$$

Two backends in v1, sharing one interface:

- **NumPy backend:** explicit forward Euler with adaptive step. Fast for small networks, easy to debug, no compilation overhead. Make this the default.
- **Brian2 backend:** for when you want to extend to LIF later or want NeuroML export. Slower startup but more extensible.

**Gotcha:** Brian2's first-run compilation overhead is ~30 s. For an interactive demo, NumPy is much friendlier. Don't ship Brian2 as the default.

**Stimulus interface:** a function `t → I_per_neuron`. Provide one or two canonical ones (ring stimulus, step input) as utilities.

**Result object:** holds rates, time points, model spec used, and convenience methods (`.plot_raster()`, `.angle_over_time()`, `.to_xarray()`).

**Checkpoint:** simulate the HD ring with a Gaussian input centered at one EPG cell. Plot the rates over time. See a bump form. Don't move on until the bump is there.

---

## Phase 5 — First validation: HD ring (1 week)

The notebook that justifies the library's existence.

Order of validation tests:

1. **Bump existence.** Stationary input → stationary bump centered on the input.
2. **Bump persistence.** Remove input after 200 ms → bump remains for several seconds.
3. **Bump tracking.** Rotating input → bump follows.
4. **Velocity integration.** Pulse one PEN subpopulation → bump moves in one direction. Pulse the other → bump moves the other way. This is the canonical test from Kim et al. 2017 and replicated in Duan/Dong/Fiete 2025.
5. **Sensitivity sweeps.** Vary global gain; show the bump regime is a finite window.

Each test gets a plot and a one-sentence interpretation in the notebook. If any of (1)–(4) fail, something is wrong with the pipeline, the defaults, or the data. Debug there before adding features.

**Reference papers (read before this phase, not during):**

- Kim, Rouault, Druckmann, Jayaraman (2017). *Ring attractor dynamics in the Drosophila central brain.* Science 356:849–853. <https://www.science.org/doi/10.1126/science.aal4835>
- Turner-Evans et al. (2020). *The neuroanatomical ultrastructure and function of a biological ring attractor.* Neuron 108:145–163. <https://www.cell.com/neuron/fulltext/S0896-6273(20)30540-0>
- Duan, Dong, Fiete (2025). *From Synapses to Dynamics.* bioRxiv 2025.05.26.655406. <https://www.biorxiv.org/content/10.1101/2025.05.26.655406v1>
- Seelig, Jayaraman (2015). *Neural dynamics for landmark orientation and angular path integration.* Nature 521:186–191.

**Checkpoint:** the validation notebook runs top-to-bottom in <2 minutes on a laptop, produces five figures, and qualitatively matches published HD-ring behavior. This is the v1 minimum.

---

## Phase 6 — Visualization (1 week)

Now, and only now, build a visual layer.

Two levels:

- **In-notebook plots.** Static and animated matplotlib/plotly, integrated with the `Result` object. This is what researchers actually use. Spend most of the time here.
- **Standalone web demo.** A small Vite + React + Three.js page that loads a pre-computed simulation and plays it back. Optional in v1; primarily for adoption.

The web demo, when it happens, should:

- Render the EB skeletons in 3D.
- Color by activity from a pre-recorded simulation (not a live one — live simulation in browser is over-engineering for v1).
- Provide a time scrubber.
- Be small enough to host on GitHub Pages or Vercel free tier.

Tech for the web demo:

- Vite + React + TypeScript.
- `react-three-fiber` + `drei` for the 3D scene.
- `uPlot` for the spike/rate raster.
- Pre-computed simulation data shipped as a static JSON or HDF5 file.

**Gotcha:** the web demo absorbs unbounded time if you let it. Cap effort at one week. If it isn't beautiful in a week, ship it functional and move on.

---

## Phase 7 — Generality test: second circuit (3–5 days)

Pick a second circuit and run the pipeline on it without modifying the library. The point is to find what breaks.

Candidates:

- **Mushroom-body Kenyon cells + APL.** ~2000 Kenyon cells, feedforward from projection neurons, single global inhibitory neuron (APL). Different topology, similar simulation.
- **Lamina / medulla motion-detection subgraph.** Smaller, well-characterized (Hassenstein-Reichardt-like).
- **Central complex fan-shaped body.** Adjacent to the EB; uses the same upstream pipeline.

Expect the library to break in interesting ways on the second circuit. Each break is a real bug or missing default. Fix them. This is the phase that converts the project from "neat HD-ring demo" to "general infrastructure."

**Checkpoint:** the second circuit runs end-to-end with at most one explicit override of a default. Documented in a second notebook.

---

## Phase 8 — Polish and ship (1 week)

Only now.

- Tag a v0.1.0 release on GitHub.
- Publish to PyPI.
- Write the README properly: what it is, who it's for, the two demonstrations, how to install, how to cite.
- Write a methods short paper (~4 pages) targeting a workshop. NeurIPS NeuroAI workshop and Cosyne abstracts are the natural venues.
- Share with three people in the field for feedback. Iterate based on what breaks.
- Post on the relevant Slacks (FlyWire, Neuromatch, computational neuro Twitter/Bluesky).

---

## Cross-cutting gotchas

- **Pinned versions everywhere.** Dataset version (`hemibrain:v1.2.1`), neuPrint client version, Brian2 version. Reproducibility is the whole point of infrastructure.
- **Cache invalidation will bite.** Have a `--refresh` flag and a documented cache directory. Don't put the cache in the package directory; use `~/.cache/circuitlab/` or `appdirs`.
- **Don't fit parameters to neural recordings in v1.** Tempting and harder than it sounds. Park in BACKLOG.
- **Cell-type labels are not always consistent across datasets.** "EPG" in hemibrain may have a different label in FlyWire. Plan for a translation layer when adding FlyWire support.
- **Asymmetry breaks ring attractors.** The default behavior on a real connectome may not exhibit a bump until symmetrization is applied. Make the symmetrization step opt-in but document it prominently.
- **The HD ring contains ring neurons (R neurons) too, not just EPG/PEN/Δ7.** Whether to include them changes the dynamics. Pick a defensible inclusion list and document it.
- **Glutamate is inhibitory in fly.** Don't apply the mammalian convention. This will silently produce wrong models.
- **Predicted neurotransmitters are sometimes confidently wrong.** When the model behaves badly, manually check the NT predictions of the most-connected neurons first.

---

## Recommendations for working with an AI coding agent

- Provide the priorities list at every session start. Re-anchor often.
- Resist letting the agent jump to "let me also add interventions." That comes in v1.5. The MVP is the pipeline + the HD ring.
- Force the agent to write the API design notebook before any real code. Push back if it skips Phase 1.
- Insist on tests for the parameterizer defaults. Each default is a small modeling decision that should be regression-tested. (Example test: "default weight matrix has signs matching NT predictions for all entries.")
- Maintain a `BACKLOG.md` and a `DESIGN_DECISIONS.md`. The agent will want to make ambient choices; surface them.
- Ask for diffs, not rewrites, once the pipeline works.
- Don't let the agent gold-plate the web demo. One week, then stop.

---

## Quick-reference links

**Data and connectome.**

- hemibrain via neuPrint (start here): <https://neuprint.janelia.org>
- `neuprint-python`: <https://connectome-neuprint.github.io/neuprint-python/docs/>
- FlyWire (for later): <https://flywire.ai>
- `navis`: <https://navis-org.github.io/navis/>
- Eckstein et al. 2024 (NT prediction): <https://www.cell.com/cell/fulltext/S0092-8674(24)00307-6>

**Simulation.**

- Brian2: <https://brian2.readthedocs.io/>
- Jaxley (for differentiable v2): <https://github.com/jaxleyverse/jaxley>
- NeuroML (for export, optional): <https://www.neuroml.org/>

**Reference papers.**

- Duan, Dong, Fiete (2025), *From Synapses to Dynamics*: <https://www.biorxiv.org/content/10.1101/2025.05.26.655406v1> · NeurIPS poster: <https://openreview.net/forum?id=zn4F6os6cq>
- Kim et al. (2017), *Ring attractor dynamics*: <https://www.science.org/doi/10.1126/science.aal4835>
- Turner-Evans et al. (2020), *Biological ring attractor*: <https://www.cell.com/neuron/fulltext/S0896-6273(20)30540-0>
- Lappalainen et al. (2024), *Connectome-constrained networks predict neural activity*: <https://www.nature.com/articles/s41586-024-07939-3>
- Shiu et al. (2024), *Drosophila computational brain model*: <https://www.nature.com/articles/s41586-024-07763-9>

**Frontend (when the time comes).**

- `react-three-fiber`: <https://docs.pmnd.rs/react-three-fiber/>
- `drei`: <https://github.com/pmndrs/drei>
- `uPlot`: <https://github.com/leeoniya/uPlot>

**Adjacent tools to study.**

- Neuroglancer: <https://github.com/google/neuroglancer>
- NetPyNE: <https://www.netpyne.org/>
- Open Source Brain: <https://www.opensourcebrain.org/>

---

## Definition of done (v1)

All true:

- A Python package, installable from PyPI, with passing CI.
- A `Connectome` abstraction with one working implementation (hemibrain via neuPrint).
- A `Parameterizer` with documented defaults, each justified by a literature citation.
- A `Simulator` running rate-model dynamics from a `ModelSpec`.
- A notebook that recovers HD-ring bump dynamics from hemibrain, matching Duan/Dong/Fiete 2025 qualitatively.
- A second notebook applying the same pipeline, unmodified, to a different circuit.
- A README that another researcher can follow start-to-finish without help.
- At least three external users have run the notebooks successfully.

Past this point, the project is shippable. Things deferred to v1.5+: FlyWire support, interventions (ablation, activity patching), biophysical (LIF/HH) models, web demo polish, parameter fitting to neural recordings.
