# FUTURE — strategy notes

This is a strategy doc, not a plan. The plan is in `BACKLOG.md`. This one
is about *what Galvani should be for*, *what makes it distinct*, and
*which research it can produce*. Re-read before any major scope decision.

The thesis comes from a comp-neuro prof who saw an early demo: the best
value Galvani can bring is **explaining mechanisms and functions**.
That single line should anchor every prioritization call below. We are
not trying to be NEST/NEURON/Brian2 (general-purpose simulators), and we
are not trying to be Neuroglancer (a viewer). We are trying to be the
tool a researcher reaches for when they ask *why does this circuit
produce this behavior, and what specifically happens if I change it?*

## 1. Where we already stand

We have a real product, not a sketch:

- A pipeline that takes a connectome subgraph and produces a runnable
  ModelSpec across four levels of detail (rate, LIF, AdEx, HH).
- Three connectome backends covering three resolution scales —
  Drosophila EM (hemibrain), human cortex EM (H01), DTI tractography
  (HCP / AAL2).
- Validated parameterization defaults — the HD-ring scenario passes the
  five qualitative tests from Kim 2017 / Duan-Dong-Fiete 2025.
- A browser demo that renders the actual EM skeletons and animates the
  simulated dynamics on top, with a click-to-zoom-in multi-compartment
  Hodgkin-Huxley view (HCN, NMDA, conduction delay, Tsodyks-Markram
  short-term plasticity, per-cell-type channel densities — see
  [`MC.md`](MC.md)) running in a Web Worker.

The thing that nobody else has assembled is **the web demo + the
multi-resolution pipeline behind it**. Most existing tools have one or
the other.

## 2. What other tools offer (and where we fit)

| Tool | What it does | What it doesn't do |
| --- | --- | --- |
| **NEURON / Brian2 / NEST / GeNN** | General-purpose simulators of arbitrary network specs | They assume you've already built the spec. None of them touch connectomes. |
| **Jaxley** (Mackelab) | Differentiable JAX multi-compartmental sim | No connectome ingestion; no UI; aimed at parameter fitting, not exploration. |
| **Flywire Codex / Neuroglancer** | Browser viewer for EM connectomes | No dynamics. You can see the wiring but not what it does. |
| **navis / neuPrint-python / caveclient** | Connectome data access | Just data — no parameterization, no simulator. |
| **OpenWorm / c302** | C. elegans whole-organism model | Tied to one species; bespoke pipeline; aging codebase. |
| **The "bespoke paper pipeline" family** — Duan-Dong-Fiete 2025, Lappalainen 2024, Shiu 2024, Pospisil 2024 | A full connectome → dynamics → behavior story for one circuit | Every paper writes its own pipeline. None share code. The choices that *are* the research sit in supplementary methods. |
| **Allen Brain Atlas / AllenSDK** | Cell-type electrophysiology + transcriptomics | Doesn't simulate; no connectome graph. |

Galvani's niche is sharp: **the only tool where a user picks a circuit
from a real connectome, sees the dynamics evolve, and can drop down
into channel-level biophysics for a focused neighborhood — all in a
browser, with the same parameterization choices the publishable
papers use.**

The narrowest existential threat is that one of the bespoke-paper labs
decides to release their pipeline as a library. That hasn't happened
yet because each lab tunes for their one circuit and doesn't want to
own the abstraction. We *should* own the abstraction.

## 3. The "explain mechanisms" thesis, made concrete

"Explaining mechanisms" is what we need to be *legibly* good at. Three
concrete things this means:

### 3.1 Causal traces, not just dynamics

Right now we show *what happens*. We don't show *why*. A real
mechanism-explanation tool would let a user:

- Click a spike and walk backwards: which presynaptic spikes within the
  last τ caused this one? Pretty cheap to implement on top of the
  existing event log.
- Compute per-edge **contribution** to a downstream cell's activity over
  a window — basically a numerical partial-derivative of the rate or
  spike-count w.r.t. an upstream input. Either by perturbation
  (silence one cell, re-run, diff) or analytically via the rate
  Jacobian.
- Render the connectome graph with edges thickness-coded by
  contribution to the highlighted cell's response. The "circuit
  diagram you can actually read."

This is the differentiator most other tools don't even attempt. It's
also exactly the thing the comp-neuro prof was pointing at.

### 3.2 Lesion / perturbation as a first-class operation

Connectome-constrained modeling's strongest claim is **counterfactual**:
"if you silence cell type X, the dynamics should look like Y." That
matches optogenetic-ablation experiments one-for-one. The UI for this
should be one click, not a re-bake.

- Ship an **`interventions` module** (BACKLOG already lists this) with
  `silence`, `patch_activity`, `scale_weights`, `delete_edge`.
- Re-run interactively (web demo: WASM or pyodide for the rate model;
  pre-baked variants for HH).
- Side-by-side dynamics: baseline vs lesion, with a delta colormap.
- Reproduce the canonical published ablations as built-in scenarios
  (APL ablation in MB, EPG silencing in HD ring) — a user landing on
  the demo should be able to *see* the canonical result in two clicks.

### 3.3 Circuit → computational-graph mapping

A really mechanism-y view would let the user see a circuit as a graph
of compute operations (sum, gain, threshold, integrator, recurrent
loop) rather than as a 3D mesh. Two complementary views from the same
ModelSpec:

- The 3D anatomical view we already have.
- A node-and-edges abstract view where each cell is a labeled
  computational primitive ("integrator τ=20ms", "thresholded sum",
  "ring of N=46 attractor units") with the wiring matrix collapsed
  into a few labeled blocks (e.g. "EPG → PEN feedforward", "Delta7
  uniform inhibition").

Both views drive the same simulator. The abstract view is what makes
it usable as a *teaching* tool and what makes a paper's "circuit
schematic" stop being a hand-drawn cartoon and start being a
projection of the actual model.

This is also the most ambitious of the three — the right level of
abstraction is itself a research question. Don't ship it until we
know what to abstract over. (Probably won't be obvious until we've
written two or three real papers with Galvani.)

## 4. Pipeline improvements toward research-tool status

In rough priority order. Each entry should be evaluated against
"does it advance one of the three points above, or does it enable a
specific paper in §5?"

### 4.1 Connectome ingestion API

The strongest argument we have is "same pipeline across resolutions and
species." We need to make adding a new connectome a one-day job, not
a one-week one.

- **`BaseConnectome` contract polish.** Today the contract is implicit
  in the three existing backends. Spell it out: required methods,
  required fields on `Neuron`, optional fields, error semantics for
  missing NT / layer / cell type. Add a `validate_backend` function
  that any new backend can call to confirm conformance.
- **First-class FlyWire (CAVE) backend.** This is in BACKLOG and is
  the most important one: it unlocks a second EM connectome at the
  same resolution as hemibrain. Cross-dataset comparisons immediately
  become possible.
- **MICrONS backend.** Same `caveclient` infrastructure as FlyWire;
  mouse visual cortex. Sets us up for cross-species comparisons.
- **C. elegans backend** — see §5.4. The full connectome has been
  public for decades; the bottleneck is just writing the loader.
  Probably the easiest backend to add and produces the highest-impact
  scenario (whole-organism simulation in the browser).

### 4.2 Jaxley backend for parameter fitting

The biggest gap between "interesting demo" and "publishable result" is
parameter justification. Right now we ship literature-anchored
defaults; reviewers will ask "did you fit?" The answer should be
"yes, optionally, via Jaxley."

- New `model.jaxley_hh` backend that constructs the same compartment
  graph our in-browser solver builds and exposes a JAX-differentiable
  loss against a recorded voltage trace.
- A `fit_to_recording` helper that takes a Subgraph + a recording (e.g.
  Allen Cell Types Database trace for a matching cell type) and
  returns a tuned `ChannelDensities` per cell type.
- Saved fits live in `parameterize/fits/<dataset>/<cell_type>.json` and
  the default parameterizer falls through to them when available.

This is the move that makes the per-cell-type channel-density table
in `hhChannels.ts` go from "literature averages" to "fitted to
published recordings." Worth a lot of reviewer goodwill.

### 4.3 Headless batch mode for sweeps

Mechanism papers always include a "we systematically swept parameter X
across N values and observed Y." Right now there's no clean way to do
that — the bake script is one-shot.

- `galvani-sweep` CLI: takes a scenario, a parameter dict, a sweep
  spec (grid / random / Sobol), and outputs a parquet of
  `(param, scenario_metric)` rows.
- Plug the same sweep machinery into the web demo as the
  "play with this parameter" slider for the cheap (rate-model)
  scenarios.

### 4.4 Publication-quality output

A `result.to_paper(figure_kind="raster" | "rate_grid" | "circuit_diagram")`
that emits a matplotlib figure at journal-resolution defaults, plus
a `to_neuroml()` / `to_nwb()` export so reviewers can run the model
in their own tooling. Reduces friction massively for the "could you
share the model" review request that always comes.

### 4.5 Reproducibility envelope

Every web-demo scenario should pin: connectome version, parameterizer
config, RNG seed, model backend, dt, solver. Bake the manifest into
the result JSON. A user asking "what produced this trace" should never
have to read source code.

## 5. Concrete research projects

These are the papers we could write *with the current tooling plus
modest extensions*. Ordered by "how clear is the contribution."

### 5.1 Fly head-direction ring: a connectome-grounded perturbation atlas

**The Duan-Dong-Fiete-style paper, but with intervention.** Duan, Dong &
Fiete (2025) showed the hemibrain CX connectome produces ring-attractor
dynamics under a simple rate model. We can take the next step:
systematically perturb the circuit and ask which perturbations break
which functional properties.

- Single-cell silencing across all EPG / PEN / Delta7 cells: which
  cells contribute to bump *existence*, which to bump *position*, which
  to *integration of angular velocity*?
- Per-cell-type pharmacology: scale GABA-A or ACh receptor density,
  reproduce or predict the Kim 2017 / Turner-Evans 2020 perturbation
  experiments.
- NT-prediction sensitivity: re-classify a percentage of synapses with
  noise and ask at what corruption level the ring attractor breaks.
- Cross-individual comparison once FlyWire backend lands — does the
  circuit's robustness vary across individuals?

The story is clean: connectome + minimal modeling assumptions →
quantitative predictions about which manipulations preserve function.
Compares directly to published optogenetic experiments. Very legible
to a neuropsych audience because the *behavior* (heading drift) is
something they already know.

**Collaboration angle.** Whoever runs fly behavioral / imaging
experiments — even just a co-author who has done two-photon Ca²⁺
imaging on EPGs — can validate predictions.

### 5.2 Mushroom-body sparse coding under realistic odor statistics

**APL ablation has been done; we can do the dose-response curve and
the cross-odor sparsity geometry.** The MB scenario already shows the
canonical "APL feedback → sparse KC code" result. Beyond that:

- Sweep APL conductance from 0 to baseline → continuous
  sparsification curve. Predict the imaging signature for partial
  APL silencing (more interesting than full ablation, which has
  already been done).
- Multi-odor decorrelation: drive the MB with realistic ORN responses
  from the Hallem-Carlson dataset; measure the geometry of KC
  population vectors under intact vs ablated APL. Connectome-
  grounded prediction of representational geometry.
- Counterfactual on PN→KC convergence: scale the connectivity sparseness
  and ask how the geometry changes.

**Collaboration angle.** Any group doing fly olfactory imaging.
Hallem-Carlson data is public; KC imaging from the Turner lab and
others is widely available.

### 5.3 H01 cortical microcircuit: a "what does L4→L2/3 actually compute" study

**The H01 fixture (~80 real cells with measured connectivity) is a
small enough subgraph to simulate biophysically and large enough to
have nontrivial population dynamics.** This is a place where Galvani
has a genuine head start — nobody else is running biophysics on real
H01 cells in a browser.

- Drive L4 spiny stellates with a pulse and trace the cascade through
  the population. Quantify L2/3 amplification gain, L5 burstiness,
  PV-mediated gain control.
- Compare to canonical-microcircuit dynamics from Douglas-Martin /
  Markram literature. Where does the real EM connectome agree, where
  does it diverge?
- Single-edge perturbations across PV → pyramidal synapses: which
  specific synapses matter most for E/I balance?

This is the riskiest of the four because H01 is one slab of one
individual. But it's also the one where we have the most distinct
tooling — no other group has a browser-based H01-realistic biophysics
pipeline.

**Collaboration angle.** Cortical-circuit modelers (Markram lineage,
Brunel lineage) for the canonical-microcircuit framing.

### 5.4 C. elegans whole-organism dynamics

**Educational + accessible. C. elegans has 302 neurons; the connectome
has been public since White 1986; the dynamics fit comfortably in our
HH-on-morphology budget.** Ship the whole organism as a scenario in
the web demo. Drive it with the canonical behaviors:

- Tap withdrawal (the classic learning paradigm)
- Chemotaxis turn-and-run
- Spontaneous behavior states (roaming / dwelling)

Reproduce the OpenWorm / Izquierdo et al. behavioral predictions and
let the user lesion specific cells live in the browser.

**Why this matters.** It's the cleanest demonstration that the pipeline
generalizes beyond the fly. It's also the most directly comparable to
existing simulation work (OpenWorm), which gives reviewers a baseline
to anchor against.

**Collaboration angle.** Any C. elegans behavioral / imaging lab — the
Bargmann/Murphy/Bringuier lineage. Educational comp-neuro courses are
also a natural distribution channel.

### 5.5 The "mechanism explainer" methods paper

Once §3.1 / §3.2 are built, write a short methods paper:

> *Galvani: an interactive tool for explaining circuit mechanisms from
> EM connectomes.*

Demonstrate on three vignettes (HD ring, MB, H01 microcircuit). Argue
that the explanatory primitives — causal trace, contribution map,
one-click lesion — are independent of the underlying model. The
contribution is the *workflow*, not the simulator.

This is the paper that makes Galvani citeable. Without it, every other
paper using Galvani has to re-explain what Galvani is in its methods
section.

## 6. On whole-brain simulation

Asked explicitly in the brainstorm. The honest answer:

- **C. elegans (302 neurons):** Fully feasible end-to-end, including HH
  biophysics. Should be a shipped scenario. See §5.4.
- **Drosophila central complex (~3K neurons):** Feasible at rate-model
  resolution, already partially done. Biophysics on the full CX would
  be a stretch for the browser but reasonable server-side.
- **Drosophila hemibrain (~25K neurons):** Rate model is feasible
  server-side; the demo would pre-bake. Biophysics is out of reach;
  the C+D coupling design in `MC.md` is the right shape for "biophysics
  on a focused neighborhood within a large rate population."
- **Drosophila whole brain via FlyWire (~140K neurons):** Same as
  hemibrain, scaled. Not the most useful target until we have a real
  scientific question that needs it.
- **Mouse / human:** Out of scope at cellular resolution. DTI-level
  region simulation is already in v1 (the `DTIConnectome` backend) and
  is the right level of abstraction for whole-brain work.

The wrong framing is "can we simulate brain X." The right framing is
"what question are we answering, and what's the smallest part of brain
X that answers it." All four research projects in §5 simulate
subgraphs, not whole brains.

## 7. Educational value

Worth its own line. Comp-neuro pedagogy is *underserved* by good
interactive tools. Courses still hand out NEURON tutorials from 2005.
A polished Galvani demo could ride into:

- Standalone scenarios designed as classroom exercises. "Build the HD
  ring step by step": start with EPGs only (no bump), add Delta7 (now
  a bump exists but doesn't move), add PENs (now angular velocity is
  integrated).
- A "guided tour" mode that walks the user through a scenario with
  pop-up explanations at each step.
- Companion materials for textbooks (Dayan-Abbott, Gerstner-Kistler).
  We'd need to actually pitch this — Cambridge/MIT Press do collaborate
  with interactive web tools.

This is not a research paper but it's plausibly the highest-impact
direction in terms of *people reached*. Also a natural funding angle
(NSF education grants, Wellcome Open Research, etc.).

## 8. Risks and non-goals

- **Do not become a general-purpose simulator.** Brian2 and NEURON
  exist. The moment we add a feature that's only justified by "to
  match Brian2's API" we've lost the plot.
- **Do not over-build the abstract-graph view (§3.3) before we know
  what abstractions matter.** Write two real papers first.
- **Do not chase "whole brain" headlines.** See §6.
- **Do not lock the visualisation to one framework.** R3F today is
  fine, but the pipeline must keep producing standard outputs (NumPy
  arrays, parquet, NeuroML) that survive the viewer being thrown out
  and rewritten.
- **Reviewer will ask "have you validated against recordings?"** The
  honest answer for v1 is "qualitatively, against published findings."
  §4.2 (Jaxley fitting) is the path to a quantitative answer.

## 9. The one-line summary

Galvani's bet: that there's a real audience for *"here's the connectome,
here's what it computes, here's exactly what would happen if you
changed it"* — and that no existing tool serves that audience. The
research projects in §5 are the proof points; the infrastructure in §4
is what makes the proof points reproducible; the mechanism-explainer
features in §3 are what justifies the time the user spends in the tool
over reading a paper.

Everything else is incidental.
