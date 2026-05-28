# TSM — positioning Galvani for a "Time Series Modelling" master practical

Strategy notes for pitching a Galvani-based project into the "Time Series
Modelling" MP offered by the neuroscience-adjacent professor whose group
fits **state-space models (SSMs) and recurrent latent-dynamics models to
brain imaging data**. 8 ECTS ≈ 240 hours ≈ one semester part-time.

This is a positioning doc, not a plan. The goal: find the framing where
Galvani is a *credible stepping stone* into that group's work — one that
(a) demonstrates the modelling skills they live on, (b) brings something
they don't already have, and (c) is scoped to fit the practical.

> **Assumption to verify.** The brief ("SSMs + recurrent models on brain
> imaging", neuroscience-adjacent, Jaxley on the radar) reads like the
> Macke-lab orbit (Tübingen) or a Linderman/Sahani-style latent-dynamics
> group. Both Jaxley (differentiable biophysics) and `sbi` (simulation-
> based inference) come from the Macke lab; if that's the professor, the
> SBI / differentiable-simulator angles below are the sharpest bridges.
> Confirm before committing.

## 1. The core angle: Galvani is a *ground-truth generator* for time-series methods

The hard problem in fitting SSMs to brain imaging is that **you never know
the true latent dynamics**. You fit an rSLDS to two-photon data, recover a
3-D latent with two fixed points, and… is that real, or an artifact of the
model class, the observation noise, the subsampling? There's no answer key.

Galvani produces neural time series from a **connectome-constrained
mechanistic model whose structure is fully known**: the weight matrix, the
effective dimensionality, the attractor topology, the bifurcation
structure under parameter sweeps. So Galvani can be the **answer key** for
time-series methods:

- Generate `(T, N)` time series from a circuit whose true dynamics we know.
- Fit the SSM the group cares about.
- Measure *how well it recovers the truth* — latent dimension, fixed
  points, manifold topology — as a function of noise, subsampling rate,
  observation model, and trial count.

This reframes "I built a simulator" into "I built a **benchmark and
validation testbed for latent-dynamics inference**", which is squarely a
methods contribution the group can use. The HD-ring circuit is the hero
example: 130 neurons collapsing onto a **1-D ring attractor** is the
cleanest possible identifiability target.

## 2. What time series can we model? (three tiers)

**Tier A — Galvani's own simulated output (primary).**
Every circuit × model backend emits a `(T, N)` array:
- `rate` — continuous firing rate per neuron (smooth, the natural SSM
  observation).
- `lif` / `adex` / `hh` — spike trains / membrane voltages (point-process
  or high-rate continuous observations).
Ground truth (W, gain, attractor type) is known exactly. This is the
testbed. No data licensing, no preprocessing, full control over noise and
sampling.

**Tier B — real neural recordings of the *same* circuits (validation).**
The fly head-direction system is one of the best-characterised dynamical
circuits in neuroscience, with public two-photon calcium imaging of the
EPG "bump":
- Seelig & Jayaraman 2015; Kim et al. 2017; Turner-Evans et al. 2020.
Fitting the *same* SSM to (i) Galvani's HD-ring simulation and (ii) real
EPG imaging, then comparing the recovered ring manifold, is the
sim-to-data bridge that makes the project land as neuroscience, not just
methods. Calcium imaging also forces a realistic observation model
(slow indicator kernel, low SNR) — exactly what the group handles.

**Tier C — macro-scale brain imaging (the group's home turf).**
Galvani already ingests HCP/AAL2 DTI tractography (94 regions) and runs
region-level rate dynamics on it. That's a whole-brain connectivity → BOLD-
like time series generator. Fitting SSMs to region-level simulated
dynamics — and, if available, to real resting-state fMRI on the same
parcellation — is the most direct match to "SSMs on brain imaging data".
Lower mechanistic fidelity, higher relevance to their existing pipeline.

## 3. The modelling toolbox (what "fit" means)

Ordered roughly classic → modern → mechanistic.

- **Linear dynamical system (LDS).** Kalman filter + EM. The baseline. Will
  it recover a linear approximation of the ring? (No — a ring attractor is
  fundamentally nonlinear; showing the LDS *fails* in an interpretable way
  is itself a result.)
- **Switching / recurrent SLDS (SLDS, rSLDS).** Linderman's `ssm` /
  `dynamax` libraries. Piecewise-linear latent dynamics with discrete
  modes. The standard tool for this question; can approximate attractors
  with enough modes. Recovering the ring's rotational structure here is
  the central positive result to aim for.
- **Deep latent dynamics — LFADS / latent SDEs / deep SSMs (S4, S5,
  Mamba).** RNN- or structured-SSM-based. More expressive, less
  interpretable. Good for "can a modern sequence model recover the
  manifold without mechanistic priors?" and connects to the recurrent-
  models half of the group's interest.
- **Simulation-based inference (`sbi`).** Treat Galvani as a black-box
  simulator with parameters θ = (gain, NT-sign rule, time constants).
  Given an observed time series (real or held-out simulation), infer the
  **posterior over θ** via neural posterior estimation. This is the most
  Macke-lab-native framing: mechanistic model + amortised Bayesian
  inference. "Can we recover the connectome's effective gain from the bump
  dynamics alone?"
- **Differentiable fitting via JAX / Jaxley.** Two variants:
  - *Phenomenological:* port Galvani's rate/LIF simulator to JAX, make it
    differentiable, fit W or the parameterization knobs to a target time
    series by gradient descent. Direct, tractable.
  - *Biophysical (Jaxley proper):* Galvani's multi-compartment HH view
    (`MC.md`) is exactly what Jaxley is built for. Re-express one circuit's
    compartmental model in Jaxley and fit channel densities / synaptic
    weights to target voltage traces. Most ambitious; closest to the
    differentiable-biophysics frontier; data-hungry and the riskiest to
    finish in 8 ECTS.

## 4. "Fit existing data" vs "analyse emergent dynamics" — you can have both

The user's instinct — *maybe we don't fit external data at all, we just
analyse the dynamics our sim emits* — is viable and de-risks the data
problem. But pure dynamical-systems analysis risks reading as "not really
time-series *modelling*" to an SSM-focused examiner. The fix is to make the
emergent-dynamics analysis the **ground truth that the SSM fit is scored
against**:

1. **Characterise the true dynamics** (no fitting). On Galvani output, do
   the dynamical-systems workup: fixed-point finding (solve `f(z*) = 0`),
   Jacobian eigenvalues at fixed points, attractor topology (ring? point?
   limit cycle?), and bifurcation diagrams as `global_gain` sweeps. For the
   HD ring this recovers "1-D continuous ring attractor, marginal stability
   around the ring, loses the bump below a critical gain". This is the
   answer key, and it's a self-contained, data-free contribution on its own.

2. **Fit the SSM and score recovery against (1).** Does the rSLDS find a 1-D
   ring? Does the recovered latent dimension match? Do the inferred fixed
   points line up? How does recovery degrade with calcium-like observation
   noise / temporal downsampling / fewer neurons observed?

This combination demonstrates *both* dynamical-systems chops *and* the
SSM-fitting skill the group runs on, while keeping the data dependency
optional (Tier B/C become "bonus validation", not critical path).

## 5. Concrete project framings

Four, ranked by impact-per-risk for *this* MP.

### P1 (recommended) — "An identifiability benchmark for latent-dynamics models, with a connectome ground truth"
Galvani generates time series from circuits of known structure (HD ring =
1-D ring attractor; mushroom body = sparse high-D code; DTI = region
network). Fit a ladder of SSMs (LDS → rSLDS → one deep SSM) and quantify
recovery of latent dimension, attractor topology, and fixed points as a
function of observation noise, sampling rate, and number of observed
neurons. **Deliverable:** a benchmark suite + a short report ("rSLDS
recovers the ring above SNR X and below subsampling Y; LDS never does").
**Why it lands:** methodologically novel, fully self-contained (Tier A
only), directly exercises the group's tools, and the connectome-as-answer-
key idea is the kind of thing that turns into a workshop paper.

### P2 — "Latent dynamics of the head-direction ring: simulation meets imaging"
Fit the same SSM to Galvani's HD-ring simulation and to real EPG calcium
imaging (Tier B), compare the recovered ring manifolds. **Deliverable:**
side-by-side latent-space comparison + a discussion of where the
mechanistic model and the data agree/diverge. **Why it lands:** it's
neuroscience, not just methods; the HD ring is famous; the comparison is a
clean story. **Risk:** wrangling and aligning real imaging data is real
work and can eat the budget — keep it as the second half, gated on P1-style
machinery working first.

### P3 — "Simulation-based inference of connectome circuit parameters"
Use `sbi` to infer the posterior over Galvani's parameters (gain, sign
rule, time constants) from observed dynamics. **Deliverable:** posteriors +
identifiability analysis ("the bump width pins the gain but leaves τ
unconstrained"). **Why it lands:** maximally Macke-lab-native if that's the
group. **Risk:** requires good summary statistics and many simulator runs;
the inference can be finicky. Strong if the professor is SBI-oriented,
overkill if not.

### P4 (stretch) — "Differentiable connectome-to-dynamics in JAX/Jaxley"
Port a Galvani circuit to a differentiable JAX simulator (rate first;
Jaxley for the biophysical MC view if time allows) and fit parameters by
gradient descent to a target time series. **Deliverable:** a differentiable
Galvani backend + a fitting demo. **Why it lands:** shows you can build the
differentiable-simulator infrastructure the frontier runs on. **Risk:**
highest; biophysical fitting is data-hungry and easy to under-deliver.
Best as an optional final module, or as the explicit "future work" that
motivates a follow-on thesis.

## 6. Recommended path + 8 ECTS scoping

Lead with **P1**, keep **P2** as the second half if time allows, name P3/P4
as future work. Rough budget (~240 h):

- **Weeks 1–2 — dynamics ground truth (data-free).** Fixed-point finder +
  Jacobian/eigenvalue analysis + bifurcation sweep on the HD ring and one
  other circuit. Produces the answer key. *Self-contained deliverable even
  if everything after slips.*
- **Weeks 3–4 — observation model + data generator.** Wrap Galvani output
  in a configurable observation model (Gaussian noise, calcium kernel,
  Poisson spikes, neuron subsampling). This is the knob the benchmark
  sweeps over.
- **Weeks 5–8 — SSM ladder.** LDS (dynamax/own Kalman-EM) → rSLDS (`ssm`)
  → one deep SSM. Wire up recovery metrics (latent-dim estimate, manifold
  topology via persistent homology or simple circular-coordinate check,
  fixed-point overlap).
- **Weeks 9–11 — the benchmark.** Sweep noise × subsampling × N-observed;
  produce the recovery curves. This is the core result.
- **Weeks 12–14 — P2 stretch / writeup.** If the machinery is solid, fit to
  real EPG imaging and compare. Otherwise deepen the benchmark and write up.

Scope discipline: **the ring attractor is the whole project's spine.** One
circuit, done thoroughly, beats five circuits done shallowly. The mushroom
body and DTI circuits are "does the method generalise?" appendices, not
core.

## 7. Credibility — what to read / cite so the pitch lands

- **Latent dynamics methods:** Macke et al. (PLDS); Linderman et al. 2017
  (rSLDS); Pandarinath et al. 2018 (LFADS); Duncker & Sahani reviews of
  latent-variable models for neural data.
- **The ring attractor:** Seelig & Jayaraman 2015; Kim et al. 2017;
  Turner-Evans et al. 2020; **Duan, Dong & Fiete 2025** (connectome →
  ring-attractor dynamics — the paper Galvani's HD-ring scenario already
  reproduces; this is your direct lineage).
- **Connectome-constrained modelling:** Lappalainen et al. 2024 (optic
  lobe), Shiu et al. 2024 (central complex).
- **The group's own tools (if Macke-orbit):** Jaxley (Deistler et al.,
  differentiable biophysics); `sbi` (Tejero-Cantero et al.); the lab's
  simulation-based-inference-for-mechanistic-models line (Gonçalves et al.
  2020).
- **Dynamical-systems framing:** Sussillo & Barak 2013 (fixed-point
  analysis of RNNs) — the template for the §4(1) workup, transplanted from
  trained RNNs onto connectome circuits.

Being able to say "Galvani already reproduces Duan-Dong-Fiete's
ring-attractor result, and I want to use it as ground truth for testing the
latent-dynamics methods your group develops" is a one-sentence pitch that
connects your existing artefact to their research programme.

## 8. Risks and non-goals

- **Don't let it become 'I built a simulator'.** The simulator exists; the
  MP is about *what you do with its time series*. Frame every deliverable
  as a modelling/analysis result, not an engineering one.
- **Don't depend on real imaging data on the critical path.** Tier A is
  self-sufficient; Tier B/C are bonuses. Data access/alignment is the
  classic time-sink that sinks practicals.
- **Don't over-promise Jaxley/biophysical fitting.** It's the flashiest and
  the riskiest; keep it as stretch/future-work unless the professor
  specifically wants it.
- **Don't sprawl across circuits.** One circuit (HD ring), deeply analysed,
  is the project. Others are generalisation checks.
- **Verify the professor's actual method focus before committing** (SLDS vs
  deep SSM vs SBI changes which framing in §3 to lead with). One
  conversation de-risks the whole pitch.

## 9. One-line pitch

*"Galvani turns a connectome into neural time series with known
ground-truth dynamics — I want to use it as an identifiability benchmark
for the latent state-space models your group fits to imaging data, with the
fly head-direction ring attractor as the hero test case."*
