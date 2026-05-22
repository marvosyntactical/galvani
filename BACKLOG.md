# Galvani BACKLOG

Things explicitly deferred. **Do not pull from here until v1 is done.**
Re-read this list at the start of every work session. If a new idea arrives, write it here, do not start it.

## v1.5

- **Real H01 cell data via cloud-volume.** Currently `connectome/h01.py` ships an H01-*inspired* canonical microcircuit (cell types + connectivity densities from Shapson-Coe 2024). The real data is hosted at `gs://h01-release/data/20210601/c3` and accessible via `cloud-volume`, but mesh / skeleton downloads are slow (multi-resolution Draco-compressed). Cleanly extracting ~50 proofread cells + their synapse predictions into a `Subgraph` is doable but needs ~1-2 days of focused integration work: fetch meshes, decimate to SWC skeletons via `kimimaro`, parse synapse-prediction annotation layers from `synapses_400nm/`. Same outline could load MICrONS mouse data via `caveclient`.


- **FlyWire / CAVE connectome backend.** Second connectome source. Forces clean abstraction.
- **Interventions module.** Ablation, activity patching, weight scaling.
- **NeuroML / Brian2 export polish.** Round-tripped `ModelSpec` to NeuroML.
- **Web demo polish.** Vite + R3F static viewer over pre-computed sim JSON.
- **Cell-type label translation layer** for cross-dataset (hemibrain ↔ FlyWire).
- **Eckstein et al. (2024) NT loading for hemibrain.** The neuPrint graph for `hemibrain:v1.2.1` does not carry the predicted NTs (paper postdates the dataset). v1 worked around it with a hand-coded `HD_RING_NT` lookup in `connectome/hemibrain.py`. Proper fix: download Eckstein's released NT predictions (Cell 2024 supplementary or zenodo), join on `bodyId`, stamp `Neuron.nt` automatically. Eliminates the per-circuit NT lookup and the type-name brittleness.
- **HD-ring bump *position* retention without input.** v1 validation passes amplitude persistence but the bump drifts to a network-preferred attractor when the stim is removed. Fixing this needs tighter weight tuning (per-cell-type bias, or a more careful symmetrization that preserves the continuous rotational symmetry). Logged but not blocking: the canonical Kim 2017 test asserts amplitude persistence, which we have.
- **Per-cell-type tau refinement.** The default 20 ms / 10 ms split is coarser than published values for the central complex. Add a literature-anchored table covering at least the HD-ring and KC types as part of the Eckstein-NT work.

## v2+

- **Biophysical (LIF / HH) models.** Beyond rate dynamics. The rate model in v1 (`tau * dr/dt = -r + phi(W*r + I + b)`) ignores spike timing, refractory periods, and ion-channel kinetics. To do biophysics we'd add a `model.lif` backend (leaky integrate-and-fire: each neuron has a membrane voltage, fires when V > threshold, has a reset and refractory period) and optionally a `model.hh` backend (Hodgkin-Huxley with gating variables for Na+/K+ channels). The `Subgraph` and `Parameterizer` are unchanged; only the simulator differs. Brian2 (already in the `[brian]` extra) is the natural backend for both. Cost: ~1 week for LIF + per-cell-type tuning; ~2 weeks for HH + per-channel parameter tables. v1 rate model is the right baseline; biophysics adds detail without changing the qualitative story for the HD ring or MB.
- **Live hyperparameter editing in the web demo.** Currently scenarios are pre-baked. Two routes: (a) ship a few pre-baked gain/stim variants and let the user pick from a slider that snaps to those values (~half day, no new infrastructure); (b) port the rate simulator to WASM / pyodide / pure-JS and let the user adjust hyperparams and re-simulate in-browser (~1 week, ~5 MB extra JS bundle, ~1-3 s recompute time for HD ring). (b) is more impressive; (a) is more practical for v1.5.
- **In-browser data upload.** A "Load custom payload" file picker that lets the user drag-drop a JSON in our schema. Useful for: testing local bake variants without redeploying, sharing arbitrary scenarios via download links, plus the obvious "show me MY connectome" use case. Lower-hanging fruit than (b) above; ~half day.
- **DTI-tractography backend.** Build a `connectome.dti` loader that consumes a tractography connectivity matrix (e.g. HCP / Glasser-360-parcellation streamline counts) and a `parameterize.region_defaults` with Wilson-Cowan or Jansen-Rit defaults at region level. The `Simulator` is unchanged. This is the killer demo for "infrastructure across resolution scales" — same pipeline drives both an EM connectome (130 neurons) and a DTI connectome (360 regions). ~1 week for synthetic / published example dataset; ~2 weeks for a clean HCP-data loader.
- **Jaxley backend** for differentiable simulation. Enables param fitting.
- **Parameter fitting to neural recordings.** Tempting and harder than it looks.
- **General-purpose subgraph editor UI.**
- **Larger circuits (whole CX, mushroom body + KC subnetworks).**

## Out of scope (likely never)

- Authentication / multi-user / cloud deployment.
- Brain simulator competing with NEURON/NEST/Brian2 directly.
- A Neuroglancer replacement.
- Mammalian connectomes (different scale, different tooling). *(Note: DTI-tractography support above sits in a gray zone — it does mammalian connectivity but at region scale, not cell scale.)*

---

**Rule:** If you find yourself implementing something from this file, stop. Either it belongs in v1 (move it to the plan) or it doesn't (leave it here). No ambient scope creep.
