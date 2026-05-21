# Galvani BACKLOG

Things explicitly deferred. **Do not pull from here until v1 is done.**
Re-read this list at the start of every work session. If a new idea arrives, write it here, do not start it.

## v1.5

- **FlyWire / CAVE connectome backend.** Second connectome source. Forces clean abstraction.
- **Interventions module.** Ablation, activity patching, weight scaling.
- **NeuroML / Brian2 export polish.** Round-tripped `ModelSpec` to NeuroML.
- **Web demo polish.** Vite + R3F static viewer over pre-computed sim JSON.
- **Cell-type label translation layer** for cross-dataset (hemibrain ↔ FlyWire).
- **Eckstein et al. (2024) NT loading for hemibrain.** The neuPrint graph for `hemibrain:v1.2.1` does not carry the predicted NTs (paper postdates the dataset). v1 worked around it with a hand-coded `HD_RING_NT` lookup in `connectome/hemibrain.py`. Proper fix: download Eckstein's released NT predictions (Cell 2024 supplementary or zenodo), join on `bodyId`, stamp `Neuron.nt` automatically. Eliminates the per-circuit NT lookup and the type-name brittleness.
- **HD-ring bump *position* retention without input.** v1 validation passes amplitude persistence but the bump drifts to a network-preferred attractor when the stim is removed. Fixing this needs tighter weight tuning (per-cell-type bias, or a more careful symmetrization that preserves the continuous rotational symmetry). Logged but not blocking: the canonical Kim 2017 test asserts amplitude persistence, which we have.
- **Per-cell-type tau refinement.** The default 20 ms / 10 ms split is coarser than published values for the central complex. Add a literature-anchored table covering at least the HD-ring and KC types as part of the Eckstein-NT work.

## v2+

- **Biophysical (LIF / HH) models.** Beyond rate dynamics.
- **Jaxley backend** for differentiable simulation. Enables param fitting.
- **Parameter fitting to neural recordings.** Tempting and harder than it looks.
- **General-purpose subgraph editor UI.**
- **Larger circuits (whole CX, mushroom body + KC subnetworks).**

## Out of scope (likely never)

- Authentication / multi-user / cloud deployment.
- Brain simulator competing with NEURON/NEST/Brian2 directly.
- A Neuroglancer replacement.
- Mammalian connectomes (different scale, different tooling).

---

**Rule:** If you find yourself implementing something from this file, stop. Either it belongs in v1 (move it to the plan) or it doesn't (leave it here). No ambient scope creep.
