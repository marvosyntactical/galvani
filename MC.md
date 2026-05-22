# Multi-compartment (MC) biophysical view

Live in-browser multi-compartment Hodgkin-Huxley simulation for the neuron
focused in the SNV plus any visible neighbors. Triggered from a button at
the top of the SNV (shortcut `M`). Spinner during compute. Bio sim is a
one-way coupled visualization layer over the outer point-neuron sim — we
don't try to make them self-consistent.

## Coupling: bio sim ⇆ outer sim

One-way. Outer drives bio, bio doesn't feed back.

For every neuron `N` in the bio neighborhood, each incoming synapse needs a
spike train. Two sources:

- **Synapse from `M` inside the bio neighborhood** → use `M`'s AIS spike
  times computed locally in the bio sim.
- **Synapse from `K` outside the neighborhood** → use the outer sim's
  output for `K`.

The "outer sim output" mapping depends on the outer model:

| Outer model | Source                        | Action                          |
| ----------- | ----------------------------- | ------------------------------- |
| HH          | `spike_times[K]`              | use directly                    |
| LIF         | `spike_times[K]`              | use directly                    |
| AdEx        | `spike_times[K]`              | use directly                    |
| Rate        | continuous `rates[t, K]`      | inhomogeneous Poisson sample    |

Single conversion function: `spikeTrainFor(payload, neuronIdx, outerModelId) → SpikeEvent[]`.
Rate is the only branch with real logic.

The locally-computed AIS spike times for cell `M` will generally **not**
agree with the outer sim's spike times for `M`. That's expected. The bio
view is "membrane dynamics given the inputs the circuit is sending each
cell," not a higher-fidelity replacement of the outer sim. A small muted
caveat in the UI surfaces this.

Inner solver stays HH regardless of outer model.

## Stages

### Stage 0 — data plumbing (Python)

- [ ] `Neuron.soma_compartment_index: int | None` — read SWC type-column
      (1 = soma). Fall back to root node or thickest-radius node.
- [ ] `Subgraph.synapse_sites: list[SynapseSite]` — surface per-synapse
      coordinates from neuPrint. Currently aggregated into `W` and thrown
      away.
- [ ] Per-synapse compartment projection: nearest skeleton point on
      post-synaptic neuron for each `(pre, post, post_xyz)` triple.
      Compute at bake time.
- [ ] Per-neuron detail JSON gains:
      - `soma_idx: int`
      - `synapse_sites_to_me: [{pre_neuron_id, post_compartment_idx, nt}]`
- [ ] Schema version bump → `schema_version: 4`. Frontend v3 fallback.
- [ ] Surface spike events: bake `spike_times[neuron] = float[]` next to
      `rates`, for LIF / AdEx / HH scenarios.

What to get right: file sizes. Synapse-site list can be long; encode
compactly (concatenated `int32` arrays per neuron, not nested objects).

### Stage 1 — pure-TS multi-compartment HH solver

- [ ] Single-compartment HH validated against textbook numbers (rheobase,
      repetitive firing at 10/20/50 nA injection).
- [ ] Multi-compartment cable: compartment tree from SWC, axial conductance
      `g_axial = π·r² / (R_a·length)` between neighbors, forward Euler on
      `C·dV/dt = -I_ion - I_axial + I_syn`.
- [ ] AIS designation: compartment 15-30 μm distal of the soma along the
      longest axon branch. 10× Na⁺/K⁺ density there. Fallback if axon is
      shorter: highest-radius compartment past the soma.
- [ ] Synaptic conductance kick: AMPA-like exponential
      (`τ_decay = 5 ms`, `g_peak ∝ synapse_count`), GABA-like for inhibitory
      NT. Sign from the outer parameterizer.
- [ ] AIS spike detection: voltage threshold crossing + minimum interspike
      interval (refractory).
- [ ] Output: `Float32Array` of `v[compartment, frame]` plus AIS spike
      times.

What to get right: numerical stability. Forward Euler at 50 μs for HH is
marginal — may need 20-25 μs (2× compute). Profile before locking.

The whole stage 1 module should compile as a pure functional core: no
DOM, no globals, no React. Web Worker entry point in stage 3.

### Stage 2 — input generation from outer sim

- [ ] `spikeTrainFor(payload, neuronIdx, outerModelId)` resolver.
      - Spiking backends: read pre-baked `spike_times[neuronIdx]`.
      - Rate backend: inhomogeneous Poisson sample from
        `payload.rates[:, neuronIdx]` at fine dt (~0.5 ms), seeded for
        reproducibility.
- [ ] Per-target routing: for focused + visible neighbor neurons, walk
      `synapse_sites_to_me`, look up each presynaptic spike train, deposit
      conductance kicks at the corresponding compartment.

### Stage 3 — Web Worker harness

- [ ] One worker per neuron in the bio neighborhood (cap at 10).
- [ ] Message format:
      `{ morphology, ais_idx, syn_compartments, input_spike_trains, dt, duration }`
      → `{ v_trace: Float32Array, ais_spikes: number[] }`
- [ ] Progress streaming: each worker posts `% complete` every ~100 ms.
- [ ] Cancellation: navigation away or Exit terminates all workers.
- [ ] In-memory cache keyed by
      `(neuron_id, outer_model_id, scenario_id, neighborhood_signature)`.

What to get right: worker startup is 50-200 ms each. Pool workers across
the neighborhood; don't spawn fresh per click.

### Stage 4 — Detail-mode UI

- [ ] **Biophysical view** button at the top of the SNV, above the
      Neighbors selector. States: `idle` / `computing` / `ready` / `error`.
- [ ] `M` keyboard shortcut, gated on form-input focus like `T`/`N`/`S`.
- [ ] Computing state: orange inline spinner + percentage; button text
      "Computing biophysics... 42%". Disabled.
- [ ] Ready state: button becomes a toggle
      `Biophysical view: ON / OFF`. OFF hides the voltage map without
      discarding the cache.
- [ ] Error state: visible reason + retry action.
- [ ] When `showConnected` is on: bio compute covers focused + all visible
      neighbors. Toggling neighbors invalidates the bio cache.
- [ ] One-way-coupling caveat shown as small muted text under the button.

### Stage 5 — Visualization

- [ ] `DetailScene` accepts optional
      `voltageMap: Map<neuronId, Float32Array>` prop.
- [ ] Per-compartment color sampled from `v_trace` at the current frame
      when present, replacing the uniform per-neuron color.
- [ ] Voltage → color: blue (rest) → cyan (subthreshold) → orange (spike) →
      white (peak). Visually distinct from the rate-mode hue ramp.
- [ ] Update via direct-buffer-write inside `useFrame`, same pattern as
      `TubesView`.
- [ ] Soma rendered as a small sphere (orienting landmark, always visible).
- [ ] Camera target snaps to soma when bio view turns on.

### Stage 6 — AIS voltage trace

A 60×280 SVG mini-trace of voltage at the AIS, same style as the existing
rate mini-trace. Replaces the rate trace while bio view is ON.

### Stage 7 — Polish

- [ ] Compartment-hover tooltip: compartment ID, voltage, m/h/n gating.
- [ ] Badges in the model panel: compartment count, AIS index, compute time.
- [ ] Cap neighborhood at top-5 in + top-5 out for v1 release; bump if
      perf allows.
- [ ] Worker code-splits into its own bundle chunk.

## Known limitations to document, not fix in v1

- **AIS placement heuristic.** Hemibrain skeletons don't tag AIS; we pick
  by distance-along-axon from soma.
- **Channel densities.** Default to squid axon values (Hodgkin & Huxley
  1952); fly central-complex neurons aren't tuned. Surface as caveat.
- **Conduction delay** from AIS along the axon arbor to synaptic-output
  compartments is real (0.5-2 ms/mm) but ignored in v1. HD-ring distances
  are sub-millisecond — acceptable for now. Add to BACKLOG.
- **Forward Euler instability** at the targeted dt — may need to drop to
  20-25 μs, with 2× compute cost.
- **Bio↔outer inconsistency** — locally-computed spike times for neighbors
  won't match the outer sim. Documented in the UI caveat.

## Scope estimate

- Stage 0 (Python): ~1 day. Synapse-site projection is the slow part.
- Stage 1 (TS solver): ~2-3 days, mostly validation.
- Stage 2 (input gen): ~0.5 day.
- Stage 3 (workers): ~1 day.
- Stage 4 (UI): ~0.5 day.
- Stage 5 (viz): ~0.5 day.
- Stages 6 & 7: ~0.5 day each.

Total ~7-8 days. Single-neuron interim (no neighborhood) ships at ~4 days.
