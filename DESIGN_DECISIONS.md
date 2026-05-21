# Design Decisions

A running log of decisions made during Galvani's development. Each entry should be short and reference the alternative considered. New decisions append to the bottom.

---

## 2026-05-21 — Project codename and package name

**Decision:** Package name is `galvani` (matches repo dir).

**Considered:** `circuitlab` (the name used in the original implementation plan). The PyPI namespace for `galvani` is taken by an unrelated package (battery file reader).

**Implication:** PyPI publishing in Phase 8 will need a different distribution name (e.g. `galvani-circuit`), but `import galvani` stays clean. This is a Phase 8 problem, not a Phase 0 one.

---

## 2026-05-21 — Python toolchain

**Decision:** `uv` for env management. Target Python 3.11+.

**Considered:** plain `venv` + pip; pyenv. `uv` is faster, manages the Python toolchain itself, and the plan recommends it.

**Implication:** all dev commands go through `uv run` / `uv sync`. CI uses `astral-sh/setup-uv`.

---

## 2026-05-21 — Testing strategy

**Decision:** Hybrid TDD, not strict.

- **Unit tests** for deterministic pieces (dataclass shape, weight construction from a tiny `Subgraph`, NT→sign mapping, `ModelSpec` JSON round-trip).
- **Golden-snapshot tests** for the parameterizer: a small fixed `Subgraph` checked in as a fixture; assert produced `ModelSpec` doesn't change unexpectedly.
- **Qualitative whole-pipeline assertions** for the HD-ring sim (`bump_width < pi/2`, `peak_drift < 0.1 rad`, …) — these are slow but cheap to keep.
- **Phase-checkpoint plots** are required deliverables. Every phase's checkpoint produces a figure.

**Considered:** strict TDD (impractical for sim correctness, which is qualitative). No tests until the end (regression risk; parameterizer defaults need pinning).

---

## 2026-05-21 — Viz cadence

**Decision:** In-notebook matplotlib plots from Phase 2 onward. Animated Plotly time-scrubber by end of Phase 4. React/three.js web demo stays at Phase 6.

**Considered:** standing up a web demo earlier; following the plan literally (viz only at Phase 6).

**Implication:** the project always has a visual feedback artifact. Every phase checkpoint is a figure or animation. The web demo, when built, plays back pre-computed sim data — not live in-browser simulation.

---

## 2026-05-21 — neuPrint auth in CI

**Decision:** `NEUPRINT_TOKEN` env var gates live tests. CI runs only `-m "not live"`. Fixtures (committed parquet files) cover the unit tests of the loader.

**Considered:** stash a token as a GitHub secret; this is risky and noisy. Skipping connectome unit tests entirely; loses coverage.

**Implication:** when the user gets a token (Phase 2), add a `Makefile` target `make refresh-fixtures` that re-pulls one HD-ring query from neuPrint and writes it under `tests/fixtures/`.

---

## 2026-05-21 — Optional dependency groups

**Decision:** Core install pulls only numpy/pandas/pyarrow/platformdirs. Extras: `[neuprint]`, `[viz]`, `[brian]`, `[all]`.

**Considered:** putting everything in core (heavy, slow installs; brian2 needs a C++ compiler). Splitting into multiple packages (premature).

**Implication:** dev uses `uv sync --all-extras --group dev`. Users opting in to Brian2 know what they're getting into.

---

## 2026-05-21 — Simulator default backend

**Decision:** NumPy (explicit Euler, fixed `dt=0.5 ms`) is the default. Brian2 is opt-in.

**Considered:** Brian2 as the default (matches plan's "two backends" idea, but Brian2 has 30s compilation overhead — bad for interactive use).

**Implication:** Brian2 backend is implemented only if needed for export/extensibility. NumPy gets the most love because it's what people will actually run.

---

## 2026-05-21 — HD-ring angular layout: spectral, not label-based

**Decision:** Ring coordinates come from the top two non-mean eigenvectors of `0.5*(W + W.T)` — `arctan2(v_sin, v_cos)`. `circuits.hd_ring.spectral_angles` is the implementation; `load_hd_ring(..., layout='spectral')` is the default.

**Considered:** Parsing the `_L<k>` / `_R<k>` PB-glomerulus suffix from hemibrain `instance` labels (the obvious thing). Naive `(k-1)/8 * 2π` collapsing of both hemispheres landed the bump ~60° off-target; the canonical L<k> ↔ R<9−k> EB-wedge flip didn't help either. The connectome's emergent ring structure doesn't line up with any label convention I tried, and the spectral embedding does the right thing automatically.

**Implication:** Layout is dataset-agnostic: any backend whose subgraph has ring topology gets correct coordinates without bespoke parsing. The label-based path (`layout='instance'`) is kept in the API for reference but is not the default.

---

## 2026-05-21 — Default simulator activation: `tanh`, not `relu`

**Decision:** The HD-ring validation uses `tanh`. `relu` stays available in `model.rate` as an option but is not the recipe in the notebook or the validation tests.

**Considered:** `relu` (the plan's implicit default — most rate-model literature). With log1p-scaled weights and spectral radius ~88 on the hemibrain HD-ring subgraph, `relu` is unbounded under recurrent excitation: activity explodes by ~10 orders of magnitude within 0.5 simulated seconds at any usable gain. `tanh` saturates at 1.0 and produces a clean bump regime.

**Implication:** Documented as the working choice in [notebooks/01_hd_ring_hemibrain.ipynb](notebooks/01_hd_ring_hemibrain.ipynb). If a future circuit calls for `relu` (sparse feedforward networks, where bounded activation distorts response), the user passes `activation=relu` — the simulator doesn't make the choice for them. The parameterizer is activation-agnostic.

---

## 2026-05-21 — HD-ring operating point and the role of symmetrization

**Decision:** v1 HD-ring validation runs at `symmetrize=True, global_gain=0.012`. Both are opt-ins from `ParameterizerOptions`; the parameterizer's defaults stay `symmetrize=False, global_gain=1.0` so they don't lie about what the raw connectome implies.

**Considered:** Leaving symmetrization off (the connectome value). On hemibrain v1.2.1, the asymmetric HD-ring matrix has a row-sum range of −30 to +92 and no clean bistable regime — bump amplitude decays within 200 ms of stim removal at every gain that doesn't saturate. Symmetrization restores a finite persistence window roughly `gain ∈ [0.011, 0.020]`. The plan flagged this risk explicitly; the empirical result confirmed it.

**Implication:** When applying the pipeline to a second circuit (Phase 7), `symmetrize=True` should be tried only when the circuit has a known underlying symmetry the connectome breaks (HD ring, lamina). For pure feedforward circuits (mushroom body) it makes no sense.

---

## 2026-05-21 — Cache the ids-only neuron query path (Phase 7 finding)

**Decision:** `HemibrainConnectome.query(ids=...)` now caches results to `neurons.by_id.<sha1>.parquet`, mirroring the adjacency cache scheme. Before this, the ids-only path went straight to neuPrint with no caching.

**Considered:** Leaving it un-cached (the original assumption: ids-only is rare and one-off). The mushroom body falsified this: APL has no `type` field in hemibrain:v1.2.1, so it can *only* be fetched by id. Without caching, fixture-only tests couldn't load it (401 because the cache was bypassed and the live token branch demanded one).

**Implication:** Any circuit whose key cells lack `type` labels — singletons, fragments, unusually-named neurons — now loads from fixtures like everything else. This is the one library change Phase 7 forced; the parameterizer, simulator, and ModelSpec were unchanged.

---

## 2026-05-21 — Per-circuit NT/cell-type patching for unlabeled singletons

**Decision:** Circuit modules (e.g. `circuits.mushroom_body._patch_apl`) own the responsibility of fixing up unlabeled or NT-less neurons after the backend returns them. The connectome backend stays "faithful to the dataset"; circuit modules know which singletons need stamping.

**Considered:** Bolting an `nt_by_id: dict[int, str]` option onto `HemibrainConnectome` so the loader could patch NT during query. Rejected because: (a) it conflates "what the backend returns" with "what a circuit-specific application needs"; (b) it doesn't help with the missing `cell_type` field, which is what makes the `nt_by_type` mechanism miss APL in the first place.

**Implication:** Future circuits with this issue (anything driven by a globally-named singleton: DPM, LAL inhibitors, etc.) get their own small fixup function next to the loader. The library convention — NT comes from the backend or a per-type lookup — remains the simple primary path.
