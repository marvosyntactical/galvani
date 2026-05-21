# galvani

**Connectome → subgraph → executable rate-model.** A Python library that takes any subgraph of a published connectome and returns a runnable rate-model simulation with biologically reasonable default parameters.

> Status: pre-alpha. Phase 0 scaffold. See [`implementation_plan.md`](../implementation_plan.md) for the full plan and [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) for the running log.

---

## What it's for

Every recent paper in connectome-constrained modeling (Duan/Dong/Fiete 2025, Lappalainen et al. 2024, Shiu et al. 2024, Pospisil et al. 2024) writes its own bespoke pipeline from connectome data to executable model. None share code. None expose their parameterization heuristics cleanly. Galvani closes that gap.

**Validation target (v1):** recover the *Drosophila* HD ring attractor from the hemibrain connectome, matching Duan/Dong/Fiete 2025 qualitatively.

## Intended usage (Phase 1 sketch — not yet implemented)

```python
from galvani import HemibrainConnectome, default_parameterizer, simulate
from galvani.stimuli import ring_stimulus

conn = HemibrainConnectome()                              # cached locally
neurons = conn.query(type=["EPG", "PEN_a", "PEN_b", "Delta7"])
subgraph = conn.subgraph(neurons)                         # weights, NTs, positions

model = default_parameterizer(subgraph)                   # ModelSpec dataclass

results = simulate(model, duration="2*second", stimulus=ring_stimulus(angle=0.5))
results.plot_raster()
results.angle_over_time().plot()
```

The shape of this snippet is the v1 API contract. If you change it, update the snippet and the design doc together.

## Development

```bash
# install uv if you haven't: https://docs.astral.sh/uv/
uv sync --all-extras --group dev

uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Repo layout

```
src/galvani/
  connectome/     # connectome interface + backends (hemibrain via neuPrint first)
  parameterize/   # synapse counts -> weights, NT -> sign, type -> tau; the actual research
  model/          # ModelSpec dataclass + simulation backends (NumPy default; Brian2 opt-in)
  circuits/       # convenience constructors (hd_ring, ...)
tests/            # unit tests + golden-snapshot fixtures
notebooks/        # validation notebooks (01_hd_ring_hemibrain, etc.)
```

## License

MIT.
