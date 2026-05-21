# galvani web demo — HD ring

Interactive playback of a pre-computed HD-ring simulation. Each neuron is
rendered as a downsampled skeleton (Three.js `LineSegments`); colors update
per frame from the activity trace baked into `public/hd_ring.json`.

## Quick start

```bash
# 1) From the repo root: build the payload (requires NEUPRINT_TOKEN the
#    first time, so skeletons can be cached). Re-runs are network-free.
uv run python scripts/build_demo_payload.py

# 2) Install and run
cd examples/web_demo
npm install
npm run dev
```

Vite prints a URL — open http://127.0.0.1:5173 in your browser.

## What you should see

A 3D ring of EPG / PEN / Delta7 skeletons in the central complex. A bump
of activity (yellow) rotates around the ring driven by the simulated
stimulus (`omega = 1 rad/s` by default). Use the scrubber for manual
control, the speed dropdown to slow it down, and the mouse to orbit /
zoom / pan.

## Data flow

`hd_ring.json` is produced by `galvani.viz.payload.build_hd_ring_payload`
and contains:

- Per-neuron downsampled skeleton (stride 20 → ~500 nodes / EPG).
- All coordinates translated and uniformly scaled into a ~10-unit cube
  (the bbox is in the payload metadata if you need raw nm back).
- A `(n_frames, n_neurons)` activity matrix sampled to 120 frames.

Schema is versioned (`schema_version: 1`). If you bump the schema in
Python, the frontend will refuse to load it (loud failure beats silent
drift).

## Iterating

Re-run `build_demo_payload.py` after changing the scenario, parameterizer
options, or stride. Vite hot-reloads the frontend; the payload load
happens on page refresh, so reload the tab after rebaking.
