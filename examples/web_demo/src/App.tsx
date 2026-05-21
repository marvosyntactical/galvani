import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stats } from "@react-three/drei";
import type {
  HyperParams,
  Manifest,
  ManifestDataset,
  ManifestScenario,
  Payload,
} from "./payload";
import { loadManifest, loadPayload } from "./payload";
import { CELL_TYPE_INFO } from "./cellTypes";
import { Infobox } from "./Infobox";
import { RingScene, type HoverInfo, type RenderMode } from "./RingScene";

export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [payloadLoading, setPayloadLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [renderMode, setRenderMode] = useState<RenderMode>("lines");
  const [hover, setHover] = useState<HoverInfo | null>(null);

  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(performance.now());

  useEffect(() => {
    loadManifest()
      .then((m) => {
        setManifest(m);
        const first = m.datasets[0];
        if (!first) {
          setErr("Manifest contains no datasets.");
          return;
        }
        setDatasetId(first.id);
        setScenarioId(first.scenarios[0]?.id ?? null);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const dataset: ManifestDataset | undefined = useMemo(
    () => manifest?.datasets.find((d) => d.id === datasetId),
    [manifest, datasetId],
  );
  const scenario: ManifestScenario | undefined = useMemo(
    () => dataset?.scenarios.find((s) => s.id === scenarioId),
    [dataset, scenarioId],
  );

  // Load the selected scenario's payload, but keep the previously rendered
  // payload on screen until the new one arrives -- so changing dropdowns
  // never blanks the UI.
  useEffect(() => {
    if (!scenario) return;
    setPayloadLoading(true);
    setHover(null);
    let cancelled = false;
    loadPayload(scenario.file)
      .then((p) => {
        if (cancelled) return;
        setPayload(p);
        setFrame(0);
        setPayloadLoading(false);
      })
      .catch((e) => {
        if (!cancelled) {
          setErr(String(e));
          setPayloadLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [scenario]);

  useEffect(() => {
    if (!payload) return;
    const totalFrames = payload.metadata.n_frames;
    const realtimeSecPerFrame = payload.metadata.duration / totalFrames;
    function tick(now: number) {
      const elapsed = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      if (playing) {
        setFrame((f) => {
          const advance = (elapsed * speed) / realtimeSecPerFrame;
          let next = f + advance;
          if (next >= totalFrames) next = next % totalFrames;
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    lastTickRef.current = performance.now();
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [payload, playing, speed]);

  if (err) return <div className="loading">Error: {err}</div>;
  if (!manifest || !dataset) return <div className="loading">Loading manifest…</div>;

  const currentFrame = payload
    ? Math.round(frame) % payload.metadata.n_frames
    : 0;
  const currentT = payload ? payload.times[currentFrame] : 0;

  const typesInPayload = payload
    ? Array.from(new Set(payload.neurons.map((n) => n.cell_type)))
    : [];

  const hoveredNeuron =
    hover && payload ? payload.neurons[hover.neuronIndex] : null;
  const hoveredStim =
    hover && payload?.stim_signal
      ? payload.stim_signal[currentFrame][hover.neuronIndex]
      : null;

  return (
    <div className="app">
      <div className="canvas-host">
        {payload && (
          <Canvas
            camera={{ position: [12, 8, 12], fov: 50, near: 0.1, far: 200 }}
            dpr={[1, 2]}
            gl={{ antialias: true }}
          >
            <color attach="background" args={["#0b0f17"]} />
            <ambientLight intensity={0.55} />
            <pointLight position={[10, 10, 10]} intensity={1.4} />
            <pointLight position={[-10, -5, -10]} intensity={0.4} />
            <OrbitControls makeDefault enableDamping />
            <RingScene
              payload={payload}
              frame={frame}
              renderMode={renderMode}
              onHover={setHover}
              hoveredIndex={hover?.neuronIndex ?? null}
            />
          </Canvas>
        )}
        {payload && <Stats />}
        {payload && (
          <div className="overlay">
            <h2>{scenario?.label}</h2>
            <p className="subtitle">
              {payload.metadata.n_neurons} neurons ·{" "}
              {payload.metadata.duration.toFixed(2)}s · dt ={" "}
              {payload.metadata.dt_sim.toExponential(1)} s
            </p>
          </div>
        )}
        {payloadLoading && (
          <div className="loading-overlay">Loading {scenario?.label}…</div>
        )}
        {!payload && !payloadLoading && (
          <div className="loading-overlay">Pick a scenario from the right →</div>
        )}
        {hover && hoveredNeuron && (
          <Tooltip
            x={hover.screen.x}
            y={hover.screen.y}
            id={hoveredNeuron.id}
            cellType={hoveredNeuron.cell_type}
            hemisphere={hoveredNeuron.hemisphere}
            rate={hover.rate}
            input={hoveredStim}
          />
        )}
      </div>

      <div className="sidebar">
        <div className="header-row">
          <h1>Galvani</h1>
          <span className="badge">v0.1</span>
        </div>

        <div className="selector">
          <label htmlFor="dataset-select">Dataset</label>
          <select
            id="dataset-select"
            value={datasetId ?? ""}
            onChange={(e) => {
              const id = e.target.value;
              setDatasetId(id);
              const ds = manifest.datasets.find((d) => d.id === id);
              setScenarioId(ds?.scenarios[0]?.id ?? null);
            }}
          >
            {manifest.datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </select>
        </div>

        <div className="selector">
          <label htmlFor="scenario-select">Scenario</label>
          <select
            id="scenario-select"
            value={scenarioId ?? ""}
            onChange={(e) => setScenarioId(e.target.value)}
          >
            {dataset.scenarios.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        <div className="selector">
          <label>Render mode</label>
          <div className="render-toggle">
            <button
              className={renderMode === "lines" ? "active" : ""}
              onClick={() => setRenderMode("lines")}
            >
              Lines
            </button>
            <button
              className={renderMode === "tubes" ? "active" : ""}
              onClick={() => setRenderMode("tubes")}
            >
              Tubes
            </button>
          </div>
        </div>

        <Infobox title="What am I looking at?">
          <p>
            A pre-computed rate-model simulation played back on top of the
            EM-traced skeletons of every neuron in the circuit. The Galvani
            pipeline goes: <code>Connectome</code> → <code>Subgraph</code> →{" "}
            <code>Parameterizer</code> → <code>Simulator</code>.
          </p>
          <p>
            <strong>Hue</strong> = cell type. <strong>Brightness</strong> =
            firing rate at the current frame. A magenta tint signals which
            neurons are receiving non-zero <strong>external input</strong> at
            this moment.
          </p>
          <p>
            <strong>Interaction:</strong> mouse-drag to orbit, scroll to zoom,
            hover a neuron for body id / rate / input. The scrubber below is
            simulated time; the speed selector controls playback rate.
          </p>
          <p>
            <strong>Render modes:</strong> "Lines" draws each skeleton as 2px
            camera-facing quads (fast). "Tubes" draws each segment as a 3D
            cylinder sized by the SWC radius (slower, closer to Neuroglancer
            aesthetic).
          </p>
        </Infobox>

        {payload && (
          <Infobox title={`Dataset · ${dataset.label}`}>
            <p>{dataset.summary}</p>
            <p>{dataset.biology}</p>
            <div className="kv">
              <span className="k">dataset_version</span>
              <span className="v">{payload.metadata.dataset_version}</span>
              <span className="k">neurons</span>
              <span className="v">{payload.metadata.n_neurons}</span>
              <span className="k">scenario</span>
              <span className="v">{payload.metadata.scenario_label}</span>
            </div>
            <p style={{ marginTop: 10 }}>{payload.metadata.description}</p>
            <strong>Cell types in this view</strong>
            <ul className="legend-list">
              {typesInPayload.map((t) => {
                const info = CELL_TYPE_INFO[t];
                const hex = info ? `#${info.baseColor.getHexString()}` : "#888";
                return (
                  <li key={t}>
                    <span className="swatch" style={{ background: hex }} />
                    <span>
                      <strong>{info?.shortLabel ?? t}</strong>
                      {info ? ` — ${info.blurb}` : ""}
                    </span>
                  </li>
                );
              })}
            </ul>
            <p style={{ marginTop: 10 }}>
              <strong>External input (magenta tint):</strong>{" "}
              {dataset.id === "hd_ring"
                ? "represents heading-related drive from upstream visual / landmark / vestibular cues that the fly is currently sensing. In the velocity-integration scenarios this stands in for the angular velocity signal carried by the L or R PEN cells."
                : "represents projection-neuron drive from the antennal lobe — i.e. the odor the fly is currently smelling. We drive ~30% of KCs to model a broad odor response."}
            </p>
          </Infobox>
        )}

        {payload && (
          <Infobox title="Hyperparameters & assumptions">
            <HyperParamsView hp={payload.metadata.hyperparams} />
            <p style={{ marginTop: 8 }}>
              <strong>Do these dynamics make sense?</strong>
            </p>
            <p>
              {dataset.id === "hd_ring"
                ? "The HD-ring operating point (gain=0.012, symmetrize=True, tanh) was selected to pass all five qualitative tests from Kim et al. 2017 / Duan-Dong-Fiete 2025: bump existence, persistence (amplitude), tracking, velocity integration, and finite-gain regime. Quantitative position retention without input is a known limitation (see BACKLOG.md). All five validation tests are pinned in tests/test_validation_hd_ring.py."
                : "The mushroom-body operating point (gain=0.08, relu, symmetrize=False) is set so that pulsing 30% of KCs settles into ~2-7% sparse activation via APL feedback. Counterfactually ablating APL's outgoing synapses (the 'APL ablated' scenario here) eliminates the sparseness — the canonical APL functional signature from Honegger et al. 2011 / Lin et al. 2014."}
            </p>
            <p>
              <strong>Things we are not modeling:</strong> biophysics (no Na+/K+
              channels, no voltage, no spikes — just rate variables);
              per-cell-type biases beyond defaults; modulatory NTs
              (octopamine / serotonin / dopamine treated as zero in v1);
              realistic input drive (PNs for the MB are approximated by
              external current on the KCs).
            </p>
            <p>
              <strong>Where this could go wrong:</strong> log1p compresses
              synapse-count dynamic range but is a guess at the synaptic-weight
              transfer function; per-cell-type tau defaults are coarse; the
              symmetrization step is opt-in and biases the HD-ring matrix
              toward an idealised ring. See <code>DESIGN_DECISIONS.md</code>
              for the full record.
            </p>
          </Infobox>
        )}
      </div>

      <div className="controls">
        <button
          className="play-btn"
          onClick={() => setPlaying((p) => !p)}
          disabled={!payload}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <div className="scrubber">
          <input
            type="range"
            min={0}
            max={payload ? payload.metadata.n_frames - 1 : 1}
            step={0.001}
            value={frame}
            disabled={!payload}
            onChange={(e) => {
              setPlaying(false);
              setFrame(parseFloat(e.target.value));
            }}
          />
          <span className="time-label">
            {payload
              ? `t = ${currentT.toFixed(3)}s (${currentFrame + 1}/${payload.metadata.n_frames})`
              : "—"}
          </span>
        </div>
        <select
          className="speed-select"
          value={speed}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
          disabled={!payload}
        >
          <option value={0.25}>0.25×</option>
          <option value={0.5}>0.5×</option>
          <option value={1.0}>1×</option>
          <option value={2.0}>2×</option>
          <option value={4.0}>4×</option>
        </select>
      </div>
    </div>
  );
}

function HyperParamsView({ hp }: { hp: HyperParams }) {
  const { stimulus, ...rest } = hp;
  return (
    <>
      <div className="kv">
        {Object.entries(rest).map(([k, v]) => (
          <RowKV key={k} k={k} v={v} />
        ))}
      </div>
      {stimulus && (
        <>
          <p style={{ marginTop: 8 }}>
            <strong>Stimulus</strong>
          </p>
          <div className="kv">
            {Object.entries(stimulus).map(([k, v]) => (
              <RowKV key={k} k={k} v={v} />
            ))}
          </div>
        </>
      )}
    </>
  );
}

function RowKV({ k, v }: { k: string; v: unknown }) {
  let formatted: string;
  if (typeof v === "number") formatted = Number.isInteger(v) ? `${v}` : v.toString();
  else if (typeof v === "boolean") formatted = v ? "true" : "false";
  else if (v === null || v === undefined) formatted = "—";
  else formatted = String(v);
  return (
    <>
      <span className="k">{k}</span>
      <span className="v">{formatted}</span>
    </>
  );
}

function Tooltip({
  x,
  y,
  id,
  cellType,
  hemisphere,
  rate,
  input,
}: {
  x: number;
  y: number;
  id: number;
  cellType: string;
  hemisphere: string | null;
  rate: number;
  input: number | null;
}) {
  const info = CELL_TYPE_INFO[cellType];
  const style = {
    left: Math.min(x + 14, window.innerWidth - 240),
    top: Math.min(y + 14, window.innerHeight - 110),
  };
  return (
    <div className="tooltip" style={style}>
      <div className="row">
        <span className="k">body_id</span>
        <span>{id}</span>
      </div>
      <div className="row">
        <span className="k">type</span>
        <span>{info?.shortLabel ?? cellType}</span>
      </div>
      <div className="row">
        <span className="k">hemi</span>
        <span>{hemisphere ?? "—"}</span>
      </div>
      <div className="row">
        <span className="k">rate</span>
        <span>{rate.toFixed(3)}</span>
      </div>
      {input !== null && (
        <div className="row">
          <span className="k">input</span>
          <span>{input.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
}
