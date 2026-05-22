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
import { UploadButton } from "./UploadButton";
import { canResim, resimulateWithGain } from "./resim";
import { Dropdown } from "./Dropdown";
import { DetailScene, type CompanionNeuron } from "./DetailScene";
import { NeuronModelInfo } from "./NeuronModelInfo";
import { HelpPopover } from "./HelpPopover";
import { loadNeuronDetail, type NeuronDetail } from "./detailLoader";

export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [payloadLoading, setPayloadLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // CRITICAL: `frame` lives in a ref, not React state.
  // The RAF loop runs at 60 fps. If we used setState here, the App would
  // re-render every 16 ms, which re-applies the `value` attribute on the
  // <select> elements every frame. Native browser dropdowns close when the
  // select gets touched -- this is why the dropdowns appeared to "flash open
  // and disappear" before. The ref-based approach keeps the App tree stable
  // during playback; only the 3D scene (which uses useFrame, not React state)
  // reads the live frame value.
  const frameRef = useRef(0);
  // Throttled mirror of frameRef for the time-label readout. Updates at 10 Hz.
  const [displayFrame, setDisplayFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [renderMode, setRenderMode] = useState<RenderMode>("lines");
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const [gainOverride, setGainOverride] = useState<number | null>(null);
  const [resimBusy, setResimBusy] = useState(false);
  const [customPayload, setCustomPayload] = useState<{ p: Payload; name: string } | null>(
    null,
  );
  // Detail mode: when set, the canvas shows only one neuron at full SWC
  // morphology, and the sidebar swaps to a per-neuron explainer.
  const [selectedNeuronIndex, setSelectedNeuronIndex] = useState<number | null>(null);
  const [detail, setDetail] = useState<NeuronDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  // Connected-neurons state.
  const [showConnected, setShowConnected] = useState(false);
  const [connectedK, setConnectedK] = useState(5);
  const [companions, setCompanions] = useState<CompanionNeuron[]>([]);
  const [companionsLoading, setCompanionsLoading] = useState(false);
  // Mobile: collapse the sidebar by default and let a hamburger toggle it.
  const [sidebarOpen, setSidebarOpen] = useState(true);

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
    // If the user uploaded a custom JSON, show that instead.
    if (customPayload) {
      setPayload(customPayload.p);
      setPayloadLoading(false);
      frameRef.current = 0;
      setDisplayFrame(0);
      setGainOverride(null);
      return;
    }
    if (!scenario) return;
    setPayloadLoading(true);
    setHover(null);
    setGainOverride(null);
    let cancelled = false;
    loadPayload(scenario.file)
      .then((p) => {
        if (cancelled) return;
        setPayload(p);
        frameRef.current = 0;
        setDisplayFrame(0);
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
  }, [scenario, customPayload]);

  // Live re-simulation when the user drags the gain slider.
  useEffect(() => {
    if (gainOverride === null || !payload || !canResim(payload)) return;
    setResimBusy(true);
    // Defer to next animation frame so the slider stays responsive.
    const handle = requestAnimationFrame(() => {
      try {
        const result = resimulateWithGain(payload, gainOverride);
        setPayload({ ...payload, rates: result.rates, times: result.times });
      } finally {
        setResimBusy(false);
      }
    });
    return () => cancelAnimationFrame(handle);
  }, [gainOverride]); // eslint-disable-line react-hooks/exhaustive-deps

  // Playback RAF loop. Mutates frameRef.current directly -- no React state
  // update, no re-render. This is the key to keeping the dropdowns
  // interactive while playback runs.
  useEffect(() => {
    if (!payload) return;
    const totalFrames = payload.metadata.n_frames;
    const realtimeSecPerFrame = payload.metadata.duration / totalFrames;
    function tick(now: number) {
      const elapsed = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      if (playing) {
        const advance = (elapsed * speed) / realtimeSecPerFrame;
        let next = frameRef.current + advance;
        if (!Number.isFinite(next)) next = 0;
        if (next >= totalFrames) next = next % totalFrames;
        frameRef.current = next;
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    lastTickRef.current = performance.now();
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [payload, playing, speed]);

  // Throttled UI mirror of frameRef.current (10 Hz). Updates the scrubber's
  // displayed time without triggering App re-renders on every animation
  // frame.
  useEffect(() => {
    if (!payload) return;
    const id = setInterval(() => {
      setDisplayFrame((prev) => {
        const next = frameRef.current;
        // Avoid setState when the rounded display value hasn't changed.
        return Math.round(next) === Math.round(prev) ? prev : next;
      });
    }, 100);
    return () => clearInterval(id);
  }, [payload]);

  // Reset connected-neurons state whenever the user enters detail mode or
  // changes the focused neuron.
  useEffect(() => {
    setCompanions([]);
    setShowConnected(false);
  }, [selectedNeuronIndex]);

  // Fetch top-k incoming/outgoing companion details when the toggle is on.
  useEffect(() => {
    if (
      !showConnected ||
      !payload ||
      selectedNeuronIndex === null ||
      !(payload as Payload & { model?: { weights: number[][] } }).model
    ) {
      setCompanions([]);
      return;
    }
    const model = (payload as Payload & {
      model: { weights: number[][]; global_gain: number };
    }).model;
    const W = model.weights;
    const N = W.length;
    const incoming: Array<{ idx: number; w: number }> = [];
    const outgoing: Array<{ idx: number; w: number }> = [];
    for (let j = 0; j < N; j++) {
      if (j === selectedNeuronIndex) continue;
      const wIn = Math.abs(W[selectedNeuronIndex][j]);
      const wOut = Math.abs(W[j][selectedNeuronIndex]);
      if (wIn > 0) incoming.push({ idx: j, w: wIn });
      if (wOut > 0) outgoing.push({ idx: j, w: wOut });
    }
    incoming.sort((a, b) => b.w - a.w);
    outgoing.sort((a, b) => b.w - a.w);
    const topIn = incoming.slice(0, connectedK).map((e) => e.idx);
    const topOut = outgoing.slice(0, connectedK).map((e) => e.idx);

    // Stimulus-input set: any neuron with non-zero stim across time.
    const stimSet = new Set<number>();
    if (payload.stim_signal) {
      for (let i = 0; i < N; i++) {
        for (const row of payload.stim_signal) {
          if (Math.abs(row[i]) > 1e-4) {
            stimSet.add(i);
            break;
          }
        }
      }
    }

    const baseUrl = import.meta.env.BASE_URL;
    setCompanionsLoading(true);
    let cancelled = false;
    Promise.all([
      ...topIn.map(async (idx) => {
        const nrn = payload.neurons[idx];
        const d = await loadNeuronDetail(
          payload.metadata.dataset_id,
          nrn.id,
          baseUrl,
        );
        return {
          detail: d,
          neuronIndex: idx,
          role: "incoming" as const,
          isStimInput: stimSet.has(idx),
        };
      }),
      ...topOut.map(async (idx) => {
        const nrn = payload.neurons[idx];
        const d = await loadNeuronDetail(
          payload.metadata.dataset_id,
          nrn.id,
          baseUrl,
        );
        return {
          detail: d,
          neuronIndex: idx,
          role: "outgoing" as const,
          isStimInput: stimSet.has(idx),
        };
      }),
    ])
      .then((arr) => {
        if (!cancelled) {
          setCompanions(arr);
          setCompanionsLoading(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          // eslint-disable-next-line no-console
          console.error("Failed to load companions", e);
          setCompanions([]);
          setCompanionsLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [showConnected, connectedK, selectedNeuronIndex, payload]);

  // Detail mode: fetch the per-neuron skeleton file when the user clicks a
  // neuron. We deliberately do not preload these -- only fetch on demand.
  useEffect(() => {
    if (selectedNeuronIndex === null || !payload) {
      setDetail(null);
      return;
    }
    const neuron = payload.neurons[selectedNeuronIndex];
    const baseUrl = import.meta.env.BASE_URL;
    setDetailLoading(true);
    let cancelled = false;
    loadNeuronDetail(payload.metadata.dataset_id, neuron.id, baseUrl)
      .then((d) => {
        if (!cancelled) {
          setDetail(d);
          setDetailLoading(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setErr(`Detail load failed for ${neuron.cell_type} #${neuron.id}: ${e}`);
          setDetailLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedNeuronIndex, payload]);

  if (err) return <div className="loading">Error: {err}</div>;
  if (!manifest || !dataset) return <div className="loading">Loading manifest…</div>;

  // UI-facing frame (already throttled to 10 Hz via displayFrame).
  const safeFrame = Number.isFinite(displayFrame) ? displayFrame : 0;
  const currentFrame = payload
    ? Math.max(
        0,
        Math.min(payload.metadata.n_frames - 1, Math.round(safeFrame)),
      )
    : 0;
  const currentT = payload ? (payload.times[currentFrame] ?? 0) : 0;

  const typesInPayload = payload
    ? Array.from(new Set(payload.neurons.map((n) => n.cell_type)))
    : [];

  const hoveredNeuron =
    hover && payload ? payload.neurons[hover.neuronIndex] : null;
  const hoveredStim =
    hover && payload?.stim_signal
      ? payload.stim_signal[currentFrame][hover.neuronIndex]
      : null;

  const inDetailMode =
    selectedNeuronIndex !== null && payload && (detail || detailLoading);

  return (
    <div className={`app ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
      <button
        className="mobile-toggle"
        aria-label={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
        onClick={() => setSidebarOpen((o) => !o)}
      >
        {sidebarOpen ? "✕" : "☰"}
      </button>
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
            {selectedNeuronIndex !== null && detail ? (
              <DetailScene
                detail={detail}
                payload={payload}
                neuronIndex={selectedNeuronIndex}
                frameRef={frameRef}
                companions={companions}
              />
            ) : (
              <RingScene
                payload={payload}
                frameRef={frameRef}
                renderMode={renderMode}
                onHover={setHover}
                hoveredIndex={hover?.neuronIndex ?? null}
                // DTI parcels don't have real per-neuron morphology, so
                // there's nothing to drill into. Only enable click for
                // single-cell datasets.
                onSelect={
                  payload.metadata.dataset_id.startsWith("dti")
                    ? undefined
                    : (i) => setSelectedNeuronIndex(i)
                }
              />
            )}
          </Canvas>
        )}
        {payload && !inDetailMode && <Stats />}
        {payload && !inDetailMode && (
          <div className="overlay">
            <h2>{scenario?.label}</h2>
            <p className="subtitle">
              {payload.metadata.n_neurons} neurons ·{" "}
              {payload.metadata.duration.toFixed(2)}s · dt ={" "}
              {payload.metadata.dt_sim.toExponential(1)} s
            </p>
            {!payload.metadata.dataset_id.startsWith("dti") && (
              <p className="subtitle" style={{ marginTop: 6 }}>
                💡 Click a neuron to see its morphology + model up close
              </p>
            )}
          </div>
        )}
        {detailLoading && (
          <div className="loading-overlay">Loading neuron morphology…</div>
        )}
        {companionsLoading && !detailLoading && (
          <div
            className="loading-overlay"
            style={{ background: "rgba(11, 15, 23, 0.4)" }}
          >
            Loading {connectedK * 2} connected neurons…
          </div>
        )}
        {payloadLoading && (
          <div className="loading-overlay">Loading {scenario?.label}…</div>
        )}
        {!payload && !payloadLoading && (
          <div className="loading-overlay">Pick a scenario from the right →</div>
        )}
        {hover && hoveredNeuron && !inDetailMode && (
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

      <div className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        {inDetailMode && payload && selectedNeuronIndex !== null && detail ? (
          <NeuronModelInfo
            payload={payload}
            neuronIndex={selectedNeuronIndex}
            frameDisplay={displayFrame}
            showConnected={showConnected}
            connectedK={connectedK}
            onToggleConnected={setShowConnected}
            onChangeK={setConnectedK}
            onClose={() => setSelectedNeuronIndex(null)}
          />
        ) : (
        <>
        <div className="header-row">
          <h1>Galvani</h1>
          <span className="badge">v0.1</span>
        </div>

        <div className="selector">
          <label htmlFor="dataset-select">Dataset</label>
          <Dropdown
            id="dataset-select"
            value={datasetId ?? ""}
            options={manifest.datasets.map((d) => ({ value: d.id, label: d.label }))}
            onChange={(id) => {
              setDatasetId(id);
              const ds = manifest.datasets.find((d) => d.id === id);
              setScenarioId(ds?.scenarios[0]?.id ?? null);
            }}
          />
        </div>

        <div className="selector">
          <label htmlFor="scenario-select">Scenario</label>
          <Dropdown
            id="scenario-select"
            value={scenarioId ?? ""}
            options={dataset.scenarios.map((s) => ({ value: s.id, label: s.label }))}
            onChange={(id) => setScenarioId(id)}
          />
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

        {payload && canResim(payload) && (
          <div className="selector">
            <label>
              Live gain ={" "}
              {(gainOverride ?? payload.metadata.hyperparams.global_gain).toFixed(4)}
              {resimBusy && <span className="badge" style={{ marginLeft: 8 }}>resimming…</span>}
            </label>
            <input
              type="range"
              min={0.001}
              max={0.05}
              step={0.0005}
              value={gainOverride ?? payload.metadata.hyperparams.global_gain}
              onChange={(e) => setGainOverride(parseFloat(e.target.value))}
            />
            {gainOverride !== null && (
              <button
                className="reset-btn"
                onClick={() => {
                  setGainOverride(null);
                  // Reload original baked payload.
                  if (scenario && !customPayload) {
                    loadPayload(scenario.file).then(setPayload);
                  }
                }}
              >
                Reset to baked gain
              </button>
            )}
          </div>
        )}

        <div className="selector">
          <label>
            Custom data{" "}
            <HelpPopover title="Upload format">
              <p>
                The file picker accepts a JSON in Galvani's payload schema v2.
                Easiest way to make one: clone the repo, run{" "}
                <code>uv run python scripts/build_demo_payload.py</code> with a
                neuPrint token, and grab any file under{" "}
                <code>examples/web_demo/public/</code>.
              </p>
              <p>
                <strong>What's inside a payload:</strong> per-neuron 3D
                skeletons (positions + radii), per-frame activation rates,
                optional stimulus signal, and weight matrix metadata.
                The weight matrix lets the in-browser live-gain slider work.
              </p>
              <p>
                <strong>Sizing:</strong> 100 neurons × 100 frames is roughly
                100 KB. 1 000 neurons × 100 frames is ~5 MB. The browser
                fetches the whole file at scenario load, so target ≤10 MB
                for snappy UX. Heavier scenarios can run but feel sluggish.
              </p>
              <p>
                <strong>Other connectome sources to try:</strong>{" "}
                <code>neuprint-python</code> (Janelia hemibrain / male-CNS /
                MANC), <code>fafbseg-py</code> (FlyWire), or HCP/AAL DTI
                matrices via <code>neurolib</code>. Each needs its own
                loader; v1 ships hemibrain + DTI.
              </p>
            </HelpPopover>
          </label>
          <UploadButton
            onLoad={(p, name) => setCustomPayload({ p, name })}
            onError={(msg) => setErr(msg)}
          />
          {customPayload && (
            <div className="custom-badge">
              <span>{customPayload.name}</span>
              <button onClick={() => setCustomPayload(null)}>×</button>
            </div>
          )}
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
            </div>
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
          </Infobox>
        )}

        {payload && (
          <Infobox title={`Scenario · ${payload.metadata.scenario_label}`}>
            <p>{payload.metadata.description}</p>
            <div className="kv">
              <span className="k">duration</span>
              <span className="v">
                {payload.metadata.duration.toFixed(2)} s
              </span>
              <span className="k">frames</span>
              <span className="v">{payload.metadata.n_frames}</span>
            </div>
            <p style={{ marginTop: 10 }}>
              <strong>How to read the viz:</strong>
            </p>
            <ul className="legend-list" style={{ marginTop: 4 }}>
              <li>
                <span className="swatch" style={{ background: "#58a6ff" }} />
                Neuron <strong>color</strong> = cell type hue, brightness =
                firing rate at the current frame.
              </li>
              <li>
                <span className="swatch" style={{ background: "#ff5cb0" }} />
                <strong>Magenta tint</strong> on a neuron = it is receiving
                non-zero external input at this frame.
              </li>
              <li>
                <span
                  className="swatch"
                  style={{ background: "transparent", border: "1px dashed #58a6ff" }}
                />
                Click any neuron to enter <strong>detail mode</strong> and see
                its full SWC morphology + single-neuron model.
              </li>
            </ul>
            <p style={{ marginTop: 10 }}>
              <strong>About the external input:</strong>{" "}
              {dataset.id === "hd_ring"
                ? "represents heading-related drive from upstream visual / landmark / vestibular cues. In the velocity-integration scenarios it stands in for angular-velocity signal carried by the L or R PEN cells."
                : dataset.id === "mushroom_body"
                  ? "represents projection-neuron drive from the antennal lobe — i.e. the odor the fly is currently smelling. We drive ~30% of KCs to model a broad odor response."
                  : "represents externally-applied drive on the chosen subset of regions (e.g. visual cortex). Activity propagates through the streamline-count matrix to the rest of the brain."}
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
        </>
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
            value={displayFrame}
            disabled={!payload}
            onChange={(e) => {
              setPlaying(false);
              const next = parseFloat(e.target.value);
              frameRef.current = next;
              setDisplayFrame(next);
            }}
          />
          <span className="time-label">
            {payload
              ? `t = ${currentT.toFixed(3)}s (${currentFrame + 1}/${payload.metadata.n_frames})`
              : "—"}
          </span>
        </div>
        <Dropdown<number>
          value={speed}
          options={[
            { value: 0.05, label: "0.05× (very slow)" },
            { value: 0.1, label: "0.1×" },
            { value: 0.25, label: "0.25×" },
            { value: 0.5, label: "0.5×" },
            { value: 1.0, label: "1× (real time)" },
            { value: 2.0, label: "2×" },
            { value: 4.0, label: "4×" },
          ]}
          onChange={setSpeed}
          disabled={!payload}
        />
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
