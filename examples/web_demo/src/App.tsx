import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stats } from "@react-three/drei";
import type {
  HyperParams,
  Manifest,
  ManifestDataset,
  ManifestScenario,
  Payload,
} from "./payload";
import { loadManifest, loadPayload, MODEL_LABELS, type ModelId } from "./payload";
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
import { runBioSession, type BioSession } from "./mc/runMcSession";

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
  const [modelId, setModelId] = useState<ModelId>("rate");
  /** Stochasticity level. Currently 0 only (baked); higher values add
   *  in-browser Gaussian jitter on top of the baked rates for visual
   *  effect. */
  const [stochasticity, setStochasticity] = useState<number>(0);
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
  // How many *unique* neurons the current companion fetch is loading.
  // Symmetrised circuits (HD ring) have top-K-in == top-K-out, so the
  // unique count is K rather than 2·K.
  const [companionsCount, setCompanionsCount] = useState(0);
  // Mobile: collapse the sidebar by default and let a hamburger toggle it.
  // We initialize from `matchMedia` so the panel starts off-canvas on phone
  // viewports without flashing visible on the first paint.
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    if (typeof window === "undefined") return true;
    return !window.matchMedia("(max-width: 720px)").matches;
  });
  // Single-open accordion: which infobox is currently expanded.
  const [openCard, setOpenCard] = useState<string | null>("overview");
  // Whether the SNV drawer is currently visible. Decoupled from
  // `selectedNeuronIndex` so the user can close the drawer (e.g. to see the
  // full canvas) without leaving detail mode. Auto-opens on desktop when a
  // neuron is selected; stays closed on mobile because the drawer would
  // otherwise cover the entire viewport.
  const [drawerOpen, setDrawerOpen] = useState(false);
  // Snapshot of the neuron-detail index that lags `selectedNeuronIndex` by
  // one frame during the close animation. Without it the drawer's content
  // would blank out the instant the user clicks the back arrow, making the
  // slide-out look broken. Cleared 260 ms after selection becomes null.
  const [drawerIndex, setDrawerIndex] = useState<number | null>(null);
  // Most-recently-opened neuron, kept across close so the `S` shortcut can
  // toggle the drawer back open without re-clicking. Survives close.
  const lastSelectedRef = useRef<number | null>(null);
  // Hidden <input> inside UploadButton; we hold the ref here so the `J`
  // keyboard shortcut can click() it from anywhere.
  const uploadInputRef = useRef<HTMLInputElement>(null);
  // FPS/memory stats panel — hidden by default, toggled with F. Hosting it
  // in our own div lets us scale it 2× via CSS without monkey-patching
  // stats.js's inline styles.
  const [statsVisible, setStatsVisible] = useState(false);
  const statsHostRef = useRef<HTMLDivElement>(null);
  // Biophysical (multi-compartment HH) session state. See MC.md.
  // `idle`     — no bio sim has been requested for this focus + neighborhood.
  // `computing`— workers are running; `bioProgress` is in [0, 1].
  // `ready`    — `bioSession` holds the result; `bioVisible` controls whether
  //              the voltage map is rendered. The user can toggle visibility
  //              off without discarding the cached result.
  // `error`    — `bioError` holds the message; user can retry.
  type BioState = "idle" | "computing" | "ready" | "error";
  const [bioState, setBioState] = useState<BioState>("idle");
  const [bioVisible, setBioVisible] = useState(false);
  const [bioProgress, setBioProgress] = useState(0);
  const [bioError, setBioError] = useState<string | null>(null);
  const [bioSession, setBioSession] = useState<BioSession | null>(null);
  // AbortController for the in-flight worker batch. We tear down on
  // detail-mode exit, focus change, or explicit user cancel.
  const bioAbortRef = useRef<AbortController | null>(null);
  // Cache keyed by `(focused, neighborSig, scenario, modelId)` so the user
  // can flip the toggle off → on without re-running.
  const bioCacheRef = useRef<Map<string, BioSession>>(new Map());
  // Latest `requestBioSim` closure, kept in a ref so the global keyboard
  // handler (defined in a useEffect with a stable dep list) can call it
  // without forcing the effect to re-attach on every render.
  const requestBioSimRef = useRef<() => void>(() => {});
  // Auto-hide bottom controls bar: show when mouse is near the bottom of
  // the canvas area or while the user is actively dragging the scrubber.
  const [controlsVisible, setControlsVisible] = useState(false);
  const controlsHoldRef = useRef(false);

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
    // Look up the file matching the selected model. Fall back to legacy
    // `file` field if the manifest is v2 (pre-Model-dropdown).
    const file =
      scenario.models?.[modelId] ??
      scenario.models?.["rate"] ??
      scenario.file;
    if (!file) {
      setErr(`Scenario ${scenario.id} has no file for model ${modelId}`);
      return;
    }
    setPayloadLoading(true);
    setHover(null);
    setGainOverride(null);
    let cancelled = false;
    loadPayload(file)
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
  }, [scenario, customPayload, modelId]);

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
    // While bio view is on, the scrubber is repurposed for bio time and
    // the user wants a slower default cadence to actually see spikes
    // unfold. Halving the effective playback rate makes the dropdown's
    // 1× behave like the previous 0.5×, 2× like the previous 1×, etc.
    const BIO_PLAYBACK_FACTOR = 0.5;
    const effectiveSpeed = bioVisible ? speed * BIO_PLAYBACK_FACTOR : speed;
    function tick(now: number) {
      const elapsed = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      if (playing) {
        const advance = (elapsed * effectiveSpeed) / realtimeSecPerFrame;
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
  }, [payload, playing, speed, bioVisible]);

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

  // Global keyboard shortcuts. Skipped while focus is in a form field.
  //   Space — play/pause
  //   T     — toggle Linear/Tubular render mode
  //   Esc   — close the SNV drawer (if open)
  //   N     — toggle Show neighbors (only meaningful while SNV is open)
  //   S     — toggle the last-selected neuron's drawer (open or close)
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      const tag = t?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || t?.isContentEditable) return;
      switch (e.code) {
        case "Space":
          e.preventDefault();
          setPlaying((p) => !p);
          return;
        case "KeyT":
          e.preventDefault();
          setRenderMode((m) => (m === "lines" ? "tubes" : "lines"));
          return;
        case "Escape":
          if (selectedNeuronIndex !== null) {
            e.preventDefault();
            setSelectedNeuronIndex(null);
          }
          return;
        case "KeyN":
          if (selectedNeuronIndex !== null) {
            e.preventDefault();
            setShowConnected((s) => !s);
          }
          return;
        case "KeyS":
          e.preventDefault();
          if (selectedNeuronIndex !== null) {
            setSelectedNeuronIndex(null);
          } else if (lastSelectedRef.current !== null) {
            setSelectedNeuronIndex(lastSelectedRef.current);
          }
          return;
        case "KeyJ":
          e.preventDefault();
          uploadInputRef.current?.click();
          return;
        case "KeyF":
          e.preventDefault();
          setStatsVisible((s) => !s);
          return;
        case "KeyM":
          if (selectedNeuronIndex !== null) {
            e.preventDefault();
            requestBioSimRef.current();
          }
          return;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedNeuronIndex]);

  // Mirror selection into the persistent ref so the `S` shortcut can re-open
  // the last-viewed neuron after Esc.
  useEffect(() => {
    if (selectedNeuronIndex !== null) {
      lastSelectedRef.current = selectedNeuronIndex;
    }
  }, [selectedNeuronIndex]);

  // Auto-hide controls bar: visible when mouse is within ~120 px of the
  // canvas bottom, or while the user is actively interacting with the bar.
  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (controlsHoldRef.current) return;
      const nearBottom = window.innerHeight - e.clientY < 120;
      setControlsVisible(nearBottom);
    }
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  // Reset connected-neurons state whenever the user enters detail mode or
  // changes the focused neuron.
  useEffect(() => {
    setCompanions([]);
    setShowConnected(false);
  }, [selectedNeuronIndex]);

  // Mirror `selectedNeuronIndex` into `drawerIndex` so the drawer's content
  // stays painted through the slide-out transition.
  useEffect(() => {
    if (selectedNeuronIndex !== null) {
      setDrawerIndex(selectedNeuronIndex);
      return;
    }
    const handle = window.setTimeout(() => setDrawerIndex(null), 260);
    return () => window.clearTimeout(handle);
  }, [selectedNeuronIndex]);

  // When the user enters or leaves detail mode, decide whether the drawer
  // should be visible by default. On desktop the drawer fits alongside the
  // canvas, so it auto-opens. On mobile a 360-px drawer covers the entire
  // viewport, so we leave it closed and let the LHS overlay button open it.
  useEffect(() => {
    if (selectedNeuronIndex === null) {
      setDrawerOpen(false);
      return;
    }
    const isMobile = window.matchMedia("(max-width: 720px)").matches;
    setDrawerOpen(!isMobile);
  }, [selectedNeuronIndex]);

  // Tear down any running bio session (and clear the result) when the user
  // leaves detail mode or changes which neuron is focused. Cache survives
  // — toggling back to a previously-computed neuron reuses it.
  useEffect(() => {
    if (selectedNeuronIndex === null) {
      bioAbortRef.current?.abort();
      bioAbortRef.current = null;
      setBioState("idle");
      setBioVisible(false);
      setBioSession(null);
      setBioError(null);
      setBioProgress(0);
    }
  }, [selectedNeuronIndex]);

  // Cache key for a given biophysical request. The neighborhood signature
  // changes when the user toggles `showConnected` or shifts `connectedK`,
  // because those change which neurons get a worker. The "anchor" — i.e.
  // the outer-time at which the bio window starts — is captured at
  // compute time, so it's not part of the auto-invalidation key (the
  // cache map below uses a key that includes it).
  const bioCacheKey = useMemo(() => {
    if (selectedNeuronIndex === null || !scenario) return null;
    const neighbors = showConnected ? `n=${connectedK}` : "n=0";
    return `${scenario.id}|${modelId}|f=${selectedNeuronIndex}|${neighbors}`;
  }, [selectedNeuronIndex, scenario, modelId, showConnected, connectedK]);

  /** Length of the bio simulation window in biological ms. Short enough
   *  that ms-scale spike events are visible when stretched across the
   *  outer scrubber. */
  const BIO_DURATION_MS = 200;

  // Invalidate when the request key changes (different neuron / neighbors
  // / model / scenario). Cancel any pending run; reset visibility but
  // keep the cache so previously-computed sessions can be revived.
  useEffect(() => {
    bioAbortRef.current?.abort();
    bioAbortRef.current = null;
    setBioVisible(false);
    setBioSession(null);
    setBioState("idle");
    setBioProgress(0);
    setBioError(null);
  }, [bioCacheKey]);

  // Kick off (or revive from cache) the biophysical computation. Called
  // from the `M` shortcut or the SNV button. Idempotent: re-pressing
  // while running is a no-op; pressing while `ready` flips visibility
  // without re-computing.
  const requestBioSim = async () => {
    if (selectedNeuronIndex === null || !payload || !bioCacheKey) return;
    // State machine — see `BioState` comment up top.
    if (bioState === "computing") return;
    if (bioState === "ready") {
      // Cheap toggle: hide / show the voltage map without recomputing.
      setBioVisible((v) => !v);
      return;
    }
    // Anchor the bio window at the user's current scrubber position.
    // Include it in the cache key so re-computing at a different outer
    // time produces a new entry; revisiting the same anchor reuses.
    const anchorOuterMs =
      (displayFrame / Math.max(1, payload.metadata.n_frames - 1)) *
      payload.metadata.duration *
      1000;
    const anchoredKey = `${bioCacheKey}|anchor=${Math.round(anchorOuterMs)}`;
    const cached = bioCacheRef.current.get(anchoredKey);
    if (cached) {
      setBioSession(cached);
      setBioVisible(true);
      setBioState("ready");
      frameRef.current = 0;
      setDisplayFrame(0);
      // Auto-play the bio sim from the start so the user sees spikes
      // unfold immediately rather than having to find Play.
      setPlaying(true);
      return;
    }
    // Build the target list: focused neuron + (top-k incoming/outgoing
    // when neighbors are visible). We dedupe and cap at 11 so we never
    // spawn more than a sensible number of workers.
    const targets: number[] = [selectedNeuronIndex];
    if (showConnected && payload.model) {
      const W = payload.model.weights;
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
      const k = Math.min(5, connectedK);
      const seen = new Set<number>([selectedNeuronIndex]);
      for (const e of incoming.slice(0, k)) {
        if (!seen.has(e.idx)) { seen.add(e.idx); targets.push(e.idx); }
      }
      for (const e of outgoing.slice(0, k)) {
        if (!seen.has(e.idx)) { seen.add(e.idx); targets.push(e.idx); }
      }
    }

    const abort = new AbortController();
    bioAbortRef.current = abort;
    setBioState("computing");
    setBioProgress(0);
    setBioError(null);
    const baseUrl = import.meta.env.BASE_URL;
    try {
      const session = await runBioSession({
        payload,
        outerModelId: modelId,
        targetIndices: targets,
        loadDetail: (i) => loadNeuronDetail(
          payload.metadata.dataset_id,
          payload.neurons[i].id,
          baseUrl,
        ),
        anchorOuterMs,
        bioDurationMs: BIO_DURATION_MS,
        signal: abort.signal,
        onProgress: (frac) => setBioProgress(frac),
      });
      if (abort.signal.aborted) return;
      bioCacheRef.current.set(anchoredKey, session);
      setBioSession(session);
      setBioVisible(true);
      setBioState("ready");
      // The scrubber is repurposed as the bio-time slider while bio is
      // on; reset it to 0 so the user starts at the beginning of the
      // computed bio window rather than wherever the outer scrubber was
      // pointing (which is on a totally different timescale now).
      frameRef.current = 0;
      setDisplayFrame(0);
      // Auto-play the bio sim from the start so the user sees spikes
      // unfold immediately rather than having to find Play.
      setPlaying(true);
    } catch (err) {
      if ((err as DOMException)?.name === "AbortError") return;
      setBioError(err instanceof Error ? err.message : String(err));
      setBioState("error");
    }
  };
  // Stash the closure for the keyboard handler.
  requestBioSimRef.current = requestBioSim;

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

    // For symmetrised circuits (HD ring) incoming and outgoing top-K are
    // the same neurons. Dedupe so we don't fetch the same skeleton twice
    // and the "Loading N companions" overlay reports the real count.
    const outSet = new Set(topOut);
    const inSet = new Set(topIn);
    const seen = new Set<number>();
    const uniqueTargets: Array<{ idx: number; role: "incoming" | "outgoing" }> = [];
    for (const idx of topIn) {
      if (seen.has(idx)) continue;
      seen.add(idx);
      uniqueTargets.push({ idx, role: outSet.has(idx) ? "outgoing" : "incoming" });
    }
    for (const idx of topOut) {
      if (seen.has(idx)) continue;
      seen.add(idx);
      uniqueTargets.push({ idx, role: inSet.has(idx) ? "incoming" : "outgoing" });
    }

    const baseUrl = import.meta.env.BASE_URL;
    setCompanionsLoading(true);
    setCompanionsCount(uniqueTargets.length);
    let cancelled = false;
    Promise.all(
      uniqueTargets.map(async ({ idx, role }) => {
        const nrn = payload.neurons[idx];
        const d = await loadNeuronDetail(
          payload.metadata.dataset_id,
          nrn.id,
          baseUrl,
        );
        return { detail: d, neuronIndex: idx, role };
      }),
    )
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
  if (!manifest || !dataset) {
    return (
      <div className="loading-fullscreen">
        <div className="spinner" />
      </div>
    );
  }

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

  // When bio view is on, the scrubber is repurposed: scrubber fraction
  // [0, 1] maps to bio time [0, bioDurationMs]. The outer scenario isn't
  // visible in detail mode anyway, so this hijack is invisible to the
  // user — they see a single time slider that controls the bio playback.
  const bioTimeMs =
    bioVisible && bioSession && payload
      ? (currentFrame / Math.max(1, payload.metadata.n_frames - 1)) *
        bioSession.bioDurationMs
      : null;

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
            // DTI brain looks best from a sagittal-like view (looking from
            // the brain's right side). The HD ring / MB look best from the
            // default upper-front angle.
            key={`canvas-${payload.metadata.dataset_id}`}
            camera={
              payload.metadata.dataset_id.startsWith("dti")
                ? { position: [30, 0, 0], fov: 35, near: 0.1, far: 500 }
                : { position: [12, 8, 12], fov: 50, near: 0.1, far: 200 }
            }
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
                bioSession={bioSession}
                bioVisible={bioVisible}
              />
            ) : (
              <RingScene
                payload={payload}
                frameRef={frameRef}
                renderMode={renderMode}
                onHover={setHover}
                hoveredIndex={hover?.neuronIndex ?? null}
                stochasticity={stochasticity}
                onSelect={
                  payload.metadata.dataset_id.startsWith("dti")
                    ? undefined
                    : (i) => setSelectedNeuronIndex(i)
                }
              />
            )}
          </Canvas>
        )}
        <div
          ref={statsHostRef}
          className="stats-host"
          style={{ display: statsVisible ? "block" : "none" }}
        />
        {payload && !inDetailMode && statsVisible && (
          <Stats parent={statsHostRef as RefObject<HTMLElement>} />
        )}
        {detailLoading && (
          <div className="loading-overlay">Loading neuron morphology…</div>
        )}
        {companionsLoading && !detailLoading && (
          <div
            className="loading-overlay"
            style={{ background: "rgba(11, 15, 23, 0.4)" }}
          >
            Loading {companionsCount} connected neuron{companionsCount === 1 ? "" : "s"}…
          </div>
        )}
        {payloadLoading && (
          <div className="loading-overlay">
            <div className="spinner" />
          </div>
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
        {selectedNeuronIndex !== null && (
          <div
            className={`detail-overlay-controls ${
              drawerOpen ? "drawer-open" : ""
            }`}
          >
            {!drawerOpen && (
              <button
                className="detail-overlay-btn detail-overlay-info"
                onClick={() => setDrawerOpen(true)}
                aria-label="Open neuron detail panel"
                title="Open neuron detail panel"
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="16" x2="12" y2="11" />
                  <line x1="12" y1="8" x2="12.01" y2="8" />
                </svg>
              </button>
            )}
            <button
              className="detail-overlay-btn detail-overlay-exit"
              onClick={() => setSelectedNeuronIndex(null)}
              aria-label="Exit single-neuron view"
              title="Exit single-neuron view"
            >
              ✕
            </button>
          </div>
        )}
      </div>

      <aside
        className={`detail-drawer ${drawerOpen ? "open" : ""}`}
        aria-hidden={!drawerOpen}
      >
        {payload && drawerIndex !== null && (
          <NeuronModelInfo
            payload={payload}
            neuronIndex={drawerIndex}
            modelId={modelId}
            frameDisplay={displayFrame}
            showConnected={showConnected}
            connectedK={connectedK}
            onToggleConnected={setShowConnected}
            onChangeK={setConnectedK}
            onClose={() => {
              // On mobile the drawer covers the canvas, so the triangle
              // means "back to canvas (stay in detail mode)". On desktop
              // closing the drawer alone is pointless — there's no LHS
              // X-button there — so the triangle exits detail mode.
              if (window.matchMedia("(max-width: 720px)").matches) {
                setDrawerOpen(false);
              } else {
                setSelectedNeuronIndex(null);
              }
            }}
            bioState={bioState}
            bioVisible={bioVisible}
            bioProgress={bioProgress}
            bioError={bioError}
            bioTotalCompartments={bioSession?.totalCompartments ?? 0}
            bioResult={bioSession?.results.get(drawerIndex) ?? null}
            bioTimeMs={bioTimeMs}
            bioAnchorOuterMs={bioSession?.anchorOuterMs ?? null}
            onTriggerBio={requestBioSim}
            onSeekBioFraction={(fraction) => {
              if (!payload) return;
              const n = payload.metadata.n_frames;
              const target = fraction * Math.max(1, n - 1);
              frameRef.current = target;
              setDisplayFrame(target);
              setPlaying(false);
            }}
          />
        )}
      </aside>

      <div className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <>
        <div className="header-row">
          <img
            className="logo"
            src={`${import.meta.env.BASE_URL}icon.png`}
            alt="Galvani logo"
          />
          <h1>
            GALVANI<small className="version">v0.2</small>
          </h1>
        </div>

        <div className="register-stack">
        <Infobox id="overview" title="Overview" openId={openCard} onToggle={setOpenCard}>
          <p>
            You're watching simulated neural activity on real
            neurons as they react to an artificial stimulus. Each neuron brightens and
            dims as its firing rate changes over time, while its color indicates its cell type.
          </p>
          <p>
            The initial simulation is of the logic of a fruit fly's
            head-direction system, but in the <strong>Menu</strong>{" "}
            you can pick different{" "} <strong>circuits</strong>{" "}
             (mushroom body, human cortex,
            whole-brain MRI tractography), <strong>stimuli</strong>{" "}
            (movement, an odor presentation, etc.), and{" "}
            <strong>models</strong> of how each neuron computes (from a
            simple firing-rate model up to detailed biophysics with real
            ion channels). 
          </p>
          <p>
            Also check out the <strong>Controls</strong> below.
          </p>
        </Infobox>

        <Infobox id="menu" title="Menu" openId={openCard} onToggle={setOpenCard}>
          <div className="controls-stack">
          <div className="selector">
            <label htmlFor="dataset-select">Circuit</label>
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
            <label htmlFor="scenario-select">Stimulus</label>
            <Dropdown
              id="scenario-select"
              value={scenarioId ?? ""}
              options={dataset.scenarios.map((s) => ({ value: s.id, label: s.label }))}
              onChange={(id) => setScenarioId(id)}
            />
          </div>

          <div className="selector">
            <label>Model</label>
            <div className="btn-toggle">
              {(["rate", "lif", "adex", "hh"] as ModelId[]).map((m) => {
                const available = scenario?.models?.[m] !== undefined;
                return (
                  <button
                    key={m}
                    className={modelId === m ? "active" : ""}
                    disabled={!available}
                    onClick={() => {
                      setModelId(m);
                      setSpeed(
                        m === "hh" ? 0.1 : m === "adex" || m === "lif" ? 0.25 : 1.0,
                      );
                    }}
                  >
                    {MODEL_LABELS[m]}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="selector">
            <label>Stochasticity</label>
            <div className="btn-toggle">
              {[
                { value: 0, label: "none" },
                { value: 0.1, label: "low" },
                { value: 0.3, label: "med" },
                { value: 0.6, label: "high" },
              ].map((opt) => (
                <button
                  key={opt.value}
                  className={stochasticity === opt.value ? "active" : ""}
                  onClick={() => setStochasticity(opt.value)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div className="selector">
            <label>Render mode</label>
            <div className="btn-toggle">
              <button
                className={renderMode === "lines" ? "active" : ""}
                onClick={() => setRenderMode("lines")}
              >
                Linear
              </button>
              <button
                className={renderMode === "tubes" ? "active" : ""}
                onClick={() => setRenderMode("tubes")}
              >
                Tubular
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
                    const file = scenario?.models?.[modelId] ?? scenario?.models?.["rate"] ?? scenario?.file;
                    if (file && !customPayload) {
                      loadPayload(file).then(setPayload);
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
              inputRef={uploadInputRef}
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
          </div>
        </Infobox>

        <Infobox id="model" title="Model" openId={openCard} onToggle={setOpenCard}>
          <ModelDeepDive modelId={modelId} />
          <p style={{ marginTop: 12, color: "var(--muted)", fontSize: 12 }}>
            Switch the active model from the <strong>Menu</strong> register
            above. Same circuit, same parameterizer, different equations per
            neuron.
          </p>
        </Infobox>

        {payload && (
          <Infobox id="circuit" title="Circuit" openId={openCard} onToggle={setOpenCard}>
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
          <Infobox id="stimulus" title="Stimulus" openId={openCard} onToggle={setOpenCard}>
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
          <Infobox id="details" title="Details" openId={openCard} onToggle={setOpenCard}>
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

        <Infobox id="controls" title="Controls" openId={openCard} onToggle={setOpenCard}>
          <p>
            Drag to orbit, scroll to zoom, hover
            a neuron for info, click one to drill in and see its full
            morphology and the model running on it. The{" "}
            <em>magenta tint</em> marks neurons currently receiving
            external input. The slider below is simulated time; the
            speed dropdown next to it controls playback rate.
          </p>
          <dl className="shortcuts-list">
            <div className="shortcut-row">
              <kbd>Space</kbd>
              <span>Play / pause</span>
            </div>
            <div className="shortcut-row">
              <kbd>T</kbd>
              <span>Toggle Linear / Tubular render</span>
            </div>
            <div className="shortcut-row">
              <kbd>S</kbd>
              <span>Toggle last-viewed neuron's detail view</span>
            </div>
            <div className="shortcut-row">
              <kbd>N</kbd>
              <span>(In detail view) Toggle Show neighbors</span>
            </div>
            <div className="shortcut-row">
              <kbd>Esc</kbd>
              <span>Close detail view</span>
            </div>
            <div className="shortcut-row">
              <kbd>J</kbd>
              <span>Upload a custom payload JSON</span>
            </div>
            <div className="shortcut-row">
              <kbd>F</kbd>
              <span>Toggle the FPS / memory stats overlay</span>
            </div>
            <div className="shortcut-row">
              <kbd>M</kbd>
              <span>
                (In detail view) Compute multi-compartment biophysics, or
                toggle the voltage map when already computed
              </span>
            </div>
          </dl>
        </Infobox>
        </div>
        <div className="sidebar-footer">
          <a
            className="sidebar-footer-link sidebar-footer-icon"
            href="https://marvosyntactical.github.io"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Marvin Koss — personal site"
            title="Marvin Koss — personal site"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="2" y1="12" x2="22" y2="12" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          </a>
          <a
            className="sidebar-footer-link sidebar-footer-coffee"
            href="https://www.buymeacoffee.com/marvinkoss"
            target="_blank"
            rel="noopener noreferrer"
            title="Tip the maintainer"
          >
            Buy me a coffee!
          </a>
          <a
            className="sidebar-footer-link github-link"
            href="https://github.com/marvosyntactical/galvani"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View source on GitHub"
            title="View source on GitHub"
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path
                fillRule="evenodd"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"
              />
            </svg>
          </a>
        </div>
        </>
      </div>

      <div
        className={`controls ${controlsVisible ? "visible" : "hidden"} ${
          drawerOpen ? "snv-open" : ""
        }`}
        onMouseEnter={() => {
          controlsHoldRef.current = true;
          setControlsVisible(true);
        }}
        onMouseLeave={() => {
          controlsHoldRef.current = false;
        }}
      >
        <button
          className="play-btn"
          onClick={() => setPlaying((p) => !p)}
          disabled={!payload}
          aria-label={playing ? "Pause" : "Play"}
          title={playing ? "Pause" : "Play"}
        >
          {playing ? (
            <span className="pause-glyph" aria-hidden />
          ) : (
            <span className="play-glyph" aria-hidden />
          )}
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
          openDirection="up"
          options={[
            { value: 0.05, label: "0.05×" },
            { value: 0.1, label: "0.1×" },
            { value: 0.25, label: "0.25×" },
            { value: 0.5, label: "0.5×" },
            { value: 1.0, label: "1×" },
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

function ModelDeepDive({ modelId }: { modelId: ModelId }) {
  switch (modelId) {
    case "rate":
      return (
        <>
          <p>
            <strong>Rate</strong>, the currently selected model, is the
            fastest of the four backends. Each neuron's state is a single
            scalar firing rate <code>r(t)</code> — no spikes, no voltage, no
            refractory period. The equation is Wilson-Cowan-style population
            dynamics:
          </p>
          <pre className="model-eq">
            {"τ · dr/dt = -r + φ(W · r + I + b)"}
          </pre>
          <p>
            Per-cell-type time constants <code>τ</code> are 10-20 ms; the
            nonlinearity <code>φ</code> is <code>tanh</code> for HD-ring
            scenarios and <code>relu</code> for the mushroom body and the
            cortical microcircuit. <code>W</code> is the parameterized
            connectome weight matrix (synapse counts → log1p → sign by NT →
            global gain); <code>I</code> is the per-frame stimulus drive.
          </p>
          <p>
            This is the model Duan, Dong & Fiete (2025) use for the fly HD
            ring and the workhorse of most Wilson-Cowan-style population
            modeling. It hides spike timing, refractoriness and bursting but
            captures attractor structure, gain, and which populations are
            co-active. Reach for it when you care about <em>what</em> fires,
            less about <em>when</em>.
          </p>
          <p>
            <strong>Stochasticity</strong> adds zero-mean Gaussian jitter to
            each neuron's displayed rate per frame — a coarse stand-in for
            synaptic noise.
          </p>
        </>
      );
    case "lif":
      return (
        <>
          <p>
            <strong>LIF</strong> (leaky integrate-and-fire), the currently
            selected model, replaces the rate variable with a membrane
            voltage <code>v(t)</code> that integrates synaptic drive and
            fires when it crosses threshold:
          </p>
          <pre className="model-eq">
            {"τ · dv/dt = -(v - v_rest) + R · (W · s + I + b)\n" +
              "if v ≥ v_thr: emit spike, v := v_reset (refractory 2 ms)"}
          </pre>
          <p>
            <code>s</code> is a per-neuron exponential synaptic-conductance
            trace kicked up by each presynaptic spike, decaying with{" "}
            <code>τ_syn ≈ 5 ms</code>. The displayed activity is a sliding
            spike-density estimate rather than the raw voltage.
          </p>
          <p>
            LIF is the textbook discrete-spike model (Gerstner & Kistler
            2002, Stein 1965). It captures threshold behavior and the
            refractory dead time that pure rate models miss, but flattens
            spike-frequency adaptation. The integration timestep needs to be
            small (~0.2 ms) to resolve threshold crossings; per-ms wall
            time it's roughly 5-10× the rate model.
          </p>
          <p>
            <strong>Stochasticity</strong> here adds Gaussian noise to the
            voltage equation — a stand-in for channel noise and synaptic
            failure.
          </p>
        </>
      );
    case "adex":
      return (
        <>
          <p>
            <strong>AdEx</strong> (adaptive exponential integrate-and-fire,
            Brette & Gerstner 2005), the currently selected model, extends
            LIF with two ingredients: an exponential spike-onset term that
            sharpens the threshold crossing, and a slow adaptation current{" "}
            <code>w</code> that builds up with each spike and decays back
            over <code>τ_w ≈ 150 ms</code>.
          </p>
          <pre className="model-eq">
            {"C · dv/dt = -g_L(v - E_L) + g_L·Δ_T·exp((v - v_T)/Δ_T)\n" +
              "             + R·(W·s + I + b) - w\n" +
              "τ_w · dw/dt = a(v - E_L) - w\n" +
              "if v ≥ v_peak: v := v_reset, w := w + Δw"}
          </pre>
          <p>
            The adaptation current is what gives AdEx its name and its
            personality. Neurons fire faster at stimulus onset, slow down as{" "}
            <code>w</code> accumulates, and can burst on rebound when{" "}
            <code>w</code> decays. AdEx is the standard middle ground when
            you want LIF's tractability but need spike-frequency adaptation
            or class-2 excitability for the science you're doing. Naud,
            Marcille, Clopath & Gerstner (2008) is the canonical reference
            for what regimes you can get out of it.
          </p>
        </>
      );
    case "hh":
      return (
        <>
          <p>
            <strong>HH</strong> (Hodgkin-Huxley 1952), the currently selected
            model, is the full biophysical formulation. There's no threshold
            and no reset rule — spikes emerge from the interaction of
            voltage-gated sodium and potassium channels. The state is{" "}
            <code>(v, m, h, n)</code>: voltage plus three gating variables.
          </p>
          <pre className="model-eq">
            {"C · dv/dt = -g_Na · m³h(v - E_Na)\n" +
              "          - g_K · n⁴(v - E_K)\n" +
              "          - g_L(v - E_L) + I_syn + I_ext\n" +
              "dx/dt = α_x(v)(1 - x) - β_x(v)·x   for x ∈ {m, h, n}"}
          </pre>
          <p>
            <code>m</code> activates Na⁺ on depolarization (fast),{" "}
            <code>h</code> inactivates it (slow), <code>n</code> activates
            K⁺ (recovery). Threshold, refractory period, and spike shape are
            all emergent. Hodgkin and Huxley got the 1963 Nobel for this; the
            squid-axon experiments date to 1952.
          </p>
          <p>
            HH needs a small timestep (~50 μs) for the gating kinetics — per
            millisecond simulated it's roughly 10× slower than LIF. Reach
            for it when subthreshold dynamics or specific ion currents
            matter, or when you need a model you can extend with extra
            channels (Ca²⁺, h-current) without changing the framework.
          </p>
        </>
      );
  }
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
