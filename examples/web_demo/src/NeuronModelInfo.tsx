/**
 * Per-neuron explainer + connected-neurons control panel for detail mode.
 *
 * Owns the "show top-k connected neurons" toggle. When enabled it asks
 * App for the relevant per-neuron detail files (via the onConnectedKChange
 * callback) and reports the lookup table back via props.
 */

import { useEffect, useMemo, useState, type MouseEvent as ReactMouseEvent } from "react";
import type { Payload } from "./payload";
import type { ModelId } from "./payload";
import type { BioNeuronResult } from "./mc/runMcSession";
import { CELL_TYPE_INFO } from "./cellTypes";

export interface ConnectedRequest {
  /** Top-k incoming and outgoing neuron indices to fetch and render. */
  incoming: number[];
  outgoing: number[];
  /** Indices that receive any stimulus input across the scenario. */
  stimulus_inputs: Set<number>;
}

interface Props {
  payload: Payload;
  neuronIndex: number;
  /** Currently selected single-neuron model — determines which equations
   *  the explainer renders. */
  modelId: ModelId;
  frameDisplay: number;
  showConnected: boolean;
  connectedK: number;
  /** Resolved by parent from connectivity; we just display the legend. */
  onToggleConnected: (next: boolean) => void;
  onChangeK: (k: number) => void;
  onClose: () => void;
  /** Biophysical (multi-compartment HH) session state — see MC.md. The
   *  parent owns the state machine; this component just renders the
   *  button and routes clicks back via `onTriggerBio`. */
  bioState?: "idle" | "computing" | "ready" | "error";
  bioVisible?: boolean;
  bioProgress?: number;
  bioError?: string | null;
  bioTotalCompartments?: number;
  bioResult?: BioNeuronResult | null;
  /** Current bio playback time in ms (0 to bioDurationMs). Drives the
   *  AIS-voltage trace cursor when bio view is on. */
  bioTimeMs?: number | null;
  /** Where the bio window is anchored in the outer scenario (ms).
   *  Surfaced in the caveat under the trigger so the user knows what
   *  outer-scenario moment is being magnified. */
  bioAnchorOuterMs?: number | null;
  onTriggerBio?: () => void;
  /** Called when the user clicks somewhere in the AIS voltage trace.
   *  `fraction` is the position in [0, 1] along the bio window. The
   *  parent remaps that to the outer scrubber position so the bio
   *  visualisation jumps to the clicked time. */
  onSeekBioFraction?: (fraction: number) => void;
}

export function NeuronModelInfo({
  payload,
  neuronIndex,
  modelId,
  frameDisplay,
  showConnected,
  connectedK,
  onToggleConnected,
  onChangeK,
  onClose,
  bioState = "idle",
  bioVisible = false,
  bioProgress = 0,
  bioError = null,
  bioTotalCompartments = 0,
  bioResult = null,
  bioTimeMs = null,
  bioAnchorOuterMs = null,
  onTriggerBio,
  onSeekBioFraction,
}: Props) {
  const neuron = payload.neurons[neuronIndex];
  const info = CELL_TYPE_INFO[neuron.cell_type];

  const model = (
    payload as Payload & {
      model?: { weights: number[][]; tau: number[]; bias: number[]; global_gain: number };
    }
  ).model;

  const activation = payload.metadata.hyperparams.activation as string;

  // Connectivity summary.
  const connectivity = useMemo(() => {
    if (!model) return null;
    const N = model.weights.length;
    const W = model.weights;
    const gain = model.global_gain;
    type Edge = { other: number; weight: number; sign: "+" | "−" };
    const incoming: Edge[] = [];
    const outgoing: Edge[] = [];
    for (let j = 0; j < N; j++) {
      const winFromJ = W[neuronIndex][j] * gain;
      if (Math.abs(winFromJ) > 1e-6 && j !== neuronIndex) {
        incoming.push({ other: j, weight: Math.abs(winFromJ), sign: winFromJ > 0 ? "+" : "−" });
      }
      const woutToJ = W[j][neuronIndex] * gain;
      if (Math.abs(woutToJ) > 1e-6 && j !== neuronIndex) {
        outgoing.push({ other: j, weight: Math.abs(woutToJ), sign: woutToJ > 0 ? "+" : "−" });
      }
    }
    incoming.sort((a, b) => b.weight - a.weight);
    outgoing.sort((a, b) => b.weight - a.weight);
    return {
      n_in: incoming.length,
      n_out: outgoing.length,
      top_in: incoming.slice(0, 5),
      top_out: outgoing.slice(0, 5),
    };
  }, [model, neuronIndex]);

  const t = Math.max(
    0,
    Math.min(payload.metadata.n_frames - 1, Math.round(frameDisplay)),
  );
  const liveRate = payload.rates[t]?.[neuronIndex] ?? 0;
  const liveInput = payload.stim_signal?.[t]?.[neuronIndex];

  const ratesForNeuron = useMemo(
    () => payload.rates.map((row) => row[neuronIndex]),
    [payload, neuronIndex],
  );
  const stimForNeuron = useMemo(
    () =>
      payload.stim_signal
        ? payload.stim_signal.map((row) => row[neuronIndex])
        : null,
    [payload, neuronIndex],
  );

  // Is this neuron itself a stimulus input?
  const isStimInput = useMemo(() => {
    if (!payload.stim_signal) return false;
    for (const row of payload.stim_signal) {
      if (Math.abs(row[neuronIndex]) > 1e-4) return true;
    }
    return false;
  }, [payload, neuronIndex]);

  return (
    <div className="detail-sidebar">
      <button
        className="drawer-close"
        aria-label="Close detail view"
        onClick={onClose}
      >
        ◀
      </button>

      {/* Biophysical view: bare button (always visible) plus a collapsible
          "Biophysics" infobox underneath that holds the explainer text +
          run status. The card starts collapsed; pressing the button auto-
          expands it the first time so the user immediately sees what the
          compute is doing. Subsequent header clicks toggle it. */}
      {onTriggerBio && (
        <BiophysicsPanel
          bioState={bioState}
          bioVisible={bioVisible}
          bioProgress={bioProgress}
          bioError={bioError}
          bioTotalCompartments={bioTotalCompartments}
          bioAnchorOuterMs={bioAnchorOuterMs}
          modelId={modelId}
          onTriggerBio={onTriggerBio}
        />
      )}

      {/* Show-neighbors control, top of the panel so it's discoverable */}
      <div className="selector">
        <label htmlFor="neighbors-switch">Neighbors</label>
        <div className="toggle-row">
          <input
            id="neighbors-switch"
            type="checkbox"
            className="toggle-switch"
            checked={showConnected}
            onChange={(e) => onToggleConnected(e.target.checked)}
          />
          <span className="toggle-status">
            {showConnected ? "showing top " + connectedK : "hidden"}
          </span>
        </div>
        {showConnected && (
          <input
            type="range"
            min={1}
            max={15}
            value={connectedK}
            onChange={(e) => onChangeK(parseInt(e.target.value, 10))}
          />
        )}
      </div>

      <div className="detail-header">
        <h2 style={{ color: `#${info?.baseColor.getHexString() ?? "ffffff"}` }}>
          {info?.shortLabel ?? neuron.cell_type}
          {isStimInput && (
            <span className="stim-badge">stim input</span>
          )}
        </h2>
        <div className="params-table" style={{ marginTop: 6 }}>
          <span className="label">body_id</span>
          <span className="value">{neuron.id}</span>
          <span className="label">hemisphere</span>
          <span className="value">{neuron.hemisphere ?? "—"}</span>
          {neuron.nt && (
            <>
              <span className="label">NT</span>
              <span className="value">{neuron.nt}</span>
            </>
          )}
        </div>
      </div>

      <div className="detail-section">
        <h3>Cell biology</h3>
        <p>{info?.blurb ?? "No description available for this cell type."}</p>
      </div>

      <div className="detail-section">
        <h3>Single-neuron model</h3>
        <ModelExplainer modelId={modelId} activation={activation} />
      </div>

      <div className="detail-section">
        <h3>
          {bioVisible && bioResult ? "AIS voltage" : "Activity over time"}
        </h3>
        {bioVisible && bioResult ? (
          <AisVoltageTrace
            bio={bioResult}
            currentTimeMs={bioTimeMs ?? 0}
            onSeek={onSeekBioFraction}
          />
        ) : (
          <MiniTrace
            rates={ratesForNeuron}
            stim={stimForNeuron}
            currentFrame={t}
          />
        )}
        <div className="legend-mini">
          {bioVisible && bioResult ? (
            <span style={{ color: "var(--accent)" }}>
              V at AIS · {bioResult.hh.aisSpikesMs.length} spikes · AIS
              compartment {bioResult.morph.aisCompartment}/
              {bioResult.morph.nCompartments}
            </span>
          ) : (
            <>
              <span className="dot dot-rate" /> firing rate &nbsp;
              {stimForNeuron && (
                <>
                  <span className="dot dot-stim" /> external input
                </>
              )}
            </>
          )}
        </div>
      </div>

      <div className="detail-section">
        <h3>Live parameters</h3>
        {model && (
          <div className="params-table">
            <span className="group-header">Per-neuron</span>
            <span className="label">tau (τ)</span>
            <span className="value">
              {(model.tau[neuronIndex] * 1000).toFixed(1)} ms
            </span>
            <span className="label">bias (b)</span>
            <span className="value">{model.bias[neuronIndex].toFixed(3)}</span>

            <span className="group-header">Global</span>
            <span className="label">gain</span>
            <span className="value">{model.global_gain.toFixed(4)}</span>
            <span className="label">activation</span>
            <span className="value">{activation}</span>

            <span className="group-header">At t = {payload.times[t]?.toFixed(3)}s</span>
            <span className="label">rate r(t)</span>
            <span className="value">{liveRate.toFixed(3)}</span>
            {liveInput !== undefined && (
              <>
                <span className="label">input I(t)</span>
                <span className="value">{liveInput.toFixed(3)}</span>
              </>
            )}
          </div>
        )}
        <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 6 }}>
          τ and bias come from the parameterizer's per-cell-type defaults;
          gain is the global multiplier set per scenario.
        </p>
      </div>

      {connectivity && (
        <div className="detail-section">
          <h3>Connectivity</h3>
          <div className="params-table">
            <span className="label">incoming edges</span>
            <span className="value">{connectivity.n_in}</span>
            <span className="label">outgoing edges</span>
            <span className="value">{connectivity.n_out}</span>
          </div>
          {connectivity.top_in.length > 0 && (
            <>
              <p style={{ marginTop: 8, marginBottom: 4 }}>
                <strong>Top 5 incoming</strong> (weight × gain):
              </p>
              <ul className="conn-list">
                {connectivity.top_in.map((e, k) => {
                  const other = payload.neurons[e.other];
                  return (
                    <li key={k}>
                      <code>
                        {e.sign}
                        {e.weight.toFixed(3)}
                      </code>{" "}
                      from {other.cell_type} #{other.id}
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * "Biophysics" register-style card pinned at the top of the SNV. The
 * "Start biophysics" button lives OUTSIDE the collapsible body so the
 * user can trigger a run without having to expand a card first. Pressing
 * the button auto-opens the body the first time so they immediately see
 * what's being computed; clicking the header toggles it from then on.
 */
function BiophysicsPanel({
  bioState,
  bioVisible,
  bioProgress,
  bioError,
  bioTotalCompartments,
  bioAnchorOuterMs,
  modelId,
  onTriggerBio,
}: {
  bioState: "idle" | "computing" | "ready" | "error";
  bioVisible: boolean;
  bioProgress: number;
  bioError: string | null;
  bioTotalCompartments: number;
  bioAnchorOuterMs: number | null;
  modelId: ModelId;
  onTriggerBio: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  // Auto-expand the first time the user kicks off a compute, so the
  // status messaging is visible without an extra click.
  useEffect(() => {
    if (bioState === "computing" || bioState === "ready" || bioState === "error") {
      setExpanded(true);
    }
  }, [bioState]);

  const handleTrigger = () => {
    setExpanded(true);
    onTriggerBio();
  };

  const buttonText =
    bioState === "computing"
      ? `Computing biophysics… ${Math.round(bioProgress * 100)}%`
      : bioState === "ready"
        ? bioVisible
          ? "Biophysical view: ON"
          : "Biophysical view: OFF"
        : bioState === "error"
          ? "Retry biophysics"
          : "Start biophysics";

  return (
    <div className="bio-section">
      <button
        className={`bio-btn bio-btn-${bioState}${bioVisible ? " on" : ""}`}
        onClick={handleTrigger}
        disabled={bioState === "computing"}
        title="Multi-compartment Hodgkin-Huxley simulation (shortcut: M)"
      >
        {bioState === "computing" ? <span className="bio-spinner" /> : null}
        <span>{buttonText}</span>
        <kbd>M</kbd>
      </button>

      <section className={`infobox bio-infobox ${expanded ? "open" : ""}`}>
        <button
          className="infobox-header"
          onClick={() => setExpanded((e) => !e)}
          aria-expanded={expanded}
        >
          <span className="infobox-marker" aria-hidden />
          <span className="infobox-title">Biophysics</span>
          <span className="infobox-chevron" aria-hidden>
            {expanded ? "−" : "+"}
          </span>
        </button>
        <div className="infobox-collapse" aria-hidden={!expanded}>
          <div className="infobox-body">
            <p>
              Pressing <strong>Start biophysics</strong> kicks off a
              multi-compartment Hodgkin-Huxley simulation rooted at{" "}
              <em>the current scrubber position</em> — that outer time
              becomes the anchor for a 200 ms window of biological time
              that gets solved at full membrane resolution.
              {bioState === "ready" && bioAnchorOuterMs !== null && (
                <>
                  {" "}
                  Currently anchored at{" "}
                  <code>{(bioAnchorOuterMs / 1000).toFixed(2)} s</code>;
                  re-anchor by scrubbing to a new spot and pressing the
                  button again.
                </>
              )}
            </p>
            <p>
              <strong>What's being simulated.</strong> The focused neuron
              plus its visible top-5 incoming and top-5 outgoing partners
              (deduped) each get a Hines-discretised cable equation on
              their full SWC morphology, aggregated into ~200-500 solver
              compartments per cell. Each compartment carries Hodgkin-
              Huxley Na⁺/K⁺/leak channels; the axon initial segment is
              picked at ~20 µm distal of the soma and gets a 10×
              Na/K-density bump so spikes initiate there.
              {bioState === "ready" && (
                <>
                  {" "}
                  This run covers{" "}
                  <code>{bioTotalCompartments}</code> compartments across
                  the neighborhood.
                </>
              )}
            </p>
            <p>
              <strong>Coupling to the rest of the circuit.</strong> Inputs
              from neurons <em>outside</em> the neighborhood enter as a
              continuous conductance proportional to the outer{" "}
              <code>{modelId.toUpperCase()}</code> sim's per-neuron firing
              rate at the corresponding outer time. This is one-way
              coupling — the bio sim doesn't feed back into the outer
              rates — and there's no Poisson-spike intermediate, so the
              slow outer rhythm enters the membrane as a slow drive
              rather than as a flicker of EPSPs.
            </p>
            <p>
              <strong>Timescales.</strong> A real spike is ~1 ms wide.
              The outer scrubber covers seconds of scenario time per
              playback. To keep spikes perceptible the 200 ms bio window
              is stretched linearly across the full scrubber: scrubber
              fraction × 200 ms = bio time. A 1 ms spike is therefore
              ~0.5 % of the scrubber. On top of that, the rendered colour
              uses a peak-hold envelope with a ~30 ms decay so individual
              spikes leave a visible afterglow.
            </p>
            <p>
              <strong>Modelling choices, briefly.</strong> Channel
              densities are the squid-axon textbook values (Hodgkin &
              Huxley 1952) — qualitatively right, not species-tuned for
              fly central complex. Compartments are aggregated from the
              fine SWC tree to ~2 µm chunks; finer would be more
              numerically stiff than necessary, coarser would hide
              dendritic structure. The cable solver is implicit (Hines
              1984) so the simulation stays stable at dt = 25 µs even
              with thin AIS compartments.
            </p>
            {bioState === "error" && bioError && (
              <p className="bio-error">Compute failed: {bioError}</p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function ModelExplainer({
  modelId,
  activation,
}: {
  modelId: ModelId;
  activation: string;
}) {
  switch (modelId) {
    case "lif":
      return <LIFExplainer />;
    case "adex":
      return <AdExExplainer />;
    case "hh":
      return <HHExplainer />;
    case "rate":
    default:
      return <RateExplainer activation={activation} />;
  }
}

function RateExplainer({ activation }: { activation: string }) {
  return (
    <>
      <p>
        Rate-model neuron. The state is a single scalar firing rate{" "}
        <code>r(t)</code>; spike timing is not modeled.
      </p>
      <pre className="model-eq">
        {"τ · dr/dt = -r + " + activation + "(W · r + I + b)"}
      </pre>
      <p>
        At each timestep, the neuron sums weighted input from upstream cells
        (the W · r term), adds external input I(t) and a bias b, applies the
        nonlinearity <code>{activation}</code>, and integrates toward that
        value on a timescale of <code>τ</code> (10-20 ms).
      </p>
    </>
  );
}

function LIFExplainer() {
  return (
    <>
      <p>
        Leaky integrate-and-fire (LIF) biophysical model. The state is a
        membrane voltage <code>v(t)</code>; spikes are discrete events.
      </p>
      <pre className="model-eq">
        {"τ · dv/dt = -(v - v_rest) + R · (W · s + I + b)\n" +
          "if v ≥ v_thr: emit spike, v := v_reset (refractory)"}
      </pre>
      <p>
        <code>s</code> is a per-neuron exponential synaptic-conductance trace
        kicked up by each presynaptic spike (decay τ_syn ≈ 5 ms). After
        crossing threshold, the cell is silenced for ~2 ms (refractory).
        The displayed rate r(t) is a spike-density estimate over a sliding
        window.
      </p>
    </>
  );
}

function AdExExplainer() {
  return (
    <>
      <p>
        Adaptive exponential integrate-and-fire (AdEx, Brette & Gerstner
        2005). Voltage v plus a slow adaptation current w that drags the
        cell back after firing — gives spike-frequency adaptation and
        bursting.
      </p>
      <pre className="model-eq">
        {"C · dv/dt = -g_L(v - E_L) + g_L·Δ_T·exp((v - v_T)/Δ_T)\n" +
          "             + R·(W·s + I + b) - w\n" +
          "τ_w · dw/dt = a(v - E_L) - w\n" +
          "if v ≥ v_peak: v := v_reset, w := w + Δw"}
      </pre>
      <p>
        The exponential term sharpens the spike onset; <code>w</code>{" "}
        accumulates each spike (jump <code>Δw</code>) and decays with
        timescale <code>τ_w</code> (~150 ms), suppressing further firing.
      </p>
    </>
  );
}

function HHExplainer() {
  return (
    <>
      <p>
        Full Hodgkin-Huxley model. Spikes emerge from the interaction of
        voltage-gated Na⁺ and K⁺ channels — no thresholding hack, no reset
        rule. The state is <code>(v, m, h, n)</code>: voltage plus three
        gating variables.
      </p>
      <pre className="model-eq">
        {"C·dv/dt = -g_Na·m³·h(v - E_Na)\n" +
          "          - g_K·n⁴(v - E_K)\n" +
          "          - g_L(v - E_L) + I_syn + I_ext\n" +
          "dx/dt = α_x(v)(1 - x) - β_x(v)·x   (x ∈ {m, h, n})"}
      </pre>
      <p>
        <code>m</code> activates Na⁺ on depolarization (fast), <code>h</code>{" "}
        inactivates it (slow), <code>n</code> activates K⁺ (recovery). The
        spike shape, threshold, and refractory period are emergent. Costly
        to simulate — ~10× slower than LIF.
      </p>
    </>
  );
}

function MiniTrace({
  rates,
  stim,
  currentFrame,
}: {
  rates: number[];
  stim: number[] | null;
  currentFrame: number;
}) {
  const W = 280;
  const H = 60;
  const N = rates.length;
  const rmax = Math.max(...rates, 0.01);
  const smax = stim ? Math.max(...stim, 0.01) : 1;
  const x = (i: number) => (W * i) / Math.max(1, N - 1);
  const yr = (v: number) => H - (H * v) / rmax;
  const ys = (v: number) => H - (H * v) / smax;
  const ratePath = rates
    .map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${yr(v).toFixed(1)}`)
    .join(" ");
  const stimPath = stim
    ? stim
        .map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${ys(v).toFixed(1)}`)
        .join(" ")
    : "";

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={H}
      style={{ background: "rgba(255,255,255,0.03)", borderRadius: 4 }}
    >
      {stim && (
        <path
          d={stimPath}
          fill="none"
          stroke="#ff5cb0"
          strokeWidth={1.2}
          opacity={0.8}
        />
      )}
      <path d={ratePath} fill="none" stroke="#58a6ff" strokeWidth={1.5} />
      <line
        x1={x(currentFrame)}
        x2={x(currentFrame)}
        y1={0}
        y2={H}
        stroke="#ffffff"
        strokeOpacity={0.5}
        strokeWidth={1}
      />
    </svg>
  );
}

/**
 * Voltage trace at the AIS compartment, plus spike markers. Replaces
 * `MiniTrace` when biophysical view is on. The y-range is fixed to the
 * canonical −90…+50 mV window so spikes are visually consistent across
 * neurons.
 *
 * Click anywhere on the trace to seek the playback to that bio time —
 * the click x-coordinate maps to a fraction in [0, 1] of the bio window
 * and the parent rebinds that to the outer scrubber.
 */
function AisVoltageTrace({
  bio,
  currentTimeMs,
  onSeek,
}: {
  bio: BioNeuronResult;
  currentTimeMs: number;
  onSeek?: (fraction: number) => void;
}) {
  const W = 280;
  const H = 60;
  const ais = bio.morph.aisCompartment;
  const nC = bio.hh.nCompartments;
  const nF = bio.hh.nFrames;
  const vTrace = bio.hh.voltageTrace;
  const tTrace = bio.hh.frameTimesMs;
  const tMin = tTrace[0] ?? 0;
  const tMax = tTrace[nF - 1] ?? 1;
  const tSpan = Math.max(1e-3, tMax - tMin);
  const vMin = -90;
  const vMax = 50;
  const xOf = (tMs: number) => ((tMs - tMin) / tSpan) * W;
  const yOf = (vMv: number) => H - ((vMv - vMin) / (vMax - vMin)) * H;

  // Render the trace with a polyline string. Sub-sample if the bio sim
  // produced more frames than pixels — drawing 8 000 path commands is
  // wasteful when the SVG only has 280 of horizontal space.
  const targetSegs = Math.min(W, nF);
  const stride = Math.max(1, Math.floor(nF / targetSegs));
  const parts: string[] = [];
  for (let i = 0; i < nF; i += stride) {
    const v = vTrace[i * nC + ais];
    parts.push(
      `${i === 0 ? "M" : "L"} ${xOf(tTrace[i]).toFixed(1)} ${yOf(v).toFixed(1)}`,
    );
  }
  const tracePath = parts.join(" ");

  const cursorX = xOf(Math.max(tMin, Math.min(tMax, currentTimeMs)));

  // Translate a mouse-x coordinate into a [0, 1] fraction of the bio
  // window. The SVG's viewBox is `0 0 W H`; the mouse-position lookup
  // uses the rendered bounding rect.
  const handleSeek = (e: ReactMouseEvent<SVGSVGElement>) => {
    if (!onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(fraction);
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={H}
      style={{
        background: "rgba(255,255,255,0.03)",
        borderRadius: 4,
        cursor: onSeek ? "ew-resize" : "default",
      }}
      onClick={handleSeek}
      onMouseDown={(e) => {
        // Allow click-and-drag scrubbing too: capture pointer events
        // until release. `e.buttons === 1` filters out non-primary drags.
        if (!onSeek) return;
        handleSeek(e);
        const onMove = (ev: MouseEvent) => {
          if (ev.buttons !== 1) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const fraction = Math.max(
            0,
            Math.min(1, (ev.clientX - rect.left) / rect.width),
          );
          onSeek(fraction);
        };
        const onUp = () => {
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
      }}
    >
      {/* Spike threshold reference. */}
      <line
        x1={0}
        x2={W}
        y1={yOf(-20)}
        y2={yOf(-20)}
        stroke="#ffffff"
        strokeOpacity={0.18}
        strokeDasharray="3 3"
      />
      {/* Spike markers as small tick marks at the top. */}
      {bio.hh.aisSpikesMs.map((tMs, i) => (
        <line
          key={`s-${i}`}
          x1={xOf(tMs)}
          x2={xOf(tMs)}
          y1={0}
          y2={6}
          stroke="var(--accent)"
          strokeWidth={1.2}
        />
      ))}
      <path d={tracePath} fill="none" stroke="var(--accent)" strokeWidth={1.4} />
      <line
        x1={cursorX}
        x2={cursorX}
        y1={0}
        y2={H}
        stroke="#ffffff"
        strokeOpacity={0.5}
        strokeWidth={1}
      />
    </svg>
  );
}
