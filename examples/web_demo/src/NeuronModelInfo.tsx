/**
 * Per-neuron explainer + connected-neurons control panel for detail mode.
 *
 * Owns the "show top-k connected neurons" toggle. When enabled it asks
 * App for the relevant per-neuron detail files (via the onConnectedKChange
 * callback) and reports the lookup table back via props.
 */

import { useMemo } from "react";
import type { Payload } from "./payload";
import type { ModelId } from "./payload";
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

      <div className="detail-section">
        <h3>Activity over time</h3>
        <MiniTrace
          rates={ratesForNeuron}
          stim={stimForNeuron}
          currentFrame={t}
        />
        <div className="legend-mini">
          <span className="dot dot-rate" /> firing rate &nbsp;
          {stimForNeuron && (
            <>
              <span className="dot dot-stim" /> external input
            </>
          )}
        </div>
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
