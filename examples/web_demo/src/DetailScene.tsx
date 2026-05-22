import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { NeuronDetail } from "./detailLoader";
import type { Payload } from "./payload";
import type { BioSession, BioNeuronResult } from "./mc/runMcSession";
import { activityColor, blendInput } from "./colormap";
import { baseColorFor } from "./cellTypes";

export interface CompanionNeuron {
  detail: NeuronDetail;
  neuronIndex: number;
  /** Direction relative to the focused neuron. Kept in the data even
   *  though we no longer differentiate visually -- the sidebar lists the
   *  top-5 incoming explicitly. */
  role: "incoming" | "outgoing";
}

interface Props {
  detail: NeuronDetail;
  payload: Payload;
  neuronIndex: number;
  frameRef: { current: number };
  companions?: CompanionNeuron[];
  /** Biophysical voltage data. When non-null and `bioVisible`, the per-
   *  compartment voltage replaces the rate-driven uniform colouring for
   *  whichever neurons have an entry in the session. */
  bioSession?: BioSession | null;
  bioVisible?: boolean;
}

/**
 * Voltage → colour ramp modulated by the cell-type baseline colour. Three
 * regimes:
 *
 *   - Sub-threshold (V ≲ -50 mV): muted version of the cell-type colour —
 *     scaled to 15-45 % brightness so a resting EPG looks like a dim
 *     pink, a resting Δ7 like a dim teal, etc. The neuron stays
 *     identifiable as its type but doesn't shout for attention.
 *   - Climbing toward threshold (V from -50 to +20 mV): ramp brightness
 *     up through the natural baseline colour into saturation.
 *   - Spike peak (V > +20 mV): blend the saturated colour toward white,
 *     so spikes pop with the same hot-white look regardless of cell
 *     type while still leaving a tint of the cell colour at the onset.
 */
function voltageToColor(
  vMv: number,
  baseColor: THREE.Color,
  out: THREE.Color,
): THREE.Color {
  const t = Math.max(0, Math.min(1, (vMv + 90) / 140));
  const br = baseColor.r;
  const bg = baseColor.g;
  const bb = baseColor.b;
  if (t < 0.5) {
    // Sub-threshold — muted cell-type colour.
    const u = t / 0.5;
    const k = 0.15 + 0.3 * u; // 0.15 → 0.45
    out.setRGB(br * k, bg * k, bb * k);
  } else if (t < 0.72) {
    // Climbing toward threshold — push brightness up to ~saturation.
    const u = (t - 0.5) / 0.22;
    const k = 0.45 + 1.05 * u; // 0.45 → 1.5
    out.setRGB(
      Math.min(1, br * k),
      Math.min(1, bg * k),
      Math.min(1, bb * k),
    );
  } else {
    // Spike — saturated cell-type colour, then blend to white.
    const u = (t - 0.72) / 0.28;
    const startR = Math.min(1, br * 1.6);
    const startG = Math.min(1, bg * 1.6);
    const startB = Math.min(1, bb * 1.6);
    out.setRGB(
      startR + (1 - startR) * u,
      startG + (1 - startG) * u,
      startB + (1 - startB) * u,
    );
  }
  return out;
}

/**
 * Build a "peak-hold with decay" envelope of the voltage trace, one
 * value per (compartment, frame). When voltage spikes briefly above its
 * recent history, the envelope tracks it; on the way back down it decays
 * exponentially with a configurable time constant toward rest.
 *
 * Why: a real spike is ~1 ms wide. At outer-sim duration ~1.5 s mapped to
 * a ~120-frame scrubber, that's well under a single rendered frame — the
 * spike is invisible. The envelope adds a ~50 ms decay tail so spikes
 * are visible at any playback speed without misrepresenting the
 * underlying physics. The mini-trace below still shows the raw voltage.
 */
function buildVoltageEnvelope(
  vTrace: Float32Array,
  nFrames: number,
  nComp: number,
  bioDtMs: number,
  decayMs: number = 30,
): Float32Array {
  const env = new Float32Array(vTrace.length);
  if (nFrames === 0) return env;
  // Per-frame retention factor. e^(-bioDt / decayMs).
  const k = Math.exp(-bioDtMs / decayMs);
  // Frame 0: just copy.
  for (let c = 0; c < nComp; c++) env[c] = vTrace[c];
  // Forward sweep. Floor of the decay is V_rest so quiet compartments
  // converge there rather than to zero.
  const vFloor = -65;
  for (let f = 1; f < nFrames; f++) {
    const baseEnv = (f - 1) * nComp;
    const baseV = f * nComp;
    const baseOut = f * nComp;
    for (let c = 0; c < nComp; c++) {
      const decayed = env[baseEnv + c] * k + vFloor * (1 - k);
      const v = vTrace[baseV + c];
      env[baseOut + c] = v > decayed ? v : decayed;
    }
  }
  return env;
}

const UNIT_CYL = new THREE.CylinderGeometry(1, 1, 1, 12, 1, false);
UNIT_CYL.translate(0, 0.5, 0);
const _Y = new THREE.Vector3(0, 1, 0);

const SPHERE_GEOM = new THREE.SphereGeometry(1, 12, 8);

/**
 * Renders the focused neuron at full SWC resolution, optionally with its
 * top-k connected companions at reduced opacity. Companions use their
 * own cell-type baseline color (same as in the overview). No tinting and
 * no stim halos -- the explainer panel in the sidebar lists incoming and
 * outgoing connections explicitly, so the 3D scene stays clean.
 */
export function DetailScene({
  detail,
  payload,
  neuronIndex,
  frameRef,
  companions = [],
  bioSession = null,
  bioVisible = false,
}: Props) {
  const maxRate = useMemo(() => {
    let mx = 0;
    for (const row of payload.rates) for (const v of row) if (v > mx) mx = v;
    return mx > 0 ? mx : 1;
  }, [payload]);
  const maxStim = useMemo(() => {
    if (!payload.stim_signal) return 1;
    let mx = 0;
    for (const row of payload.stim_signal) for (const v of row) if (v > mx) mx = v;
    return mx > 0 ? mx : 1;
  }, [payload]);

  const focusedBio = bioVisible && bioSession
    ? bioSession.results.get(neuronIndex) ?? null
    : null;
  // Used by NeuronMesh to convert the playback scrubber position to bio
  // time. Passed as a number (not on the bio prop) so a single value
  // covers focused + companion meshes uniformly.
  const bioDurationMs = bioSession?.bioDurationMs ?? 0;

  return (
    <group>
      <NeuronMesh
        detail={detail}
        baseColor={baseColorFor(payload.neurons[neuronIndex].cell_type)}
        index={neuronIndex}
        payload={payload}
        frameRef={frameRef}
        maxRate={maxRate}
        maxStim={maxStim}
        opacity={1.0}
        emissiveBoost={0.5}
        bio={focusedBio}
        bioDurationMs={bioDurationMs}
      />
      {companions.map((c) => {
        const cBio = bioVisible && bioSession
          ? bioSession.results.get(c.neuronIndex) ?? null
          : null;
        return (
          <NeuronMesh
            key={`c-${c.neuronIndex}`}
            detail={c.detail}
            baseColor={baseColorFor(payload.neurons[c.neuronIndex].cell_type)}
            index={c.neuronIndex}
            payload={payload}
            frameRef={frameRef}
            maxRate={maxRate}
            maxStim={maxStim}
            // Bio-visible companions need to stay readable, so we boost
            // opacity above the rate-mode 0.18 baseline.
            opacity={cBio ? 0.85 : 0.18}
            emissiveBoost={cBio ? 0.3 : 0.05}
            bio={cBio}
            bioDurationMs={bioDurationMs}
          />
        );
      })}
    </group>
  );
}

interface MeshProps {
  detail: NeuronDetail;
  baseColor: THREE.Color;
  index: number;
  payload: Payload;
  frameRef: { current: number };
  maxRate: number;
  maxStim: number;
  opacity: number;
  emissiveBoost: number;
  /** When non-null, per-compartment voltage replaces the uniform colour. */
  bio: BioNeuronResult | null;
  /** Length of the bio window in biological ms. Drives the outer-frame
   *  → bio-time mapping inside useFrame. */
  bioDurationMs: number;
}

function NeuronMesh({
  detail,
  baseColor,
  index,
  payload,
  frameRef,
  maxRate,
  maxStim,
  opacity,
  emissiveBoost,
  bio,
  bioDurationMs,
}: MeshProps) {
  const cylRef = useRef<THREE.InstancedMesh>(null);
  const sphRef = useRef<THREE.InstancedMesh>(null);
  const { positions, radii, edges } = detail;
  const nNodes = detail.n_nodes;
  const nSegs = edges.length / 2;
  // SWC node index → solver compartment index, for the voltage lookup.
  // Only populated when bio is on; null otherwise.
  const nodeToComp = useMemo(
    () => (bio ? bio.morph.nodeToCompartment : null),
    [bio],
  );
  // Pre-built peak-hold envelope of the bio voltage trace. Memoised so we
  // only pay the O(nFrames · nComp) cost once per bio result.
  const voltageEnvelope = useMemo(() => {
    if (!bio) return null;
    const nF = bio.hh.nFrames;
    const nC = bio.hh.nCompartments;
    const bioDtMs = nF > 1
      ? bio.hh.frameTimesMs[1] - bio.hh.frameTimesMs[0]
      : 1;
    return buildVoltageEnvelope(bio.hh.voltageTrace, nF, nC, bioDtMs, 30);
  }, [bio]);

  // Static instance matrices.
  useEffect(() => {
    const cylMesh = cylRef.current;
    const sphMesh = sphRef.current;
    if (!cylMesh || !sphMesh) return;
    const mat = new THREE.Matrix4();
    const p0 = new THREE.Vector3();
    const p1 = new THREE.Vector3();
    for (let s = 0; s < nSegs; s++) {
      const parentIdx = edges[s * 2];
      const childIdx = edges[s * 2 + 1];
      p0.set(
        positions[parentIdx * 3],
        positions[parentIdx * 3 + 1],
        positions[parentIdx * 3 + 2],
      );
      p1.set(
        positions[childIdx * 3],
        positions[childIdx * 3 + 1],
        positions[childIdx * 3 + 2],
      );
      const dir = new THREE.Vector3().subVectors(p1, p0);
      const len = dir.length();
      const r = Math.max(0.001, 0.5 * (radii[parentIdx] + radii[childIdx]));
      if (len < 1e-9) {
        mat.makeScale(0, 0, 0);
        mat.setPosition(p0);
      } else {
        dir.divideScalar(len);
        const q = new THREE.Quaternion().setFromUnitVectors(_Y, dir);
        mat.compose(p0, q, new THREE.Vector3(r, len, r));
      }
      cylMesh.setMatrixAt(s, mat);
    }
    cylMesh.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < nNodes; i++) {
      const r = Math.max(0.002, radii[i] * 0.9);
      mat.makeScale(r, r, r);
      mat.setPosition(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      sphMesh.setMatrixAt(i, mat);
    }
    sphMesh.instanceMatrix.needsUpdate = true;
  }, [edges, positions, radii, nNodes, nSegs]);

  const tmp = useMemo(() => new THREE.Color(), []);
  useFrame(() => {
    const f = frameRef.current;
    const tOuter = Math.max(
      0,
      Math.min(payload.metadata.n_frames - 1, Math.round(Number.isFinite(f) ? f : 0)),
    );

    if (bio && nodeToComp && voltageEnvelope && cylRef.current && sphRef.current) {
      // Biophysical view: per-compartment voltage colours, sampled from
      // the peak-hold envelope so spikes stay visible at any playback
      // speed.
      //
      // C+D design: bio time is decoupled from outer simulated time.
      // The scrubber position [0, 1] linearly maps to bio time
      // [0, bioDurationMs], stretching a short biological window across
      // the full outer scrubber range so spikes are perceptible.
      const scrubberFraction = tOuter / Math.max(1, payload.metadata.n_frames - 1);
      const bioTimeMs = scrubberFraction * bioDurationMs;
      const bioNf = bio.hh.nFrames;
      const bioTimes = bio.hh.frameTimesMs;
      let lo = 0;
      let hi = bioNf - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (bioTimes[mid] <= bioTimeMs) lo = mid;
        else hi = mid;
      }
      const bioFrame =
        bioTimeMs - bioTimes[lo] <= bioTimes[hi] - bioTimeMs ? lo : hi;
      const nComp = bio.hh.nCompartments;
      const off = bioFrame * nComp;

      const cyl = cylRef.current;
      for (let s = 0; s < nSegs; s++) {
        const childNode = edges[s * 2 + 1];
        const comp = nodeToComp[childNode];
        const safeComp = comp >= 0 ? comp : 0;
        const vMv = voltageEnvelope[off + safeComp];
        voltageToColor(Number.isFinite(vMv) ? vMv : -65, baseColor, tmp);
        cyl.setColorAt(s, tmp);
      }
      if (cyl.instanceColor) cyl.instanceColor.needsUpdate = true;

      const sph = sphRef.current;
      for (let i = 0; i < nNodes; i++) {
        const comp = nodeToComp[i];
        const safeComp = comp >= 0 ? comp : 0;
        const vMv = voltageEnvelope[off + safeComp];
        voltageToColor(Number.isFinite(vMv) ? vMv : -65, baseColor, tmp);
        sph.setColorAt(i, tmp);
      }
      if (sph.instanceColor) sph.instanceColor.needsUpdate = true;

      // Material colour stays white so per-instance colours multiply
      // through. Emissive carries a dim cell-type-tinted glow, so the
      // neuron still reads as its type at rest; the per-instance colour
      // (which already encodes cell type + voltage) dominates the look.
      for (const ref of [cyl, sph]) {
        const m = ref.material as THREE.MeshStandardMaterial;
        m.color.setRGB(1, 1, 1);
        m.emissive.setRGB(baseColor.r * 0.12, baseColor.g * 0.12, baseColor.b * 0.12);
        m.emissiveIntensity = 0.45;
        m.opacity = opacity;
        m.transparent = opacity < 1;
      }
      return;
    }

    // Rate-driven uniform colouring (the original path).
    const v = payload.rates[tOuter][index] / maxRate;
    activityColor(v, baseColor, tmp);
    const stimRow = payload.stim_signal?.[tOuter];
    if (stimRow) {
      const inputLevel = stimRow[index] / maxStim;
      if (inputLevel > 0.05) blendInput(tmp, inputLevel, tmp);
    }
    for (const ref of [cylRef.current, sphRef.current]) {
      if (!ref?.material) continue;
      const m = ref.material as THREE.MeshStandardMaterial;
      m.color.copy(tmp);
      m.emissive.copy(tmp);
      m.emissiveIntensity = emissiveBoost + 0.5 * Math.max(v, 0);
      m.opacity = opacity;
      m.transparent = opacity < 1;
    }
  });

  return (
    <group>
      <instancedMesh ref={cylRef} args={[UNIT_CYL, undefined, nSegs]}>
        <meshStandardMaterial
          color={baseColor}
          roughness={0.45}
          metalness={0.1}
          emissive={baseColor}
          emissiveIntensity={emissiveBoost}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </instancedMesh>
      <instancedMesh ref={sphRef} args={[SPHERE_GEOM, undefined, nNodes]}>
        <meshStandardMaterial
          color={baseColor}
          roughness={0.45}
          metalness={0.1}
          emissive={baseColor}
          emissiveIntensity={emissiveBoost}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </instancedMesh>
    </group>
  );
}
