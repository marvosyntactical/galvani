import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import type { PayloadNeuron, Payload } from "./payload";
import { activityColor, blendInput } from "./colormap";
import { baseColorFor } from "./cellTypes";

export type RenderMode = "lines" | "tubes";

export interface HoverInfo {
  neuronIndex: number;
  rate: number;
  screen: { x: number; y: number };
}

interface Props {
  payload: Payload;
  /** Live frame index, mutated by the App's RAF loop. The Scene reads
   *  `.current` each useFrame call -- no React state involved. */
  frameRef: { current: number };
  renderMode: RenderMode;
  lineWidth?: number;
  onHover?: (info: HoverInfo | null) => void;
  hoveredIndex?: number | null;
  onSelect?: (neuronIndex: number) => void;
  /** Browser-side Gaussian noise added to each neuron's rate per frame.
   *  0 = deterministic, 0.3 = noticeable jitter. */
  stochasticity?: number;
}

/* ------------------------------------------------------------------------ */
/* helpers                                                                   */
/* ------------------------------------------------------------------------ */

function pointsFromFlat(flat: number[]): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let i = 0; i < flat.length; i += 3) {
    out.push([flat[i], flat[i + 1], flat[i + 2]]);
  }
  return out;
}

const UNIT_CYLINDER = new THREE.CylinderGeometry(1, 1, 1, 6, 1, false);
UNIT_CYLINDER.translate(0, 0.5, 0); // base at origin, tip at +Y
const _Y = new THREE.Vector3(0, 1, 0);

// Module-level scratch vectors so the per-segment matrix build doesn't
// allocate. The old code did `new Vector3()` per segment which created
// thousands of GC-tracked objects on each mode switch.
const _SCRATCH = {
  dir: new THREE.Vector3(),
  q: new THREE.Quaternion(),
  scale: new THREE.Vector3(),
};

function segmentMatrix(
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  radius: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const { dir, q, scale } = _SCRATCH;
  dir.subVectors(p1, p0);
  const len = dir.length();
  if (len < 1e-9) {
    out.makeScale(0, 0, 0);
    out.setPosition(p0);
    return out;
  }
  dir.divideScalar(len);
  q.setFromUnitVectors(_Y, dir);
  scale.set(radius, len, radius);
  out.compose(p0, q, scale);
  return out;
}

/* ------------------------------------------------------------------------ */
/* per-neuron Line component (renderMode === "lines")                        */
/* ------------------------------------------------------------------------ */

type LineColorBearer = { material: unknown } | null;

interface NeuronLineProps {
  neuron: PayloadNeuron;
  index: number;
  registerRef: (i: number, obj: LineColorBearer) => void;
  lineWidth: number;
  onPointerOver?: (e: { clientX: number; clientY: number }) => void;
  onPointerOut?: () => void;
  onClick?: () => void;
  isHovered: boolean;
}

function NeuronLine({
  neuron,
  index,
  registerRef,
  lineWidth,
  onPointerOver,
  onPointerOut,
  onClick,
  isHovered,
}: NeuronLineProps) {
  const points = useMemo(() => pointsFromFlat(neuron.segments), [neuron.segments]);
  return (
    <Line
      points={points}
      segments
      lineWidth={isHovered ? lineWidth * 2 : lineWidth}
      color="#202020"
      transparent
      opacity={isHovered ? 1.0 : 0.92}
      onPointerOver={(e: { clientX: number; clientY: number; stopPropagation?: () => void }) => {
        e.stopPropagation?.();
        onPointerOver?.(e);
      }}
      onPointerOut={onPointerOut}
      onClick={(e: { stopPropagation?: () => void }) => {
        e.stopPropagation?.();
        onClick?.();
      }}
      ref={(el) => {
        registerRef(index, el ? (el as unknown as LineColorBearer) : null);
      }}
    />
  );
}

/* ------------------------------------------------------------------------ */
/* TubesView -- ONE InstancedMesh for the whole scene                        */
/* Why: one mesh per neuron used to mount hundreds of InstancedMeshes        */
/* whenever the user flipped to "tubes", which froze the page for >1 s.      */
/* A single mesh is one draw call, one material compile, and one matrix      */
/* allocation pass.                                                          */
/* ------------------------------------------------------------------------ */

interface TubesViewProps {
  payload: Payload;
  frameRef: { current: number };
  baseColors: THREE.Color[];
  maxRate: number;
  maxStim: number;
  stochasticity: number;
  hoveredIndex: number | null;
  onHover?: (info: HoverInfo | null) => void;
  onSelect?: (i: number) => void;
}

function TubesView({
  payload,
  frameRef,
  baseColors,
  maxRate,
  maxStim,
  stochasticity,
  hoveredIndex,
  onHover,
  onSelect,
}: TubesViewProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  // Per-payload lookup tables: which instance range belongs to which neuron,
  // and the reverse lookup so a pointer event's `instanceId` -> neuron index.
  const { totalSegments, neuronOffsets, neuronCounts, neuronByInstance } =
    useMemo(() => {
      const N = payload.neurons.length;
      const offsets = new Int32Array(N);
      const counts = new Int32Array(N);
      let total = 0;
      for (let i = 0; i < N; i++) {
        const c = (payload.neurons[i].segments.length / 6) | 0;
        offsets[i] = total;
        counts[i] = c;
        total += c;
      }
      const lookup = new Int32Array(total);
      for (let i = 0; i < N; i++) {
        const off = offsets[i];
        const cnt = counts[i];
        for (let s = 0; s < cnt; s++) lookup[off + s] = i;
      }
      return {
        totalSegments: total,
        neuronOffsets: offsets,
        neuronCounts: counts,
        neuronByInstance: lookup,
      };
    }, [payload]);

  // One-shot: write instance matrices + initial base colors.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const mat = new THREE.Matrix4();
    const p0 = new THREE.Vector3();
    const p1 = new THREE.Vector3();
    let idx = 0;
    for (let i = 0; i < payload.neurons.length; i++) {
      const n = payload.neurons[i];
      const segs = n.segments;
      const radii = n.radii;
      const count = neuronCounts[i];
      for (let s = 0; s < count; s++) {
        const base = s * 6;
        p0.set(segs[base], segs[base + 1], segs[base + 2]);
        p1.set(segs[base + 3], segs[base + 4], segs[base + 5]);
        const r = Math.max(0.005, radii[s] ?? 0.02);
        segmentMatrix(p0, p1, r, mat);
        mesh.setMatrixAt(idx, mat);
        idx++;
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    // Seed instance colors so the InstancedBufferAttribute exists; useFrame
    // overwrites them every frame.
    for (let i = 0; i < payload.neurons.length; i++) {
      const off = neuronOffsets[i];
      const cnt = neuronCounts[i];
      for (let s = 0; s < cnt; s++) {
        mesh.setColorAt(off + s, baseColors[i]);
      }
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [payload, neuronCounts, neuronOffsets, baseColors]);

  // Per-frame color update. Writes directly into the InstancedBufferAttribute
  // backing array so we never allocate or call setColorAt 5 000+ times per
  // frame (which was significant per-frame GC pressure).
  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh || !mesh.instanceColor) return;
    const f = frameRef.current;
    const t = Math.max(
      0,
      Math.min(payload.metadata.n_frames - 1, Math.round(Number.isFinite(f) ? f : 0)),
    );
    const row = payload.rates[t];
    const stimRow = payload.stim_signal?.[t];
    const buf = mesh.instanceColor.array as Float32Array;

    for (let i = 0; i < payload.neurons.length; i++) {
      let v = row[i] / maxRate;
      if (stochasticity > 0) {
        const u = Math.max(1e-6, Math.random());
        const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
        v = Math.max(0, Math.min(1.2, v + stochasticity * g * 0.4));
      }
      activityColor(v, baseColors[i], tmpColor);
      if (stimRow) {
        const inputLevel = stimRow[i] / maxStim;
        if (inputLevel > 0.05) blendInput(tmpColor, inputLevel, tmpColor);
      }
      if (i === hoveredIndex) {
        // Boost hovered neuron without going off-gamut.
        tmpColor.r = Math.min(1, tmpColor.r * 1.6 + 0.1);
        tmpColor.g = Math.min(1, tmpColor.g * 1.6 + 0.1);
        tmpColor.b = Math.min(1, tmpColor.b * 1.6 + 0.1);
      }
      const off = neuronOffsets[i];
      const cnt = neuronCounts[i];
      const r = tmpColor.r;
      const g = tmpColor.g;
      const b = tmpColor.b;
      for (let s = 0; s < cnt; s++) {
        const o = (off + s) * 3;
        buf[o] = r;
        buf[o + 1] = g;
        buf[o + 2] = b;
      }
    }
    mesh.instanceColor.needsUpdate = true;
  });

  // Pointer events: instancedmesh exposes instanceId; map to neuron via the
  // precomputed lookup. We translate pointerMove to "hover" (single
  // continuous handler) and pointerOut to "unhover".
  return (
    <instancedMesh
      ref={meshRef}
      args={[UNIT_CYLINDER, undefined, totalSegments]}
      onPointerMove={(e) => {
        const inst = e.instanceId;
        if (inst === undefined) return;
        e.stopPropagation();
        const i = neuronByInstance[inst];
        const f = frameRef.current;
        const t = Math.max(
          0,
          Math.min(
            payload.metadata.n_frames - 1,
            Math.round(Number.isFinite(f) ? f : 0),
          ),
        );
        onHover?.({
          neuronIndex: i,
          rate: payload.rates[t][i],
          screen: { x: e.clientX, y: e.clientY },
        });
      }}
      onPointerOut={() => onHover?.(null)}
      onClick={(e) => {
        const inst = e.instanceId;
        if (inst === undefined) return;
        e.stopPropagation();
        const i = neuronByInstance[inst];
        onSelect?.(i);
      }}
    >
      <meshStandardMaterial roughness={0.55} metalness={0.05} />
    </instancedMesh>
  );
}

/* ------------------------------------------------------------------------ */
/* main                                                                      */
/* ------------------------------------------------------------------------ */

export function RingScene({
  payload,
  frameRef,
  renderMode,
  lineWidth = 2,
  onHover,
  hoveredIndex,
  onSelect,
  stochasticity = 0,
}: Props) {
  const lineRefs = useRef<Array<LineColorBearer>>([]);
  const tmp = useMemo(() => new THREE.Color(), []);
  const { size } = useThree();

  const baseColors = useMemo(
    () => payload.neurons.map((n) => baseColorFor(n.cell_type)),
    [payload],
  );

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

  // Keep Line2 resolution in sync with the canvas (lines mode only).
  useEffect(() => {
    if (renderMode !== "lines") return;
    for (const obj of lineRefs.current) {
      if (!obj) continue;
      const mat = (obj as unknown as { material?: { resolution?: THREE.Vector2 } }).material;
      if (mat && "resolution" in mat && mat.resolution) {
        mat.resolution.set(size.width, size.height);
      }
    }
  }, [size.width, size.height, payload, renderMode]);

  // Lines-mode per-frame color update (unchanged from before -- the
  // per-neuron <Line> components are cheap to mount).
  useFrame(() => {
    if (renderMode !== "lines") return;
    const f = frameRef.current;
    const t = Math.max(
      0,
      Math.min(payload.metadata.n_frames - 1, Math.round(Number.isFinite(f) ? f : 0)),
    );
    const row = payload.rates[t];
    const stimRow = payload.stim_signal?.[t];
    for (let i = 0; i < lineRefs.current.length; i++) {
      const obj = lineRefs.current[i];
      if (!obj) continue;
      let v = row[i] / maxRate;
      if (stochasticity > 0) {
        const u = Math.max(1e-6, Math.random());
        const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
        v = Math.max(0, Math.min(1.2, v + stochasticity * g * 0.4));
      }
      activityColor(v, baseColors[i], tmp);
      if (stimRow) {
        const inputLevel = stimRow[i] / maxStim;
        if (inputLevel > 0.05) blendInput(tmp, inputLevel, tmp);
      }
      const mat = (obj as unknown as { material?: { color?: THREE.Color } }).material;
      if (mat && mat.color) {
        mat.color.copy(tmp);
      }
    }
  });

  const registerLineRef = (i: number, obj: LineColorBearer) => {
    lineRefs.current[i] = obj;
  };

  const handlePointerOver = (i: number) => (e: { clientX: number; clientY: number }) => {
    const f = frameRef.current;
    const t = Math.max(
      0,
      Math.min(payload.metadata.n_frames - 1, Math.round(Number.isFinite(f) ? f : 0)),
    );
    onHover?.({
      neuronIndex: i,
      rate: payload.rates[t][i],
      screen: { x: e.clientX, y: e.clientY },
    });
  };
  const handlePointerOut = () => onHover?.(null);

  // DTI datasets: render explicit edges between region centroids. The
  // skeletons-as-stars are too austere to show what's actually connected
  // to what -- the edges carry the structural information.
  const isDTI = payload.metadata.dataset_id.startsWith("dti");
  const dtiEdges = useMemo(() => {
    if (!isDTI) return null;
    const model = (payload as unknown as { model?: { weights: number[][]; global_gain: number } }).model;
    if (!model) return null;
    const W = model.weights;
    const gain = model.global_gain;
    const points: number[] = [];
    const N = payload.neurons.length;
    const flat: number[] = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const w = Math.abs(W[i][j]) * gain;
      if (w > 0) flat.push(w);
    }
    flat.sort((a, b) => a - b);
    const threshold = flat[Math.floor(flat.length * 0.7)] ?? 0;
    for (let i = 0; i < N; i++) {
      const sa = payload.neurons[i].soma;
      if (!sa) continue;
      for (let j = i + 1; j < N; j++) {
        const sb = payload.neurons[j].soma;
        if (!sb) continue;
        const w = Math.abs(W[i][j]) * gain;
        if (w <= threshold) continue;
        points.push(sa[0], sa[1], sa[2], sb[0], sb[1], sb[2]);
      }
    }
    return new Float32Array(points);
  }, [payload, isDTI]);

  // DTI: rotate the whole brain so MNI's superior axis (z) becomes
  // world up (y), and MNI's anterior axis (y) becomes world -z (toward
  // the viewer / screen-left).
  const groupRotation: [number, number, number] = isDTI
    ? [-Math.PI / 2, 0, Math.PI / 2]
    : [0, 0, 0];

  return (
    <group rotation={groupRotation}>
      {dtiEdges && dtiEdges.length > 0 && (
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[dtiEdges, 3]}
              count={dtiEdges.length / 3}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial
            color="#3b6ec9"
            transparent
            opacity={0.32}
            depthWrite={false}
          />
        </lineSegments>
      )}
      {renderMode === "lines"
        ? payload.neurons.map((n, i) => (
            <NeuronLine
              key={`l-${n.id}`}
              neuron={n}
              index={i}
              registerRef={registerLineRef}
              lineWidth={lineWidth}
              onPointerOver={handlePointerOver(i)}
              onPointerOut={handlePointerOut}
              onClick={onSelect ? () => onSelect(i) : undefined}
              isHovered={hoveredIndex === i}
            />
          ))
        : (
          <TubesView
            payload={payload}
            frameRef={frameRef}
            baseColors={baseColors}
            maxRate={maxRate}
            maxStim={maxStim}
            stochasticity={stochasticity}
            hoveredIndex={hoveredIndex ?? null}
            onHover={onHover}
            onSelect={onSelect}
          />
        )}
    </group>
  );
}
