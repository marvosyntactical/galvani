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

function segmentMatrix(
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  radius: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const dir = new THREE.Vector3().subVectors(p1, p0);
  const len = dir.length();
  if (len < 1e-9) {
    out.makeScale(0, 0, 0);
    out.setPosition(p0);
    return out;
  }
  dir.divideScalar(len);
  const q = new THREE.Quaternion().setFromUnitVectors(_Y, dir);
  out.compose(p0, q, new THREE.Vector3(radius, len, radius));
  return out;
}

/* ------------------------------------------------------------------------ */
/* per-neuron components                                                     */
/* ------------------------------------------------------------------------ */

type ColorBearer = { material: unknown } | null;

interface NeuronProps {
  neuron: PayloadNeuron;
  index: number;
  baseColor: THREE.Color;
  registerRef: (i: number, obj: ColorBearer) => void;
  lineWidth: number;
  onPointerOver?: (e: { clientX: number; clientY: number }) => void;
  onPointerOut?: () => void;
  isHovered: boolean;
}

function NeuronLine({
  neuron,
  index,
  registerRef,
  lineWidth,
  onPointerOver,
  onPointerOut,
  isHovered,
}: NeuronProps) {
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
      ref={(el) => {
        registerRef(index, el ? (el as unknown as ColorBearer) : null);
      }}
    />
  );
}

function NeuronTubes({
  neuron,
  index,
  baseColor,
  registerRef,
  onPointerOver,
  onPointerOut,
  isHovered,
}: NeuronProps) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const nSegments = neuron.segments.length / 6;

  // Precompute instance matrices.
  useEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const mat = new THREE.Matrix4();
    const p0 = new THREE.Vector3();
    const p1 = new THREE.Vector3();
    const segs = neuron.segments;
    const radii = neuron.radii;
    for (let s = 0; s < nSegments; s++) {
      const base = s * 6;
      p0.set(segs[base + 0], segs[base + 1], segs[base + 2]);
      p1.set(segs[base + 3], segs[base + 4], segs[base + 5]);
      const r = Math.max(0.005, radii[s] ?? 0.02);
      segmentMatrix(p0, p1, r, mat);
      mesh.setMatrixAt(s, mat);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, [neuron.segments, neuron.radii, nSegments]);

  useEffect(() => {
    if (ref.current) {
      registerRef(index, ref.current);
    }
    return () => registerRef(index, null);
  }, [index, registerRef]);

  return (
    <instancedMesh
      ref={ref}
      args={[UNIT_CYLINDER, undefined, nSegments]}
      onPointerOver={(e) => {
        e.stopPropagation();
        onPointerOver?.({ clientX: e.clientX, clientY: e.clientY });
      }}
      onPointerOut={onPointerOut}
    >
      <meshStandardMaterial
        color={baseColor}
        roughness={0.6}
        metalness={0.05}
        emissive={isHovered ? baseColor : new THREE.Color(0, 0, 0)}
        emissiveIntensity={isHovered ? 0.4 : 0.0}
      />
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
}: Props) {
  const refs = useRef<Array<ColorBearer>>([]);
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

  // Keep Line2 resolution in sync with the canvas.
  useEffect(() => {
    if (renderMode !== "lines") return;
    for (const obj of refs.current) {
      if (!obj) continue;
      const mat = (obj as unknown as { material?: { resolution?: THREE.Vector2 } }).material;
      if (mat && "resolution" in mat && mat.resolution) {
        mat.resolution.set(size.width, size.height);
      }
    }
  }, [size.width, size.height, payload, renderMode]);

  useFrame(() => {
    const f = frameRef.current;
    const t = Math.max(
      0,
      Math.min(payload.metadata.n_frames - 1, Math.round(Number.isFinite(f) ? f : 0)),
    );
    const row = payload.rates[t];
    const stimRow = payload.stim_signal?.[t];
    for (let i = 0; i < refs.current.length; i++) {
      const obj = refs.current[i];
      if (!obj) continue;
      const v = row[i] / maxRate;
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

  const registerRef = (i: number, obj: typeof refs.current[number]) => {
    refs.current[i] = obj;
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

  return (
    <group>
      {payload.neurons.map((n, i) => {
        const common = {
          neuron: n,
          index: i,
          baseColor: baseColors[i],
          registerRef,
          lineWidth,
          onPointerOver: handlePointerOver(i),
          onPointerOut: handlePointerOut,
          isHovered: hoveredIndex === i,
        };
        return renderMode === "lines" ? (
          <NeuronLine key={`l-${n.id}`} {...common} />
        ) : (
          <NeuronTubes key={`t-${n.id}`} {...common} />
        );
      })}
    </group>
  );
}
