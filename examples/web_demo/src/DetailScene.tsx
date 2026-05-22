import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { NeuronDetail } from "./detailLoader";
import type { Payload } from "./payload";
import { activityColor, blendInput } from "./colormap";
import { baseColorFor } from "./cellTypes";

export interface CompanionNeuron {
  detail: NeuronDetail;
  neuronIndex: number;
  /** Direction relative to the focused neuron. */
  role: "incoming" | "outgoing";
  /** Whether this companion receives external stimulus at any frame. */
  isStimInput: boolean;
}

interface Props {
  detail: NeuronDetail;
  payload: Payload;
  neuronIndex: number;
  frameRef: { current: number };
  companions?: CompanionNeuron[];
}

const UNIT_CYL = new THREE.CylinderGeometry(1, 1, 1, 12, 1, false);
UNIT_CYL.translate(0, 0.5, 0);
const _Y = new THREE.Vector3(0, 1, 0);

const SPHERE_GEOM = new THREE.SphereGeometry(1, 12, 8);

/* Subdued tints for companions, blended with the cell-type baseColor. */
const INCOMING_TINT = new THREE.Color("#4dd6ff"); // cyan
const OUTGOING_TINT = new THREE.Color("#7ee87a"); // green
const FOCUS_TINT = new THREE.Color("#ffd95c"); // gold

/**
 * Renders one focused neuron at full SWC resolution, optionally surrounded
 * by its top-k incoming + outgoing companions. The focus neuron is gold;
 * incoming-to-focus get a cyan tint; outgoing-from-focus get a green tint.
 * Per-frame activity modulates brightness for every neuron. Stim-input
 * companions get an extra magenta halo.
 */
export function DetailScene({
  detail,
  payload,
  neuronIndex,
  frameRef,
  companions = [],
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

  return (
    <group>
      <NeuronMesh
        detail={detail}
        baseColor={mixColors(baseColorFor(payload.neurons[neuronIndex].cell_type), FOCUS_TINT, 0.5)}
        index={neuronIndex}
        payload={payload}
        frameRef={frameRef}
        maxRate={maxRate}
        maxStim={maxStim}
        opacity={1.0}
        emissiveBoost={0.5}
      />
      {companions.map((c) => {
        const tint = c.role === "incoming" ? INCOMING_TINT : OUTGOING_TINT;
        const base = mixColors(
          baseColorFor(payload.neurons[c.neuronIndex].cell_type),
          tint,
          0.6,
        );
        return (
          <NeuronMesh
            key={`${c.role}-${c.neuronIndex}`}
            detail={c.detail}
            baseColor={base}
            index={c.neuronIndex}
            payload={payload}
            frameRef={frameRef}
            maxRate={maxRate}
            maxStim={maxStim}
            opacity={0.35}
            emissiveBoost={0.15}
            stimHalo={c.isStimInput}
          />
        );
      })}
    </group>
  );
}

function mixColors(a: THREE.Color, b: THREE.Color, t: number): THREE.Color {
  return new THREE.Color().copy(a).lerp(b, t);
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
  stimHalo?: boolean;
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
  stimHalo,
}: MeshProps) {
  const cylRef = useRef<THREE.InstancedMesh>(null);
  const sphRef = useRef<THREE.InstancedMesh>(null);
  const haloRef = useRef<THREE.Mesh>(null);
  const { positions, radii, edges } = detail;
  const nNodes = detail.n_nodes;
  const nSegs = edges.length / 2;

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

  // Compute the neuron's centroid for halo placement.
  const centroid = useMemo(() => {
    let cx = 0,
      cy = 0,
      cz = 0;
    for (let i = 0; i < nNodes; i++) {
      cx += positions[i * 3];
      cy += positions[i * 3 + 1];
      cz += positions[i * 3 + 2];
    }
    return new THREE.Vector3(cx / nNodes, cy / nNodes, cz / nNodes);
  }, [positions, nNodes]);

  const tmp = useMemo(() => new THREE.Color(), []);
  useFrame(() => {
    const f = frameRef.current;
    const t = Math.max(
      0,
      Math.min(payload.metadata.n_frames - 1, Math.round(Number.isFinite(f) ? f : 0)),
    );
    const v = payload.rates[t][index] / maxRate;
    activityColor(v, baseColor, tmp);
    const stimRow = payload.stim_signal?.[t];
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
      {stimHalo && (
        <mesh ref={haloRef} position={centroid}>
          <sphereGeometry args={[0.3, 16, 12]} />
          <meshBasicMaterial
            color="#ff5cb0"
            transparent
            opacity={0.18}
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  );
}
