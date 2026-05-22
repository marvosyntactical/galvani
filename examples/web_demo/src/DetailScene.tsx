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
        baseColor={baseColorFor(payload.neurons[neuronIndex].cell_type)}
        index={neuronIndex}
        payload={payload}
        frameRef={frameRef}
        maxRate={maxRate}
        maxStim={maxStim}
        opacity={1.0}
        emissiveBoost={0.5}
      />
      {companions.map((c) => (
        <NeuronMesh
          key={`c-${c.neuronIndex}`}
          detail={c.detail}
          baseColor={baseColorFor(payload.neurons[c.neuronIndex].cell_type)}
          index={c.neuronIndex}
          payload={payload}
          frameRef={frameRef}
          maxRate={maxRate}
          maxStim={maxStim}
          opacity={0.18}
          emissiveBoost={0.05}
        />
      ))}
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
}: MeshProps) {
  const cylRef = useRef<THREE.InstancedMesh>(null);
  const sphRef = useRef<THREE.InstancedMesh>(null);
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
    </group>
  );
}
