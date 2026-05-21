/**
 * Color helpers for activity rendering.
 *
 * Two functions:
 *   - `colormap(v)`: standalone sequential scale (dark blue -> yellow).
 *   - `activityColor(v, base)`: modulates a cell-type baseline color
 *      between a dim version (low activity) and a hot saturated bloom
 *      (high activity). Preserves cell-type identity through hue.
 */

import * as THREE from "three";

const STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [0.05, 0.05, 0.1]],
  [0.25, [0.12, 0.18, 0.5]],
  [0.5, [0.12, 0.43, 0.92]],
  [0.75, [0.6, 0.78, 0.3]],
  [1.0, [1.0, 0.92, 0.22]],
];

const _color = new THREE.Color();
const _hot = new THREE.Color("#fff39c");

export function colormap(value: number, out?: THREE.Color): THREE.Color {
  const target = out ?? _color;
  const v = Math.max(0, Math.min(1, value));
  for (let i = 0; i < STOPS.length - 1; i++) {
    const [t0, c0] = STOPS[i];
    const [t1, c1] = STOPS[i + 1];
    if (v <= t1) {
      const f = (v - t0) / (t1 - t0);
      target.setRGB(
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f,
      );
      return target;
    }
  }
  const last = STOPS[STOPS.length - 1][1];
  target.setRGB(last[0], last[1], last[2]);
  return target;
}

/**
 * Modulate a cell-type baseline color by activity.
 *   v=0   -> ~20% baseColor brightness (cell still visible but dim)
 *   v=0.5 -> baseColor at full brightness
 *   v=1   -> baseColor blended ~60% toward a hot saturated white-yellow
 */
export function activityColor(
  value: number,
  base: THREE.Color,
  out?: THREE.Color,
): THREE.Color {
  const target = out ?? _color;
  const v = Math.max(0, Math.min(1, value));
  const brightness = 0.2 + 0.8 * v;
  target.copy(base).multiplyScalar(brightness);
  if (v > 0.5) {
    const hotMix = (v - 0.5) * 1.2; // 0..0.6
    target.lerp(_hot, hotMix);
  }
  return target;
}
