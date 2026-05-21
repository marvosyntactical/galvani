/** Runtime types for v2 JSON payloads written by galvani.viz.payload. */

export interface PayloadNeuron {
  id: number;
  cell_type: string;
  hemisphere: string | null;
  angle: number | null;
  soma: [number, number, number] | null;
  /** Flat (x0,y0,z0, x1,y1,z1, ...) line segments, normalised to ~10-unit cube. */
  segments: number[];
  /** Per-segment radius (length n_segments). Same normalised units as `segments`. */
  radii: number[];
}

export interface HyperParams {
  global_gain: number;
  symmetrize: boolean;
  weight_heuristic: string;
  activation: string;
  dt_sim: number;
  tau_default_ms: number;
  stimulus?: Record<string, unknown>;
  apl_ablated?: boolean;
  n_kc_subset?: number;
  kc_drive_fraction?: number;
  [k: string]: unknown;
}

export interface PayloadMetadata {
  schema_version: number;
  dataset_id: string;
  scenario_id: string;
  scenario_label: string;
  description: string;
  hyperparams: HyperParams;
  dataset_version: string;
  n_neurons: number;
  n_frames: number;
  dt_sim: number;
  duration: number;
}

export interface Payload {
  metadata: PayloadMetadata;
  bbox: {
    min: [number, number, number];
    max: [number, number, number];
    center: [number, number, number];
    scale: number;
  };
  neurons: PayloadNeuron[];
  times: number[];
  /** (n_frames, n_neurons) */
  rates: number[][];
}

export interface ManifestScenario {
  id: string;
  label: string;
  file: string;
  description: string;
}

export interface ManifestDataset {
  id: string;
  label: string;
  summary: string;
  biology: string;
  scenarios: ManifestScenario[];
}

export interface Manifest {
  schema_version: number;
  datasets: ManifestDataset[];
}

export async function loadManifest(url = "/manifest.json"): Promise<Manifest> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load manifest: ${r.status}`);
  return (await r.json()) as Manifest;
}

export async function loadPayload(url: string): Promise<Payload> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  const payload = (await response.json()) as Payload;
  if (payload.metadata.schema_version !== 2) {
    throw new Error(
      `Unsupported payload schema_version ${payload.metadata.schema_version} (expected 2)`,
    );
  }
  return payload;
}
