/** Per-neuron detail file: full SWC skeleton, normalised to the dataset's bbox. */

export interface NeuronDetail {
  body_id: number;
  n_nodes: number;
  /** Flat (x,y,z) coordinates, length = 3 * n_nodes. */
  positions: number[];
  /** Per-node radius, length = n_nodes. */
  radii: number[];
  /** Flat (parent_idx, child_idx) edges, length = 2 * (n_nodes - 1). */
  edges: number[];
}

const cache = new Map<string, NeuronDetail>();

export async function loadNeuronDetail(
  datasetId: string,
  bodyId: number,
  baseUrl: string,
): Promise<NeuronDetail> {
  const key = `${datasetId}/${bodyId}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const url = `${baseUrl.replace(/\/$/, "")}/neurons/${datasetId}/${bodyId}.json`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load ${url}: ${r.status}`);
  const d = (await r.json()) as NeuronDetail;
  cache.set(key, d);
  return d;
}
