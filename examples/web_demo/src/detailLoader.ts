/** Per-neuron detail file: full SWC skeleton, normalised to the dataset's bbox. */

export interface NeuronDetail {
  body_id: number;
  n_nodes: number;
  /** Flat (x,y,z) coordinates in demo space, length = 3 * n_nodes. */
  positions: number[];
  /** Per-node radius in demo space, length = n_nodes. */
  radii: number[];
  /** Flat (parent_idx, child_idx) edges, length = 2 * (n_nodes - 1).
   *  Redundant with `parents` but cheaper to iterate when drawing lines. */
  edges: number[];
  /** Parent compartment index per node (-1 for the soma / disconnected
   *  roots). Forms the spanning tree the multi-compartment HH solver
   *  uses for axial coupling. Newer payloads only — guard with a presence
   *  check on the consumer side. */
  parents?: number[];
  /** Euclidean distance in raw nm from each node to its parent (0 for
   *  roots). Used to compute axial conductance `g_axial = π r² / (R_a · L)`
   *  in the bio sim. Newer payloads only. */
  compartment_length_nm?: number[];
  /** Index of the inferred soma (largest-radius node). Newer payloads
   *  only; older payloads fall back to compartment 0. */
  soma_idx?: number;
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
