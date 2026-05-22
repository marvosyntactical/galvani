/**
 * Morphology helpers for the multi-compartment HH solver.
 *
 * The per-neuron detail JSON we ship is at SWC-node resolution: ~10 nodes
 * per micrometre of skeleton, which is way too fine for the cable
 * equation. Two reasons:
 *
 *  1. Numerical stiffness. At 0.1 μm compartment length the axial
 *     time constant is in the picoseconds — forward Euler at any sane dt
 *     diverges immediately.
 *  2. Compute. A 7 000-compartment forward-Euler at dt = 25 μs over 2 s is
 *     ~600 M float ops per neuron; we want to run a handful of these in
 *     parallel in workers.
 *
 * So we aggregate the SWC nodes into ~1-5 μm "solver compartments" before
 * running the HH math, and map results back to the fine nodes for
 * visualisation. `buildSolverCompartments` does the aggregation.
 *
 * The AIS finder picks the compartment ~20 μm distal of the soma along
 * the longest descendant path — a reasonable proxy for the spike-
 * initiation zone in the absence of any AIS labelling in the EM data.
 */

import type { NeuronDetail } from "../detailLoader";

/** Demo-space scale factor used when normalising raw nm → demo units. */
export const SCALE_NM_PER_UM = 1000;

export interface SolverCompartment {
  /** Index in the SWC-node list of this compartment's "anchor" node. */
  anchorNode: number;
  /** All SWC node indices grouped into this compartment. */
  members: number[];
  /** Total Euclidean length in raw nm (sum of edge lengths inside). */
  lengthNm: number;
  /** Effective cylindrical radius in raw nm (radius-weighted mean over
   *  members; cylinder approximation). */
  radiusNm: number;
  /** Parent compartment index (-1 for compartments that contain a root). */
  parent: number;
}

export interface SolverMorphology {
  nCompartments: number;
  compartments: SolverCompartment[];
  /** Inverse map: SWC node index → solver compartment index. */
  nodeToCompartment: Int32Array;
  /** Compartment containing the soma SWC node. */
  somaCompartment: number;
  /** Compartment chosen as the AIS for spike-initiation channel densities. */
  aisCompartment: number;
}

/**
 * Greedy walk from the soma: merge SWC nodes into solver compartments of
 * roughly `targetLengthUm` (μm) along the parent chain. Branch points end
 * a compartment so each compartment is a single unbranched segment.
 *
 * The returned compartment tree is rooted at the soma even when the
 * underlying SWC has disconnected fragments — we ignore fragments not
 * reachable from the soma via the parent graph.
 */
export function buildSolverCompartments(
  detail: NeuronDetail,
  targetLengthUm: number = 2.0,
): SolverMorphology {
  const N = detail.n_nodes;
  const parents = detail.parents ?? defaultParents(detail);
  const compLen = detail.compartment_length_nm ?? new Float32Array(N);
  const radii = detail.radii;
  const somaIdx = detail.soma_idx ?? 0;

  // Build an undirected neighbour list so we can walk outward from the soma
  // regardless of which direction the SWC tree was rooted (the SWC root is
  // not necessarily the soma).
  const neighbours: number[][] = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) {
    const p = parents[i];
    if (p >= 0) {
      neighbours[i].push(p);
      neighbours[p].push(i);
    }
  }

  // Edge length lookup: from i to its parent it's compLen[i]. Symmetric:
  // when walking from a parent to a child the length is compLen[child].
  const edgeLen = (a: number, b: number): number => {
    if (parents[a] === b) return compLen[a];
    if (parents[b] === a) return compLen[b];
    return 0; // disconnected — treat as zero-length bridge
  };

  // BFS from soma. We grow compartments greedily along each branch: keep
  // appending the next node into the current compartment until either
  // (a) cumulative length passes targetLengthUm, or (b) the node has
  // multiple unvisited neighbours (branch point), or (c) the node has no
  // unvisited neighbours (leaf).
  const targetNm = targetLengthUm * SCALE_NM_PER_UM;
  const visited = new Uint8Array(N);
  const nodeToCompartment = new Int32Array(N).fill(-1);
  const compartments: SolverCompartment[] = [];

  // Each queue entry: continue building a compartment starting at `start`
  // (whose parent compartment is `parentComp`).
  type Pending = { start: number; parentComp: number };
  const queue: Pending[] = [{ start: somaIdx, parentComp: -1 }];
  visited[somaIdx] = 1;

  while (queue.length > 0) {
    const { start, parentComp } = queue.shift()!;

    const members: number[] = [start];
    let cumNm = 0;
    let radiusAccum = 0;
    let cursor = start;
    radiusAccum += radii[cursor];

    // Walk linearly. Stop on branch / leaf / target-length.
    while (true) {
      const nextCandidates = neighbours[cursor].filter((j) => !visited[j]);
      if (nextCandidates.length === 0) {
        // Leaf: end the compartment.
        break;
      }
      if (nextCandidates.length > 1) {
        // Branch point: end the current compartment, queue every branch.
        break;
      }
      const nxt = nextCandidates[0];
      const step = edgeLen(cursor, nxt);
      if (members.length > 1 && cumNm + step > targetNm) {
        // Compartment full; start a new one from nxt (don't add it here).
        break;
      }
      members.push(nxt);
      cumNm += step;
      radiusAccum += radii[nxt];
      visited[nxt] = 1;
      cursor = nxt;
    }

    const compIdx = compartments.length;
    const radiusMean = radiusAccum / members.length;
    compartments.push({
      anchorNode: start,
      members,
      // A compartment of length 0 (one-node compartment) gets a tiny floor
      // so axial-conductance division is well-defined.
      lengthNm: Math.max(cumNm, 1),
      radiusNm: radiusMean,
      parent: parentComp,
    });
    for (const m of members) nodeToCompartment[m] = compIdx;

    // Queue continuations: any unvisited neighbour of the terminal cursor
    // starts a new compartment whose parent is the one we just built.
    for (const j of neighbours[cursor]) {
      if (!visited[j]) {
        visited[j] = 1;
        queue.push({ start: j, parentComp: compIdx });
      }
    }
  }

  // Map SWC soma → its solver compartment.
  const somaCompartment = nodeToCompartment[somaIdx] >= 0
    ? nodeToCompartment[somaIdx]
    : 0;

  const aisCompartment = findAis(compartments, somaCompartment, 20.0);

  return {
    nCompartments: compartments.length,
    compartments,
    nodeToCompartment,
    somaCompartment,
    aisCompartment,
  };
}

/**
 * Walk the solver-compartment tree downstream from the soma and return
 * the compartment whose cumulative distance from the soma is closest to
 * `targetDistanceUm` *along the longest descendant path* (the one most
 * likely to be the axon). Falls back to the soma itself if the tree is
 * shorter than the target.
 */
export function findAis(
  compartments: SolverCompartment[],
  somaCompartment: number,
  targetDistanceUm: number,
): number {
  if (compartments.length <= 1) return somaCompartment;

  const N = compartments.length;
  const children: number[][] = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) {
    const p = compartments[i].parent;
    if (p >= 0) children[p].push(i);
  }

  // DFS from the soma collecting distance-from-soma and a predecessor link
  // along the longest-descendant path. We score each descendant by max
  // distance reachable from it, so the longest-branch path beats short
  // dendrites.
  const dist = new Float32Array(N).fill(-1);
  const longestThrough = new Float32Array(N).fill(0);

  function fillLongest(i: number): number {
    if (children[i].length === 0) {
      longestThrough[i] = 0;
      return 0;
    }
    let best = 0;
    for (const c of children[i]) {
      const branch = compartments[c].lengthNm + fillLongest(c);
      if (branch > best) best = branch;
    }
    longestThrough[i] = best;
    return best;
  }
  fillLongest(somaCompartment);

  // Walk along the longest path from the soma, accumulating distance.
  // Pick the compartment whose distance is closest to the target.
  const targetNm = targetDistanceUm * SCALE_NM_PER_UM;
  let bestComp = somaCompartment;
  let bestDelta = Infinity;
  let cursor = somaCompartment;
  let dAccum = 0;
  dist[cursor] = 0;
  for (let safety = 0; safety < N; safety++) {
    const delta = Math.abs(dAccum - targetNm);
    if (delta < bestDelta) {
      bestDelta = delta;
      bestComp = cursor;
    }
    const cs = children[cursor];
    if (cs.length === 0) break;
    // Pick the child whose downstream extent is largest (longest branch).
    let nextChild = cs[0];
    let nextScore = compartments[cs[0]].lengthNm + longestThrough[cs[0]];
    for (let k = 1; k < cs.length; k++) {
      const s = compartments[cs[k]].lengthNm + longestThrough[cs[k]];
      if (s > nextScore) {
        nextScore = s;
        nextChild = cs[k];
      }
    }
    dAccum += compartments[nextChild].lengthNm;
    dist[nextChild] = dAccum;
    cursor = nextChild;
  }
  return bestComp;
}

/** Fallback parent list when the detail JSON is too old to ship one. */
function defaultParents(detail: NeuronDetail): Int32Array {
  const N = detail.n_nodes;
  const parents = new Int32Array(N).fill(-1);
  const edges = detail.edges;
  for (let i = 0; i < edges.length; i += 2) {
    const parent = edges[i];
    const child = edges[i + 1];
    parents[child] = parent;
  }
  return parents;
}
