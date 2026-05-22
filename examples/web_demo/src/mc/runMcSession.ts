/**
 * Orchestrator for a multi-compartment biophysical session.
 *
 * Coordinates the four stages of MC.md:
 *   - load per-neuron detail JSON for every neighbor
 *   - aggregate SWC nodes into solver compartments
 *   - build presynaptic spike trains from the outer-sim payload
 *   - route them onto each target neuron's compartments
 *   - spawn one worker per target, collect voltage traces
 *
 * Returns once every worker has reported its `done` message. Progress is
 * aggregated across all workers and reported through `onProgress` (a
 * single fraction in [0, 1] over the whole session). Cancellation is
 * supported via the optional AbortSignal — any pending worker is
 * terminated; already-completed neurons are kept in the partial result.
 */

import type { Payload, ModelId } from "../payload";
import type { NeuronDetail } from "../detailLoader";
import { buildSolverCompartments, type SolverMorphology } from "./morphology";
import {
  buildConductanceSchedule,
  type SynapseRoutingConfig,
} from "./synapseRouter";
import type { HhResult, HhRunConfig } from "./hhSolver";

export interface BioSessionRequest {
  payload: Payload;
  outerModelId: ModelId;
  /** Neurons (by outer-payload index) to run multi-compartment HH on.
   *  Typically the focused neuron + its visible top-k neighbors. */
  targetIndices: number[];
  /** Loader for the per-neuron skeleton detail JSON. */
  loadDetail: (neuronIdx: number) => Promise<NeuronDetail>;
  /** Where in the outer scenario this bio window is anchored, in ms.
   *  Bio time 0 corresponds to outer time `anchorOuterMs`. Default 0. */
  anchorOuterMs?: number;
  /** Length of the bio simulation in BIOLOGICAL milliseconds. Short
   *  (default 200) so the user can perceive ms-scale spike dynamics
   *  when the result is stretched across the scrubber. */
  bioDurationMs?: number;
  /** Integration step in ms. Default 25 μs. */
  dtMs?: number;
  /** Output frame stride (integration steps per recorded frame). */
  frameStride?: number;
  routing?: SynapseRoutingConfig;
  /** Compartment target length (μm). Smaller = finer voltage map but
   *  more compute. Defaults to 2 μm. */
  compartmentTargetUm?: number;
  /** Progress callback, fraction in [0, 1] across the whole neighborhood. */
  onProgress?: (fraction: number) => void;
  /** Abort signal — terminates workers mid-run. */
  signal?: AbortSignal;
}

export interface BioNeuronResult {
  /** Outer-payload neuron index. */
  neuronIdx: number;
  morph: SolverMorphology;
  hh: HhResult;
}

export interface BioSession {
  /** One entry per target neuron, keyed by outer-payload index. */
  results: Map<number, BioNeuronResult>;
  /** Wall-clock ms the session took. */
  elapsedMs: number;
  /** Total compartments across all targets — useful for badges. */
  totalCompartments: number;
  /** Outer-scenario time at which the bio window is anchored (ms). */
  anchorOuterMs: number;
  /** Bio-window length in biological ms. */
  bioDurationMs: number;
}

/** Eager worker factory. Bundlers (Vite) handle `?worker` to produce a
 *  Worker constructor whose URL is bundled separately. */
async function spawnWorker(): Promise<Worker> {
  // Lazy-import the worker via the `?worker` query so the worker bundle
  // is only built when this orchestrator is actually used.
  const mod = await import("./hhWorker?worker");
  return new mod.default();
}

/** Run one target neuron in its own worker; returns when the worker
 *  finishes (`done`) or throws on error / abort. */
function runOneInWorker(
  id: number,
  morph: SolverMorphology,
  config: HhRunConfig,
  schedule: ReturnType<typeof buildConductanceSchedule>,
  onProgress: (frac: number) => void,
  signal?: AbortSignal,
): Promise<HhResult> {
  return new Promise<HhResult>(async (resolve, reject) => {
    let worker: Worker;
    try {
      worker = await spawnWorker();
    } catch (err) {
      reject(err);
      return;
    }
    const abortHandler = () => {
      worker.terminate();
      reject(new DOMException("aborted", "AbortError"));
    };
    if (signal) {
      if (signal.aborted) {
        abortHandler();
        return;
      }
      signal.addEventListener("abort", abortHandler, { once: true });
    }

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data as
        | { type: "progress"; id: number; fraction: number }
        | { type: "done"; id: number; result: HhResult }
        | { type: "error"; id: number; message: string };
      if (msg.id !== id) return;
      if (msg.type === "progress") {
        onProgress(msg.fraction);
      } else if (msg.type === "done") {
        signal?.removeEventListener("abort", abortHandler);
        worker.terminate();
        resolve(msg.result);
      } else if (msg.type === "error") {
        signal?.removeEventListener("abort", abortHandler);
        worker.terminate();
        reject(new Error(msg.message));
      }
    };
    worker.onerror = (e) => {
      signal?.removeEventListener("abort", abortHandler);
      worker.terminate();
      reject(new Error(e.message));
    };

    worker.postMessage({
      type: "run",
      id,
      morph,
      synEvents: [],
      config,
      gSchedule: schedule,
    });
  });
}

/**
 * Drive the full session. Resolves once every target has either finished
 * or errored. The caller is expected to surface partial results on abort.
 */
export async function runBioSession(req: BioSessionRequest): Promise<BioSession> {
  const start = performance.now();
  const anchorOuterMs = req.anchorOuterMs ?? 0;
  const bioDurationMs = req.bioDurationMs ?? 200;
  const dtMs = req.dtMs ?? 0.025;
  const frameStride = req.frameStride ?? 40;
  const compartmentTargetUm = req.compartmentTargetUm ?? 2.0;
  const config: HhRunConfig = { durationMs: bioDurationMs, dtMs, frameStride };

  // 1. Load per-target morphologies in parallel.
  const morphs = new Map<number, SolverMorphology>();
  const details = await Promise.all(
    req.targetIndices.map((idx) => req.loadDetail(idx).then((d) => [idx, d] as const)),
  );
  let totalCompartments = 0;
  for (const [idx, detail] of details) {
    const m = buildSolverCompartments(detail, compartmentTargetUm);
    morphs.set(idx, m);
    totalCompartments += m.nCompartments;
  }

  // 2. Per-target smooth conductance schedule. C+D design: outside-the-
  //    neighborhood drive enters as a slow per-compartment conductance
  //    derived from the outer-sim rates around `anchorOuterMs`. No
  //    Poisson sampling, no discrete events.
  const progressByTarget = new Map<number, number>();
  for (const idx of req.targetIndices) progressByTarget.set(idx, 0);
  const reportProgress = () => {
    if (!req.onProgress) return;
    let sum = 0;
    for (const f of progressByTarget.values()) sum += f;
    req.onProgress(sum / Math.max(1, progressByTarget.size));
  };

  const workerJobs = req.targetIndices.map(async (idx) => {
    const morph = morphs.get(idx)!;
    const schedule = buildConductanceSchedule(
      req.payload,
      idx,
      morph,
      anchorOuterMs,
      bioDurationMs,
      /* bioFrameStrideMs */ 1,
      req.routing,
    );
    const hh = await runOneInWorker(
      idx,
      morph,
      config,
      schedule,
      (f) => {
        progressByTarget.set(idx, f);
        reportProgress();
      },
      req.signal,
    );
    progressByTarget.set(idx, 1);
    reportProgress();
    return { neuronIdx: idx, morph, hh } satisfies BioNeuronResult;
  });

  // 3. Collect results. Use allSettled so one failure doesn't drop the
  //    whole batch — caller can decide whether to surface partial state.
  const settled = await Promise.allSettled(workerJobs);
  const results = new Map<number, BioNeuronResult>();
  for (const s of settled) {
    if (s.status === "fulfilled") {
      results.set(s.value.neuronIdx, s.value);
    }
  }

  return {
    results,
    elapsedMs: performance.now() - start,
    totalCompartments,
    anchorOuterMs,
    bioDurationMs,
  };
}
