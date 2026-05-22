/**
 * Web Worker entry point for the multi-compartment HH bio sim.
 *
 * Two request shapes:
 *
 *   { type: "run" }       — single-neuron run via runMcHH. Used by the
 *                           validation tests and as a fallback.
 *   { type: "runCoupled" } — multi-neuron run via runCoupledMcHH, with
 *                            within-neighborhood spike coupling. This is
 *                            what the orchestrator uses now.
 *
 * Replies for "runCoupled" come back as a single { type: "done", results }
 * carrying every target's HhResult, since we need consistent V^n across
 * the neighborhood — splitting across workers would lose the coupling.
 */

import {
  runCoupledMcHH,
  runMcHH,
  type CoupledIntraEdge,
  type CoupledTarget,
  type GScheduleInput,
  type HhResult,
  type HhRunConfig,
  type SynEvent,
} from "./hhSolver";
import type { SolverMorphology } from "./morphology";

interface RunRequest {
  type: "run";
  id: number;
  morph: SolverMorphology;
  synEvents: SynEvent[];
  config: HhRunConfig;
  iExt?: { data: Float32Array; nFrames: number };
  /** Smooth conductance schedule from outside the neighborhood (C+D). */
  gSchedule?: GScheduleInput;
}

interface RunCoupledRequest {
  type: "runCoupled";
  id: number;
  targets: CoupledTarget[];
  intraEdges: CoupledIntraEdge[];
  config: HhRunConfig;
}

type AnyRequest = RunRequest | RunCoupledRequest;

type WorkerMsg =
  | { type: "progress"; id: number; fraction: number }
  | { type: "done"; id: number; result: HhResult }
  | { type: "doneCoupled"; id: number; results: HhResult[] }
  | { type: "error"; id: number; message: string };

self.onmessage = (e: MessageEvent<AnyRequest>) => {
  const msg = e.data;
  let lastProgressPosted = 0;
  const reportProgress = (fraction: number) => {
    const now = performance.now();
    if (now - lastProgressPosted < 80) return;
    lastProgressPosted = now;
    const out: WorkerMsg = { type: "progress", id: msg.id, fraction };
    self.postMessage(out);
  };

  try {
    if (msg.type === "run") {
      const { id, morph, synEvents, config, iExt, gSchedule } = msg;
      const result = runMcHH(
        morph,
        synEvents,
        config,
        iExt ?? null,
        reportProgress,
        gSchedule ?? null,
      );
      const out: WorkerMsg = { type: "done", id, result };
      self.postMessage(out, {
        transfer: [
          result.voltageTrace.buffer,
          result.frameTimesMs.buffer,
        ],
      });
      return;
    }

    if (msg.type === "runCoupled") {
      const { id, targets, intraEdges, config } = msg;
      const out = runCoupledMcHH(targets, intraEdges, config, reportProgress);
      const transferables: Transferable[] = [];
      for (const r of out.results) {
        transferables.push(r.voltageTrace.buffer);
        transferables.push(r.frameTimesMs.buffer);
      }
      const reply: WorkerMsg = { type: "doneCoupled", id, results: out.results };
      self.postMessage(reply, { transfer: transferables });
      return;
    }
  } catch (err) {
    const out: WorkerMsg = {
      type: "error",
      id: msg.id,
      message: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(out);
  }
};

// Make the file a module (required by Vite worker tooling).
export {};
