/**
 * Web Worker entry point for one multi-compartment HH run.
 *
 * The orchestrator (`runMcSession.ts`) spawns one worker per neuron in the
 * biophysical neighborhood. Each worker is self-contained: it receives a
 * pre-built `SolverMorphology`, a sorted SynEvent list, and a config, and
 * posts back voltage trace + AIS spike times.
 *
 * Why a worker per neuron rather than a single worker for the whole
 * neighborhood: forward-Euler HH is mostly memory-bandwidth-bound on the
 * compartment count, so splitting into ~10 small simulations parallelises
 * cleanly across cores, while keeping each worker's state small enough
 * (< 1 MB) that postMessage transfer is cheap.
 *
 * Communication protocol:
 *   main → worker: { type: "run", id, morph, synEvents, config, iExt? }
 *   worker → main: { type: "progress", id, fraction }
 *                  { type: "done", id, result }
 *                  { type: "error", id, message }
 */

import {
  runMcHH,
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

type WorkerMsg =
  | { type: "progress"; id: number; fraction: number }
  | { type: "done"; id: number; result: HhResult }
  | { type: "error"; id: number; message: string };

self.onmessage = (e: MessageEvent<RunRequest>) => {
  const msg = e.data;
  if (msg.type !== "run") return;
  const { id, morph, synEvents, config, iExt, gSchedule } = msg;

  let lastProgressPosted = 0;
  try {
    const result = runMcHH(
      morph,
      synEvents,
      config,
      iExt ?? null,
      (fraction) => {
        // Throttle progress messages so we don't flood the main thread
        // when the inner loop is fast.
        const now = performance.now();
        if (now - lastProgressPosted < 80) return;
        lastProgressPosted = now;
        const out: WorkerMsg = { type: "progress", id, fraction };
        self.postMessage(out);
      },
      gSchedule ?? null,
    );

    // Transfer ownership of the big typed arrays to avoid copying.
    const transferables: Transferable[] = [
      result.voltageTrace.buffer,
      result.frameTimesMs.buffer,
    ];
    const out: WorkerMsg = { type: "done", id, result };
    self.postMessage(out, { transfer: transferables });
  } catch (err) {
    const out: WorkerMsg = {
      type: "error",
      id,
      message: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(out);
  }
};

// Make the file a module (required by Vite worker tooling).
export {};
