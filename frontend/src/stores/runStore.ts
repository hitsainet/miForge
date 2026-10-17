/**
 * A run's liveness, from the record and only from the record (FR-009.4; ADR-007).
 *
 * ## The one function, and why its signature is the guard
 *
 * `deriveLiveness(record, nowMs)` takes **the record and a clock. Nothing else.** No socket,
 * no transport state, no "time of the last event". Mutation **M3** - `isLive =
 * socket.connected` - therefore cannot be written without changing this signature, which
 * makes it visible in review as well as red in the suite.
 *
 * ADR-007 is the reason: *the socket is a convenience, the database is the truth.* miStudio
 * reaped a live 5.8-hour job as "worker lost" because progress went only over the socket, and
 * the inverse failure is a screen that reports a dead run as healthy because a socket is
 * still open.
 *
 * ## The threshold is the janitor's, read and never chosen
 *
 * `GET /api/v1/runs/{id}` carries a `heartbeat` block - verified against
 * `backend/src/workers/heartbeat.py::liveness_of`, not taken from a document - with
 * `age_seconds`, `longest_gap_seconds`, **`lapse_threshold_seconds`** and **`lapsed`**. The
 * server computes the verdict and publishes its own threshold so that no reader derives a
 * second one. This module reads both and chooses neither.
 *
 * **Why that matters more than it sounds.** The two alternatives were both written here and
 * both withdrawn. Trusting the socket shows a stalled run as live. Deriving a threshold from
 * the 60 s heartbeat throttle shows a **healthy** run as stalled - because 60 s is *a minimum
 * interval between writes, not a maximum gap*, and the loop's own sub-operations exceed it: a
 * test pass is generation-bound over hundreds of prompts, and a 2.4 GB save is seconds to
 * tens of seconds. That threshold would have marked every healthy run stalled during every
 * test pass and every save, on the most-watched screen in the product, in the false-positive
 * direction of the exact failure the screen exists to prevent.
 *
 * The janitor's value is **provisional** upstream (`janitor_lapse_seconds`, 900 s, pending
 * foundation 6.4b's longest-observed gaps). Which is why this module reports the age, the
 * run's own longest observed gap **and** the threshold alongside the verdict: an operator who
 * can see that this run has already shown a 700 s gap reads a 900 s lapse differently.
 *
 * ## A socket event for a terminal run is discarded and logged
 *
 * `applyEvent` checks the record's state first and drops the event. It does not reorder,
 * merge or catch up. ADR-007's heartbeat *"refuses to move a terminal row"*; the store honours
 * the same rule client-side. Mutation **M4.**
 */
import { create } from 'zustand';
import type { ApiFailure } from '@/api/errors';
import { fetchBatches, fetchRun, fetchSteps } from '@/api/runs';
import { fetchRunCheckpoints } from '@/api/checkpoints';
import { fetchScores } from '@/api/scores';
import type { BatchRecord, RunCheckpoints, RunRecord, ScoreSeries, StepRecord } from '@/types/run';
import { pollIntervalMs, type TransportState } from './transport';

/**
 * The run statuses this build knows, **from `backend/src/models/run.py`** rather than from a
 * feature document.
 *
 * The two disagree and the divergence is reported rather than reconciled here: the column's
 * `RUN_STATUSES` is `{pending, running, completed, failed, cancelled}` while F6's FR-006.13
 * declares the enum as `queued → running → {completed | failed}`. `queued` is kept in the
 * pill's map so F6's spelling renders if it lands; `pending` and `cancelled` are there
 * because they are what the database can actually hold today.
 */
export const TERMINAL_STATUSES: readonly string[] = ['completed', 'failed', 'cancelled'];

export function isTerminal(status: string): boolean {
  return TERMINAL_STATUSES.includes(status);
}

export type Liveness =
  | {
      readonly kind: 'running';
      readonly step: number | null;
      readonly ageSeconds: number | null;
      readonly longestGapSeconds: number | null;
      readonly thresholdSeconds: number;
      readonly clockSkewSeconds: number | null;
    }
  | {
      readonly kind: 'stalled';
      readonly lastStep: number | null;
      readonly sinceMs: number | null;
      readonly longestGapSeconds: number | null;
      readonly thresholdSeconds: number;
      readonly clockSkewSeconds: number | null;
    }
  | {
      readonly kind: 'terminal';
      readonly state: string;
      readonly lastStep: number | null;
    };

/** Beyond this, the browser's clock and the server's disagree enough to say so. */
export const CLOCK_SKEW_TOLERANCE_SECONDS = 30;

/**
 * The record's own verdict, read.
 *
 * `nowMs` is used for exactly one thing: comparing the browser's idea of how old the
 * heartbeat is against the server's `age_seconds`. A browser whose clock is minutes out would
 * otherwise render "3 minutes ago" for a beat that was seconds old, and the operator would
 * have no way to tell. It is **never** used to decide liveness - that is the server's
 * `lapsed`, computed against the janitor's threshold.
 */
export function deriveLiveness(record: RunRecord, nowMs: number): Liveness {
  const beat = record.heartbeat;
  if (isTerminal(record.status)) {
    // A terminal run is not stalled - it is finished, and its heartbeat is *expected* to be
    // old. Reporting a completed run as lapsed is the same class of false positive as a
    // threshold taken from the 60 s throttle.
    return { kind: 'terminal', state: record.status, lastStep: record.last_step };
  }

  const skew = clockSkewSeconds(beat.heartbeat_at, beat.age_seconds, nowMs);

  if (beat.lapsed) {
    return {
      kind: 'stalled',
      lastStep: record.last_step,
      sinceMs: beat.age_seconds === null ? null : Math.round(beat.age_seconds * 1000),
      longestGapSeconds: beat.longest_gap_seconds,
      thresholdSeconds: beat.lapse_threshold_seconds,
      clockSkewSeconds: skew,
    };
  }
  return {
    kind: 'running',
    step: record.last_step,
    ageSeconds: beat.age_seconds,
    longestGapSeconds: beat.longest_gap_seconds,
    thresholdSeconds: beat.lapse_threshold_seconds,
    clockSkewSeconds: skew,
  };
}

function clockSkewSeconds(
  heartbeatAt: string | null,
  ageSeconds: number | null,
  nowMs: number,
): number | null {
  if (heartbeatAt === null || ageSeconds === null) return null;
  const parsed = Date.parse(heartbeatAt);
  if (Number.isNaN(parsed)) return null;
  const browserAge = (nowMs - parsed) / 1000;
  const difference = browserAge - ageSeconds;
  return Math.abs(difference) > CLOCK_SKEW_TOLERANCE_SECONDS ? Math.round(difference) : null;
}

/**
 * The sentence the screen shows beside the state.
 *
 * It states the last recorded step, the time since it, the run's own longest observed gap and
 * the threshold - rather than only the verdict - because the threshold is provisional and an
 * operator reading "stalled" needs the figures it was reached from.
 */
export function describeLiveness(liveness: Liveness): string {
  switch (liveness.kind) {
    case 'terminal':
      return `This run is ${liveness.state}. Its last recorded step was ${
        liveness.lastStep ?? 'none'
      }; its heartbeat is expected to be old.`;
    case 'running': {
      const age =
        liveness.ageSeconds === null
          ? 'no heartbeat has been recorded yet'
          : `last heartbeat ${Math.round(liveness.ageSeconds)} s ago`;
      const gap =
        liveness.longestGapSeconds === null
          ? 'no gap has been observed yet'
          : `longest gap so far ${liveness.longestGapSeconds} s`;
      return `Stepping: ${age}, ${gap}, reaped after ${liveness.thresholdSeconds} s.`;
    }
    case 'stalled': {
      const since =
        liveness.sinceMs === null
          ? 'no heartbeat has ever been recorded'
          : `nothing for ${Math.round(liveness.sinceMs / 1000)} s`;
      const gap =
        liveness.longestGapSeconds === null
          ? 'this run has shown no gap before now'
          : `its longest gap before now was ${liveness.longestGapSeconds} s`;
      return `Last recorded step ${liveness.lastStep ?? 'none'}, ${since} — past the ${
        liveness.thresholdSeconds
      } s the janitor reaps at, and ${gap}. A long save or a generation-bound test pass can look like this; the janitor's threshold is the one that decides.`;
    }
  }
}

/* ------------------------------------------------------------------------------------ *
 * The store
 * ------------------------------------------------------------------------------------ */

export interface RunViewState {
  readonly runId: string | null;
  readonly record: RunRecord | null;
  readonly steps: readonly StepRecord[];
  readonly batches: readonly BatchRecord[];
  readonly scores: ScoreSeries | null;
  readonly checkpoints: RunCheckpoints | null;
  readonly phase: 'idle' | 'loading' | 'ready' | 'failed';
  readonly failure: ApiFailure | null;
  /** Per-region failures, so one dead endpoint degrades one named panel (task 10.5). */
  readonly regionFailures: Readonly<Record<string, ApiFailure>>;
  readonly transport: TransportState;
  /** Socket events dropped because the record says the run is terminal (M4). */
  readonly discardedEvents: number;
  readonly loadedAtMs: number | null;
}

export interface RunViewActions {
  readonly open: (runId: string) => void;
  readonly load: () => Promise<void>;
  readonly setTransport: (transport: TransportState) => void;
  readonly applyEvent: (event: RunSocketEvent) => void;
  readonly nextPollMs: () => number;
  readonly reset: () => void;
}

/**
 * The six event names, **from the emitters** (F6 FPRD section 7, plus F4's and F5's).
 *
 * This document's own earlier list - `step | score | save | signal` - was this feature's
 * invention and is withdrawn. *A screen subscribing to names the emitter does not emit is
 * silently dead and looks connected*, which is the failure direction FR-009.4 exists to
 * prevent, so the names are taken from the features that emit them and nowhere else.
 */
export const RUN_EVENT_NAMES = [
  'boundary',
  'refresh',
  'run_terminal',
  'score',
  'retention_floor_breached',
  'checkpoint_saved',
] as const;
export type RunEventName = (typeof RUN_EVENT_NAMES)[number];

export interface RunSocketEvent {
  readonly name: string;
  readonly payload: Readonly<Record<string, unknown>>;
}

const EMPTY: Omit<RunViewState, 'runId'> = {
  record: null,
  steps: [],
  batches: [],
  scores: null,
  checkpoints: null,
  phase: 'idle',
  failure: null,
  regionFailures: {},
  transport: 'connecting',
  discardedEvents: 0,
  loadedAtMs: null,
};

export const useRunStore = create<RunViewState & RunViewActions>((set, get) => ({
  runId: null,
  ...EMPTY,

  open: (runId) => set({ runId, ...EMPTY }),

  load: async () => {
    const runId = get().runId;
    if (runId === null) return;
    set((state) => ({ phase: state.record === null ? 'loading' : state.phase }));

    const record = await fetchRun(runId);
    if (!record.ok) {
      set({ phase: 'failed', failure: record.failure });
      return;
    }

    // Each region is read independently and each failure is recorded against its own name,
    // so a 500 from one endpoint degrades one panel rather than the screen (task 10.5).
    const [steps, batches, scores, checkpoints] = await Promise.all([
      fetchSteps(runId, { limit: 200 }),
      fetchBatches(runId, { limit: 50 }),
      fetchScores(runId),
      fetchRunCheckpoints(runId),
    ]);

    const regionFailures: Record<string, ApiFailure> = {};
    if (!steps.ok) regionFailures['steps'] = steps.failure;
    if (!batches.ok) regionFailures['batches'] = batches.failure;
    if (!scores.ok) regionFailures['scores'] = scores.failure;
    if (!checkpoints.ok) regionFailures['checkpoints'] = checkpoints.failure;

    set((state) => ({
      phase: 'ready',
      failure: null,
      record: record.value,
      steps: steps.ok ? steps.value.steps : state.steps,
      batches: batches.ok ? batches.value.batches : state.batches,
      scores: scores.ok ? scores.value : state.scores,
      checkpoints: checkpoints.ok ? checkpoints.value : state.checkpoints,
      regionFailures,
      loadedAtMs: Date.now(),
    }));
  },

  setTransport: (transport) => set({ transport }),

  /**
   * Apply a socket event, or discard it.
   *
   * An event for a run the **record** says is terminal is dropped and counted, never applied.
   * It is not reordered and not merged: the record is the truth, and a late `boundary` for a
   * finished run would otherwise move a completed run back to running on the screen - which
   * is M4.
   */
  applyEvent: (event) => {
    const state = get();
    const record = state.record;
    if (record === null) return;
    if (isTerminal(record.status)) {
      // Logged, not silent: a dropped event that nobody can see is indistinguishable from one
      // that never arrived, and the two have different causes.
      console.warn(
        `[runStore] discarding socket event ${event.name} for ${record.run_id}: the run ` +
          `record says ${record.status}. The record is the truth (ADR-007).`,
      );
      set({ discardedEvents: state.discardedEvents + 1 });
      return;
    }
    // Events move the *series* between polls and never the liveness verdict, which is
    // derived from the record alone. A `run_terminal` event triggers a re-read rather than
    // setting the status here: the record is what says a run finished.
    if (event.name === 'run_terminal') {
      void get().load();
      return;
    }
    const step = event.payload['step'];
    if (event.name === 'boundary' && typeof step === 'number') {
      const loss = event.payload['loss'];
      const next: StepRecord = {
        step,
        loss: typeof loss === 'number' ? loss : null,
        grad_norm: null,
        learning_rate: null,
        wall_time_ms: null,
        batch_source_hashes: null,
      };
      set({ steps: mergeStep(state.steps, next) });
      return;
    }
    if (event.name === 'score' || event.name === 'checkpoint_saved') {
      // The payload is a notification, not a record. Re-read the series it touched rather
      // than synthesising a row, so the screen only ever shows what an endpoint served.
      void get().load();
    }
  },

  nextPollMs: () => pollIntervalMs(get().transport),

  reset: () => set({ runId: null, ...EMPTY }),
}));

/** Newest first, replacing a step already held rather than appending a duplicate. */
export function mergeStep(steps: readonly StepRecord[], next: StepRecord): readonly StepRecord[] {
  const without = steps.filter((step) => step.step !== next.step);
  return [next, ...without].sort((a, b) => b.step - a.step);
}

/**
 * The steps at which a checkpoint **exists** (FR-009.11).
 *
 * From F5's listing, and the store holds **no `save_every` field** - so M11's mutation
 * (`step % save_every === 0`) requires adding one, which is the point. `steps_on_disk` is
 * preferred over the index rows because it is read from the filesystem, which is the
 * authority (ADR-004): a run whose index was dropped still reports its artifacts honestly.
 */
export function savePoints(checkpoints: RunCheckpoints | null): readonly number[] {
  if (checkpoints === null) return [];
  const steps = new Set<number>(checkpoints.steps_on_disk);
  for (const row of checkpoints.checkpoints) steps.add(row.step);
  return [...steps].sort((a, b) => a - b);
}
