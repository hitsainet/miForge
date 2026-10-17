/**
 * The run record, its series, and **R-02.53's create and validate paths**.
 *
 * ## Re-checked against the live route table, not against an earlier reading of it
 *
 * An earlier version of this module declared `POST /api/v1/runs/validate` absent and refused
 * locally. **It is mounted** - F6's `runs_loop.py` landed it, with `POST /runs/define` and
 * `GET /runs/{id}/boundaries`, while this screen was being built. Found by hitting the app
 * rather than by re-reading this file, which is the asymmetry this project keeps recording:
 * *a cross-document claim has a shelf life, including your own from earlier the same session.*
 *
 * ## `define`, not `runs`
 *
 * **`POST /api/v1/runs` is F1's acquisition half** and R-02.53's create path is
 * `POST /api/v1/runs/define`. F6 says so in its own module docstring: F1's route *"submits
 * the acquisition half - no pack, no stop condition, no scoring - which is precisely the
 * shape R-02.53 forbids"*, and a run with no recorded definition is refused by name in the
 * worker. So the form posts to `/define`, which is the one path that reaches a running run
 * and the one the gate is on.
 *
 * ## `validate` returns **200 either way**
 *
 * Asking whether a definition is valid is a **successful question**. The body carries
 * `{ok, may_be_more_to_fix, steps_checked, failures[], skipped[], resolved}` - not an error
 * envelope - so `ok: false` is a 200 and the failures are read from the body. Decoding that
 * as a transport-level failure would make a perfectly good answer look like a broken request.
 *
 * `skipped` is the half that is easy to drop and should not be: a step the validator *could
 * not run* names what it waited on, so a form showing only `failures` would tell an operator
 * they had one problem when they may have three.
 */
import { apiGet, apiPost, type ApiResult } from './client';
import { V1 } from './v1';
import type { BatchPage, RunRecord, StepPage } from '@/types/run';

export const RUN_PATH = (runId: string): string => `${V1}/runs/${encodeURIComponent(runId)}`;

export const fetchRun = (runId: string): Promise<ApiResult<RunRecord>> =>
  apiGet<RunRecord>(RUN_PATH(runId));

export const fetchSteps = (
  runId: string,
  options: { readonly limit?: number; readonly beforeStep?: number } = {},
): Promise<ApiResult<StepPage>> =>
  apiGet<StepPage>(`${RUN_PATH(runId)}/steps`, {
    query: { limit: options.limit, before_step: options.beforeStep },
  });

export const fetchBatches = (
  runId: string,
  options: { readonly limit?: number } = {},
): Promise<ApiResult<BatchPage>> =>
  apiGet<BatchPage>(`${RUN_PATH(runId)}/batches`, { query: { limit: options.limit } });

export const cancelRun = (runId: string): Promise<ApiResult<unknown>> =>
  apiPost<unknown>(`${RUN_PATH(runId)}/cancel`);

/** F6's boundary records. What the run screen reads for the run's own observed gaps. */
export const fetchBoundaries = (
  runId: string,
  options: { readonly limit?: number } = {},
): Promise<ApiResult<{ readonly boundaries: readonly Readonly<Record<string, unknown>>[] }>> =>
  apiGet<{ readonly boundaries: readonly Readonly<Record<string, unknown>>[] }>(
    `${RUN_PATH(runId)}/boundaries`,
    { query: { limit: options.limit } },
  );

/** What will stop this run, and how close it is. Run-scoped: it needs a run to exist. */
export const fetchStopConditions = (
  runId: string,
): Promise<ApiResult<Readonly<Record<string, unknown>>>> =>
  apiGet<Readonly<Record<string, unknown>>>(`${RUN_PATH(runId)}/stop-conditions`);

/** The recorded definition: what was asked for **and** what it resolved to. */
export const fetchRunDefinition = (
  runId: string,
): Promise<ApiResult<Readonly<Record<string, unknown>>>> =>
  apiGet<Readonly<Record<string, unknown>>>(`${RUN_PATH(runId)}/definition`);

/* ------------------------------------------------------------------------------------ *
 * R-02.53: define and validate
 * ------------------------------------------------------------------------------------ */

export const RUN_DEFINE_PATH = `${V1}/runs/define`;
export const RUN_VALIDATE_PATH = `${V1}/runs/validate`;

/** One failure from the validator. **The owner's own code, unchanged.** */
export interface DefinitionFailure {
  readonly step: string;
  readonly code: string;
  readonly message: string;
  readonly details: Readonly<Record<string, unknown>>;
}

/** A step the validator could not run, naming what it waited on. */
export interface SkippedStep {
  readonly step: string;
  readonly status: string;
  readonly waited_on: readonly string[];
  readonly message: string;
}

/** `POST /api/v1/runs/validate`'s body - **200 whether or not the definition is valid**. */
export interface ValidationOutcome {
  readonly ok: boolean;
  readonly may_be_more_to_fix: boolean;
  readonly steps_checked: readonly string[];
  readonly failures: readonly DefinitionFailure[];
  readonly skipped: readonly SkippedStep[];
  /** The resolved summary: pinned versions, the budget, the footprint, absent services. */
  readonly resolved: Readonly<Record<string, unknown>>;
  readonly footprint?: Readonly<Record<string, unknown>>;
}

/**
 * Validate the whole definition without creating anything.
 *
 * Read-only: it allocates nothing, creates no row and writes no run-log entry - and it is
 * **the same validator the create path calls**, which is what makes the answer truthful.
 *
 * **A green answer enables nothing.** `/define` re-runs the same validator, and a part can
 * stop resolving or free VRAM can change in between, so the form never caches this as
 * permission and never skips the submit refusal path because a check passed.
 */
export const validateRun = (definition: unknown): Promise<ApiResult<ValidationOutcome>> =>
  apiPost<ValidationOutcome>(RUN_VALIDATE_PATH, definition);

/**
 * Define and start a run: **one body, one POST** (R-02.53, FR-009.34).
 *
 * `202`, because the run has been *accepted* rather than performed - the parameters are
 * acquired in the worker, under the lease. Nothing is staged, chunked or pre-committed:
 * R-02.53's guarantee is that the *whole* definition is validated before anything starts, and
 * a form submitting in pieces breaks exactly that.
 *
 * Refuses `run_definition_invalid` with **every** failure in `details.failures` - a list
 * rather than a merged message, *"so a form can put each refusal beside the field it belongs
 * to."*
 */
export const defineRun = (
  definition: unknown,
): Promise<ApiResult<{ readonly run_id: string; readonly [key: string]: unknown }>> =>
  apiPost<{ readonly run_id: string; readonly [key: string]: unknown }>(
    RUN_DEFINE_PATH,
    definition,
  );

export const RUN_LIST_PATH = `${V1}/runs`;

/**
 * Every run, for the index (FR-009.19).
 *
 * **Still absent.** `GET /api/v1/runs` is not a route - the path carries only
 * `POST /api/v1/runs`, so a GET there answers **405**, not 404, which is worth knowing
 * because a client that treated a 405 as "empty list" would show an empty index as a fact.
 * F9's **F-2** records that no requirement describes a run list; the navigational minimum
 * still needs an enumeration from somewhere. Rather than fabricate rows, this refuses by name
 * and `RunIndexPage` renders the refusal with what to do next.
 */
export const listRuns = async (): Promise<ApiResult<{ readonly runs: readonly RunRecord[] }>> => ({
  ok: false,
  failure: {
    kind: 'refusal',
    status: 0,
    code: 'run_list_endpoint_not_mounted',
    message:
      'This build serves no run list: there is no GET /api/v1/runs, only ' +
      'GET /api/v1/runs/{run_id}. Open a run by its identifier, or start one from New run.',
    details: { endpoint: `GET ${RUN_LIST_PATH}`, answers: '405 Method Not Allowed', owner: 'F6' },
  },
});

export interface SurfaceGap {
  readonly endpoint: string;
  readonly owner: string;
  readonly what: string;
}

/** The gaps a screen states in words, so it reads as honest rather than broken. */
export const runsSurfaceGaps = (): readonly SurfaceGap[] => [
  {
    endpoint: `GET ${RUN_LIST_PATH}`,
    owner: 'F6',
    what: 'enumerating runs, which the run index needs (the path answers 405, not 404)',
  },
];
