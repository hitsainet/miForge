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

/** What `POST /runs/{id}/archive` and `/unarchive` answer (`services/archive.py`). */
export interface RunArchiveOutcome {
  readonly run_id: string;
  readonly status: string;
  readonly archived: boolean;
  readonly archived_at: string | null;
  readonly changed: boolean;
}

/**
 * Hide a finished run from the default lists (operator decision 2026-10-20). **Deletes
 * nothing**: checkpoints, trainer state and history stay, and the run view still opens it.
 * A pending or running run is refused `run_not_archivable` (409).
 */
export const archiveRun = (runId: string): Promise<ApiResult<RunArchiveOutcome>> =>
  apiPost<RunArchiveOutcome>(`${RUN_PATH(runId)}/archive`);

/** Return an archived run to the default lists. */
export const unarchiveRun = (runId: string): Promise<ApiResult<RunArchiveOutcome>> =>
  apiPost<RunArchiveOutcome>(`${RUN_PATH(runId)}/unarchive`);

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

/** One row of `GET /api/v1/runs` - `api/v1/endpoints/runs_index.py::run_row`, exactly. */
export interface RunListRow {
  readonly run_id: string;
  readonly status: string;
  readonly run_type: string;
  readonly created_at: string | null;
  readonly finished_at: string | null;
  readonly last_step: number | null;
  /** The column `mark_completed` writes. The run view reads the run log, which wins. */
  readonly stop_reason: string | null;
  readonly failure_code: string | null;
  /**
   * When the operator archived the run, or `null` while it is listed (operator decision
   * 2026-10-20). An archived row reaches the list only when `includeArchived` asked for it.
   * Optional so a backend that predates archive reads as "not archived".
   */
  readonly archived_at?: string | null;
}

export interface RunListPage {
  readonly runs: readonly RunListRow[];
  readonly total: number;
  readonly limit: number;
  readonly offset: number;
  /** The filter vocabularies, served so the screen lists them rather than typing them. */
  readonly statuses: readonly string[];
  readonly run_types: readonly string[];
}

export interface RunListQuery {
  readonly limit?: number;
  readonly offset?: number;
  readonly status?: string;
  readonly runType?: string;
  /** Also list archived runs. Omitted (not `false`) when unset, so the default is the route's. */
  readonly includeArchived?: boolean;
}

/**
 * Every run, newest first, a page at a time (FR-009.19; task 14.0, AC-009.40).
 *
 * **Absent until 2026-10-18**: the path carried only `POST /api/v1/runs`, so a GET answered
 * 405 and this function refused locally by name. `api/v1/endpoints/runs_index.py` serves it
 * now. An empty filter is omitted, never sent as `status=`, because the backend refuses an
 * unknown status by name and the empty string is one.
 */
export const listRuns = async (query: RunListQuery = {}): Promise<ApiResult<RunListPage>> => {
  const result = await apiGet<RunListPage>(RUN_LIST_PATH, {
    query: {
      limit: query.limit,
      offset: query.offset,
      status: query.status === undefined || query.status.length === 0 ? undefined : query.status,
      run_type:
        query.runType === undefined || query.runType.length === 0 ? undefined : query.runType,
      include_archived: query.includeArchived === true ? true : undefined,
    },
  });
  return result.ok ? checkRunListPage(result.value) : result;
};

/**
 * A 200 that is not a run list is **malformed**, named - not an empty list and not a crash.
 *
 * Found by `budgets.spec.ts`, whose catch-all stub answers `{}` to every path: the Runs
 * screen read `{}.runs` as a list and took the whole app down, on the screen a reload lands
 * on. A proxy's or a mismatched build's answer reaches the same line.
 */
export function checkRunListPage(value: unknown): ApiResult<RunListPage> {
  const page = value as Partial<RunListPage> | null;
  const ok =
    page !== null &&
    typeof page === 'object' &&
    Array.isArray(page.runs) &&
    typeof page.total === 'number' &&
    Array.isArray(page.statuses) &&
    Array.isArray(page.run_types);
  return ok
    ? { ok: true, value: page as RunListPage }
    : {
        ok: false,
        failure: {
          kind: 'malformed',
          status: 200,
          reason: 'the response is not a run list (runs, total, statuses, run_types)',
          body: JSON.stringify(value).slice(0, 2000),
        },
      };
}
