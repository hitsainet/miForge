/**
 * Weight-space programming (WSP) runs, read-only (fix 2026-11-20).
 *
 * Until 2026-11-20 nothing on the frontend read `/wsp/`: a WSP run was reachable only through
 * Approvals (whose "Open run" link sent its id to the TRAINING run view, which answered
 * `run_not_found`) and the API. These are the three reads the WSP runs screens make:
 *
 * - `GET /api/v1/wsp/runs` - miForge's own list rows, newest first (`list_runs`);
 * - `GET /api/v1/wsp/runs/{id}` - one run as `miforge.wsp-run/v1` (the vendored contract);
 * - `GET /api/v1/wsp/runs/{id}/transcript` - the editor's transcript, every line DATA;
 * - `GET /api/v1/wsp/runs/{id}/progress` - live progress, miForge's own read beside the
 *   contract (live progress, 2026-11-20): the worker's last heartbeat, the gateway's metered
 *   usage so far, each budget against its cap, a queued run's place in the queue;
 * - `GET /api/v1/wsp/runs/{id}/calls` - one row per editor call: counts and codes, never text;
 * - `GET /api/v1/wsp/pool` - the worker pool as the driver last read it (feature 019 P3): busy,
 *   idle and not answering, the pool size and its provisioned maximum, the queue per endpoint.
 *
 * No write is made from these screens. Cancel exists as `POST /wsp/runs/{id}/cancel`, but this
 * app gates no run action behind its own buttons for WSP; agents and the API do that.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

export const WSP_RUNS_PATH = `${V1}/wsp/runs`;
export const WSP_RUN_PATH = (runId: string): string =>
  `${WSP_RUNS_PATH}/${encodeURIComponent(runId)}`;
export const WSP_POOL_PATH = `${V1}/wsp/pool`;
/** The route's own maximum page (`MAX_PAGE` in `api/v1/endpoints/wsp_runs.py`). */
export const WSP_RUNS_LIMIT = 200;

/** A WSP run's id: `wsp_` and twelve hex digits (`miforge-wsp-run-v1`'s `run_id` pattern). */
const WSP_RUN_ID = /^wsp_[0-9a-f]{12}$/;

/** True for a WSP run's id, which the training run view cannot open. */
export function isWspRunId(id: string): boolean {
  return WSP_RUN_ID.test(id);
}

/** `{code, message}`: why a run stopped or failed, as the run records it. */
export interface WspFailure {
  readonly code: string;
  readonly message: string;
}

export interface WspUsage {
  readonly requests: number;
  readonly input_tokens: number;
  readonly output_tokens: number;
  readonly cost_usd: number;
}

/** The run states with nothing left to observe (`TERMINAL_STATUSES` in `wsp/models.py`). */
export const WSP_TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  'completed',
  'failed',
  'cancelled',
]);

/**
 * True while a run can still change (queued, pushing, running, collecting, or any state this
 * client does not know): the screens poll such a run. Decided from the RECORD's status only.
 */
export function isLiveWspStatus(status: string): boolean {
  return !WSP_TERMINAL_STATUSES.has(status);
}

/** One budget against its cap (`wsp/progress.py` `_budget`). `steps` is an estimate. */
export interface WspBudget {
  readonly name: 'tokens' | 'cost_usd' | 'time_s' | 'steps' | string;
  readonly unit?: string;
  readonly spent: number | null;
  readonly cap: number | null;
  readonly fraction: number | null;
  readonly estimate: boolean;
}

/**
 * Feature 019 (FR-019.2, FR-019.17): the window and output ceiling a run's push used
 * (`wsp.window_read.run_fields`), in its list row and its progress read. All optional: a run
 * from before 019, or an older backend, has none (the run page reads "not recorded").
 */
export interface WspRunWindow {
  readonly window_tokens?: number | null;
  readonly window_source?: 'stated' | 'reported' | 'assumed' | string | null;
  readonly window_reported_tokens?: number | null;
  readonly output_ceiling_tokens?: number | null;
  readonly output_ceiling_lowered?: boolean | null;
  readonly window_warnings?: readonly string[];
  /** Debt D3 (2026-11-28): when the run began waiting for its editor model (the claim read no
   * window); a queued run with it set is waiting now. Kept after the wait; null if it never
   * waited; absent from an older backend. */
  readonly editor_wait_since?: string | null;
}

/** `GET /api/v1/wsp/runs/{id}/progress`. */
export interface WspProgress extends WspRunWindow {
  readonly run_id: string;
  readonly status: string;
  readonly live: boolean;
  /** The server's clock when it answered: elapsed times are counted from it. */
  readonly now: string;
  readonly created_at: string | null;
  readonly started_at: string | null;
  readonly finished_at: string | null;
  readonly clock_from: 'started_at' | 'created_at';
  readonly elapsed_s: number | null;
  /** 2026-11-23: why a finished run whose patch changed weights is `no_change` (the patch is kept
   * as evidence, not a candidate); null otherwise, and on runs finished before it was stored. */
  readonly outcome_reason?: string | null;
  readonly worker: {
    readonly state: string | null;
    readonly elapsed_s: number | null;
    readonly last_heartbeat_at: string | null;
    readonly heartbeat_age_s: number | null;
  } | null;
  readonly metered: {
    readonly calls: number;
    readonly refused: number;
    readonly input_tokens: number;
    readonly output_tokens: number;
    readonly tokens: number;
    readonly cost_usd: number;
    readonly last_call_at: string | null;
    /** Feature 020 (FR-020.21, DEP-4): the run's cache totals, summed from its calls; null when
     * no call recorded one ("not recorded"); absent from an older backend. */
    readonly cache_read_tokens?: number | null;
    readonly cache_write_tokens?: number | null;
  };
  readonly in_flight: {
    readonly worst_case_tokens: number;
    readonly worst_case_usd: number;
    readonly since: string | null;
  } | null;
  readonly budgets: readonly WspBudget[];
  readonly queue: {
    readonly position: number;
    readonly queued: number;
    readonly active_run_id: string | null;
  } | null;
}

/** The list row's compact progress (`wsp/progress.py` `compact`); null on a finished run. */
export interface WspProgressCompact {
  readonly elapsed_s: number | null;
  readonly clock_from: 'started_at' | 'created_at';
  readonly worker_state: string | null;
  readonly heartbeat_age_s: number | null;
  readonly calls: number;
  readonly refused: number;
  readonly tokens: number;
  readonly cost_usd: number;
  readonly last_call_at: string | null;
  readonly queue_position: number | null;
  readonly budgets: readonly WspBudget[];
}

/** One editor call (`GET /api/v1/wsp/runs/{id}/calls`): counts and codes, never text. */
export interface WspCall {
  readonly seq: number;
  readonly at: string | null;
  /** `served`, `upstream_error`, `upstream_failed`, `session_ended`, `refused`,
   * `closed_at_worst`, or null for a call recorded before outcomes were (migration 0030). */
  readonly outcome: string | null;
  readonly http_status: number | null;
  readonly latency_ms: number | null;
  readonly finish_reason: string | null;
  readonly refusal_code: string | null;
  readonly input_tokens: number;
  readonly output_tokens: number;
  readonly cost_usd: number;
  /** Feature 020 (FR-020.20): the call's cache reads and writes; null on a call from before 020
   * ("not recorded"); absent from an older backend. */
  readonly cache_read_tokens?: number | null;
  readonly cache_write_tokens?: number | null;
}

export interface WspCallPage {
  readonly run_id: string;
  readonly status: string;
  readonly live: boolean;
  readonly calls: readonly WspCall[];
  readonly more: boolean;
}

/**
 * Feature 020 (FR-020.21, F20-D23, DEP-4): a run's cost in USD and its cache-read and cache-write
 * totals, summed from its call rows by the runs route (never `run_document`). A cache total is
 * null when no call recorded one.
 */
export interface WspMetered {
  readonly cost_usd: number;
  readonly cache_read_tokens: number | null;
  readonly cache_write_tokens: number | null;
}

/** One row of `GET /api/v1/wsp/runs`. */
export interface WspRunRow extends WspRunWindow {
  readonly run_id: string;
  readonly status: string;
  readonly kind: string;
  readonly mode: string;
  readonly caller: string | null;
  readonly caller_ref: string | null;
  readonly endpoint_id: string | null;
  readonly stop_reason: string | null;
  readonly outcome: string | null;
  /** 2026-11-23: why a run whose patch changed weights is `no_change`; null otherwise. */
  readonly outcome_reason?: string | null;
  readonly created_at: string | null;
  readonly provider: string | null;
  readonly provider_version?: string | null;
  readonly failure?: WspFailure | null;
  readonly usage?: WspUsage;
  readonly started_at?: string | null;
  readonly finished_at?: string | null;
  readonly forward_passes_used: number | null;
  /** 2026-11-21: the thinking frozen in the request, and whether any reply carried reasoning
   * (`null` unless thinking was on and a result was read). */
  readonly thinking?: string | null;
  readonly thinking_tokens?: number | null;
  readonly output_cap?: string;
  readonly reasoning_turns?: number | null;
  readonly reasoning_returned?: boolean | null;
  /** Repeated turns: the repeat guard's actions not run because they repeated one on the same
   * weights (2026-11-21), plus, in a run whose transcript header has `edit_repeats: 1`, edits
   * that re-made weights already measured and reverted (fix 4, `0cdc1de7`, 2026-11-28). Those
   * edits run; the engine only adds a notice to the answer and counts them here. A repeated edit
   * does not count toward the in-a-row stop (decision F4). */
  readonly repeated_turns?: number | null;
  /** Paging (2026-11-21): the frozen setting (`off`, or `null`: not frozen, so on for
   * `miwsp_engine`), and the engine's counters: results paged out, recalls, ledger rows and the
   * largest prompt it estimated, in tokens. */
  readonly paging?: string | null;
  readonly pages_out?: number | null;
  readonly recalls?: number | null;
  readonly ledger_rows?: number | null;
  readonly peak_prompt_tokens?: number | null;
  /** The engine's handoffs (a context guard's restarts), from the result; null before one. */
  readonly handoffs?: number | null;
  /** Feature 020 (FR-020.21): cost and cache totals; absent from an older backend. */
  readonly metered?: WspMetered;
  /** Live progress (2026-11-20): on a live row; null on a finished one. */
  readonly progress?: WspProgressCompact | null;
}

export interface WspRunListing {
  readonly runs: readonly WspRunRow[];
}

/** The fields of `miforge.wsp-run/v1` the detail screen reads. The rest stay unread. */
export interface WspRunDocument {
  readonly kind: string;
  readonly run_id: string;
  readonly status: string;
  readonly created_at: string;
  readonly caller_ref: string | null;
  readonly stop_reason: string | null;
  readonly outcome: string | null;
  readonly failure: WspFailure | null;
  readonly usage: WspUsage & { readonly metered_by?: string };
  readonly forward_passes_used: number | null;
  readonly request: Readonly<Record<string, unknown>>;
  readonly request_sha256: string;
  readonly approval: { readonly approval_id: string; readonly action: string } | null;
  readonly candidate_id: string | null;
  readonly patch_sha256: string | null;
  readonly content_sha256: string | null;
  readonly transcript_sha256: string | null;
  readonly verdict: Readonly<Record<string, unknown>> | null;
  readonly lineage: {
    readonly bundle_sha256: string;
    readonly provider: string;
    readonly provider_version: string;
    readonly chain: readonly string[];
  };
  readonly replay: Readonly<Record<string, unknown>> | null;
  /** 018 FU-2 / fix 2 (2026-11-27): the worker's check of the submitted answer, absent when
   * none was stored. `measured` false on an inspect that answered before any forward pass. */
  readonly submitted_check?: WspSubmittedCheck;
}

/** The run contract's `SubmittedCheck` (the fields the run page reads). */
export interface WspSubmittedCheck {
  readonly measured: boolean;
  readonly stated: boolean;
  readonly compared: boolean;
  readonly differs: boolean | null;
  readonly split: string | null;
  readonly stated_split: string | null;
  readonly checks: readonly Readonly<Record<string, unknown>>[];
}

/** `GET /wsp/runs/{id}/transcript`: entries are untrusted data, rendered as text only. */
export interface WspTranscript {
  readonly run_id: string;
  readonly transcript_sha256: string | null;
  readonly entries: readonly unknown[];
  readonly partial: boolean;
}

export const listWspRuns = (
  options: { readonly limit?: number; readonly status?: string } = {},
): Promise<ApiResult<WspRunListing>> =>
  apiGet<WspRunListing>(WSP_RUNS_PATH, {
    query: {
      limit: options.limit ?? WSP_RUNS_LIMIT,
      status: options.status === undefined || options.status === '' ? undefined : options.status,
    },
  });

export const fetchWspRun = (runId: string): Promise<ApiResult<WspRunDocument>> =>
  apiGet<WspRunDocument>(WSP_RUN_PATH(runId));

export const fetchWspTranscript = (runId: string): Promise<ApiResult<WspTranscript>> =>
  apiGet<WspTranscript>(`${WSP_RUN_PATH(runId)}/transcript`);

export const fetchWspProgress = (runId: string): Promise<ApiResult<WspProgress>> =>
  apiGet<WspProgress>(`${WSP_RUN_PATH(runId)}/progress`);

/** The calls after `afterSeq` (all of them when it is null), oldest first. */
export const fetchWspCalls = (
  runId: string,
  afterSeq: number | null = null,
): Promise<ApiResult<WspCallPage>> =>
  apiGet<WspCallPage>(
    `${WSP_RUN_PATH(runId)}/calls`,
    afterSeq === null ? undefined : { query: { after_seq: afterSeq } },
  );

/** One provisioned worker, as the driver's last pass read it (`wsp/pool.py`). */
export interface WspPoolWorker {
  readonly index: number;
  /** busy (a run holds it), idle, restarting (its session ended and its pod is being replaced),
   * or not_answering (its last read failed). */
  readonly state: string;
  /** What the worker's own `/ready` said (idle, running, finished, unknown, not_answering...). */
  readonly worker_state?: string;
  readonly run_id: string | null;
  readonly boot_id: string | null;
  readonly read_at: string | null;
}

/** `GET /api/v1/wsp/pool` (TDD 019 §5.3). */
export interface WspPool {
  readonly pool_size: number;
  readonly maximum: number;
  readonly effective: number;
  readonly busy: number;
  readonly idle: number;
  /** Absent from a backend before 2026-11-26. */
  readonly restarting?: number;
  readonly not_answering: number;
  readonly workers: readonly WspPoolWorker[];
  readonly queued_by_endpoint: readonly { readonly endpoint_id: string; readonly queued: number }[];
  readonly queued_without_endpoint: number;
  readonly shared_base_urls: readonly {
    readonly base_url: string;
    readonly endpoint_ids: readonly string[];
    readonly run_ids: readonly string[];
  }[];
}

export const getWspPool = (): Promise<ApiResult<WspPool>> => apiGet<WspPool>(WSP_POOL_PATH);

/** The approval actions that start a WSP run (`ApprovalCard`'s labels name them). */
export const WSP_APPROVAL_ACTIONS: ReadonlySet<string> = new Set([
  'wsp_session',
  'wsp_session_external',
]);
