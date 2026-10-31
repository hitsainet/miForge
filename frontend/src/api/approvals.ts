/**
 * The approval queue (F12; FR-012.12 - .18). The operator's half of the gate.
 *
 * **This module never sends `X-Miforge-Agent`** - nothing in the frontend does - so every call
 * here is the operator's, which is exactly what the backend requires of approve and reject
 * (an agent is refused `403 agent_cannot_decide`). `approvals.test.ts` pins that the request
 * carries no such header.
 */
import { apiGet, apiPost, type ApiResult } from './client';
import { V1 } from './v1';

export type ApprovalAction = 'gpu_start' | 'hotupdate_attach' | 'delete' | 'config_write';
export type ApprovalStatus =
  'pending' | 'executing' | 'executed' | 'rejected' | 'expired' | 'failed';

export interface ApprovalError {
  readonly code: string;
  readonly message: string;
  readonly details?: Readonly<Record<string, unknown>>;
}

export interface Approval {
  readonly approval_id: string;
  readonly action: ApprovalAction;
  readonly status: ApprovalStatus;
  readonly summary: Readonly<Record<string, unknown>>;
  readonly requested_by: string;
  readonly decided_by: string | null;
  readonly reason: string | null;
  readonly result: unknown;
  readonly error: ApprovalError | null;
  readonly created_at: string;
  readonly decided_at: string | null;
  readonly expires_at: string;
  /** Every argument the agent sent, as stored (the "All arguments" list on the card). */
  readonly payload?: unknown;
  readonly request_digest?: string;
  /** Set when the operator approved this with other settings changes in one action (D79). */
  readonly batch_id?: string | null;
}

export interface ApprovalListing {
  readonly approvals: readonly Approval[];
  readonly count: number;
  readonly pending: number;
  readonly limit: number;
}

export const APPROVALS_PATH = `${V1}/approvals`;

/** The API's ceiling on one listing (`GET /approvals?limit=` is `1..500`). */
export const APPROVALS_MAX_LIMIT = 500;

export interface ListApprovalsQuery {
  readonly status?: ApprovalStatus;
  readonly limit?: number;
}

export const listApprovals = (
  query: ListApprovalsQuery = {},
): Promise<ApiResult<ApprovalListing>> =>
  apiGet<ApprovalListing>(
    APPROVALS_PATH,
    query.status === undefined && query.limit === undefined
      ? undefined
      : { query: { status: query.status, limit: query.limit } },
  );

export const approveApproval = (id: string): Promise<ApiResult<Approval>> =>
  apiPost<Approval>(`${APPROVALS_PATH}/${encodeURIComponent(id)}/approve`);

/** The reason the operator gave, or ENH-008 D8c's default when the field was left blank. */
export const DEFAULT_REJECT_REASON = 'Rejected by the operator.';
/** The longest reason the UI sends (the API keeps up to 2000; the suite's field allows 1000). */
export const REJECT_REASON_MAX = 1000;

export const rejectApproval = (id: string, reason?: string): Promise<ApiResult<Approval>> => {
  const trimmed = (reason ?? '').trim().slice(0, REJECT_REASON_MAX);
  return apiPost<Approval>(`${APPROVALS_PATH}/${encodeURIComponent(id)}/reject`, {
    reason: trimmed === '' ? DEFAULT_REJECT_REASON : trimmed,
  });
};

/**
 * The actions that change a setting (ENH-006 D79). `config_write` covers both settings writes
 * the backend gates: set (`forge_set_setting`) and reset (`forge_reset_setting`).
 */
export const SETTINGS_CHANGE_ACTIONS: ReadonlySet<string> = new Set(['config_write']);

export const isSettingsChange = (approval: Approval): boolean =>
  approval.status === 'pending' && SETTINGS_CHANGE_ACTIONS.has(approval.action);

/** One request's outcome inside a batch: the stored row, plus whether it ran. */
export interface ApprovalBatchResult extends Approval {
  readonly ok: boolean;
}

export interface ApprovalBatch {
  readonly batch_id: string;
  readonly count: number;
  readonly results: readonly ApprovalBatchResult[];
}

/**
 * Approve every named waiting settings change with one operator action (D79). The backend
 * refuses the whole batch (409 `approval_batch_refused`) before approving anything when an id
 * is not a waiting settings change; otherwise each runs through the single-approve path.
 */
export const approveBatch = (ids: readonly string[]): Promise<ApiResult<ApprovalBatch>> =>
  apiPost<ApprovalBatch>(`${APPROVALS_PATH}/approve-batch`, { approval_ids: [...ids] });
