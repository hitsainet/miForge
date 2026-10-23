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
}

export interface ApprovalListing {
  readonly approvals: readonly Approval[];
  readonly count: number;
  readonly pending: number;
  readonly limit: number;
}

export const APPROVALS_PATH = `${V1}/approvals`;

export const listApprovals = (): Promise<ApiResult<ApprovalListing>> =>
  apiGet<ApprovalListing>(APPROVALS_PATH);

export const approveApproval = (id: string): Promise<ApiResult<Approval>> =>
  apiPost<Approval>(`${APPROVALS_PATH}/${encodeURIComponent(id)}/approve`);

export const rejectApproval = (id: string): Promise<ApiResult<Approval>> =>
  apiPost<Approval>(`${APPROVALS_PATH}/${encodeURIComponent(id)}/reject`, {});
