/**
 * `GET /api/v1/attachment` - the serving process's attachment state, read live by the backend.
 *
 * `configured: false` is the standalone default: nothing can be attached, and the shell shows
 * no chip. `reachable: false` names why the read failed. Otherwise the fields are the serving
 * process's status as read for this request, stamped `observed_at`.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

export interface AttachmentStatus {
  readonly configured: boolean;
  readonly serving_base_url?: string;
  readonly reachable?: boolean;
  /** Present when `reachable` is false: the refusal code of the failed read. */
  readonly reason?: string;
  readonly attached?: boolean;
  readonly holder?: string | null;
  readonly resident_model?: string | null;
  readonly weights_state?: string | null;
  readonly stale?: boolean | null;
  readonly observed_at?: string;
}

export const ATTACHMENT_STATUS_PATH = `${V1}/attachment`;

export const fetchAttachmentStatus = (): Promise<ApiResult<AttachmentStatus>> =>
  apiGet<AttachmentStatus>(ATTACHMENT_STATUS_PATH);
