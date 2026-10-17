/**
 * F5's checkpoint surface, consumed (FR-009.11, .17, .31).
 *
 * `fetchRunCheckpoints` is what `SavePointMarks` is fed from: **the steps where a checkpoint
 * exists**, never a cadence. An out-of-cadence save (`POST .../checkpoints` requests one at
 * the next step boundary) would be invisible to a `step % save_every` computation, and a
 * cadence step whose save failed would be marked as though it had succeeded - the dangerous
 * direction. That is mutation **M11**, and the reason this module exposes no `save_every`.
 */
import { apiGet, apiPost, type ApiResult } from './client';
import { V1 } from './v1';
import type { CheckpointRow, RunCheckpoints } from '@/types/run';

export const fetchRunCheckpoints = (runId: string): Promise<ApiResult<RunCheckpoints>> =>
  apiGet<RunCheckpoints>(`${V1}/runs/${encodeURIComponent(runId)}/checkpoints`);

export interface CheckpointListing {
  readonly checkpoints: readonly CheckpointRow[];
  readonly count: number;
  readonly total: number;
}

export const fetchCheckpoints = (
  options: { readonly runId?: string; readonly domain?: string; readonly limit?: number } = {},
): Promise<ApiResult<CheckpointListing>> =>
  apiGet<CheckpointListing>(`${V1}/checkpoints`, {
    query: { run_id: options.runId, domain: options.domain, limit: options.limit },
  });

/**
 * One checkpoint's manifest **as written on disk**, plus the labelled provider join.
 *
 * Read from disk because the manifest is the authority (ADR-004); the endpoint's own
 * docstring says the index is not consulted for this answer at all. That is what makes
 * FR-009.17's "every key present is rendered" meaningful - the view is looking at the
 * artifact, not at a projection of it.
 */
export interface CheckpointManifestResponse {
  readonly run_id: string;
  readonly step: number;
  readonly manifest: Readonly<Record<string, unknown>>;
  readonly manifest_sha256: string;
  readonly provider_sources: Readonly<Record<string, unknown>>;
}

export const fetchCheckpointManifest = (
  runId: string,
  step: number,
): Promise<ApiResult<CheckpointManifestResponse>> =>
  apiGet<CheckpointManifestResponse>(`${V1}/runs/${encodeURIComponent(runId)}/checkpoints/${step}`);

/** F5's branch. The tree links to it; F9 defines no capability of its own (FR-009.31). */
export const branchFromCheckpoint = (
  runId: string,
  step: number,
  body: { readonly run_id: string },
): Promise<ApiResult<unknown>> =>
  apiPost<unknown>(`${V1}/runs/${encodeURIComponent(runId)}/checkpoints/${step}/branch`, body);
