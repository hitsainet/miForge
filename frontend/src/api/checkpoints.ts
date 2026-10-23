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

/**
 * A 2xx whose `manifest` is not an object is a BROKEN CONTRACT, returned as `malformed` -
 * never handed to the view. The view's empty state is `null`; an `undefined` or an array
 * slipped past it and threw inside a render, which unmounted the WHOLE app to a blank page
 * (found 2026-10-18 by a CI Playwright run: `Cannot read properties of undefined (reading
 * 'kind')`, the run view going blank after its curves had painted).
 */
export const fetchCheckpointManifest = async (
  runId: string,
  step: number,
): Promise<ApiResult<CheckpointManifestResponse>> => {
  const result = await apiGet<CheckpointManifestResponse>(
    `${V1}/runs/${encodeURIComponent(runId)}/checkpoints/${step}`,
  );
  if (!result.ok) return result;
  const manifest: unknown = (result.value as { manifest?: unknown } | null)?.manifest;
  if (typeof manifest !== 'object' || manifest === null || Array.isArray(manifest)) {
    return {
      ok: false,
      failure: {
        kind: 'malformed',
        status: 200,
        reason: 'a checkpoint response carried no manifest object',
        body: JSON.stringify(result.value ?? null).slice(0, 2000),
      },
    };
  }
  return result;
};

/** F5's branch. The tree links to it; F9 defines no capability of its own (FR-009.31). */
export const branchFromCheckpoint = (
  runId: string,
  step: number,
  body: { readonly run_id: string },
): Promise<ApiResult<unknown>> =>
  apiPost<unknown>(`${V1}/runs/${encodeURIComponent(runId)}/checkpoints/${step}/branch`, body);

/**
 * `POST /runs/{id}/continue` with `submit: true` - **submit** a continue of an attached run
 * (D52, F11 FTASKS 18.0). The backend refuses by name when the served model is not the
 * checkpoint (`resume_served_model_mismatch`, inside `run_definition_invalid`'s failure list)
 * and names the operator action; this module reports it unchanged.
 */
export interface ContinueSubmitted {
  readonly submitted: true;
  readonly run_status: string;
  readonly plan: {
    readonly step: number;
    readonly name: string;
    readonly optimizer: { readonly state: string; readonly reason: string };
  };
  readonly resume: {
    readonly superseded_steps: { readonly first: number; readonly last: number } | null;
  };
}

export const submitContinue = (
  runId: string,
  options: { readonly step?: number } = {},
): Promise<ApiResult<ContinueSubmitted>> =>
  apiPost<ContinueSubmitted>(`${V1}/runs/${encodeURIComponent(runId)}/continue`, {
    ...(options.step === undefined ? {} : { step: options.step }),
    submit: true,
  });
