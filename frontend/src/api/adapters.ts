/**
 * F8's adapter surface, consumed (FR-009.22, .23, .31).
 *
 * `mergeAdapter` is F8's, and **its refusal is rendered rather than re-derived** (FR-009.31):
 * R-02.45 refuses a merge when no checkpoint of the unmerged base exists, and that check
 * lives in `services/runtypes_service.preflight_merge`. A tree that decided for itself
 * whether a merge were possible would be a second answer to F8's question.
 */
import { apiGet, apiPost, type ApiResult } from './client';
import { V1 } from './v1';
import type { AdapterRow } from '@/types/tree';

export interface AdapterListing {
  readonly adapters: readonly AdapterRow[];
  readonly count: number;
}

export const fetchAdapters = (
  options: { readonly runId?: string } = {},
): Promise<ApiResult<AdapterListing>> =>
  apiGet<AdapterListing>(`${V1}/adapters`, { query: { run_id: options.runId } });

export interface AdapterDetail {
  readonly manifest?: Readonly<Record<string, unknown>>;
  readonly [key: string]: unknown;
}

export const fetchAdapter = (name: string): Promise<ApiResult<AdapterDetail>> =>
  apiGet<AdapterDetail>(`${V1}/adapters/${encodeURI(name)}`);

export const mergeAdapter = (name: string): Promise<ApiResult<unknown>> =>
  apiPost<unknown>(`${V1}/adapters/${encodeURI(name)}/merge`);
