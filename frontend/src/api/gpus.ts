/**
 * `GET /api/v1/gpus` - the visible cards, **by UUID** (task 14.0, AC-009.42).
 *
 * No index is in this type because none is in the response: indices inverted on the suite's
 * GPU node on 2026-09-18, and a response is where one would be copied from.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

export interface GpuRow {
  readonly uuid: string;
  readonly name: string;
  /** `null` when the driver answered the device list but not the free figure. */
  readonly free_memory_bytes: number | null;
}

export interface GpuListing {
  readonly devices: readonly GpuRow[];
  /** `false` when this process could not read the driver - named in `unavailable`. */
  readonly available: boolean;
  readonly unavailable: { readonly code: string; readonly message: string } | null;
  /** `auto` is always a valid request; the backend resolves it at submit. */
  readonly auto: boolean;
}

export const GPUS_PATH = `${V1}/gpus`;

export const fetchGpus = (): Promise<ApiResult<GpuListing>> => apiGet<GpuListing>(GPUS_PATH);
