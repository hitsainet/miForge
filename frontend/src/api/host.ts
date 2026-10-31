/**
 * `GET /api/v1/host/resources` - what the shell's resource strip shows on every screen.
 *
 * Each GPU is keyed by UUID; no index is in the response. A source the backend could not read
 * is listed in `unavailable` with a short reason, and its reading is `null`.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

export interface GpuTelemetry {
  readonly uuid: string;
  readonly name: string;
  readonly utilization_percent: number | null;
  readonly memory_used_bytes: number | null;
  readonly memory_total_bytes: number | null;
  readonly temperature_c: number | null;
}

export interface HostUnavailable {
  readonly metric: 'cpu' | 'memory' | 'gpu' | string;
  readonly code: string;
  /** Short and operator-facing, e.g. "no GPU visible". Rendered verbatim. */
  readonly message: string;
}

export interface HostResources {
  readonly observed_at: string;
  readonly cpu: { readonly percent: number; readonly logical_cores: number | null } | null;
  readonly memory: { readonly used_bytes: number; readonly total_bytes: number } | null;
  readonly gpus: readonly GpuTelemetry[];
  readonly unavailable: readonly HostUnavailable[];
}

export const HOST_RESOURCES_PATH = `${V1}/host/resources`;

export const fetchHostResources = (): Promise<ApiResult<HostResources>> =>
  apiGet<HostResources>(HOST_RESOURCES_PATH);
