/**
 * R-02.46's acquisition modes, **from the live registry** (FR-009.33's first row).
 *
 * `GET /api/v1/models/sources` lists every mode, available or not, each unavailable one
 * carrying its reason and the capabilities it would have provided - *"an option missing from a
 * response is indistinguishable from one that is broken"* (ADR-013). `attach` appears here
 * even in a build without the hot-update mode, reported unavailable, which is why the form
 * shows it greyed with its reason rather than hiding it.
 *
 * `default` is marked by the server so this screen does not have to know which mode R-02.46
 * names - and the form **does not preselect it**: a non-nullable field with an initial value
 * *is* a default (FR-009.39, M19), and `default: true` on a listing row is information about
 * the backend, not a choice the operator made.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

export interface ModelSourceMode {
  readonly mode: string;
  readonly available: boolean;
  readonly default: boolean;
  readonly reason: string | null;
  readonly message: string | null;
  readonly capabilities_unavailable: readonly string[];
  readonly sources: readonly string[];
}

export interface ModelSourceListing {
  readonly modes: readonly ModelSourceMode[];
  readonly default: string;
}

export const MODEL_SOURCES_PATH = `${V1}/models/sources`;

export const fetchModelSources = (): Promise<ApiResult<ModelSourceListing>> =>
  apiGet<ModelSourceListing>(MODEL_SOURCES_PATH);
