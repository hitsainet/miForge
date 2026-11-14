/**
 * The **live** run-type registry (FR-009.33's table, M22).
 *
 * `GET /api/v1/run-types` answers from `training.runtypes.known()`, which F8 built to be a
 * reachability target: removing a `register(...)` line must shorten this body. So the form
 * reads it and **never** holds a literal `['change','create','adapter']` - a hard-coded list
 * is a second registry, and a fourth registered type must appear with no code change here.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

export interface RunTypeEntry {
  readonly name: string;
  readonly stored_as: string;
  /** One line per type, supplied by the registry so the form writes no copy of its own. */
  readonly changes: string;
}

export interface RunTypeListing {
  readonly run_types: readonly RunTypeEntry[];
  readonly count: number;
}

export const RUN_TYPES_PATH = `${V1}/run-types`;

export const fetchRunTypes = (): Promise<ApiResult<RunTypeListing>> =>
  apiGet<RunTypeListing>(RUN_TYPES_PATH);
