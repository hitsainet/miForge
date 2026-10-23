/**
 * `GET /api/v1/stop-conditions` - R-02.25's conditions from the **live registry** (task 14.0).
 *
 * F6 added this route for the New run form, and until 2026-10-18 the form did not read it: it
 * offered a hard-coded list and filed each condition's one value under a key table that was
 * wrong for two of the five (`step_count` reads `total`, not `steps`; `retention_floor` takes
 * no parameter). The parameter names are now the registry's.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

/** One entry - `runs_loop.py::list_stop_conditions`. */
export interface StopConditionEntry {
  readonly condition: string;
  /** Whether the condition takes parameters at all. */
  readonly configured: boolean;
  /** The parameter names, in the order a refusal names them. No defaults: each is required. */
  readonly required_parameters: readonly string[];
  readonly precedence_rank: number;
  readonly summary: string;
  /** `loss_plateau` only: the closed set its `metric` parameter takes. */
  readonly metrics?: readonly string[];
}

export interface StopConditionRegistry {
  readonly conditions: readonly StopConditionEntry[];
  readonly count: number;
  readonly closed: boolean;
  readonly precedence: readonly string[];
  readonly minimum_enabled: number;
}

export const STOP_CONDITIONS_PATH = `${V1}/stop-conditions`;

export const fetchStopConditionRegistry = (): Promise<ApiResult<StopConditionRegistry>> =>
  apiGet<StopConditionRegistry>(STOP_CONDITIONS_PATH);
