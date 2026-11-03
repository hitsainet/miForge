/**
 * F8's budget estimator (FR-009.36, M20).
 *
 * **The estimate is a preview, not a permission.** `estimateBudget` returns the figures and
 * nothing else; there is no `onValid`, no boolean and no verdict-to-disabled mapping anywhere
 * downstream of it. F8's estimator is *the same code path* as the gate, which is what makes
 * the preview truthful - and still not the decision. The authoritative refusal is the create
 * path's; moving it into the browser is **M20**, written before the preview was wired.
 *
 * `verdict` is `unknown` rather than `fits` on a machine with no card, because *"I could not
 * read the limit, so it fits"* is the pass-through failure wearing the estimator's name.
 */
import { apiPost, type ApiResult } from './client';
import { V1 } from './v1';

export interface BudgetTerm {
  readonly name: string;
  readonly bytes: number;
  readonly [key: string]: unknown;
}

/** Where the server read the model's size: a directory, a checkpoint, the Hub, a config. */
export interface BudgetSource {
  readonly kind: string;
  readonly param_count?: number | null;
  readonly from_config?: boolean;
  readonly path?: string;
  readonly repo?: string;
  readonly revision_resolved?: string;
  readonly [key: string]: unknown;
}

/** One card the run could be placed on, with its own verdict. Any number of cards. */
export interface BudgetCard {
  readonly uuid: string | null;
  readonly name: string | null;
  readonly free_bytes: number | null;
  readonly verdict: string;
}

export interface BudgetEstimate {
  /** The figure a start compares: the upper end of the estimate. */
  readonly estimate_bytes?: number;
  readonly estimate_low_bytes?: number;
  readonly terms?: readonly BudgetTerm[];
  readonly free_bytes?: number | null;
  readonly verdict?: string;
  readonly gpu_uuid?: string | null;
  readonly gpu_name?: string | null;
  readonly source?: BudgetSource | null;
  readonly cards?: readonly BudgetCard[];
  readonly [key: string]: unknown;
}

export interface BudgetRequest {
  readonly run_type: string;
  readonly config: Readonly<Record<string, unknown>>;
  /**
   * The run definition's model source, exactly as the form submits it. The server reads the
   * model's size from it - this screen never computes or guesses a size.
   */
  readonly model_source?: Readonly<Record<string, unknown>>;
  readonly architecture_config?: Readonly<Record<string, unknown>> | null;
  readonly param_count?: number | null;
  readonly adapter?: Readonly<Record<string, unknown>> | null;
  readonly gpu_request?: string;
}

export const BUDGET_ESTIMATE_PATH = `${V1}/budget/estimate`;

export const estimateBudget = (request: BudgetRequest): Promise<ApiResult<BudgetEstimate>> =>
  apiPost<BudgetEstimate>(BUDGET_ESTIMATE_PATH, request);
