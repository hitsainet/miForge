/**
 * F3's capability listing (FR-009.37's render-time courtesy).
 *
 * Availability is computed **per call from live settings**, so an operator who configures
 * miDataworks sees the human-review type become available without a restart. An unavailable
 * type is **listed, naming the service it needs** - never omitted and never a mute disabled
 * control, because an option missing from a response is indistinguishable from one that is
 * broken (ADR-013).
 *
 * This is a courtesy only. The authoritative availability answer is F6's validator
 * (FR-006.60, step 6), and the form performs no check of its own.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

export interface SelectorCapability {
  readonly name: string;
  readonly forms: readonly string[];
  readonly requires: string;
  readonly summary: string;
  readonly available: boolean;
  /** The refusal code a call would raise. `null` exactly when `available`. */
  readonly unavailable_reason: string | null;
}

export const SELECTORS_PATH = `${V1}/selectors`;

export const fetchSelectors = (): Promise<ApiResult<readonly SelectorCapability[]>> =>
  apiGet<readonly SelectorCapability[]>(SELECTORS_PATH);
