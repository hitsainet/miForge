/**
 * F4's score series (FR-009.10, .12, .13, .14).
 *
 * The response carries `provider_discontinuities` - **the steps at which the instrument
 * changed** - computed server-side from each row's `provider_discontinuity` flag. The curve
 * breaks there rather than joining two providers' numbers into one line (F4 FR-004.10,
 * FR-009.13); joining them is mutation **M18**. It also carries `exposure_window_steps`,
 * which is why a retention floor is a measurement with a lag rather than a guarantee.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';
import type { ScoreSeries } from '@/types/run';

export const fetchScores = (runId: string): Promise<ApiResult<ScoreSeries>> =>
  apiGet<ScoreSeries>(`${V1}/runs/${encodeURIComponent(runId)}/scores`);
