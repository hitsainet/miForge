/**
 * Decimation that **names its own sample** (FR-009.20; ADR-012).
 *
 * The hook returns `{points, ratio, rawCount}` rather than only points, because a hook that
 * returned only points could not satisfy ADR-012's rule that every number names its sample:
 * the chart has to be able to render *"1 point per 25 steps, 50,000 steps"*. A decimated
 * curve presented as the raw series is a number quoted without its sample.
 *
 * **Nulls survive decimation.** The two kinds of gap this chart must show - a provider
 * discontinuity and a step where no test ran - are explicit `null` points, and a decimator
 * that dropped them would close both gaps. So a window containing any null yields a null,
 * which is the conservative direction: a break that should not be there is visible, a break
 * that should be there and is not would be a false join (M18's shape).
 */
import { useMemo } from 'react';

export interface DecimatedSeries<T> {
  readonly points: readonly T[];
  /** How many raw points each rendered point stands for. 1 means undecimated. */
  readonly ratio: number;
  readonly rawCount: number;
}

export const DEFAULT_MAX_POINTS = 2000;

/**
 * @param rows the raw series, in step order.
 * @param maxPoints the budget. 2,000 is a browser-side figure, not a measurement: a Recharts
 *   `LineChart` with three series at 2,000 points each is 6,000 SVG path segments, which
 *   paints without stalling on this project's own hardware. The figure is named here so the
 *   next person measures it rather than guessing at why it is what it is (ADR-024).
 */
export function decimate<T>(
  rows: readonly T[],
  isNull: (row: T) => boolean,
  maxPoints: number = DEFAULT_MAX_POINTS,
): DecimatedSeries<T> {
  const rawCount = rows.length;
  if (rawCount <= maxPoints) return { points: rows, ratio: 1, rawCount };

  const ratio = Math.ceil(rawCount / maxPoints);
  const points: T[] = [];
  for (let start = 0; start < rawCount; start += ratio) {
    const window = rows.slice(start, start + ratio);
    // Any null in the window wins. See the module docstring: closing a real gap is the
    // dangerous direction, and showing one window earlier than it began is not.
    //
    // `findIndex`, not `find`. A `null` point IS the value being looked for, and
    // `window.find(...) ?? window[0]` discards it - `??` reads the found `null` as "nothing
    // found" and falls back to the first value, which silently closes every gap. The test
    // caught it; reading the line did not.
    const nulledAt = window.findIndex((row) => isNull(row));
    const chosen = nulledAt >= 0 ? window[nulledAt] : window[0];
    if (nulledAt >= 0 || chosen !== undefined) points.push(chosen as T);
  }
  return { points, ratio, rawCount };
}

export function useDecimatedSeries<T>(
  rows: readonly T[],
  isNull: (row: T) => boolean,
  maxPoints: number = DEFAULT_MAX_POINTS,
): DecimatedSeries<T> {
  return useMemo(() => decimate(rows, isNull, maxPoints), [rows, isNull, maxPoints]);
}

/** `1 point per 25 steps, 50,000 steps` - or the honest thing when nothing was dropped. */
export function describeDecimation(series: DecimatedSeries<unknown>, unit = 'steps'): string {
  const raw = series.rawCount.toLocaleString('en-GB');
  if (series.ratio === 1) return `every point shown, ${raw} ${unit}`;
  return `1 point per ${series.ratio} ${unit}, ${raw} ${unit}`;
}
