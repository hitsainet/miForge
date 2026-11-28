/**
 * Decimation that names its own sample (FR-009.20; AC-009.23).
 *
 * The property that matters beyond "fewer points": **a null survives decimation.** The two
 * gaps this chart must show are explicit `null` points, and a decimator that dropped them
 * would close both - which is M18's failure arriving by a different route.
 */
import { describe, expect, it } from 'vitest';
import { DEFAULT_MAX_POINTS, decimate, describeDecimation } from './useDecimatedSeries';

const isNull = (row: number | null): boolean => row === null;

describe('[decimate]', () => {
  it('should leave a series under the budget untouched and say so', () => {
    const rows = [1, 2, 3];
    const series = decimate<number | null>(rows, isNull, 10);
    expect(series.points).toBe(rows);
    expect(series.ratio).toBe(1);
    expect(series.rawCount).toBe(3);
    expect(describeDecimation(series)).toBe('every point shown, 3 steps');
  });

  it('should report the ratio and the raw count, so the chart can name its sample', () => {
    // A hook that returned only points could not satisfy ADR-012's rule.
    const rows = Array.from({ length: 50_000 }, (_, index) => index);
    const series = decimate<number | null>(rows, isNull, 2000);
    expect(series.rawCount).toBe(50_000);
    expect(series.ratio).toBe(25);
    expect(series.points.length).toBeLessThanOrEqual(2000);
    expect(describeDecimation(series)).toBe('1 point per 25 steps, 50,000 steps');
  });

  it('should keep a null in preference to a value when a window holds both', () => {
    // Closing a real gap is the dangerous direction; showing one a window earlier than it
    // began is not. So any null in the window wins.
    const rows: (number | null)[] = [1, 2, null, 4, 5, 6];
    const series = decimate<number | null>(rows, isNull, 2);
    expect(series.points).toContain(null);
  });

  it('should never drop every point of a long series', () => {
    const rows = Array.from({ length: 1000 }, (_, index) => index);
    expect(decimate<number | null>(rows, isNull, 10).points.length).toBeGreaterThan(0);
  });

  it('should name its default budget rather than hiding it', () => {
    // ADR-024: a number quoted without its topology is not a measurement, so the figure is a
    // named constant with its reasoning in the module rather than a literal in a call.
    expect(DEFAULT_MAX_POINTS).toBe(2000);
  });
});
