import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  HotUpdateTimings,
  P999_MIN_SAMPLES,
  hasHotUpdateTimings,
  nearestRank,
  summarise,
} from './HotUpdateTimings';
import type { StepRecord } from '@/types/run';

function step(n: number, hold: number | null, commit: number | null): StepRecord {
  return {
    step: n,
    loss: 1,
    grad_norm: 1,
    learning_rate: 1e-5,
    wall_time_ms: 100,
    batch_source_hashes: null,
    quiesce_hold_ms: hold,
    commit_duration_ms: commit,
  };
}

describe('[HotUpdateTimings]', () => {
  it('should use nearest rank, the backend definition, never an interpolated value', () => {
    expect(nearestRank([10, 20, 30], 50)).toBe(20);
    expect(nearestRank([10, 20, 30, 40], 50)).toBe(20);
    expect(nearestRank([10, 20, 30], 100)).toBe(30);
    expect(nearestRank([], 50)).toBeNull();
  });

  it('should withhold p99.9 below the sample it needs, and give it at that sample', () => {
    const small = summarise(Array.from({ length: P999_MIN_SAMPLES - 1 }, (_, i) => i));
    expect(small.p999).toBeNull();
    expect(small.max).toBe(P999_MIN_SAMPLES - 2);
    const big = summarise(Array.from({ length: P999_MIN_SAMPLES }, (_, i) => i + 1));
    // `Math.ceil(99.9 / 100 * 1000)` is 1000 in IEEE doubles, and Python's
    // `math.ceil(99.9 / 100.0 * 1000)` is 1000 too - the panel and
    // `run_log.commit_duration_percentile` agree, which is the property that matters.
    expect(big.p999).toBe(1000);
    expect(big.n).toBe(P999_MIN_SAMPLES);
  });

  it('should not render on a standalone run', () => {
    expect(hasHotUpdateTimings([step(1, null, null), step(2, null, null)])).toBe(false);
    expect(hasHotUpdateTimings([step(1, null, 12.5)])).toBe(true);
  });

  it('should label each series with its own sample, and say it is a page', () => {
    render(
      <HotUpdateTimings steps={[step(3, 230, 12.6), step(2, 231, null), step(1, 229, 13.1)]} />,
    );
    const panel = screen.getByTestId('hotupdate-timings');
    expect(panel).toHaveTextContent('3 steps loaded on this page');
    // Three holds but only two commits: each series counts its own samples.
    expect(screen.getByTestId('series-quiesce-hold')).toHaveTextContent('3 holds');
    expect(screen.getByTestId('series-commit-duration')).toHaveTextContent('2 commits');
    expect(screen.getByTestId('series-commit-duration-p999-withheld')).toHaveTextContent(
      'p99.9 not reported',
    );
  });
});
