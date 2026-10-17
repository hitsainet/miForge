/**
 * The two extra step-record series for a §5.10 run (F11 task 16.4; R-02.14; ADR-012).
 *
 * `quiesce_hold_ms` and `commit_duration_ms` are written on every step of an attached run
 * and are `null` on every standalone step - that `null` is the assertion that no bracket
 * was held, so this panel is not rendered at all on the default path.
 *
 * **Every figure names its sample** (ADR-012): *"p99.9 = 14.4 ms over 10,000 commits,
 * serving concurrently"*, never *"14.4 ms"*. And a p99.9 is **not reported below 1,000
 * samples**: over fifty steps the 99.9th percentile is simply the maximum, and quoting it
 * as a percentile would hand miLLM's watchdog review a number that is not what it says.
 * The sample is the steps **loaded on this page**, said in words, because the step listing
 * is paginated newest-first and a page is not the run.
 */
import type { ReactNode } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Card } from '@/components/common/Card';
import { NumberWithSample, formatCount } from '@/components/common/NumberWithSample';
import type { StepRecord } from '@/types/run';

/** Below this many samples a p99.9 is the maximum by another name, and is not reported. */
export const P999_MIN_SAMPLES = 1000;

export interface SeriesSummary {
  readonly n: number;
  readonly p50: number | null;
  readonly p999: number | null;
  readonly max: number | null;
}

/**
 * Nearest-rank percentile, the same definition as the backend's
 * `run_log.commit_duration_percentile` - never interpolated, because an interpolated
 * duration is one that was never measured.
 */
export function nearestRank(sorted: readonly number[], percentile: number): number | null {
  if (sorted.length === 0) return null;
  const rank = Math.max(1, Math.ceil((percentile / 100) * sorted.length));
  return sorted[Math.min(rank, sorted.length) - 1] ?? null;
}

export function summarise(values: readonly (number | null)[]): SeriesSummary {
  const present = values.filter((v): v is number => v !== null).sort((a, b) => a - b);
  return {
    n: present.length,
    p50: nearestRank(present, 50),
    p999: present.length >= P999_MIN_SAMPLES ? nearestRank(present, 99.9) : null,
    max: present.length === 0 ? null : (present[present.length - 1] ?? null),
  };
}

/** Whether any loaded step carries a §5.10 timing - the panel's own render condition. */
export function hasHotUpdateTimings(steps: readonly StepRecord[]): boolean {
  return steps.some(
    (s) => (s.quiesce_hold_ms ?? null) !== null || (s.commit_duration_ms ?? null) !== null,
  );
}

function SummaryLine({
  title,
  summary,
  sampleLabel,
  testId,
}: {
  readonly title: string;
  readonly summary: SeriesSummary;
  readonly sampleLabel: string;
  readonly testId: string;
}): ReactNode {
  if (summary.n === 0) {
    return (
      <div className="text-hint text-dim" data-testid={testId}>
        {title}: no samples on the loaded steps.
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1" data-testid={testId}>
      <span className="text-heading">{title}</span>
      <NumberWithSample
        value={(summary.p50 ?? 0).toFixed(1)}
        unit="ms"
        sample={summary.n}
        sampleLabel={sampleLabel}
        qualifier="p50"
      />
      {summary.p999 === null ? (
        <span className="text-hint text-dim" data-testid={`${testId}-p999-withheld`}>
          p99.9 not reported: {formatCount(summary.n)} {sampleLabel} is below the{' '}
          {formatCount(P999_MIN_SAMPLES)} a 99.9th percentile needs
        </span>
      ) : (
        <NumberWithSample
          value={summary.p999.toFixed(1)}
          unit="ms"
          sample={summary.n}
          sampleLabel={sampleLabel}
          qualifier="p99.9"
        />
      )}
      <NumberWithSample
        value={(summary.max ?? 0).toFixed(1)}
        unit="ms"
        sample={summary.n}
        sampleLabel={sampleLabel}
        qualifier="max"
      />
    </div>
  );
}

export interface HotUpdateTimingsProps {
  readonly steps: readonly StepRecord[];
}

export function HotUpdateTimings({ steps }: HotUpdateTimingsProps): ReactNode {
  const holds = summarise(steps.map((s) => s.quiesce_hold_ms ?? null));
  const commits = summarise(steps.map((s) => s.commit_duration_ms ?? null));
  const points = [...steps]
    .sort((a, b) => a.step - b.step)
    .map((s) => ({
      step: s.step,
      hold: s.quiesce_hold_ms ?? null,
      commit: s.commit_duration_ms ?? null,
    }));
  return (
    <Card className="mb-4" data-testid="hotupdate-timings">
      <div className="mb-1 text-heading">Commit bracket timings (§5.10)</div>
      <div className="mb-3 text-hint text-dim">
        Over the {formatCount(steps.length)} steps loaded on this page, newest first — a page, not
        the whole run. Measured while miLLM served concurrently; miLLM&apos;s commit watchdog is
        checked against the commit duration&apos;s p99.9.
      </div>
      <SummaryLine
        title="Quiesce hold"
        summary={holds}
        sampleLabel="holds"
        testId="series-quiesce-hold"
      />
      <SummaryLine
        title="Commit duration"
        summary={commits}
        sampleLabel="commits"
        testId="series-commit-duration"
      />
      <div className="mt-3 h-48" data-testid="hotupdate-timings-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--mf-line)" />
            <XAxis dataKey="step" stroke="var(--mf-dim)" />
            <YAxis stroke="var(--mf-dim)" unit=" ms" />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="commit"
              name={`Commit duration (${formatCount(commits.n)} commits)`}
              stroke="var(--mf-cyan)"
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="hold"
              name={`Quiesce hold (${formatCount(holds.n)} holds)`}
              stroke="var(--mf-dim)"
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
