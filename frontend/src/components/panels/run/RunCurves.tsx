/**
 * Three series, one chart, two kinds of gap (FR-009.10, .11, .12, .13, .20).
 *
 * ## `connectNulls` is off, and the gaps are explicit nulls
 *
 * Two things must render as breaks, and both are **built as `null` points rather than by
 * filtering the array**: a provider discontinuity (FR-009.13) and a step where no test ran
 * (S1). A filtered array makes Recharts join straight across the gap, and joining a score
 * series across a provider change is mutation **M18** - *a score is comparable only against
 * another taken the same way.*
 *
 * ## The floor is drawn only when it is recorded
 *
 * `ScoreRecord.required` is `[set, score, n]`; `floor` is optional. An absent floor renders
 * `AbsentValue reason="no_floor_recorded"` in the legend area and **no line** - because a
 * line at zero is a policy the run does not have (**M12**).
 *
 * ## Colour
 *
 * Loss **accent** - the one series ADR-012 permits it on. Test green, retention cyan, save
 * marks dashed `dim`, the floor amber. *FPRD F-7 is live and visible here: cyan is also the
 * suite's reserved colour for miLLM state, and after the standalone reframe the retention
 * curve has nothing to do with miLLM. Both ADR-012 and the mockup assign cyan to this curve,
 * so it is kept - and the finding is left in this comment, because a reader enforcing the
 * reservation will otherwise read the curve as a violation.*
 */
import type { ReactNode } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Card } from '@/components/common/Card';
import { AbsentValue } from '@/components/common/AbsentValue';
import { MonoId } from '@/components/common/MonoId';
import { describeDecimation, useDecimatedSeries } from '@/hooks/useDecimatedSeries';
import type { ScoreSeriesRow, StepRecord } from '@/types/run';
import { SavePointMarks } from './SavePointMarks';

export interface CurvePoint {
  readonly step: number;
  readonly loss: number | null;
  readonly test: number | null;
  readonly retention: number | null;
}

export interface RunCurvesProps {
  readonly steps: readonly StepRecord[];
  readonly scores: readonly ScoreSeriesRow[];
  /** The steps at which a checkpoint exists. Passed straight through to `SavePointMarks`. */
  readonly savePoints: readonly number[];
  /** The steps at which the instrument changed, from F4's own response. */
  readonly providerDiscontinuities: readonly number[];
  readonly floor: number | null;
}

/**
 * The legend reads Loss, Test, Retention - the order of the stat tiles above the chart.
 * Recharts 3 sorts legend items ALPHABETICALLY by default (`itemSorter="value"`), which turned
 * it into Loss, Retention, Test on the upgrade (2026-10-21); `null` leaves the order unstable
 * between renders, so the order is stated here. Pinned by `e2e/run-curves.spec.ts`.
 */
const LEGEND_ORDER: readonly string[] = ['loss', 'test', 'retention'];
export const legendOrder = (item: { readonly dataKey?: unknown }): number =>
  LEGEND_ORDER.indexOf(String(item.dataKey));

/**
 * Build the plotted series.
 *
 * A step with no test gets `test: null` - **present in the array with a null value**, which is
 * what makes the gap a gap rather than a join. A step at a provider discontinuity gets a null
 * for the kind that changed, immediately before its new value, so the line breaks there and
 * the two providers' numbers are never joined into one segment.
 */
export function buildCurve(
  steps: readonly StepRecord[],
  scores: readonly ScoreSeriesRow[],
  discontinuities: readonly number[],
): readonly CurvePoint[] {
  const breaks = new Set(discontinuities);
  const byStep = new Map<
    number,
    { loss: number | null; test: number | null; retention: number | null }
  >();

  for (const step of steps) {
    byStep.set(step.step, { loss: step.loss, test: null, retention: null });
  }
  for (const score of scores) {
    const entry = byStep.get(score.step) ?? { loss: null, test: null, retention: null };
    // A discontinuity's own step keeps its value; the break is drawn by the null point
    // inserted *before* it below, so the new provider's first reading is still plotted.
    if (score.kind === 'test') entry.test = score.score;
    if (score.kind === 'retention') entry.retention = score.score;
    byStep.set(score.step, entry);
  }

  const ordered = [...byStep.entries()].sort((a, b) => a[0] - b[0]);
  const points: CurvePoint[] = [];
  for (const [step, entry] of ordered) {
    if (breaks.has(step)) {
      // An explicit null point at a half-step. Recharts plots it, finds no value, and ends
      // the segment - which is the break. Filtering instead would join across it (M18).
      points.push({ step: step - 0.5, loss: null, test: null, retention: null });
    }
    points.push({ step, loss: entry.loss, test: entry.test, retention: entry.retention });
  }
  return points;
}

/** The providers on either side of a break, so the label names both (FR-009.13). */
export function describeBreak(
  scores: readonly ScoreSeriesRow[],
  step: number,
): { readonly before: string | null; readonly after: string | null } {
  const sorted = [...scores].sort((a, b) => a.step - b.step);
  const index = sorted.findIndex((row) => row.step === step);
  if (index < 0) return { before: null, after: null };
  const after = sorted[index]?.provider ?? null;
  let before: string | null = null;
  for (let i = index - 1; i >= 0; i -= 1) {
    const candidate = sorted[i];
    if (candidate !== undefined && candidate.provider !== null) {
      before = candidate.provider;
      break;
    }
  }
  return { before, after };
}

export function RunCurves({
  steps,
  scores,
  savePoints,
  providerDiscontinuities,
  floor,
}: RunCurvesProps): ReactNode {
  const raw = buildCurve(steps, scores, providerDiscontinuities);
  const decimated = useDecimatedSeries(
    raw,
    (point) => point.loss === null && point.test === null && point.retention === null,
  );

  if (raw.length === 0) {
    return (
      <Card data-testid="run-curves-empty">
        <div className="mb-1 text-heading">Loss, test and retention</div>
        <AbsentValue
          reason="no_test_at_this_step"
          variant="block"
          detail="this run has recorded no steps and no scores yet"
        />
      </Card>
    );
  }

  return (
    <Card data-testid="run-curves">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-heading">Loss, test and retention over steps</div>
        {/* The decimation and the underlying sample, named. A decimated curve presented as
            the raw series is a number quoted without its sample (ADR-012, FR-009.20). */}
        <div className="text-hint text-dim" data-testid="curve-decimation">
          {describeDecimation(decimated, 'steps')}
        </div>
      </div>

      <div style={{ height: 280 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={[...decimated.points]}
            margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
          >
            <CartesianGrid stroke="var(--mf-line)" strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              type="number"
              domain={['dataMin', 'dataMax']}
              stroke="var(--mf-dim)"
              tick={{ fontSize: 12 }}
            />
            <YAxis stroke="var(--mf-dim)" tick={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{
                background: 'var(--mf-card)',
                border: '1px solid var(--mf-line)',
                borderRadius: 8,
              }}
            />
            <Legend itemSorter={legendOrder} />
            <SavePointMarks steps={savePoints} />
            {/* Only when recorded. No line, and a named absence, otherwise (E-1, M12). */}
            {floor === null ? null : (
              <ReferenceLine
                y={floor}
                stroke="var(--mf-amber)"
                strokeDasharray="6 3"
                label={{
                  value: `floor ${floor.toFixed(3)}`,
                  fill: 'var(--mf-amber)',
                  fontSize: 12,
                }}
              />
            )}
            <Line
              type="monotone"
              dataKey="loss"
              name="Loss"
              /* accent-ok:loss-curve - ADR-012 names "the loss curve" as a permitted accent
                 surface. It is the only series on this chart that may be orange. */
              stroke="var(--mf-accent)"
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="test"
              name="Test"
              stroke="var(--mf-green)"
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="retention"
              name="Retention"
              stroke="var(--mf-cyan)"
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 stack-y-1 text-hint text-dim">
        {floor === null ? (
          <AbsentValue reason="no_floor_recorded" variant="block" data-testid="curve-no-floor" />
        ) : null}
        {savePoints.length === 0 ? (
          <div data-testid="curve-no-saves">
            No checkpoint exists for this run yet, so no save points are marked. The marks come from
            the checkpoints on disk, never from the save cadence.
          </div>
        ) : (
          <div data-testid="curve-saves">
            {savePoints.length} save point{savePoints.length === 1 ? '' : 's'} marked, at the steps
            where a checkpoint exists.
          </div>
        )}
        {providerDiscontinuities.map((step) => {
          const sides = describeBreak(scores, step);
          return (
            <div key={step} className="text-amber" data-testid={`curve-break-${step}`}>
              Series broken at step {step}: the provider changed from{' '}
              <MonoId value={sides.before ?? 'unrecorded'} /> to{' '}
              <MonoId value={sides.after ?? 'unrecorded'} />. A score is comparable only against
              another taken the same way, so the two are not joined.
            </div>
          );
        })}
      </div>
    </Card>
  );
}
