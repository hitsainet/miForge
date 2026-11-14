/**
 * Four tiles, each naming its scale, its sample and the step it is from (FR-009.14).
 *
 * Test and retention run on a **cadence** and loss does not, so the latest test score is
 * frequently not from the run's current step. Every tile therefore carries `atStep` when the
 * value's step differs from the run's, because `0.84` shown beside `step 590` when the score
 * was taken at 500 is a number quoted against the wrong sample.
 *
 * **The retention tile renders the floor only when one is recorded.** `ScoreRecord.required`
 * is `[set, score, n]` - `floor` is optional, parsed from the vendored schema - so an absent
 * floor goes through `AbsentValue`. `floor ?? 0` is mutation **M12**, and it is the worst
 * available rendering: `0` says *"nothing is protected and that is the recorded policy"*,
 * which is the opposite of *"no floor was recorded"*.
 */
import type { ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { AbsentValue } from '@/components/common/AbsentValue';
import { NumberWithSample, formatCount } from '@/components/common/NumberWithSample';
import type { ScoreSeriesRow, StepRecord } from '@/types/run';

export interface StatTilesProps {
  readonly currentStep: number | null;
  readonly totalSteps: number | null;
  readonly latestStep: StepRecord | null;
  readonly latestTest: ScoreSeriesRow | null;
  readonly latestRetention: ScoreSeriesRow | null;
}

function Tile({
  label,
  children,
  testId,
}: {
  readonly label: string;
  readonly children: ReactNode;
  readonly testId: string;
}): ReactNode {
  return (
    <Card className="min-w-0" data-testid={testId}>
      <div className="mb-1 text-hint text-dim">{label}</div>
      <div className="text-heading">{children}</div>
    </Card>
  );
}

export function StatTiles({
  currentStep,
  totalSteps,
  latestStep,
  latestTest,
  latestRetention,
}: StatTilesProps): ReactNode {
  return (
    <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <Tile label="Step" testId="tile-step">
        {currentStep === null ? (
          // No step recorded is not step zero. A run that has not stepped has no step.
          <AbsentValue reason="no_test_at_this_step" detail="no step recorded yet" />
        ) : totalSteps === null ? (
          // `590 of 800` needs the 800, and no response carries it yet - F6's `total_steps`
          // is not on `GET /api/v1/runs/{id}`. The step is stated with its sample being
          // itself rather than invented against a total nothing published.
          <NumberWithSample
            value={formatCount(currentStep)}
            sample={currentStep}
            sampleLabel="steps recorded"
          />
        ) : (
          <NumberWithSample
            value={`${formatCount(currentStep)} of ${formatCount(totalSteps)}`}
            sample={totalSteps}
            sampleLabel="steps planned"
          />
        )}
      </Tile>

      <Tile label="Loss" testId="tile-loss">
        {latestStep === null || latestStep.loss === null ? (
          <AbsentValue reason="no_test_at_this_step" detail="no loss recorded" />
        ) : (
          <NumberWithSample
            value={latestStep.loss.toFixed(4)}
            sample={1}
            sampleLabel="step"
            {...(currentStep !== null && latestStep.step !== currentStep
              ? { atStep: latestStep.step }
              : {})}
          />
        )}
      </Tile>

      <Tile label="Test" testId="tile-test">
        {latestTest === null ? (
          <AbsentValue reason="no_test_at_this_step" />
        ) : (
          <NumberWithSample
            value={latestTest.score.toFixed(3)}
            sample={latestTest.n}
            sampleLabel="prompts"
            {...(latestTest.provider === null ? {} : { provider: latestTest.provider })}
            {...(currentStep !== null && latestTest.step !== currentStep
              ? { atStep: latestTest.step }
              : {})}
          />
        )}
      </Tile>

      <Tile label="Retention" testId="tile-retention">
        {latestRetention === null ? (
          <AbsentValue reason="no_test_at_this_step" />
        ) : (
          <NumberWithSample
            value={latestRetention.score.toFixed(3)}
            sample={latestRetention.n}
            sampleLabel="prompts"
            {...(latestRetention.provider === null ? {} : { provider: latestRetention.provider })}
            {...(currentStep !== null && latestRetention.step !== currentStep
              ? { atStep: latestRetention.step }
              : {})}
            {...(latestRetention.floor === null
              ? {}
              : { qualifier: `floor ${latestRetention.floor.toFixed(3)}` })}
          />
        )}
        {latestRetention !== null && latestRetention.floor === null ? (
          <AbsentValue reason="no_floor_recorded" variant="block" className="mt-1" />
        ) : null}
      </Tile>
    </div>
  );
}

/** The newest row of one score kind, or `null`. Out-of-cadence rows are included: they are */
/* real evaluations, recorded with `out_of_cadence: true` so they never silently join the
   step series - but the *latest* value is still the latest value. */
export function latestOfKind(rows: readonly ScoreSeriesRow[], kind: string): ScoreSeriesRow | null {
  let best: ScoreSeriesRow | null = null;
  for (const row of rows) {
    if (row.kind !== kind) continue;
    if (best === null || row.step >= best.step) best = row;
  }
  return best;
}
