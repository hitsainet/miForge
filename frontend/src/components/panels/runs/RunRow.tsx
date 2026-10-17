/**
 * One run, enough to open it (FR-009.19).
 *
 * State comes **from the run record** and goes through `StatePill`'s default branch, which
 * stays because an enum grows and a pill that throws takes a screen down. The mockup's `held`
 * is defined by no requirement and is not here.
 *
 * A stopped run shows F6's **`stop_reason`** rather than a bare "done" - which of the five
 * stop conditions fired is the one thing an operator wants from a finished run. The field is
 * **not on this response yet**, so it renders as a named absence rather than as a guess.
 *
 * **A running run's progress bar is accent; every other state's is not** (FR-009.3). That is
 * one of the two mockup bugs this feature fixes: its bar was accent for every non-`held` run
 * including `done` ones, and its `done` pill was green while the bar above it was orange.
 */
import type { ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { MonoId } from '@/components/common/MonoId';
import { StatePill } from '@/components/common/Pill';
import { Button } from '@/components/common/Button';
import { AbsentValue } from '@/components/common/AbsentValue';
import { NumberWithSample, formatCount } from '@/components/common/NumberWithSample';
import { describeLiveness, deriveLiveness, isTerminal } from '@/stores/runStore';
import type { RunRecord, ScoreSeriesRow } from '@/types/run';

export interface RunRowProps {
  readonly record: RunRecord;
  readonly latestTest: ScoreSeriesRow | null;
  readonly latestRetention: ScoreSeriesRow | null;
  readonly onOpen: (runId: string) => void;
}

/**
 * The progress bar.
 *
 * Accent **only** while running. `total` is `null` in this build - no response carries a step
 * total - so there is no proportion to draw and the bar is an indeterminate marker rather than
 * a fraction of an invented denominator.
 */
export function ProgressBar({
  running,
  step,
  total,
}: {
  readonly running: boolean;
  readonly step: number | null;
  readonly total: number | null;
}): ReactNode {
  const fraction =
    total === null || total === 0 || step === null ? null : Math.min(1, step / total);
  return (
    <div
      className="h-1.5 w-full overflow-hidden rounded-full bg-panel"
      data-testid="run-progress"
      data-running={running ? 'true' : 'false'}
      role="presentation"
    >
      <div
        /* accent-ok:running-progress - ADR-012 names "running progress bars". The ternary is
           the rule: every other state's bar is `dim`, which is the mockup bug FR-009.3 fixes. */
        className={`h-full ${running ? 'bg-accent' : 'bg-dim'}`}
        style={{ width: fraction === null ? '100%' : `${Math.round(fraction * 100)}%` }}
      />
    </div>
  );
}

export function RunRow({ record, latestTest, latestRetention, onOpen }: RunRowProps): ReactNode {
  const liveness = deriveLiveness(record, Date.now());
  const running = liveness.kind === 'running';

  return (
    <Card as="li" className="min-w-0" data-testid={`run-row-${record.run_id}`}>
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <MonoId value={record.run_id} />
          <div className="mt-1 flex flex-wrap items-center gap-2 text-hint text-dim">
            <StatePill state={record.status} />
            <span>{record.run_type ?? 'run type not recorded'}</span>
          </div>
        </div>
        <Button onClick={() => onOpen(record.run_id)}>Open this run</Button>
      </div>

      <div className="mt-3">
        <ProgressBar running={running} step={record.last_step} total={record.total_steps ?? null} />
      </div>

      <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 text-hint sm:grid-cols-2">
        <div>
          <dt className="inline text-dim">step: </dt>
          <dd className="inline">
            {record.last_step === null ? (
              <AbsentValue reason="no_test_at_this_step" detail="no step recorded yet" />
            ) : (
              <NumberWithSample
                value={formatCount(record.last_step)}
                sample={record.last_step}
                sampleLabel="steps recorded"
              />
            )}
          </dd>
        </div>
        <div>
          <dt className="inline text-dim">liveness: </dt>
          <dd className="inline" data-liveness={liveness.kind}>
            {describeLiveness(liveness)}
          </dd>
        </div>
        <div>
          <dt className="inline text-dim">test: </dt>
          <dd className="inline">
            {latestTest === null ? (
              <AbsentValue reason="no_test_at_this_step" />
            ) : (
              <NumberWithSample
                value={latestTest.score.toFixed(3)}
                sample={latestTest.n}
                sampleLabel="prompts"
                {...(latestTest.provider === null ? {} : { provider: latestTest.provider })}
                atStep={latestTest.step}
              />
            )}
          </dd>
        </div>
        <div>
          <dt className="inline text-dim">retention: </dt>
          <dd className="inline">
            {latestRetention === null ? (
              <AbsentValue reason="no_test_at_this_step" />
            ) : (
              <NumberWithSample
                value={latestRetention.score.toFixed(3)}
                sample={latestRetention.n}
                sampleLabel="prompts"
                atStep={latestRetention.step}
                {...(latestRetention.floor === null
                  ? {}
                  : { qualifier: `floor ${latestRetention.floor.toFixed(3)}` })}
              />
            )}
          </dd>
        </div>
      </dl>

      {isTerminal(record.status) ? (
        record.stop_reason == null ? (
          <div className="mt-2">
            <AbsentValue
              reason="missing_required_key"
              variant="block"
              detail="stop_reason — not carried by GET /api/v1/runs/{id} in this build (F6 FR-006.10)"
            />
          </div>
        ) : (
          <div className="mt-2 text-hint text-dim">
            Stopped: <MonoId value={record.stop_reason} />
          </div>
        )
      ) : null}
    </Card>
  );
}
