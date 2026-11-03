/**
 * The step log (FR-009.15; ADR-012).
 *
 * Monospace, newest first, bounded **with the bound rendered**, one line per recorded event,
 * coloured by type. It reads F1's step records, F4's score series, F5's checkpoint list and
 * F2's batch records - never a client-side reconstruction of any of them.
 *
 * ## Three things it has to get right
 *
 * 1. **Counts are three-valued.** `Accepted 41, rejected 11, undecided 12 of 64`. *A single
 *    place that treats `undecided` as a reject makes the health record lie about an unattended
 *    queue*, and this log is one of those places. The counts come from F2's batch record,
 *    which carries all three precisely so the two-valued form is not expressible.
 * 2. **The §5.10 events do not exist here.** `quiesce`, `commit`, `torn`, "miLLM resumed",
 *    "step retried" are F11's. The mockup's `quiesce timeout · miLLM resumed · step retried`
 *    line belongs to F11's screens. There is **no case** for any of them: one arriving renders
 *    through the default branch as an unrecognised event naming its type, which is also what
 *    the test asserts.
 * 3. **The default branch.** An event type this build does not know names itself rather than
 *    rendering as blank or throwing (FR-009.8).
 *
 * ## One deliberate divergence from this feature's own FTID, reported not silently taken
 *
 * F9's FTID section 3.8 and FTASKS 7.1 both say *"save accent"*. **ADR-012 and this feature's
 * own FR-009.3 permit the accent on seven surfaces and a step-log line is not one of them** -
 * FR-009.3 enumerates "the active nav item, primary buttons, the logo tile, list-row icon
 * tiles, the loss curve, a running run's progress bar, and a callout banner tint", and the
 * accent test enforces exactly that list. So a save renders at **full `text` weight** - the
 * only event type that does, which distinguishes it - and the divergence is reported rather
 * than resolved by widening the allowlist for one line of a log.
 */
import type { ReactNode } from 'react';
import { MonoId } from '@/components/common/MonoId';
import { Card } from '@/components/common/Card';
import { selectionCountsText } from '@/components/common/SelectionCounts';
import type { BatchRecord, ScoreSeriesRow, StepRecord } from '@/types/run';

export type LogEventType = 'step' | 'batch' | 'score' | 'save' | 'warning';

export interface LogEvent {
  readonly step: number;
  /** A string, not a union, so an event this build does not know can still be carried. */
  readonly type: string;
  readonly text: string;
  readonly detail?: string;
}

/** Tone per known type. Nothing here is accent - see the module docstring. */
const TONES: Readonly<Record<LogEventType, string>> = {
  step: 'text-dim',
  batch: 'text-dim',
  score: 'text-green',
  // The only line at full text weight, which is how a save is distinguished without orange.
  save: 'text-text font-semibold',
  warning: 'text-amber',
};

export function toneFor(type: string): string {
  const known = TONES[type as LogEventType];
  // The default branch. An unrecognised type is rendered and named, never specially styled -
  // which is what keeps a §5.10 event from acquiring a presentation here (ADR-022).
  return known ?? 'text-dim';
}

export function isKnownEventType(type: string): boolean {
  return Object.prototype.hasOwnProperty.call(TONES, type);
}

export interface StepLogProps {
  readonly steps: readonly StepRecord[];
  readonly scores: readonly ScoreSeriesRow[];
  readonly batches: readonly BatchRecord[];
  readonly savePoints: readonly number[];
  /** Events from elsewhere, e.g. a socket payload. Carried so the default branch is reachable. */
  readonly extra?: readonly LogEvent[];
  readonly limit?: number;
}

export const DEFAULT_LOG_LIMIT = 200;

/** Newest first, bounded. The bound is rendered by the component, not implied. */
export function buildLog(props: StepLogProps): readonly LogEvent[] {
  const events: LogEvent[] = [];

  for (const step of props.steps) {
    events.push({
      step: step.step,
      type: 'step',
      text:
        step.loss === null
          ? `step ${step.step} · no loss recorded`
          : `step ${step.step} · loss ${step.loss.toFixed(4)}`,
      ...(step.wall_time_ms === null ? {} : { detail: `${step.wall_time_ms} ms` }),
    });
  }

  for (const batch of props.batches) {
    events.push({
      step: batch.step,
      type: 'batch',
      // Three-valued, from the record. See hint 1 in the module docstring.
      // The step is named on the line, not only in the sort key: a log line an operator can
      // copy out of the screen has to say which step it belongs to.
      text: `step ${batch.step} · batch ${batch.source} · ${selectionCountsText({
        n: batch.n_prompts,
        accepted: batch.verdicts.accept,
        rejected: batch.verdicts.reject,
        undecided: batch.verdicts.undecided,
      })}`,
      detail: batch.batch_sha256.slice(0, 12),
    });
  }

  for (const score of props.scores) {
    events.push({
      step: score.step,
      // A breach is a warning; an ordinary evaluation is not. The flag is the server's.
      type: score.breached ? 'warning' : 'score',
      text: score.message,
      ...(score.out_of_cadence ? { detail: 'out of cadence' } : {}),
    });
  }

  for (const step of props.savePoints) {
    events.push({ step, type: 'save', text: `checkpoint saved at step ${step}` });
  }

  for (const event of props.extra ?? []) events.push(event);

  return events.sort((a, b) => b.step - a.step || a.type.localeCompare(b.type));
}

export function StepLog(props: StepLogProps): ReactNode {
  const limit = props.limit ?? DEFAULT_LOG_LIMIT;
  const all = buildLog(props);
  const shown = all.slice(0, limit);

  return (
    <Card data-testid="step-log">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-heading">Step log</div>
        {/* The bound, rendered. A page quietly shorter than the run reads as the end of it. */}
        <div className="text-hint text-dim" data-testid="step-log-bound">
          {shown.length === all.length
            ? `${all.length} event${all.length === 1 ? '' : 's'}, all shown`
            : `newest ${shown.length} of ${all.length} events`}
        </div>
      </div>
      {shown.length === 0 ? (
        <div className="text-hint text-dim">
          Nothing recorded yet. The log fills from the run&apos;s own step, batch, evaluation and
          checkpoint records as they are written.
        </div>
      ) : (
        <ol className="stack-y-0.5">
          {shown.map((event, index) => (
            <li
              key={`${event.step}-${event.type}-${index}`}
              className={`font-mono text-ident ${toneFor(event.type)}`}
              data-event-type={event.type}
              data-event-known={isKnownEventType(event.type) ? 'true' : 'false'}
            >
              {isKnownEventType(event.type) ? null : (
                <span className="mr-1 text-dim">[unrecognised event {event.type}]</span>
              )}
              {event.text}
              {event.detail === undefined ? null : (
                <MonoId value={event.detail} className="ml-2 text-dim" />
              )}
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}
