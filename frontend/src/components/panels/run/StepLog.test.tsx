/**
 * The step log (FR-009.15), and the three things it has to get right.
 *
 * 1. **Three-valued counts.** `Accepted 41, rejected 11, undecided 12 of 64`. A log line that
 *    shows "accepted 41 / 64" when 12 are pending at a human-review queue is a lie about an
 *    unattended run.
 * 2. **No section 5.10 event has a case.** A `quiesce`, `commit` or `torn` event renders
 *    through the default branch as an unrecognised event naming its type - it acquires no
 *    presentation here, because those are F11's (ADR-022, handoff section 6c).
 * 3. **The bound is rendered.** A page quietly shorter than the run reads as the end of it.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StepLog, buildLog, isKnownEventType, toneFor } from './StepLog';
import type { BatchRecord, ScoreSeriesRow, StepRecord } from '@/types/run';

const steps: readonly StepRecord[] = [
  {
    step: 412,
    loss: 1.2,
    grad_norm: null,
    learning_rate: null,
    wall_time_ms: 410,
    batch_source_hashes: null,
  },
];

const batches: readonly BatchRecord[] = [
  {
    batch_sha256: 'b'.repeat(64),
    step: 412,
    source: 'self_generation',
    resolved_model: 'LFM2-1.2B',
    parts: null,
    n_prompts: 64,
    verdicts: { accept: 41, reject: 11, undecided: 12 },
    selection_health: {
      candidates_generated: 128,
      candidates_admitted: 41,
      acceptance_rate: 0.640625,
      exhausted: false,
    },
    selector_config_sha256: null,
    grader_config_sha256: null,
    created_at: '2026-10-14T12:00:00Z',
  },
];

const scores: readonly ScoreSeriesRow[] = [
  {
    step: 400,
    kind: 'retention',
    set: 'retain@2',
    score: 0.68,
    n: 180,
    floor: 0.72,
    floor_layer: 'pack',
    breached: true,
    provider: 'in-process',
    provider_kind: 'in_process',
    provider_model: 'LFM2-1.2B',
    aggregation: 'mean',
    out_of_cadence: false,
    provider_discontinuity: false,
    message: 'Retention 0.680 on 180 prompts via in-process (floor 0.720) — below the floor.',
  },
];

describe('[StepLog]', () => {
  it('should render three-valued counts, never two', () => {
    render(<StepLog steps={steps} scores={[]} batches={batches} savePoints={[]} />);
    const log = screen.getByTestId('step-log');
    expect(log).toHaveTextContent('Accepted 41, rejected 11, undecided 12 of 64');
    // The two-valued form must not appear: "accepted 41 of 64" discards the 12 pending.
    expect(log.textContent ?? '').not.toContain('Accepted 41 of 64');
  });

  it('should show a retention breach in amber as a warning, using the servers sentence', () => {
    render(<StepLog steps={[]} scores={scores} batches={[]} savePoints={[]} />);
    const line = screen.getByTestId('step-log').querySelector('[data-event-type="warning"]');
    expect(line).not.toBeNull();
    expect(line?.className).toContain('text-amber');
    expect(line?.textContent).toContain('below the floor');
  });

  it('should render newest first', () => {
    render(<StepLog steps={steps} scores={scores} batches={batches} savePoints={[300]} />);
    const types = [...screen.getByTestId('step-log').querySelectorAll('li')].map(
      (node) => node.textContent ?? '',
    );
    expect(types[0]).toContain('412');
    expect(types[types.length - 1]).toContain('300');
  });

  it('should render the bound rather than implying it', () => {
    render(<StepLog steps={steps} scores={scores} batches={batches} savePoints={[]} limit={1} />);
    expect(screen.getByTestId('step-log-bound')).toHaveTextContent('newest 1 of 3 events');
  });

  it('should give no section 5.10 event a case of its own', () => {
    // ADR-022's fence, at the rendering layer. A quiesce/commit/torn event is NOT specially
    // rendered: it goes through the default branch and names its type.
    for (const type of ['quiesce', 'commit', 'torn', 'step_retried', 'millm_resumed']) {
      expect(isKnownEventType(type), type).toBe(false);
      expect(toneFor(type), type).toBe('text-dim');
    }
    render(
      <StepLog
        steps={[]}
        scores={[]}
        batches={[]}
        savePoints={[]}
        extra={[{ step: 412, type: 'torn', text: 'weights torn' }]}
      />,
    );
    const line = screen.getByTestId('step-log').querySelector('[data-event-type="torn"]');
    expect(line?.getAttribute('data-event-known')).toBe('false');
    expect(line?.textContent).toContain('[unrecognised event torn]');
    // And it is not painted red or amber, which would give it a presentation it must not have.
    expect(line?.className).not.toContain('text-red');
    expect(line?.className).not.toContain('text-amber');
  });

  it('should paint no line with the accent', () => {
    // FR-009.3 permits the accent on seven surfaces and a step-log line is not one of them.
    // F9's FTID section 3.8 says "save accent"; the requirement and ADR-012 do not. Reported;
    // a save is distinguished by full text weight instead.
    render(<StepLog steps={steps} scores={scores} batches={batches} savePoints={[300]} />);
    for (const line of screen.getByTestId('step-log').querySelectorAll('li')) {
      expect(line.className).not.toContain('accent');
    }
    const save = screen.getByTestId('step-log').querySelector('[data-event-type="save"]');
    expect(save?.className).toContain('font-semibold');
  });

  it('should say nothing has been recorded rather than rendering an empty list', () => {
    render(<StepLog steps={[]} scores={[]} batches={[]} savePoints={[]} />);
    expect(screen.getByTestId('step-log')).toHaveTextContent('Nothing recorded yet');
  });
});

describe('[buildLog]', () => {
  it('should mark an out-of-cadence evaluation as such rather than hiding it', () => {
    const events = buildLog({
      steps: [],
      scores: [{ ...scores[0]!, out_of_cadence: true, breached: false }],
      batches: [],
      savePoints: [],
    });
    expect(events[0]?.detail).toBe('out of cadence');
    expect(events[0]?.type).toBe('score');
  });

  it('should render a step with no loss as having none rather than as zero', () => {
    const events = buildLog({
      steps: [{ ...steps[0]!, loss: null }],
      scores: [],
      batches: [],
      savePoints: [],
    });
    expect(events[0]?.text).toContain('no loss recorded');
  });
});
