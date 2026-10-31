/**
 * Liveness from the record (FR-009.4), and **M3 and M4**.
 *
 * The two mutations here are the two wrong answers that were actually written and withdrawn
 * during this feature's design:
 *
 * - **M3** — `isLive = socket.connected`. Shows a **stalled run as live**. Blocked
 *   structurally: `deriveLiveness` takes `(record, nowMs)` and nothing else, so the mutation
 *   requires changing the signature.
 * - **M4** — let an event move a terminal run back to running. ADR-007's heartbeat *"refuses
 *   to move a terminal row"*; the store honours the same rule client-side.
 *
 * And the third wrong answer, which is why the threshold is read and not chosen: a 60 s rule
 * marks a **healthy** run stalled during every test pass and every save. The tests below use
 * the record's own `lapse_threshold_seconds` and `lapsed`, never a local constant - a test
 * that hard-coded 60 s would pin the defect rather than prevent it.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunRecord } from '@/types/run';
import {
  CLOCK_SKEW_TOLERANCE_SECONDS,
  deriveLiveness,
  describeLiveness,
  isTerminal,
  mergeStep,
  savePoints,
  useRunStore,
} from './runStore';

function record(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    run_id: 'run-a',
    status: 'running',
    run_type: 'change',
    last_step: 590,
    cancel_requested_at: null,
    failure_code: null,
    heartbeat: {
      heartbeat_at: '2026-10-14T12:00:00+00:00',
      age_seconds: 12,
      longest_gap_seconds: 180,
      lapse_threshold_seconds: 900,
      lapsed: false,
      ...(overrides.heartbeat ?? {}),
    },
    ...overrides,
  };
}

const NOW = Date.parse('2026-10-14T12:00:12+00:00');

describe('[deriveLiveness]', () => {
  it('should take only the record and a clock', () => {
    // M3's mutation requires changing this signature, which makes it visible in review as
    // well as red in the suite. No socket, no transport, no "last event at".
    expect(deriveLiveness.length).toBe(2);
    const source = deriveLiveness.toString();
    expect(source).not.toContain('socket');
    expect(source).not.toContain('connected');
    expect(source).not.toContain('transport');
  });

  it('should report a healthy run as running, with the servers figures', () => {
    const liveness = deriveLiveness(record(), NOW);
    expect(liveness.kind).toBe('running');
    expect(liveness.kind === 'running' && liveness.step).toBe(590);
    expect(liveness.kind === 'running' && liveness.thresholdSeconds).toBe(900);
    expect(liveness.kind === 'running' && liveness.longestGapSeconds).toBe(180);
  });

  it('should report a lapsed heartbeat as stalled even with everything else healthy', () => {
    // AC-009.7, tested against the SERVER's verdict rather than against a 60 s guess.
    const liveness = deriveLiveness(
      record({ heartbeat: { ...record().heartbeat, age_seconds: 1200, lapsed: true } }),
      NOW,
    );
    expect(liveness.kind).toBe('stalled');
    expect(liveness.kind === 'stalled' && liveness.lastStep).toBe(590);
    expect(liveness.kind === 'stalled' && liveness.sinceMs).toBe(1_200_000);
  });

  it('should NOT report a gap longer than 60 s as stalled on its own', () => {
    // The withdrawn derivation, pinned as a negative. 60 s is a MINIMUM interval between
    // writes, not a maximum gap: a generation-bound test pass and a 2.4 GB save both exceed
    // it legitimately. A rule of "60 s plus a margin" would make this case stalled and would
    // show every healthy run as dead during every test pass.
    const liveness = deriveLiveness(
      record({ heartbeat: { ...record().heartbeat, age_seconds: 240, lapsed: false } }),
      NOW,
    );
    expect(liveness.kind).toBe('running');
  });

  it('should read the threshold from the record and hold no constant of its own', () => {
    // Two thresholds are two answers to "is this run alive", and the one the operator sees
    // would disagree with the one that reaps.
    const liveness = deriveLiveness(
      record({ heartbeat: { ...record().heartbeat, lapse_threshold_seconds: 1800 } }),
      NOW,
    );
    expect(liveness.kind === 'running' && liveness.thresholdSeconds).toBe(1800);
    // Behavioural, not a scrape of `deriveLiveness.toString()`: that scrape went red under
    // Vite 8, whose transform keeps comments in the function body, on the comment that
    // *explains* the 60 s throttle. A source regex measures the transformer, not the rule.
    // Instead: across thresholds and ages on both sides of 60 and 900, the verdict follows
    // the record's `lapsed` and the threshold shown is the record's - so no constant of the
    // module's own can be deciding either.
    for (const threshold of [7, 60, 900, 1800]) {
      for (const age of [0, 61, 901, 5000]) {
        for (const lapsed of [false, true]) {
          const verdict = deriveLiveness(
            record({
              heartbeat: {
                ...record().heartbeat,
                age_seconds: age,
                lapsed,
                lapse_threshold_seconds: threshold,
              },
            }),
            NOW,
          );
          expect(verdict.kind).toBe(lapsed ? 'stalled' : 'running');
          expect(verdict.kind !== 'terminal' && verdict.thresholdSeconds).toBe(threshold);
        }
      }
    }
  });

  it('should report a terminal run as terminal, never as stalled', () => {
    // A completed run's heartbeat is EXPECTED to be old. Reporting it as lapsed is the same
    // class of false positive as a threshold taken from the throttle.
    for (const status of ['completed', 'failed', 'cancelled']) {
      const liveness = deriveLiveness(
        record({ status, heartbeat: { ...record().heartbeat, age_seconds: 99_999, lapsed: true } }),
        NOW,
      );
      expect(liveness.kind, status).toBe('terminal');
    }
  });

  it('should report a never-beaten run with null age rather than as zero seconds old', () => {
    const liveness = deriveLiveness(
      record({
        heartbeat: {
          heartbeat_at: null,
          age_seconds: null,
          longest_gap_seconds: null,
          lapse_threshold_seconds: 900,
          lapsed: false,
        },
      }),
      NOW,
    );
    expect(liveness.kind === 'running' && liveness.ageSeconds).toBeNull();
    expect(describeLiveness(liveness)).toContain('no heartbeat has been recorded yet');
  });

  it('should report clock skew when the browsers clock disagrees with the servers age', () => {
    // nowMs's one honest use. A browser minutes out would otherwise render "3 minutes ago"
    // for a beat seconds old, with nothing to tell the operator why.
    const skewed = NOW + (CLOCK_SKEW_TOLERANCE_SECONDS + 60) * 1000;
    const liveness = deriveLiveness(record(), skewed);
    expect(liveness.kind === 'running' && liveness.clockSkewSeconds).not.toBeNull();

    const fine = deriveLiveness(record(), NOW);
    expect(fine.kind === 'running' && fine.clockSkewSeconds).toBeNull();
  });
});

describe('[describeLiveness]', () => {
  it('should state the last step, the age, the longest gap and the threshold when stalled', () => {
    // The threshold is provisional upstream, so an operator reading "stalled" is given the
    // figures it was reached from - including that this run has already shown a 700 s gap.
    const sentence = describeLiveness(
      deriveLiveness(
        record({
          heartbeat: {
            ...record().heartbeat,
            age_seconds: 1000,
            longest_gap_seconds: 700,
            lapsed: true,
          },
        }),
        NOW,
      ),
    );
    expect(sentence).toContain('590');
    expect(sentence).toContain('1000 s');
    expect(sentence).toContain('700 s');
    expect(sentence).toContain('900 s');
  });
});

describe('[isTerminal]', () => {
  it('should match the backends own terminal set', () => {
    // `models/run.py::TERMINAL_STATUSES` is {completed, failed, cancelled}.
    expect(['completed', 'failed', 'cancelled'].every(isTerminal)).toBe(true);
    expect(['pending', 'running', 'queued'].some(isTerminal)).toBe(false);
  });
});

describe('[useRunStore.applyEvent]', () => {
  beforeEach(() => {
    useRunStore.getState().reset();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  it('should discard a socket event for a run the record says is terminal, and log it', () => {
    // M4. It is not reordered, not merged and not caught up: the record is the truth.
    useRunStore.setState({ runId: 'run-a', record: record({ status: 'completed' }), steps: [] });
    useRunStore.getState().applyEvent({ name: 'boundary', payload: { step: 600, loss: 1.2 } });

    expect(useRunStore.getState().steps).toEqual([]);
    expect(useRunStore.getState().discardedEvents).toBe(1);
    expect(console.warn).toHaveBeenCalled();
  });

  it('should apply a boundary event to the series for a live run', () => {
    useRunStore.setState({ runId: 'run-a', record: record(), steps: [] });
    useRunStore.getState().applyEvent({ name: 'boundary', payload: { step: 600, loss: 1.25 } });
    expect(useRunStore.getState().steps[0]).toMatchObject({ step: 600, loss: 1.25 });
    expect(useRunStore.getState().discardedEvents).toBe(0);
  });

  it('should never change the run status from an event', () => {
    // Liveness is derived from the record alone. An event may move the series; it may not
    // move the verdict, which is what FR-009.4 forbids in both directions.
    useRunStore.setState({ runId: 'run-a', record: record({ status: 'running' }), steps: [] });
    useRunStore.getState().applyEvent({ name: 'boundary', payload: { step: 601 } });
    expect(useRunStore.getState().record?.status).toBe('running');
  });

  it('should ignore an event with no record loaded rather than inventing one', () => {
    useRunStore.setState({ runId: 'run-a', record: null, steps: [] });
    useRunStore.getState().applyEvent({ name: 'boundary', payload: { step: 1 } });
    expect(useRunStore.getState().steps).toEqual([]);
  });
});

describe('[mergeStep]', () => {
  it('should keep the series newest first and replace rather than duplicate a step', () => {
    const merged = mergeStep(
      [
        {
          step: 2,
          loss: 2,
          grad_norm: null,
          learning_rate: null,
          wall_time_ms: null,
          batch_source_hashes: null,
        },
        {
          step: 1,
          loss: 1,
          grad_norm: null,
          learning_rate: null,
          wall_time_ms: null,
          batch_source_hashes: null,
        },
      ],
      {
        step: 2,
        loss: 9,
        grad_norm: null,
        learning_rate: null,
        wall_time_ms: null,
        batch_source_hashes: null,
      },
    );
    expect(merged.map((step) => step.step)).toEqual([2, 1]);
    expect(merged[0]?.loss).toBe(9);
  });
});

describe('[savePoints]', () => {
  it('should come from the checkpoints that exist and hold no save cadence', () => {
    // M11's mutation requires adding a `saveEvery` field to this store, which it does not
    // have. An out-of-cadence save would be invisible to a `step % save_every` computation,
    // and a cadence step whose save failed would be marked as though it had succeeded.
    expect(savePoints.toString()).not.toContain('save_every');
    expect(savePoints.toString()).not.toContain('saveEvery');

    expect(
      savePoints({
        run_id: 'run-a',
        checkpoints: [],
        latest_step: 400,
        // From disk, so a run whose index was dropped still reports its artifacts honestly.
        steps_on_disk: [100, 250, 400],
      }),
    ).toEqual([100, 250, 400]);
  });

  it('should return nothing for a run with no checkpoint, rather than a guessed cadence', () => {
    expect(savePoints(null)).toEqual([]);
  });
});
