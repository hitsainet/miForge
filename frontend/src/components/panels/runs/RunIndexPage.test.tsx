/**
 * The navigational minimum (FR-009.19; FPRD **F-2**), and the accent rule on a progress bar.
 *
 * `GET /api/v1/runs` is **not mounted** in this build, so this screen states the gap by name
 * and opens a run by identifier rather than fabricating rows. The two assertions that matter
 * beyond that: a **running** run's bar is accent and **every other state's is not** - which is
 * one of the two mockup bugs FR-009.3 fixes - and a stopped run shows `stop_reason` or says it
 * is absent rather than reading "done".
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProgressBar, RunRow } from './RunRow';
import { RunIndexPage } from './RunIndexPage';
import { NavigationProvider } from '@/contexts/NavigationContext';
import type { RunRecord } from '@/types/run';

function record(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    run_id: 'md-notary-1',
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
    },
    ...overrides,
  };
}

describe('[ProgressBar]', () => {
  it('should be accent for a running run and NOT for any other state', () => {
    // FR-009.3, and the mockup's bug: its bar was accent for every non-`held` run including
    // `done` ones, while its `done` pill was green.
    const { container } = render(
      <>
        <ProgressBar running step={590} total={800} />
        <ProgressBar running={false} step={800} total={800} />
      </>,
    );
    const bars = [...container.querySelectorAll('[data-running]')].map((node) => ({
      running: node.getAttribute('data-running'),
      fill: node.firstElementChild?.className ?? '',
    }));
    expect(bars.find((bar) => bar.running === 'true')?.fill).toContain('bg-accent');
    expect(bars.find((bar) => bar.running === 'false')?.fill).not.toContain('accent');
  });

  it('should draw a full-width marker rather than a fraction of an invented total', () => {
    // No response carries a step total in this build, so there is no denominator to divide by
    // and `590 / 590` would be an invented one.
    const { container } = render(<ProgressBar running step={590} total={null} />);
    expect((container.querySelector('[data-running] > div') as HTMLElement).style.width).toBe(
      '100%',
    );
  });
});

describe('[RunRow]', () => {
  it('should name every score with its sample and the step it came from', () => {
    render(
      <RunRow
        record={record()}
        latestTest={{
          step: 500,
          kind: 'test',
          set: 'held-out@3',
          score: 0.84,
          n: 320,
          floor: null,
          floor_layer: null,
          breached: false,
          provider: 'in-process',
          provider_kind: 'in_process',
          provider_model: null,
          aggregation: 'mean',
          out_of_cadence: false,
          provider_discontinuity: false,
          message: '',
        }}
        latestRetention={null}
        onOpen={() => {}}
      />,
    );
    const row = screen.getByTestId('run-row-md-notary-1');
    expect(row).toHaveTextContent('320 prompts');
    expect(row).toHaveTextContent('at step 500');
    expect(row).toHaveTextContent('no test at this step');
  });

  it('should show a stopped runs stop_reason, or say the field is absent', () => {
    // FR-009.19: which of the five conditions fired is the one thing an operator wants from a
    // finished run, and a bare "done" does not say.
    const { unmount } = render(
      <RunRow
        record={record({ status: 'completed' })}
        latestTest={null}
        latestRetention={null}
        onOpen={() => {}}
      />,
    );
    expect(screen.getByTestId('run-row-md-notary-1')).toHaveTextContent('stop_reason');
    unmount();

    render(
      <RunRow
        record={record({ status: 'completed', stop_reason: 'retention_floor' })}
        latestTest={null}
        latestRetention={null}
        onOpen={() => {}}
      />,
    );
    expect(screen.getByTestId('run-row-md-notary-1')).toHaveTextContent('retention_floor');
  });

  it('should render a pending run through the pill rather than as unrecognised', () => {
    // `models/run.py` defaults a submitted run's status to `pending`, so this is the most
    // common state of a new run - and it rendered "pending (unrecognised)" until the pill's
    // map carried it.
    render(
      <RunRow
        record={record({ status: 'pending' })}
        latestTest={null}
        latestRetention={null}
        onOpen={() => {}}
      />,
    );
    expect(screen.getByTestId('run-row-md-notary-1')).toHaveTextContent('Pending');
    expect(screen.getByTestId('run-row-md-notary-1').textContent ?? '').not.toContain(
      'unrecognised',
    );
  });

  it('should render an unrecognised state without throwing', () => {
    expect(() =>
      render(
        <RunRow
          record={record({ status: 'reaped' })}
          latestTest={null}
          latestRetention={null}
          onOpen={() => {}}
        />,
      ),
    ).not.toThrow();
    expect(screen.getByTestId('run-row-md-notary-1')).toHaveTextContent('reaped (unrecognised)');
  });

  it('should show a stalled run as stalled, from the record', () => {
    render(
      <RunRow
        record={record({ heartbeat: { ...record().heartbeat, age_seconds: 1200, lapsed: true } })}
        latestTest={null}
        latestRetention={null}
        onOpen={() => {}}
      />,
    );
    expect(
      screen.getByTestId('run-row-md-notary-1').querySelector('[data-liveness="stalled"]'),
    ).not.toBeNull();
  });
});

describe('[RunIndexPage]', () => {
  it('should name the missing endpoints rather than looking broken', () => {
    render(
      <NavigationProvider>
        <RunIndexPage />
      </NavigationProvider>,
    );
    expect(screen.getByText(/serves no run listing/)).toBeInTheDocument();
    expect(screen.getByText(/GET \/v1\/runs/)).toBeInTheDocument();
    expect(screen.getByTestId('run-index-empty')).toHaveTextContent('No run is open');
  });

  it('should offer a New run action, which is a link rather than a dialog', () => {
    // FR-009.19: "A New run action belongs here and is no longer unowned" - R-02.53 defines
    // the form, and this list performs no part of the submission.
    render(
      <NavigationProvider>
        <RunIndexPage />
      </NavigationProvider>,
    );
    expect(screen.getByRole('button', { name: 'New run' })).toBeInTheDocument();
  });

  it('should offer no Detach, no resident-model card and no attachment chip', () => {
    // FPRD section 3.9: all section 5.10's.
    render(
      <NavigationProvider>
        <RunIndexPage />
      </NavigationProvider>,
    );
    expect(document.body.textContent ?? '').not.toMatch(/detach|resident model|quiesce/i);
  });

  it('should offer no Pause and no held state', () => {
    // R-02.25 gives an operator stop and ADR-007 cooperative cancellation; neither is a
    // resumable pause-in-place, and F6's enum has no `held`.
    render(
      <NavigationProvider>
        <RunIndexPage />
      </NavigationProvider>,
    );
    expect(screen.queryByRole('button', { name: /pause/i })).toBeNull();
    expect(document.body.textContent ?? '').not.toMatch(/\bheld\b/i);
  });
});
