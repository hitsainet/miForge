/**
 * The run index (FR-009.19; FPRD **F-2**; task 14.0, AC-009.43), and the accent rule on a
 * progress bar.
 *
 * **The list is `GET /api/v1/runs`** (task 14.0): newest first, paged, filtered by status and
 * run type, each row opening its run - injected here as `list`, so the assertions are on the
 * **query each action sends and how many times**. Opening by identifier stays as the fallback.
 * The two assertions that matter beyond that: a **running** run's bar is accent and **every other state's is not** - which is
 * one of the two mockup bugs FR-009.3 fixes - and a stopped run shows `stop_reason` or says it
 * is absent rather than reading "done".
 */
import { describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ProgressBar, RunRow } from './RunRow';
import { RUN_LIST_PAGE, RUN_PICKER_LIMIT, RunIndexPage } from './RunIndexPage';
import { NavigationProvider, useNavigation } from '@/contexts/NavigationContext';
import type { RunRecord } from '@/types/run';
import type { ApiResult } from '@/api/client';
import { checkRunListPage, type RunListPage, type RunListQuery, type RunListRow } from '@/api/runs';
import { choose, openPicker, pickedValue, pickerValues } from '@/components/common/pickerTestKit';

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
  it('should list every run newest first from GET /runs, each row opening its run', async () => {
    const list = vi.fn((query: RunListQuery = {}) => Promise.resolve(page([RUN_B, RUN_A], query)));
    renderIndex(list);
    await screen.findByTestId('run-list-rows');
    // The route's order, which is newest first; the screen does not re-sort.
    expect(
      [
        ...screen.getByTestId('run-list-rows').querySelectorAll('[data-testid^="run-list-row-"]'),
      ].map((node) => node.getAttribute('data-testid')),
    ).toEqual(['run-list-row-run-b', 'run-list-row-run-a']);
    expect(screen.getByTestId('run-list-count')).toHaveTextContent('Showing 2 of 2 runs');
    // The first read: one page, no filter sent - not `status=` (an unknown status is refused).
    expect(pageCalls(list)).toHaveLength(1);
    expect(pageCalls(list)).toContainEqual({
      limit: RUN_LIST_PAGE,
      offset: 0,
      status: '',
      runType: '',
      includeArchived: false,
    });

    // A finished run says why it ended; a failed one names its code.
    expect(screen.getByTestId('run-list-row-run-a')).toHaveTextContent('stopped: step_count');
    expect(screen.getByTestId('run-list-row-run-b')).toHaveTextContent(
      'failed: tokenizer_unavailable',
    );

    fireEvent.click(screen.getByTestId('run-list-row-run-a'));
    expect(screen.getByTestId('where')).toHaveTextContent('run:run-a');
  });

  it('should send the chosen filter as the id the route accepts, and re-read', async () => {
    const list = vi.fn((query: RunListQuery = {}) => Promise.resolve(page([RUN_A], query)));
    renderIndex(list);
    await screen.findByTestId('run-list-rows');
    await choose('run-filter-status', 'failed');
    await waitFor(() => expect(pageCalls(list)).toHaveLength(2));
    expect(pageCalls(list).at(-1)).toEqual({
      limit: RUN_LIST_PAGE,
      offset: 0,
      status: 'failed',
      runType: '',
      includeArchived: false,
    });
    expect(pickedValue('run-filter-status')).toBe('failed');

    await choose('run-filter-type', 'adapter');
    await waitFor(() => expect(pageCalls(list)).toHaveLength(3));
    expect(pageCalls(list).at(-1)).toEqual({
      limit: RUN_LIST_PAGE,
      offset: 0,
      status: 'failed',
      runType: 'adapter',
      includeArchived: false,
    });

    fireEvent.click(screen.getByTestId('run-filter-clear'));
    await waitFor(() => expect(pageCalls(list)).toHaveLength(4));
    expect(pageCalls(list).at(-1)).toEqual({
      limit: RUN_LIST_PAGE,
      offset: 0,
      status: '',
      runType: '',
      includeArchived: false,
    });
  });

  it('should never let a slower answer to an older filter replace a newer one (M50)', async () => {
    // Choose `failed`, then `completed` before the `failed` page arrives. The `completed`
    // answer is the one on screen when the late `failed` answer lands - without the ticket
    // guard the stale page overwrote it, and the list showed runs the filter excluded.
    let releaseFailed: (value: ApiResult<RunListPage>) => void = () => {};
    const list = vi.fn((query: RunListQuery = {}) => {
      if (query.status === 'failed')
        return new Promise<ApiResult<RunListPage>>((resolve) => {
          releaseFailed = resolve;
        });
      if (query.status === 'completed') return Promise.resolve(page([RUN_A], query));
      return Promise.resolve(page([RUN_A, RUN_B], query));
    });
    renderIndex(list);
    await screen.findByTestId('run-list-row-run-b');
    await choose('run-filter-status', 'failed');
    await choose('run-filter-status', 'completed');
    await waitFor(() => expect(screen.queryByTestId('run-list-row-run-b')).toBeNull());
    await act(async () => {
      releaseFailed(page([RUN_B], { status: 'failed' }));
    });
    expect(screen.getByTestId('run-list-row-run-a')).toBeInTheDocument();
    expect(screen.queryByTestId('run-list-row-run-b')).toBeNull();
  });

  it('should page older runs onto the list rather than replacing it', async () => {
    const first = Array.from({ length: RUN_LIST_PAGE }, (_, index) => row(`run-${index}`));
    const list = vi.fn((query: RunListQuery = {}) =>
      Promise.resolve(
        (query.offset ?? 0) === 0
          ? page(first, query, RUN_LIST_PAGE + 1)
          : page([row('run-oldest')], query, RUN_LIST_PAGE + 1),
      ),
    );
    renderIndex(list);
    await screen.findByTestId('run-list-more');
    expect(screen.getByTestId('run-list-count')).toHaveTextContent(
      `Showing ${RUN_LIST_PAGE} of ${RUN_LIST_PAGE + 1} runs`,
    );
    await act(async () => {
      fireEvent.click(screen.getByTestId('run-list-more'));
    });
    await screen.findByTestId('run-list-row-run-oldest');
    expect(pageCalls(list).at(-1)).toEqual({
      limit: RUN_LIST_PAGE,
      offset: RUN_LIST_PAGE,
      status: '',
      runType: '',
      includeArchived: false,
    });
    expect(screen.getByTestId('run-list-row-run-0')).toBeInTheDocument();
    expect(screen.queryByTestId('run-list-more')).toBeNull();
  });

  it('should say no run is listed when the list is empty, and offer New run', async () => {
    // Archived runs are hidden by default, so an empty default list is "none listed", not
    // "none started" - which would be false while archived runs exist (2026-10-20).
    renderIndex(() => Promise.resolve(page([], {})));
    const empty = await screen.findByTestId('run-list-empty');
    expect(empty).toHaveTextContent('No run is listed. Start one from New run');
    expect(empty).toHaveTextContent('Show archived runs');
    fireEvent.click(screen.getByTestId('run-show-archived'));
    await waitFor(() =>
      expect(screen.getByTestId('run-list-empty')).toHaveTextContent('No run has been started yet'),
    );
  });

  it('should name the list as unavailable when it cannot be read, never show it empty', async () => {
    renderIndex(() =>
      Promise.resolve({ ok: false, failure: { kind: 'transport', status: 0, cause: 'refused' } }),
    );
    expect(await screen.findByTestId('run-list-failure')).toHaveTextContent(/the run list/i);
    expect(screen.queryByTestId('run-list-empty')).toBeNull();
  });

  it('should name a 200 that is not a run list as malformed, not crash or show it empty', async () => {
    // budgets.spec.ts's catch-all answers `{}` everywhere; the screen took the app down.
    renderIndex(() => Promise.resolve(checkRunListPage({})));
    expect(await screen.findByTestId('run-list-failure')).toHaveTextContent(/the run list/i);
    expect(screen.queryByTestId('run-list-empty')).toBeNull();
  });

  it('should list every run in the Open-a-run dropdown and open the one chosen', async () => {
    // Operator, 2026-10-18: "I want the available runs to be listed in a DROPDOWN. I don't
    // want to have to know the id from memory."
    const list = vi.fn((query: RunListQuery = {}) => Promise.resolve(page([RUN_B, RUN_A], query)));
    renderIndex(list);
    expect(await pickerValues('run-id-input')).toEqual(['run-b', 'run-a']);
    // Its own read: the newest runs, unfiltered by the list's filters.
    expect(list).toHaveBeenCalledWith({
      limit: RUN_PICKER_LIMIT,
      offset: 0,
      includeArchived: false,
    });
    const listbox = await openPicker('run-id-input');
    expect(listbox).toHaveTextContent('run-a — completed · change · created');
    await choose('run-id-input', 'run-a');
    expect(screen.getByTestId('where')).toHaveTextContent('run:run-a');
  });

  it('should keep pasting an id as the secondary way in, with its reason', async () => {
    renderIndex(() => Promise.resolve(page([], {})));
    const box = screen.getByTestId('run-id-input');
    expect(box.getAttribute('role')).toBe('combobox');
    expect(box.getAttribute('data-free-text-reason')).toMatch(/Paste an id/);
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

/* ------------------------------------------------------------------------------------ */

function row(runId: string, overrides: Partial<RunListRow> = {}): RunListRow {
  return {
    run_id: runId,
    status: 'running',
    run_type: 'change',
    created_at: '2026-10-17T12:00:00+00:00',
    finished_at: null,
    last_step: 3,
    stop_reason: null,
    failure_code: null,
    ...overrides,
  };
}

const RUN_A = row('run-a', {
  status: 'completed',
  stop_reason: 'step_count',
  finished_at: '2026-10-17T13:00:00+00:00',
  last_step: 40,
});
const RUN_B = row('run-b', { status: 'failed', failure_code: 'tokenizer_unavailable' });

function page(
  runs: readonly RunListRow[],
  query: RunListQuery,
  total = runs.length,
): ApiResult<RunListPage> {
  return {
    ok: true,
    value: {
      runs,
      total,
      limit: query.limit ?? RUN_LIST_PAGE,
      offset: query.offset ?? 0,
      statuses: ['cancelled', 'completed', 'failed', 'pending', 'running'],
      run_types: ['adapter', 'change', 'create'],
    },
  };
}

function Where() {
  const navigation = useNavigation();
  return (
    <span data-testid="where">
      {navigation.screen}:{navigation.params['runId'] ?? ''}
    </span>
  );
}

function renderIndex(list: (query?: RunListQuery) => Promise<ApiResult<RunListPage>>) {
  return render(
    <NavigationProvider>
      <RunIndexPage list={list as never} />
      <Where />
    </NavigationProvider>,
  );
}

/** The paged list's reads - not the dropdown's own (limit RUN_PICKER_LIMIT). */
function pageCalls(list: { mock: { calls: unknown[][] } }): RunListQuery[] {
  return list.mock.calls
    .map((call) => (call[0] ?? {}) as RunListQuery)
    .filter((query) => query.limit === RUN_LIST_PAGE);
}
