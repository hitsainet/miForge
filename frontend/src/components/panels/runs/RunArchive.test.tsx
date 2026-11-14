/**
 * Archive (operator decision 2026-10-20) on the Runs screen and the run view.
 *
 * The assertions are on **the request each action sends and how many times**: "Show archived
 * runs" must reach both the paged list and the Open-a-run dropdown, Archive must POST to the
 * run's own archive route and then re-read the list, and an active run must offer no Archive
 * at all - the backend refuses one (`run_not_archivable`).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { RUN_LIST_PAGE, RUN_PICKER_LIMIT, RunIndexPage } from './RunIndexPage';
import { RunArchiveAction, offersArchiveAction } from './RunArchiveAction';
import { NavigationProvider } from '@/contexts/NavigationContext';
import type { ApiResult } from '@/api/client';
import { listRuns, type RunListPage, type RunListQuery, type RunListRow } from '@/api/runs';

function row(runId: string, overrides: Partial<RunListRow> = {}): RunListRow {
  return {
    run_id: runId,
    status: 'failed',
    run_type: 'change',
    created_at: '2026-10-17T12:00:00+00:00',
    finished_at: '2026-10-17T13:00:00+00:00',
    last_step: null,
    stop_reason: null,
    failure_code: 'weights_rebuild_failed',
    archived_at: null,
    ...overrides,
  };
}

function page(runs: readonly RunListRow[], query: RunListQuery): ApiResult<RunListPage> {
  return {
    ok: true,
    value: {
      runs,
      total: runs.length,
      limit: query.limit ?? RUN_LIST_PAGE,
      offset: query.offset ?? 0,
      statuses: ['cancelled', 'completed', 'failed', 'pending', 'running'],
      run_types: ['adapter', 'change', 'create'],
    },
  };
}

let posts: string[] = [];
let originalFetch: typeof globalThis.fetch;
let fetched: string[] = [];

beforeEach(() => {
  window.localStorage.clear();
  posts = [];
  fetched = [];
  originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
    fetched.push(url.pathname + url.search);
    if ((init?.method ?? 'GET').toUpperCase() === 'POST') posts.push(url.pathname);
    const runId = url.pathname.split('/')[4] ?? '';
    const archived = url.pathname.endsWith('/archive');
    return Promise.resolve(
      new Response(
        JSON.stringify({
          run_id: runId,
          status: 'failed',
          archived,
          archived_at: archived ? '2026-10-20T12:00:00+00:00' : null,
          changed: true,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
  }) as typeof globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  window.localStorage.clear();
});

function renderIndex(list: (query?: RunListQuery) => Promise<ApiResult<RunListPage>>) {
  return render(
    <NavigationProvider>
      <RunIndexPage list={list as never} />
    </NavigationProvider>,
  );
}

const calls = (list: { mock: { calls: unknown[][] } }): RunListQuery[] =>
  list.mock.calls.map((call) => (call[0] ?? {}) as RunListQuery);

describe('[Runs] archive', () => {
  it('should hide archived runs by default and send include_archived to BOTH reads when ticked', async () => {
    const list = vi.fn((query: RunListQuery = {}) => Promise.resolve(page([row('r1')], query)));
    renderIndex(list);
    await screen.findByTestId('run-list-rows');
    expect(calls(list).every((query) => query.includeArchived === false)).toBe(true);

    fireEvent.click(screen.getByTestId('run-show-archived'));

    await waitFor(() =>
      expect(calls(list).filter((query) => query.includeArchived === true)).toHaveLength(2),
    );
    const ticked = calls(list).filter((query) => query.includeArchived === true);
    expect(ticked.map((query) => query.limit).sort()).toEqual(
      [RUN_LIST_PAGE, RUN_PICKER_LIMIT].sort(),
    );
    expect(screen.getByTestId('run-list-count')).toHaveTextContent('archived included');
  });

  it('should POST Archive to the runs own route once, then re-read the list and dropdown', async () => {
    const list = vi.fn((query: RunListQuery = {}) =>
      Promise.resolve(page([row('r1'), row('live', { status: 'running' })], query)),
    );
    renderIndex(list);
    await screen.findByTestId('run-list-rows');
    const before = list.mock.calls.length;
    // An active run offers no Archive: the backend would refuse it.
    expect(screen.queryByTestId('run-archive-action-live')).toBeNull();

    fireEvent.click(screen.getByTestId('run-archive-action-r1'));

    await waitFor(() => expect(posts).toEqual(['/api/v1/runs/r1/archive']));
    await waitFor(() => expect(list.mock.calls.length).toBeGreaterThanOrEqual(before + 2));
    const after = calls(list).slice(before);
    expect(after.some((query) => query.limit === RUN_LIST_PAGE && query.offset === 0)).toBe(true);
    expect(after.some((query) => query.limit === RUN_PICKER_LIMIT)).toBe(true);
  });

  it('should mark an archived row and offer Unarchive, which POSTs unarchive', async () => {
    const list = vi.fn((query: RunListQuery = {}) =>
      Promise.resolve(page([row('r1', { archived_at: '2026-10-20T12:00:00+00:00' })], query)),
    );
    renderIndex(list);
    await screen.findByTestId('run-list-rows');
    expect(screen.getByTestId('run-list-archived')).toHaveTextContent('Archived');
    const action = screen.getByTestId('run-archive-action-r1');
    expect(action).toHaveTextContent('Unarchive');

    fireEvent.click(action);

    await waitFor(() => expect(posts).toEqual(['/api/v1/runs/r1/unarchive']));
  });
});

describe('[RunArchiveAction]', () => {
  it('should offer an action for a finished or archived run and none for an active one', () => {
    expect(offersArchiveAction('failed', null)).toBe(true);
    expect(offersArchiveAction('completed', undefined)).toBe(true);
    expect(offersArchiveAction('cancelled', null)).toBe(true);
    expect(offersArchiveAction('running', null)).toBe(false);
    expect(offersArchiveAction('pending', null)).toBe(false);
    // An archived row always offers Unarchive.
    expect(offersArchiveAction('running', '2026-10-20T12:00:00+00:00')).toBe(true);
  });

  it('should name a refusal rather than swallow it', async () => {
    const archive = vi.fn(() =>
      Promise.resolve({
        ok: false as const,
        failure: {
          kind: 'refusal' as const,
          status: 409,
          code: 'run_not_archivable',
          message: 'Run r1 is running.',
          details: {},
        },
      }),
    );
    const onDone = vi.fn();
    render(
      <RunArchiveAction
        runId="r1"
        status="failed"
        archivedAt={null}
        onDone={onDone}
        archive={archive as never}
      />,
    );
    fireEvent.click(screen.getByTestId('run-archive-action-r1'));
    expect(await screen.findByTestId('run-archive-failure-r1')).toHaveTextContent(
      'run_not_archivable',
    );
    expect(archive).toHaveBeenCalledTimes(1);
    expect(onDone).not.toHaveBeenCalled();
  });
});

describe('[listRuns] include_archived', () => {
  it('should send include_archived=true only when asked, and omit it otherwise', async () => {
    await listRuns({ limit: 5 });
    await listRuns({ limit: 5, includeArchived: true });
    expect(fetched[0]).not.toContain('include_archived');
    expect(fetched[1]).toContain('include_archived=true');
  });
});
