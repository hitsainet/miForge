/**
 * Archive on the run view and the part index on the Packs screen (operator decision
 * 2026-10-20), through the mounted App - so removing the action from `RunViewPage` or the
 * `<PartIndex />` line from `PacksPage` turns this file red.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { App } from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { useRunStore } from '@/stores/runStore';

const HEARTBEAT = {
  heartbeat_at: null,
  age_seconds: null,
  longest_gap_seconds: null,
  lapse_threshold_seconds: 300,
  lapsed: false,
};

let originalFetch: typeof globalThis.fetch;
let posts: string[] = [];
let runReads = 0;

function serve(status: string, archivedAt: string | null): void {
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    const method = (init?.method ?? 'GET').toUpperCase();
    let body: unknown = {};
    if (method === 'POST') {
      posts.push(new URL(url, 'http://local').pathname);
      body = { run_id: 'run-a', status, archived: false, archived_at: null, changed: true };
    } else if (/\/api\/v1\/parts(\?|$)/.test(url)) {
      body = {
        items: [
          {
            id: 'part-9',
            part_kind: 'retention_set',
            source: 'midataworks_contract',
            contract_reference: { type: 'retention_set', name: 'general', version: '1' },
            rendered: 'retention_set/general@1',
            dataset: 'general',
            version_number: 1,
            row_count: 0,
            caveats: [],
            licence_classes: [],
            imported_at: '2026-10-18T00:00:56+00:00',
            archived_at: null,
          },
        ],
        total: 1,
      };
    } else if (/\/api\/v1\/packs(\?|$)/.test(url)) {
      body = { packs: [] };
    } else if (/\/checkpoints/.test(url)) {
      body = { run_id: 'run-a', checkpoints: [], latest_step: null, steps_on_disk: [] };
    } else if (/\/scores/.test(url)) {
      body = {
        run_id: 'run-a',
        scores: [],
        test_every: 10,
        retention_every: 10,
        exposure_window_steps: 10,
        provider_discontinuities: [],
      };
    } else if (/\/batches/.test(url)) {
      body = { run_id: 'run-a', batches: [], count: 0, limit: 50, next_before: null };
    } else if (/\/api\/v1\/runs\/run-a\/steps/.test(url)) {
      body = { run_id: 'run-a', steps: [], count: 0, limit: 200, next_before_step: null };
    } else if (/\/api\/v1\/runs\/run-a(\?|$)/.test(url)) {
      runReads += 1;
      body = {
        run_id: 'run-a',
        status,
        run_type: 'change',
        last_step: 2,
        cancel_requested_at: null,
        failure_code: null,
        heartbeat: HEARTBEAT,
        attachment: null,
        archived_at: archivedAt,
      };
    }
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  }) as typeof globalThis.fetch;
}

beforeEach(() => {
  originalFetch = globalThis.fetch;
  posts = [];
  runReads = 0;
  window.localStorage.clear();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  act(() => useRunStore.getState().open(''));
  window.localStorage.clear();
});

const inTheme = (element: React.ReactElement) => (
  <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
);

describe('[RunViewPage] archive', () => {
  it('should mark an archived run and Unarchive it with one POST, then re-read', async () => {
    serve('failed', '2026-10-20T12:00:00+00:00');
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    const pill = await screen.findByTestId('run-archived');
    expect(pill).toHaveTextContent('Archived Oct 20, 8:00 AM ET');
    expect(pill).toHaveTextContent('nothing was deleted');
    const action = screen.getByTestId('run-archive-action-run-a');
    expect(action).toHaveTextContent('Unarchive');
    const reads = runReads;

    fireEvent.click(action);

    await waitFor(() => expect(posts).toEqual(['/api/v1/runs/run-a/unarchive']));
    await waitFor(() => expect(runReads).toBeGreaterThan(reads));
  });

  it('should offer Archive on a finished run and nothing on a running one', async () => {
    serve('completed', null);
    const first = render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    expect(await screen.findByTestId('run-archive-action-run-a')).toHaveTextContent('Archive');
    expect(screen.queryByTestId('run-archived')).toBeNull();
    first.unmount();
    act(() => useRunStore.getState().open(''));

    serve('running', null);
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    await screen.findByTestId('run-liveness');
    expect(screen.queryByTestId('run-archive-action-run-a')).toBeNull();
  });
});

describe('[PacksPage] part index', () => {
  it('should show the part index with Archive on the Packs screen', async () => {
    serve('completed', null);
    render(inTheme(<App initialScreen="packs" />));
    expect(await screen.findByTestId('part-row-part-9')).toHaveTextContent(
      'retention_set/general@1',
    );
    fireEvent.click(screen.getByTestId('part-archive-action-part-9'));
    await waitFor(() => expect(posts).toEqual(['/api/v1/parts/part-9/archive']));
  });
});
