/**
 * The run page's way back to the list (operator request 2026-10-21): an "All runs" button in
 * the page head that always opens Runs, whether or not the browser has history to go back
 * to. Mounted through the real App, so dropping the `back` prop from `RunViewPage` or the
 * button from `PageHead` turns this file red.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { App } from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { useRunStore } from '@/stores/runStore';

let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  // Minimal valid bodies; this test is about navigation, not data.
  globalThis.fetch = ((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    let body: unknown = {};
    if (/\/checkpoints/.test(url)) {
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
      body = {
        run_id: 'run-a',
        status: 'completed',
        run_type: 'change',
        last_step: 2,
        cancel_requested_at: null,
        failure_code: null,
        attachment: null,
        archived_at: null,
        heartbeat: {
          heartbeat_at: null,
          age_seconds: null,
          longest_gap_seconds: null,
          lapse_threshold_seconds: 300,
          lapsed: false,
        },
      };
    } else if (/\/api\/v1\/runs(\?|$)/.test(url)) {
      body = { items: [], total: 0 };
    }
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  }) as typeof globalThis.fetch;
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

describe('[RunViewPage] back to the list', () => {
  it('should open Runs from an open run, with no history behind it', async () => {
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    const back = await screen.findByTestId('page-head-back');
    expect(back).toHaveTextContent('All runs');
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('run-a');

    fireEvent.click(back);

    expect(await screen.findByRole('heading', { level: 1, name: 'Runs' })).toBeInTheDocument();
    expect(screen.queryByTestId('page-head-back')).toBeNull();
  });

  it('should offer the same way back when no run is open', async () => {
    render(inTheme(<App initialScreen="run" initialParams={{}} />));
    fireEvent.click(await screen.findByTestId('page-head-back'));
    expect(await screen.findByRole('heading', { level: 1, name: 'Runs' })).toBeInTheDocument();
  });
});
