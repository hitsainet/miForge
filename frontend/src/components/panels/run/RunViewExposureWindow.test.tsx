/**
 * The run page's exposure-window line counts its steps in the singular for one (train F4,
 * 2026-11-27): "Retention runs every 1 step, so up to 1 step of training ...", never "1 steps".
 * Mounted through the real App with the HTTP boundary stubbed, as `RunViewBack.test.tsx` is.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, render, screen } from '@testing-library/react';
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
        retention_every: 1,
        exposure_window_steps: 1,
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

describe('[RunViewPage] exposure window', () => {
  it('should count one step in the singular', async () => {
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    const line = await screen.findByTestId('exposure-window');
    expect(line.textContent?.replace(/\s+/g, ' ')).toContain(
      'Retention runs every 1 step, so up to 1 step of training can occur',
    );
  });
});
