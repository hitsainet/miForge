/**
 * The run view shows the run's own record of miLLM's state as history; the shell's chip does
 * not take it (F11 16.4, corrected 2026-10-19).
 *
 * Production, 2026-10-18: run `f11w2-r8` failed at 11:36 UTC while attached, its record kept
 * `attached: true, weights_state: clean, observed_at 11:35`, and the shell's chip - fed from
 * that record - read "Attached to miLLM ... step-900 ... Weights clean" the next day while
 * miLLM held no attachment. Driven through the real `App`, the real run store and the real
 * fetch path, stubbed only at the HTTP boundary.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
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

function record(runId: string, attachment: unknown, status = 'running') {
  return {
    run_id: runId,
    status,
    run_type: 'change',
    last_step: 2,
    cancel_requested_at: null,
    failure_code: null,
    heartbeat: HEARTBEAT,
    attachment,
  };
}

const ATTACHMENT = {
  serving_base_url: 'http://millm-backend.millm.svc:8000',
  resident_model: null,
  weights_state: null,
  holder: null,
  stale: null,
};

/** The production record of `f11w2-r8`, verbatim but for the host. */
const LAST_OBSERVED = {
  serving_base_url: 'http://millm-backend.millm.svc.cluster.local:8000',
  resident_model: 'LiquidAI/LFM2.5-1.2B-Instruct:f11w2-r8:step-900',
  weights_state: 'clean',
  holder: null,
  stale: false,
  attached: true,
  observed_at: '2026-10-18T11:35:22.843344+00:00',
};

let originalFetch: typeof globalThis.fetch;

function serve(
  attachment: unknown,
  steps: unknown[] = [],
  status = 'running',
  live: unknown = { configured: false },
) {
  globalThis.fetch = ((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    let body: unknown = {};
    if (/\/api\/v1\/attachment$/.test(url)) {
      body = live;
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
      body = { run_id: 'run-a', steps, count: steps.length, limit: 200, next_before_step: null };
    } else if (/\/api\/v1\/runs\/run-a(\?|$)/.test(url)) {
      body = record('run-a', attachment, status);
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
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  act(() => useRunStore.getState().open(''));
});

const inTheme = (element: React.ReactElement) => (
  <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
);

describe('[RunViewPage] attachment record', () => {
  it('should show an attached runs record on the run view, not as the shell chip', async () => {
    serve(ATTACHMENT);
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    await waitFor(() => expect(screen.getByTestId('run-attachment')).toBeInTheDocument());
    expect(screen.getByTestId('hotupdate-timings')).toBeInTheDocument();
    expect(screen.getByTestId('run-attachment-observed')).toHaveTextContent(
      'not recorded miLLM’s state yet',
    );
    // Nothing configured on the live read, so the strip shows no chip whatever the record says.
    expect(screen.queryByTestId('attachment-chip')).toBeNull();
  });

  it('should never put a failed runs last observation in the strip as an attachment', async () => {
    serve(LAST_OBSERVED, [], 'failed', {
      configured: true,
      reachable: true,
      serving_base_url: 'http://millm-backend.millm.svc.cluster.local:8000',
      attached: false,
      holder: null,
      resident_model: 'Llama-3.1-8B-Instruct',
      weights_state: 'clean',
      stale: null,
      observed_at: '2026-10-19T13:00:00+00:00',
    });
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    await waitFor(() => expect(screen.getByTestId('run-attachment')).toBeInTheDocument());
    await waitFor(() =>
      expect(screen.getByTestId('attachment-chip')).toHaveAttribute('data-attachment', 'detached'),
    );
    const strip = screen.getByTestId('shell-resource-strip');
    expect(strip.textContent ?? '').not.toMatch(/step-900/);
    expect(strip.textContent ?? '').not.toMatch(/Attached to/);

    const card = screen.getByTestId('run-attachment');
    expect(card).toHaveTextContent('Ran attached to miLLM');
    expect(screen.getByTestId('run-attachment-observed')).toHaveTextContent(
      'Last observed by this run Oct 18, 7:35 AM ET',
    );
    expect(screen.getByTestId('run-attachment-observed')).toHaveTextContent(
      'The run has ended, so this is the last state it saw',
    );
  });

  it('should show no attachment card and no timings panel for a run that did not attach', async () => {
    serve(null);
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    await waitFor(() => expect(screen.getByTestId('run-liveness')).toBeInTheDocument());
    expect(screen.queryByTestId('run-attachment')).toBeNull();
    expect(screen.queryByTestId('hotupdate-timings')).toBeNull();
  });
});
