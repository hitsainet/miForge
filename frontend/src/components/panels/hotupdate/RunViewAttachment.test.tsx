/**
 * The run view hands the open run's attachment to the shell, and takes it back (F11 16.4).
 *
 * Driven through the real `App`, the real run store and the real fetch path, stubbed only at
 * the HTTP boundary - so removing the run view's effect, or the shell's slot, turns it red.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import { App } from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { useAttachmentStore } from '@/stores/attachmentStore';
import { useRunStore } from '@/stores/runStore';

const HEARTBEAT = {
  heartbeat_at: null,
  age_seconds: null,
  longest_gap_seconds: null,
  lapse_threshold_seconds: 300,
  lapsed: false,
};

function record(runId: string, attachment: unknown) {
  return {
    run_id: runId,
    status: 'running',
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

let originalFetch: typeof globalThis.fetch;

function serve(attachment: unknown, steps: unknown[] = []) {
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
      body = { run_id: 'run-a', steps, count: steps.length, limit: 200, next_before_step: null };
    } else if (/\/api\/v1\/runs\/run-a(\?|$)/.test(url)) {
      body = record('run-a', attachment);
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
  act(() => useAttachmentStore.getState().show(null));
  act(() => useRunStore.getState().open(''));
});

const inTheme = (element: React.ReactElement) => (
  <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
);

describe('[RunViewPage] attachment hand-off', () => {
  it('should put an attached runs chip in the strip, and remove it on leaving', async () => {
    serve(ATTACHMENT);
    const { unmount } = render(
      inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />),
    );
    await waitFor(() => expect(screen.getByTestId('attachment-chip')).toBeInTheDocument());
    expect(
      screen.getByTestId('shell-resource-strip').contains(screen.getByTestId('attachment-chip')),
    ).toBe(true);
    expect(screen.getByTestId('hotupdate-timings')).toBeInTheDocument();
    unmount();
    expect(useAttachmentStore.getState().attachment).toBeNull();
  });

  it('should show no chip and no timings panel for a run that did not attach', async () => {
    serve(null);
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    await waitFor(() => expect(screen.getByTestId('run-liveness')).toBeInTheDocument());
    expect(screen.queryByTestId('attachment-chip')).toBeNull();
    expect(screen.queryByTestId('hotupdate-timings')).toBeNull();
  });
});
