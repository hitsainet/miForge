/**
 * The run view offers "Continue" for an attached run that ended, and submits it (D52).
 *
 * Through the real `App` and the real fetch path, stubbed at the HTTP boundary - so removing
 * the run view's mount, or the API call, turns it red. The POST is asserted by URL, body and
 * count.
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

interface Posted {
  readonly url: string;
  readonly body: unknown;
}

let originalFetch: typeof globalThis.fetch;
let posts: Posted[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function serve(record: Record<string, unknown>, answer: () => Response): void {
  posts = [];
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (init?.method === 'POST') {
      posts.push({ url, body: JSON.parse(String(init.body ?? 'null')) });
      return Promise.resolve(answer());
    }
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
    } else if (/\/steps/.test(url)) {
      body = { run_id: 'run-a', steps: [], count: 0, limit: 200, next_before_step: null };
    } else if (/\/api\/v1\/runs\/run-a(\?|$)/.test(url)) {
      body = {
        run_id: 'run-a',
        run_type: 'change',
        last_step: 5,
        cancel_requested_at: null,
        failure_code: null,
        heartbeat: HEARTBEAT,
        attachment: null,
        ...record,
      };
    }
    return Promise.resolve(json(body));
  }) as typeof globalThis.fetch;
}

beforeEach(() => {
  originalFetch = globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  act(() => useRunStore.getState().open(''));
});

const open = () =>
  render(
    <ThemeProvider initialTheme="dark">
      <App initialScreen="run" initialParams={{ runId: 'run-a' }} />
    </ThemeProvider>,
  );

const SUBMITTED = {
  submitted: true,
  run_status: 'pending',
  plan: {
    step: 100,
    name: 'LFM2:run-a:step-100',
    optimizer: { state: 'restored', reason: 'trainer_state_restored' },
  },
  resume: { superseded_steps: { first: 101, last: 103 } },
};

describe('[ContinueRun]', () => {
  it('should submit a continue for an attached run that failed', async () => {
    serve({ status: 'failed', acquisition_mode: 'attach' }, () => json(SUBMITTED, 202));
    open();
    const button = await screen.findByRole('button', { name: 'Continue from the last save' });
    fireEvent.click(button);
    await waitFor(() => expect(screen.getByTestId('continue-submitted')).toBeInTheDocument());
    expect(posts).toEqual([{ url: '/api/v1/runs/run-a/continue', body: { submit: true } }]);
    expect(screen.getByTestId('continue-submitted').textContent).toContain(
      'Steps 101–103 of the previous attempt are superseded',
    );
  });

  it('should show the served-model refusal by name, with the operator action', async () => {
    serve({ status: 'failed', acquisition_mode: 'attach' }, () =>
      json(
        {
          error: {
            code: 'run_definition_invalid',
            message: 'This run definition has 1 problem(s): resume_served_model_mismatch.',
            details: {
              failures: [
                {
                  step: 'budget',
                  code: 'resume_served_model_mismatch',
                  message: 'The model miLLM is serving is not checkpoint LFM2:run-a:step-100.',
                  details: {
                    operator_action: {
                      call: 'POST /api/weights/load',
                      body: { name: 'LFM2:run-a:step-100' },
                    },
                  },
                },
              ],
            },
          },
        },
        400,
      ),
    );
    open();
    fireEvent.click(await screen.findByRole('button', { name: 'Continue from the last save' }));
    const nested = await screen.findByTestId('continue-failure');
    expect(nested.textContent).toContain('resume_served_model_mismatch');
    expect(nested.textContent).toContain('POST /api/weights/load');
    expect(posts).toHaveLength(1);
  });

  it.each([
    ['a running attached run', { status: 'running', acquisition_mode: 'attach' }],
    ['a failed load run', { status: 'failed', acquisition_mode: 'load' }],
    ['a record from an older backend', { status: 'failed' }],
  ])('should not offer it for %s', async (_label, record) => {
    serve(record, () => json(SUBMITTED, 202));
    open();
    await waitFor(() => expect(screen.getByTestId('run-liveness')).toBeInTheDocument());
    expect(screen.queryByRole('button', { name: 'Continue from the last save' })).toBeNull();
  });
});
