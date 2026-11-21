/**
 * The WSP screens keep themselves current, in a real browser (live progress, 2026-11-20).
 *
 * The backend is served at the network: a run that is `running` on the first reads and
 * `completed` from the third, with a growing call timeline. The list is reached through the nav;
 * the spec waits for the list and the run page to change WITHOUT any click on a re-read button,
 * which is the operator's complaint (both pages read once, and had to be refreshed by hand).
 * It also asserts the waiting WSP request appears on the list and links to Approvals, and that
 * no call row carries message text.
 */
import { expect, test, type Page, type Route } from '@playwright/test';

const RUN = 'wsp_0123456789ab';
const SENTINEL = 'SENTINEL-message-text';

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

interface World {
  /** Reads of the run document so far. */
  runReads: number;
  listReads: number;
  transcriptReads: number;
}

function status(world: World): string {
  return world.runReads >= 3 ? 'completed' : 'running';
}

function progress(world: World) {
  const live = status(world) !== 'completed';
  return {
    run_id: RUN,
    status: status(world),
    live,
    now: new Date().toISOString(),
    created_at: '2026-11-20T15:00:00Z',
    started_at: '2026-11-20T15:00:05Z',
    finished_at: live ? null : '2026-11-20T15:05:00Z',
    clock_from: 'started_at',
    elapsed_s: 60 + world.runReads * 5,
    worker: {
      state: live ? 'running' : 'finished',
      elapsed_s: 55 + world.runReads * 5,
      last_heartbeat_at: '2026-11-20T15:01:00Z',
      heartbeat_age_s: live ? 2 : null,
    },
    metered: {
      calls: world.runReads,
      refused: 0,
      input_tokens: 100 * world.runReads,
      output_tokens: 50 * world.runReads,
      tokens: 150 * world.runReads,
      cost_usd: 0,
      last_call_at: '2026-11-20T15:01:00Z',
    },
    in_flight: null,
    budgets: [
      {
        name: 'tokens',
        unit: 'tokens',
        spent: 150 * world.runReads,
        cap: 1000,
        fraction: (150 * world.runReads) / 1000,
        estimate: false,
      },
      { name: 'cost_usd', unit: 'USD', spent: 0, cap: null, fraction: null, estimate: false },
      { name: 'time_s', unit: 's', spent: 60, cap: 600, fraction: 0.1, estimate: false },
      {
        name: 'steps',
        unit: 'editor calls',
        spent: world.runReads,
        cap: 50,
        fraction: world.runReads / 50,
        estimate: true,
      },
    ],
    queue: null,
  };
}

function document(world: World) {
  return {
    kind: 'miforge.wsp-run/v1',
    run_id: RUN,
    status: status(world),
    created_at: '2026-11-20T15:00:00Z',
    request: { kind: 'repair', mode: 'session', task: 'Repair the probe.' },
    request_sha256: 'a'.repeat(64),
    caller_ref: null,
    approval: null,
    usage: { requests: 0, input_tokens: 0, output_tokens: 0, cost_usd: 0 },
    forward_passes_used: null,
    stop_reason: status(world) === 'completed' ? 'submitted' : null,
    outcome: status(world) === 'completed' ? 'no_change' : null,
    failure: null,
    content_sha256: null,
    transcript_sha256: status(world) === 'completed' ? 'c'.repeat(64) : null,
    patch_sha256: null,
    candidate_id: null,
    verdict: null,
    lineage: {
      bundle_sha256: 'b'.repeat(64),
      provider: 'weightspace',
      provider_version: '0.3.0',
      chain: [],
    },
    replay: null,
  };
}

function row(world: World) {
  const live = world.listReads < 3;
  return {
    run_id: RUN,
    status: live ? 'running' : 'completed',
    kind: 'repair',
    mode: 'session',
    caller: 'agent:forge-mcp',
    caller_ref: null,
    endpoint_id: 'ep-local',
    stop_reason: live ? null : 'submitted',
    outcome: live ? null : 'no_change',
    created_at: '2026-11-20T15:00:00Z',
    provider: 'weightspace',
    provider_version: '0.3.0',
    failure: null,
    usage: { requests: 0, input_tokens: 0, output_tokens: 0, cost_usd: 0 },
    started_at: '2026-11-20T15:00:05Z',
    finished_at: live ? null : '2026-11-20T15:05:00Z',
    forward_passes_used: null,
    progress: live
      ? {
          elapsed_s: 60,
          clock_from: 'started_at',
          worker_state: 'running',
          heartbeat_age_s: 2,
          calls: world.listReads,
          refused: 0,
          tokens: 150 * world.listReads,
          cost_usd: 0,
          last_call_at: '2026-11-20T15:01:00Z',
          queue_position: null,
          budgets: [],
        }
      : null,
  };
}

const WAITING = {
  approval_id: 'appr-wsp-waiting',
  action: 'wsp_session',
  status: 'pending',
  summary: { kind: 'inspect', mode: 'session', endpoint: { model: 'qwen-editor' } },
  requested_by: 'agent:forge-mcp',
  decided_by: null,
  reason: null,
  result: null,
  error: null,
  created_at: '2026-11-20T15:00:00Z',
  decided_at: null,
  expires_at: '2099-11-21T15:00:00Z',
};

async function serve(page: Page): Promise<World> {
  const world: World = { runReads: 0, listReads: 0, transcriptReads: 0 };
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path === '/api/v1/approvals')
      return json(route, { approvals: [WAITING], count: 1, pending: 1, limit: 500 });
    if (path === '/api/v1/wsp/runs') {
      world.listReads += 1;
      return json(route, { runs: [row(world)] });
    }
    if (path === `/api/v1/wsp/runs/${RUN}`) {
      world.runReads += 1;
      return json(route, document(world));
    }
    if (path === `/api/v1/wsp/runs/${RUN}/progress`) return json(route, progress(world));
    if (path === `/api/v1/wsp/runs/${RUN}/calls`) {
      const after = Number(url.searchParams.get('after_seq') ?? '-1');
      const calls = Array.from({ length: world.runReads }, (_, seq) => ({
        seq,
        at: '2026-11-20T15:01:00Z',
        outcome: 'served',
        http_status: 200,
        latency_ms: 1200,
        finish_reason: 'tool_calls',
        refusal_code: null,
        input_tokens: 100,
        output_tokens: 50,
        cost_usd: 0,
      })).filter((c) => c.seq > after);
      return json(route, { run_id: RUN, status: status(world), live: true, calls, more: false });
    }
    if (path === `/api/v1/wsp/runs/${RUN}/transcript`) {
      world.transcriptReads += 1;
      return json(route, {
        run_id: RUN,
        transcript_sha256: 'c'.repeat(64),
        entries: [{ role: 'assistant', content: SENTINEL }],
        partial: false,
      });
    }
    return json(route, { error: { code: 'not_stubbed', message: path, details: {} } }, 404);
  });
  return world;
}

test('the WSP run list updates on its own and shows the waiting request', async ({ page }) => {
  test.setTimeout(60_000);
  const world = await serve(page);
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'WSP runs', exact: true })
    .click();
  const listed = page.getByTestId(`wsp-run-row-${RUN}`);
  await expect(listed.getByTestId('wsp-run-progress')).toContainText('1 call');
  await expect(page.getByTestId('wsp-runs-updated')).toContainText('updated');
  // No click: the next reads arrive on their own, and the run finishes.
  await expect(listed.getByTestId('wsp-run-progress')).toContainText('2 calls', {
    timeout: 15_000,
  });
  await expect(listed.getByTestId('wsp-run-progress')).toHaveCount(0, { timeout: 15_000 });
  expect(world.listReads).toBeGreaterThanOrEqual(3);
  // The waiting request is a row of its own, and it opens Approvals.
  await page.getByTestId('wsp-pending-appr-wsp-waiting').click();
  await expect(page).toHaveURL(/#\/approvals/);
});

test('the WSP run page follows a live run to its end, then reads the transcript once', async ({
  page,
}) => {
  test.setTimeout(60_000);
  const world = await serve(page);
  await page.goto(`/#/wsp/runs/${RUN}`);
  await expect(page.getByTestId('wsp-progress')).toHaveAttribute('data-live', 'true');
  await expect(page.getByTestId('wsp-transcript-pending')).toBeVisible();
  await expect(page.getByTestId('wsp-call')).toHaveCount(1);
  // On their own: a second call, then the terminal read and the transcript.
  await expect(page.getByTestId('wsp-call')).toHaveCount(2, { timeout: 15_000 });
  await expect(page.getByTestId('wsp-progress')).toHaveAttribute('data-live', 'false', {
    timeout: 15_000,
  });
  await expect(page.getByTestId('wsp-transcript-entry')).toHaveCount(1);
  const reads = world.runReads;
  await page.waitForTimeout(7_000);
  expect(world.runReads).toBe(reads); // stopped at terminal
  expect(world.transcriptReads).toBe(1);
  // The timeline is counts and codes: the transcript's text is in the transcript only.
  await expect(page.getByTestId('wsp-calls')).not.toContainText(SENTINEL);
});
