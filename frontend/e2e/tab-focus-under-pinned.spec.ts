/**
 * A control reached by Tab lands clear of everything pinned at the top of `shell-main`
 * (Tab-focus fix, 2026-11-26), in a real browser: jsdom has no layout and no focus scrolling.
 *
 * Chrome scrolls a focused element into the scroller's padded viewport, so `shell-main`'s
 * `scroll-padding-top` is the pinned stack's total (`layout/pinned.ts`). Before the fix only the
 * WSP runs list counted itself; on every other screen a row reached by Shift+Tab could land behind
 * the approvals dock, and on the WSP run page behind its floating header. Each test scrolls to the
 * end, focuses the last control and walks back with Shift+Tab, requiring every focused element's
 * top to sit at or below the bottom of the lowest pinned element.
 */
import { expect, test, type Page, type Route } from '@playwright/test';
import backend from './fixtures/backend.json' with { type: 'json' };

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

const notStubbed = (route: Route, path: string) =>
  json(route, { error: { code: 'not_stubbed', message: path, details: {} } }, 404);

const WAITING = {
  approval_id: 'appr-tab-focus',
  action: 'wsp_session',
  status: 'pending',
  summary: { kind: 'repair', mode: 'session', endpoint: { model: 'qwen-editor' } },
  requested_by: 'agent:forge-mcp',
  decided_by: null,
  reason: null,
  result: null,
  error: null,
  created_at: '2026-11-26T10:17:00Z',
  decided_at: null,
  expires_at: '2099-11-27T10:17:00Z',
};

const approvals = (route: Route, waiting: boolean) => {
  const rows = waiting ? [WAITING] : [];
  return json(route, { approvals: rows, count: rows.length, pending: rows.length, limit: 500 });
};

// The training run list, from the generated fixture's own row shape (as archive-scroll does).
const SEEDED = (backend as Record<string, unknown>)['/api/v1/runs?limit=25&offset=0'] as {
  runs: Record<string, unknown>[];
  [key: string]: unknown;
};
const RUNS = Array.from({ length: 25 }, (_, index) => ({
  ...SEEDED.runs[0],
  run_id: `run-tab-${index + 1}`,
  archived_at: null,
}));

async function openRuns(page: Page): Promise<void> {
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/v1/approvals') return approvals(route, true);
    if (url.pathname === '/api/v1/runs')
      return json(route, { ...SEEDED, runs: RUNS, total: RUNS.length, limit: 25, offset: 0 });
    return notStubbed(route, url.pathname);
  });
  await page.goto('/#/runs');
  await expect(page.getByTestId('approvals-banner-dock')).toBeVisible();
  await expect(
    page.getByTestId('run-list-rows').locator('[data-testid^="run-list-row-"]'),
  ).toHaveCount(25);
}

const RUN = 'wsp_0123456789ab';
const STEPS = 12;

async function openRun(page: Page, waiting: boolean): Promise<void> {
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', (route) => {
    const path = new URL(route.request().url()).pathname;
    const base = `/api/v1/wsp/runs/${RUN}`;
    if (path === '/api/v1/approvals') return approvals(route, waiting);
    if (path === base)
      return json(route, {
        kind: 'miforge.wsp-run/v1',
        run_id: RUN,
        status: 'completed',
        created_at: '2026-11-26T11:20:28Z',
        request: { kind: 'repair', mode: 'session', task: 'Repair the probe.' },
        request_sha256: 'a'.repeat(64),
        caller_ref: null,
        approval: null,
        usage: { requests: 1, input_tokens: 10, output_tokens: 5, cost_usd: 0 },
        forward_passes_used: 1,
        stop_reason: 'submitted',
        outcome: 'no_change',
        failure: null,
        content_sha256: null,
        transcript_sha256: 'c'.repeat(64),
        patch_sha256: null,
        candidate_id: null,
        verdict: null,
        lineage: {
          bundle_sha256: 'b'.repeat(64),
          provider: 'miwsp_engine',
          provider_version: '0.1.0',
          chain: [],
        },
        replay: null,
      });
    if (path === `${base}/progress`)
      return json(route, {
        run_id: RUN,
        status: 'completed',
        live: false,
        now: '2026-11-26T12:31:37Z',
        created_at: '2026-11-26T11:20:28Z',
        started_at: '2026-11-26T11:24:00Z',
        finished_at: '2026-11-26T12:31:00Z',
        clock_from: 'started_at',
        elapsed_s: 4020,
        worker: null,
        metered: {
          calls: 1,
          refused: 0,
          input_tokens: 10,
          output_tokens: 5,
          tokens: 15,
          cost_usd: 0,
          last_call_at: null,
        },
        in_flight: null,
        budgets: [],
        queue: null,
      });
    if (path === `${base}/calls`)
      return json(route, { run_id: RUN, status: 'completed', live: false, calls: [], more: false });
    if (path === `${base}/transcript`)
      return json(route, {
        run_id: RUN,
        transcript_sha256: 'c'.repeat(64),
        partial: false,
        entries: Array.from({ length: STEPS }, (_, n) => ({
          role: 'assistant',
          content: Array.from({ length: 12 }, (__, i) => `step ${n + 1} line ${i}`).join('\n'),
          reasoning: { text: `why ${n + 1}`, source: 'reasoning_content' },
        })),
      });
    return notStubbed(route, path);
  });
  await page.goto(`/#/wsp/runs/${RUN}`);
  await expect(page.getByTestId('wsp-transcript-entry')).toHaveCount(STEPS);
  if (waiting) await expect(page.getByTestId('approvals-banner-dock')).toBeVisible();
}

/** Scroll to the end and focus the last focusable control in `shell-main`. */
async function focusLast(page: Page): Promise<void> {
  await page.getByTestId('shell-main').evaluate((main) => {
    main.scrollTop = main.scrollHeight;
    const focusable = main.querySelectorAll<HTMLElement>(
      'button:not([disabled]), a[href], input:not([disabled]), summary, [tabindex="0"]',
    );
    focusable[focusable.length - 1]!.focus();
  });
}

interface Reading {
  /** The focused element's top and the lowest pinned element's bottom, in viewport pixels. */
  readonly top: number;
  readonly covered: number;
  /** The focused element is itself pinned (the walk has reached the pinned stack). */
  readonly pinned: boolean;
  readonly label: string;
}

const reading = (page: Page, pinnedIds: readonly string[]) =>
  page.evaluate((ids): Reading => {
    const active = document.activeElement as HTMLElement;
    const covers = ids
      .map((id) => document.querySelector(`[data-testid="${id}"]`))
      .filter((element): element is Element => element !== null);
    return {
      top: active.getBoundingClientRect().top,
      covered: Math.max(...covers.map((element) => element.getBoundingClientRect().bottom)),
      pinned: covers.some((element) => element.contains(active)),
      label: `${active.tagName} ${active.textContent?.slice(0, 40) ?? ''}`,
    };
  }, pinnedIds);

/** Walk back `presses` times with Shift+Tab; every control reached lands clear of the pins. */
async function walkBack(page: Page, pinnedIds: readonly string[], presses: number) {
  let checked = 0;
  for (let n = 0; n < presses; n += 1) {
    await page.keyboard.press('Shift+Tab');
    // Let the focus scroll settle before reading.
    await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => done(null))));
    const now = await reading(page, pinnedIds);
    if (now.pinned) break;
    expect(now.top, `${now.label} lands below the pinned stack`).toBeGreaterThanOrEqual(
      now.covered - 1,
    );
    checked += 1;
  }
  // The walk checked real controls, not none.
  expect(checked).toBeGreaterThanOrEqual(8);
}

test('a training run row reached by Shift+Tab lands clear of the approvals dock', async ({
  page,
}) => {
  await openRuns(page);
  await focusLast(page);
  await walkBack(page, ['approvals-banner-dock'], 20);
});

test('a WSP run control reached by Shift+Tab lands clear of the floating header', async ({
  page,
}) => {
  await openRun(page, false);
  await focusLast(page);
  await walkBack(page, ['wsp-run-header'], 24);
});

test('a WSP run control reached by Shift+Tab lands clear of the dock and the header', async ({
  page,
}) => {
  await openRun(page, true);
  await focusLast(page);
  await walkBack(page, ['approvals-banner-dock', 'wsp-run-header'], 24);
});
