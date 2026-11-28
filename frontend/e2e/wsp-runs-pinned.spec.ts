/**
 * The WSP runs list's pinned header, in a real browser (operator request 2026-11-25). jsdom has
 * no layout, so this is where the pinning itself is proven:
 *
 * - everything from the title down to the "N runs, newest first." label stays put while
 *   `shell-main` scrolls the rows, opaque and on top of them;
 * - on a short window (600 px) with many workers the pinned block is capped and scrolls inside,
 *   so the rows keep at least 40 % of the viewport and the count label stays in view;
 * - a row reached by Shift+Tab is scrolled into view below the pinned block, never behind it.
 */
import { expect, test, type Page, type Route } from '@playwright/test';

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

const ROWS = 40;
const runId = (n: number) => `wsp_${String(n).padStart(12, '0')}`;

const row = (n: number) => ({
  run_id: runId(n),
  status: 'completed',
  kind: 'repair',
  mode: 'session',
  caller: 'agent:forge-mcp',
  caller_ref: `probe-${n}`,
  endpoint_id: 'ep-local',
  stop_reason: 'submitted',
  outcome: 'no_change',
  outcome_reason: null,
  created_at: '2026-11-25T15:00:00+00:00',
  started_at: '2026-11-25T15:00:05+00:00',
  finished_at: '2026-11-25T15:10:00+00:00',
  provider: 'miwsp_engine',
  provider_version: '0.1.0',
  failure: null,
  usage: { requests: 3, input_tokens: 7000, output_tokens: 3200, cost_usd: 0 },
  forward_passes_used: 2,
  metered: { cost_usd: 0, cache_read_tokens: null, cache_write_tokens: null },
});

const pool = (workers: number) => ({
  pool_size: workers,
  maximum: workers,
  effective: workers,
  busy: 0,
  idle: workers,
  not_answering: 0,
  workers: Array.from({ length: workers }, (_, index) => ({
    index,
    state: 'idle',
    worker_state: 'idle',
    run_id: null,
    boot_id: 'b'.repeat(32),
    read_at: '2026-11-25T15:00:00+00:00',
  })),
  queued_by_endpoint: [],
  queued_without_endpoint: 0,
  shared_base_urls: [],
});

/** A WSP request waiting in Approvals: the approvals dock opens, and the list shows its row. */
const WAITING = {
  approval_id: 'appr-wsp-waiting',
  action: 'wsp_session',
  status: 'pending',
  summary: { kind: 'repair', mode: 'session', endpoint: { model: 'qwen-editor' } },
  requested_by: 'agent:forge-mcp',
  decided_by: null,
  reason: null,
  result: null,
  error: null,
  created_at: '2026-11-25T10:17:00Z',
  decided_at: null,
  expires_at: '2099-11-26T10:17:00Z',
};

async function open(page: Page, workers: number, waiting = false): Promise<void> {
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/v1/approvals') {
      const rows = waiting ? [WAITING] : [];
      return json(route, { approvals: rows, count: rows.length, pending: rows.length, limit: 500 });
    }
    if (path === '/api/v1/wsp/runs')
      return json(route, { runs: Array.from({ length: ROWS }, (_, n) => row(n)) });
    if (path === '/api/v1/wsp/pool') return json(route, pool(workers));
    return json(route, { error: { code: 'not_stubbed', message: path, details: {} } }, 404);
  });
  await page.goto('/#/wsp/runs');
  await expect(page.getByTestId('wsp-runs-count')).toHaveText(`${ROWS} runs, newest first.`);
  await expect(page.getByTestId('wsp-pool-workers').getByRole('listitem')).toHaveCount(workers);
}

async function box(page: Page, testId: string) {
  const found = await page.getByTestId(testId).boundingBox();
  if (found === null) throw new Error(`${testId} has no box`);
  return found;
}

const scrollTo = (page: Page, at: number | 'end') =>
  page.getByTestId('shell-main').evaluate((main, where) => {
    main.scrollTop = where === 'end' ? main.scrollHeight : where;
  }, at);

/** Nothing is painted over the pinned block: each probe point hits the block itself. */
const pinnedIsOnTop = (page: Page) =>
  page.getByTestId('wsp-runs-pinned').evaluate((pinned) => {
    const r = pinned.getBoundingClientRect();
    return [0.05, 0.5, 0.95].every((fy) =>
      [0.1, 0.5, 0.9].every((fx) =>
        pinned.contains(document.elementFromPoint(r.x + r.width * fx, r.y + r.height * fy)),
      ),
    );
  });

test('the header and the count label stay put while the rows scroll', async ({
  page,
}, testInfo) => {
  await open(page, 4);
  // Opaque in this project's theme: no row shows through.
  const background = await page
    .getByTestId('wsp-runs-pinned')
    .evaluate((pinned) => getComputedStyle(pinned).backgroundColor);
  expect(background).toMatch(/^rgb\(/);

  await scrollTo(page, 200);
  const main = await box(page, 'shell-main');
  await expect
    .poll(async () => Math.abs((await box(page, 'wsp-runs-pinned')).y - main.y))
    .toBeLessThan(2);
  const pinned = await box(page, 'wsp-runs-pinned');
  const label = await box(page, 'wsp-runs-count');
  const title = await box(page, 'shell-page-head');
  for (const at of [800, 'end'] as const) {
    await scrollTo(page, at);
    await expect.poll(async () => (await box(page, 'wsp-runs-pinned')).y).toBeCloseTo(pinned.y, 0);
    expect((await box(page, 'wsp-runs-count')).y).toBeCloseTo(label.y, 0);
    expect((await box(page, 'shell-page-head')).y).toBeCloseTo(title.y, 0);
    expect(await pinnedIsOnTop(page)).toBe(true);
    await expect(page.getByTestId('wsp-runs-reread')).toBeInViewport();
    await expect(page.getByTestId('wsp-runs-count')).toBeInViewport();
  }
  // The rows did move: the last one is now in view, and the first has gone under the block.
  await expect(page.getByTestId(`wsp-run-row-${runId(ROWS - 1)}`)).toBeInViewport();
  await expect(page.getByTestId(`wsp-run-row-${runId(0)}`)).not.toBeInViewport();
  await page.screenshot({ path: testInfo.outputPath('wsp-runs-scrolled-to-end.png') });
});

test('on a 600 px window with many workers the rows keep 40 % of the viewport', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 1280, height: 600 });
  await open(page, 8);
  await scrollTo(page, 600);
  const main = await box(page, 'shell-main');
  const pinned = await box(page, 'wsp-runs-pinned');
  expect(pinned.height).toBeLessThanOrEqual(0.6 * 600 - 64 + 1);
  const listShare = (main.y + main.height - (pinned.y + pinned.height)) / 600;
  expect(listShare).toBeGreaterThanOrEqual(0.4);
  // The workers overflow the cap, so the part above the label scrolls inside the block...
  const upper = page.getByTestId('wsp-runs-pinned-scroll');
  expect(await upper.evaluate((e) => e.scrollHeight > e.clientHeight)).toBe(true);
  // ...and the count label stays in view, the last thing in the block.
  await expect(page.getByTestId('wsp-runs-count')).toBeInViewport();
  const label = await box(page, 'wsp-runs-count');
  expect(label.y + label.height).toBeLessThanOrEqual(pinned.y + pinned.height + 1);
  await page.screenshot({ path: testInfo.outputPath('wsp-runs-600px-scrolled.png') });
  // Everything in the block is reachable: its own scroll brings the last worker into view.
  await upper.evaluate((e) => {
    e.scrollTop = e.scrollHeight;
  });
  await expect(page.getByTestId('wsp-pool-queue')).toBeInViewport();
  await expect(page.getByTestId('wsp-pool-worker-7')).toBeInViewport();
  expect((await box(page, 'wsp-runs-pinned')).y).toBeCloseTo(pinned.y, 0);
  await page.screenshot({ path: testInfo.outputPath('wsp-runs-600px-pinned-scrolled.png') });
});

test('with a request waiting the block sits below the approvals dock, its row pinned', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 600 });
  await open(page, 8, true);
  await expect(page.getByTestId('approvals-banner-dock')).toBeVisible();
  await expect(page.getByTestId('wsp-pending-rows')).toBeVisible();
  await scrollTo(page, 900);
  const gap = async () => {
    const dock = await box(page, 'approvals-banner-dock');
    return Math.abs((await box(page, 'wsp-runs-pinned')).y - (dock.y + dock.height));
  };
  await expect.poll(gap).toBeLessThan(2);
  expect(await pinnedIsOnTop(page)).toBe(true);
  await expect(page.getByTestId('wsp-runs-count')).toBeInViewport();
  // The dock (expanded: its own cap, 40vh) and the block together leave the rows 40 % of the
  // window: the block's cap is 60dvh - 4rem less the dock, never under 4rem.
  const main = await box(page, 'shell-main');
  const pinned = await box(page, 'wsp-runs-pinned');
  expect((main.y + main.height - (pinned.y + pinned.height)) / 600).toBeGreaterThanOrEqual(0.4);
  // The pending request is in the block, before the label, reachable by its own scroll.
  const upper = page.getByTestId('wsp-runs-pinned-scroll');
  await upper.evaluate((e) => {
    e.scrollTop = e.scrollHeight;
  });
  await expect(page.getByTestId('wsp-pending-rows')).toBeInViewport();
});

test('a row reached by Shift+Tab lands below the pinned block', async ({ page }) => {
  await open(page, 4);
  await scrollTo(page, 'end');
  await page.getByTestId(`wsp-run-row-${runId(ROWS - 1)}`).focus();
  for (let n = ROWS - 2; n >= ROWS - 12; n -= 1) {
    await page.keyboard.press('Shift+Tab');
    const focused = page.getByTestId(`wsp-run-row-${runId(n)}`);
    await expect(focused).toBeFocused();
    const pinned = await box(page, 'wsp-runs-pinned');
    await expect
      .poll(async () => (await box(page, `wsp-run-row-${runId(n)}`)).y)
      .toBeGreaterThanOrEqual(pinned.y + pinned.height - 1);
  }
});
