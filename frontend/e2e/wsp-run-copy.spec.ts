/**
 * The WSP run page's copy buttons and floating header, in a real browser (operator request
 * 2026-11-22). jsdom has neither layout nor a clipboard, so this is where both are proven:
 *
 * - the header stays at the top of `shell-main` while it scrolls, opaque and on top;
 * - with a request waiting it sits just below the approvals dock, and follows it when it changes;
 * - Copy works over plain HTTP (no `navigator.clipboard`, as on the LAN), read back by pasting,
 *   and does not move the page.
 */
import { expect, test, type Page, type Route } from '@playwright/test';
import { fontsSettled } from './fonts';

const RUN = 'wsp_0123456789ab';
const STEP = (n: number) =>
  Array.from({ length: 40 }, (_, i) => `step ${n} line ${i}  <kept>`).join('\n');
const STEPS = 12;

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

const RUN_DOCUMENT = {
  kind: 'miforge.wsp-run/v1',
  run_id: RUN,
  status: 'completed',
  created_at: '2026-11-22T11:20:28Z',
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
};

const PROGRESS = {
  run_id: RUN,
  status: 'completed',
  live: false,
  now: '2026-11-22T12:31:37Z',
  created_at: '2026-11-22T11:20:28Z',
  started_at: '2026-11-22T11:24:00Z',
  finished_at: '2026-11-22T12:31:00Z',
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
};

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
  created_at: '2026-11-22T10:17:00Z',
  decided_at: null,
  expires_at: '2099-11-23T10:17:00Z',
};

async function open(page: Page, waiting: boolean): Promise<void> {
  // As on the LAN: plain HTTP, so the browser offers no clipboard API.
  await page.addInitScript(() => {
    Object.defineProperty(Navigator.prototype, 'clipboard', {
      configurable: true,
      get: () => undefined,
    });
  });
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', (route) => {
    const path = new URL(route.request().url()).pathname;
    const base = `/api/v1/wsp/runs/${RUN}`;
    if (path === '/api/v1/approvals') {
      const rows = waiting ? [WAITING] : [];
      return json(route, { approvals: rows, count: rows.length, pending: rows.length, limit: 500 });
    }
    if (path === base) return json(route, RUN_DOCUMENT);
    if (path === `${base}/progress`) return json(route, PROGRESS);
    if (path === `${base}/calls`)
      return json(route, { run_id: RUN, status: 'completed', live: false, calls: [], more: false });
    if (path === `${base}/transcript`)
      return json(route, {
        run_id: RUN,
        transcript_sha256: 'c'.repeat(64),
        partial: false,
        entries: Array.from({ length: STEPS }, (_, n) => ({
          role: 'assistant',
          content: STEP(n + 1),
          reasoning: { text: `why ${n + 1}`, source: 'reasoning_content' },
        })),
      });
    return json(route, { error: { code: 'not_stubbed', message: path, details: {} } }, 404);
  });
  await page.goto(`/#/wsp/runs/${RUN}`);
  await expect(page.getByTestId('wsp-transcript-entry')).toHaveCount(STEPS);
  if (waiting) await expect(page.getByTestId('approvals-banner-dock')).toBeVisible();
  // A face landing later re-lays out the transcript, and scroll anchoring then moves
  // `shell-main` (3696 to 3717 on the mirror's runner): measure after the fonts have landed.
  await fontsSettled(page);
}

const scrollTo = (page: Page, where: 'middle' | 'end') =>
  page.getByTestId('shell-main').evaluate((main, at) => {
    main.scrollTop = at === 'end' ? main.scrollHeight : main.scrollHeight / 2;
  }, where);

async function box(page: Page, testId: string) {
  const found = await page.getByTestId(testId).boundingBox();
  if (found === null) throw new Error(`${testId} has no box`);
  return found;
}

/** Nothing is painted over the header: the element at each probe point is the header's own. */
const headerIsOnTop = (page: Page) =>
  page.getByTestId('wsp-run-header').evaluate((header) => {
    const r = header.getBoundingClientRect();
    return [0.1, 0.5, 0.9].every((f) =>
      header.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height * f)),
    );
  });

async function pasted(page: Page): Promise<string> {
  await page.evaluate(() => {
    const area = document.createElement('textarea');
    area.id = 'paste-target';
    document.body.appendChild(area);
    area.focus({ preventScroll: true });
  });
  await page.keyboard.press('ControlOrMeta+V');
  const target = page.locator('#paste-target');
  const value = await target.inputValue();
  await target.evaluate((area) => area.remove());
  return value;
}

test('the header floats above the scroll', async ({ page }) => {
  await open(page, false);
  for (const where of ['middle', 'end'] as const) {
    await scrollTo(page, where);
    await expect
      .poll(async () =>
        Math.abs((await box(page, 'wsp-run-header')).y - (await box(page, 'shell-main')).y),
      )
      .toBeLessThan(2);
    expect(await headerIsOnTop(page)).toBe(true);
    await expect(page.getByTestId('wsp-copy-page')).toBeInViewport();
  }
});

test('with a request waiting the header sits below the approvals dock', async ({ page }) => {
  await open(page, true);
  await scrollTo(page, 'middle');
  const gap = async () => {
    const dock = await box(page, 'approvals-banner-dock');
    return Math.abs((await box(page, 'wsp-run-header')).y - (dock.y + dock.height));
  };
  await expect.poll(gap).toBeLessThan(2);
  expect(await headerIsOnTop(page)).toBe(true);
  // The dock collapses to one line; the header follows it up.
  const tall = (await box(page, 'approvals-banner-dock')).height;
  await page
    .getByTestId('approvals-banner')
    .getByRole('button', { name: /collapse|hide/i })
    .click();
  await expect
    .poll(async () => (await box(page, 'approvals-banner-dock')).height)
    .toBeLessThan(tall);
  await expect.poll(gap).toBeLessThan(2);
});

test('a step copies over plain HTTP without moving the page', async ({ page }) => {
  await open(page, false);
  const button = page.getByRole('button', { name: 'Copy step 7 content', exact: true });
  await button.evaluate((element) => element.scrollIntoView({ block: 'center' }));
  const main = page.getByTestId('shell-main');
  const before = await main.evaluate((element) => element.scrollTop);
  await button.click();
  await expect(page.getByTestId('wsp-transcript-entry').nth(6)).toContainText('Copied');
  expect(await main.evaluate((element) => element.scrollTop)).toBe(before);
  expect(await pasted(page)).toBe(STEP(7));
});

test('Copy page copies the whole run over plain HTTP', async ({ page }) => {
  await open(page, false);
  await scrollTo(page, 'end');
  await page.getByTestId('wsp-copy-page').click();
  await expect(page.getByTestId('wsp-copy-page')).toContainText('Copied');
  const text = await pasted(page);
  expect(text).toContain(RUN);
  expect(text).toContain('Repair the probe.');
  for (let n = 1; n <= STEPS; n += 1) {
    expect(text).toContain(STEP(n));
    expect(text).toContain(`why ${n}`); // collapsed on screen
  }
});
