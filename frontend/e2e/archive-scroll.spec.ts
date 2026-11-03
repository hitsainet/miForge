/**
 * "Show archived runs" works inside the page's own scroll container (operator decisions
 * 2026-10-20: archive, and "the pages should scroll independently from the sidebar").
 *
 * The archive toggle was written while the document scrolled; the layout fix then made
 * `shell-main` the only scroller. This spec runs the toggle there, in a real browser: the run
 * list is long enough that `shell-main` overflows, the page is scrolled to the bottom first,
 * and the click must reach the toggle, send `include_archived=true` (once per list it feeds),
 * show the archived rows marked - all without the document scrolling or the sidebar moving.
 *
 * The run list is served at the network from the generated fixture's own row shape
 * (`fixtures/backend.json`, pinned by the backend), with only the run ids and `archived_at`
 * varied.
 */
import { expect, test, type Locator, type Page, type Route } from '@playwright/test';
import backend from './fixtures/backend.json' with { type: 'json' };

interface RunRow {
  readonly run_id: string;
  readonly archived_at: string | null;
  readonly [key: string]: unknown;
}

const SEEDED = (backend as Record<string, unknown>)['/api/v1/runs?limit=25&offset=0'] as {
  runs: RunRow[];
  [key: string]: unknown;
};
const TEMPLATE = SEEDED.runs[0] as RunRow;
const LISTED: RunRow[] = Array.from({ length: 30 }, (_, index) => ({
  ...TEMPLATE,
  run_id: `run-listed-${index + 1}`,
  archived_at: null,
}));
const ARCHIVED: RunRow[] = Array.from({ length: 3 }, (_, index) => ({
  ...TEMPLATE,
  run_id: `run-archived-${index + 1}`,
  archived_at: '2026-10-20T09:00:00+00:00',
}));

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

/** Serve the run list (archived rows first, only when asked). Returns every list query. */
async function serve(page: Page): Promise<string[]> {
  const queries: string[] = [];
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === 'GET' && url.pathname === '/api/v1/runs') {
      queries.push(url.search);
      const include = url.searchParams.get('include_archived') === 'true';
      const limit = Number(url.searchParams.get('limit') ?? '50');
      const offset = Number(url.searchParams.get('offset') ?? '0');
      const all = include ? [...ARCHIVED, ...LISTED] : LISTED;
      return json(route, {
        ...SEEDED,
        include_archived: include,
        runs: all.slice(offset, offset + limit),
        total: all.length,
        limit,
        offset,
      });
    }
    return json(route, { error: { code: 'not_stubbed', message: url.pathname, details: {} } }, 404);
  });
  return queries;
}

async function box(locator: Locator) {
  const found = await locator.boundingBox();
  if (found === null) throw new Error('element has no box');
  return found;
}

test('Show archived runs works from a scrolled page inside the scroll container', async ({
  page,
}) => {
  const queries = await serve(page);
  await page.goto('/#/runs');
  const main = page.getByTestId('shell-main');
  const rows = page.getByTestId('run-list-rows').locator('[data-testid^="run-list-row-"]');
  await expect(rows).toHaveCount(25);
  await expect(page.getByTestId('run-list-archived')).toHaveCount(0);

  // Precondition: the list overflows the window, so scrolling means something.
  await expect
    .poll(() => main.evaluate((element) => element.scrollHeight - element.clientHeight))
    .toBeGreaterThan(400);
  const sidebarBefore = await box(page.getByTestId('shell-sidebar'));

  // Scroll the page to its end; the toggle sits above, out of the window.
  await main.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  await expect.poll(() => main.evaluate((element) => element.scrollTop)).toBeGreaterThan(400);
  const toggle = page.getByTestId('run-show-archived');
  await expect(toggle).not.toBeInViewport();

  const before = queries.length;
  await toggle.check();
  await expect(toggle).toBeChecked();

  // The click reached it: the list (and the open-by-id picker) re-read with archived rows.
  await expect(page.getByTestId('run-list-archived')).toHaveCount(3);
  await expect(page.getByTestId('run-list-row-run-archived-1')).toBeVisible();
  const asked = queries.slice(before);
  expect(asked.filter((query) => query.includes('include_archived=true'))).toHaveLength(2);
  expect(asked.filter((query) => !query.includes('include_archived=true'))).toHaveLength(0);

  // The scroll stayed in shell-main: the document did not move and neither did the sidebar.
  expect(await page.evaluate(() => window.scrollY)).toBe(0);
  expect(await box(page.getByTestId('shell-sidebar'))).toEqual(sidebarBefore);

  // And back: unticking hides them again, still without a document scroll.
  await toggle.uncheck();
  await expect(page.getByTestId('run-list-archived')).toHaveCount(0);
  expect(await page.evaluate(() => window.scrollY)).toBe(0);
});
