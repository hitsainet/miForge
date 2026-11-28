/**
 * At phone width, no text runs past its card or off the screen (operator, 2026-11-26, iPhone:
 * the checkpoint tree's ids ran past each card's right edge and were cut off at the viewport).
 *
 * Every screen in `config/routes.ts` (`src/config/phonePages.ts`, pinned to `ROUTES` by
 * `src/config/phonePages.test.ts`) is opened at 390 x 844 in both theme projects, against a backend whose
 * identifiers are as long as real ones (a Hub revision, a run name and a step). Then, for every
 * rendered element:
 * - its right edge, and the right edge of what it holds (`scrollWidth`), stay inside the card it
 *   sits in (`.rounded-card`) and inside the viewport;
 * - an element that clips (`overflow: hidden`) clips nothing, so no id hides behind an ellipsis.
 *
 * A region that scrolls sideways on purpose (`overflow-x: auto`, such as a wide table) is its
 * own boundary: what is inside it is measured against it, and it is measured against its card.
 */
import { writeFile } from 'node:fs/promises';
import { expect, test, type Page, type Route } from '@playwright/test';
import { answer } from '../src/testing/stubBackend';
import approvalsFixture from './fixtures/approvals.json' with { type: 'json' };
import backendFixture from './fixtures/backend.json' with { type: 'json' };
import treeFixture from './fixtures/tree.json' with { type: 'json' };
import { fontsSettled } from './fonts';
import { LONG, PHONE_PAGES } from '../src/config/phonePages';

/** The keys whose string values are identifiers; each gets {@link LONG} appended. */
const ID_KEYS = new Set([
  'approval_id',
  'base',
  'base_model',
  'boot_id',
  'caller_ref',
  'candidate_id',
  'checkpoint',
  'endpoint_id',
  'id',
  'model',
  'name',
  'path',
  'relative_path',
  'rendered',
  'repo',
  'run_id',
  'target',
  'tokenizer',
]);

function longIds(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(longIds);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [
      key,
      ID_KEYS.has(key) && typeof item === 'string' && item.length > 0
        ? `${item}${LONG}`
        : // A WSP run's budget meters are named by what they meter (tokens, cost_usd).
          key === 'budgets'
          ? item
          : longIds(item),
    ]),
  );
}

const recorded = backendFixture as Record<string, unknown>;
/** `Sidebar.tsx`'s SIDEBAR_COLLAPSED_KEY (the app module cannot load here). */
const SIDEBAR_KEY = 'miforge.sidebar.collapsed';

/** The recorded backend first, then the shared stub; ids in the request are shortened back. */
function bodyFor(method: string, url: URL): { status: number; body: unknown } {
  const path = decodeURIComponent(url.pathname).split(LONG).join('');
  if (method === 'GET' && path === '/api/v1/tree') return { status: 200, body: treeFixture };
  if (method === 'GET' && path === '/api/v1/approvals')
    return { status: 200, body: approvalsFixture };
  const exact = recorded[`${path}${url.search}`] ?? recorded[path];
  if (method === 'GET' && exact !== undefined) return { status: 200, body: exact };
  return answer(method, path);
}

/**
 * Answers whose `name` is a vocabulary word or a hardware model, not an identifier (a run type,
 * a signal source, a selector, a stop condition, a GPU), so they stay as recorded.
 */
const VOCABULARY =
  /\/api\/v1\/(gpus|host|run-types|signal-sources|selectors|stop-conditions|models\/sources)\b/;
const vocabulary = (url: string): boolean => VOCABULARY.test(new URL(url).pathname);

async function serve(page: Page): Promise<void> {
  // No stored sidebar choice: a phone width collapses it by itself (operator, 2026-11-27;
  // expanded, it leaves the page 166 of 390 px). `sidebar-phone.spec.ts` covers a stored one.
  await page.addInitScript((key) => window.localStorage.removeItem(key), SIDEBAR_KEY);
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', (route: Route) => {
    const { status, body } = bodyFor(route.request().method(), new URL(route.request().url()));
    return route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(
        status < 300 && !vocabulary(route.request().url()) ? longIds(body) : body,
      ),
    });
  });
}

interface Overflow {
  readonly what: string;
  readonly right: number;
  readonly limit: number;
  readonly against: string;
  /** No flagged element inside it: the content that overflows, not an ancestor it widened. */
  readonly innermost: boolean;
}

/** Every element whose box or content runs past its card, its sideways scroller or the screen. */
function overflows(page: Page): Promise<Overflow[]> {
  return page.evaluate(() => {
    const viewport = document.documentElement.clientWidth;
    const flagged: { element: Element; entry: Omit<Overflow, 'innermost'> }[] = [];
    const describe = (element: Element): string => {
      const id = element.getAttribute('data-testid');
      const text = (element.textContent ?? '').trim().slice(0, 70);
      return `<${element.tagName.toLowerCase()}${id === null ? '' : ` ${id}`}> "${text}"`;
    };
    const scrollsSideways = (element: Element): boolean => {
      if (element === document.documentElement || element === document.body) return false;
      // The content area scrolls vertically, which computes overflow-x to auto; it is the page,
      // not a deliberate sideways region, so the viewport stays its bound.
      if (element.getAttribute('data-testid') === 'shell-main') return false;
      return /auto|scroll/.test(getComputedStyle(element).overflowX);
    };
    for (const element of document.querySelectorAll('body *')) {
      const box = element.getBoundingClientRect();
      if (box.width === 0 && box.height === 0) continue;
      // An icon's parts are measured as the icon.
      if (element.closest('svg') !== element && element.closest('svg') !== null) continue;
      const style = getComputedStyle(element);
      if (style.visibility === 'hidden' || style.position === 'fixed') continue;
      // Screen-reader-only text is clipped to a pixel by design.
      if (element.classList.contains('sr-only')) continue;
      const right =
        element instanceof HTMLElement && style.display !== 'inline'
          ? Math.max(box.right, box.left + element.scrollWidth)
          : box.right;
      let bound: { right: number; name: string } = { right: viewport, name: 'the viewport' };
      for (let up = element.parentElement; up !== null; up = up.parentElement) {
        if (scrollsSideways(up)) {
          bound = { right: up.getBoundingClientRect().right, name: `its scroller ${describe(up)}` };
          break;
        }
        if (up.classList.contains('rounded-card')) {
          bound = { right: up.getBoundingClientRect().right, name: `its card ${describe(up)}` };
          break;
        }
      }
      if (right > bound.right + 0.5)
        flagged.push({
          element,
          entry: { what: describe(element), right, limit: bound.right, against: bound.name },
        });
      else if (
        element instanceof HTMLElement &&
        /hidden|clip/.test(style.overflowX) &&
        element.scrollWidth > element.clientWidth + 1
      )
        flagged.push({
          element,
          entry: {
            what: describe(element),
            right: box.left + element.scrollWidth,
            limit: box.right,
            against: 'itself (clipped)',
          },
        });
    }
    return flagged.map(({ element, entry }) => ({
      ...entry,
      innermost: !flagged.some(
        (other) => other.element !== element && element.contains(other.element),
      ),
    }));
  });
}

test.use({ viewport: { width: 390, height: 844 } });

for (const [screen, hash] of Object.entries(PHONE_PAGES)) {
  test(`${screen} (${hash}): nothing runs past its card or the screen at 390 px`, async ({
    page,
  }) => {
    await serve(page);
    await page.goto(`/${hash}`);
    // Attached, not visible: an overflowing title is exactly what may not read as visible here.
    await expect(page.getByRole('heading', { level: 1 }).first()).toBeAttached();
    await expect(page.getByTestId('shell-sidebar')).toHaveAttribute('data-collapsed', 'true');
    await page.waitForLoadState('networkidle');
    // What a collapsed <details> holds is measured too: the operator opens it on a phone.
    await page.evaluate(() => {
      for (const details of document.querySelectorAll('details')) details.open = true;
    });
    await fontsSettled(page);
    const found = await overflows(page);
    await writeFile(test.info().outputPath('overflows.json'), JSON.stringify(found, null, 1));
    // The head's actions drop below the title rather than squeezing it to a few words a line
    // (the Checkpoints title sat under its own Re-read button before; that is no overflow, so
    // the walk above cannot see it).
    const actions = page.getByTestId('shell-page-head-actions');
    if ((await actions.count()) > 0) {
      const title = await page.getByTestId('shell-page-head').locator('h1').boundingBox();
      const below = await actions.boundingBox();
      if (title === null || below === null) throw new Error('the page head has no box');
      expect(below.y, 'the actions sit below the title').toBeGreaterThanOrEqual(
        title.y + title.height,
      );
    }
    // Only the innermost offenders are listed; every ancestor they widened is in overflows.json.
    expect(
      found
        .filter((entry) => entry.innermost)
        .map(
          ({ what, right, limit, against }) =>
            `${what} ends at ${right}, past ${limit} (${against})`,
        ),
    ).toEqual([]);
  });
}
