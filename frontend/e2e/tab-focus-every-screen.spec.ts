/**
 * On every screen, a control reached by the keyboard lands fully visible: clear of everything
 * pinned at the top of its scroller and of everything pinned at its bottom (Tab-focus debt,
 * suite board "Tracked debt found by trains", 2026-11-27).
 *
 * `tab-focus-under-pinned.spec.ts` walks the two screens the fix was written for. This walks
 * every screen in `config/routes.ts` (`src/config/phonePages.ts`, pinned to `ROUTES` by
 * `phonePages.test.ts`) with requests waiting, so the approvals dock is open on every screen but
 * Approvals, on a window short enough that each screen scrolls.
 *
 * What is pinned is read from the live page, not from a list: every element whose computed
 * `position` is `sticky` (or `fixed`) with a `top` or a `bottom`. A pinned header added later that
 * forgets to register with `layout/pinned.ts` is caught here without this file changing.
 *
 * Each screen is walked twice. Backwards from the end with Shift+Tab, which is how a control comes
 * into view under a pinned header; and forwards from the top with Tab, which is how one comes into
 * view above a pinned footer (an approval card's decision row, inside the dock's region and on
 * Approvals). Going backwards, every control outside a pinned element must start below the lowest
 * pinned top of every scroller it is inside; going forwards, every control (those in the dock
 * too) must end above the highest pinned bottom. A sticky element sticks inside its scroller's
 * padding, so that is where it is read as stuck.
 *
 * Controls, each RED in the dark project (2026-11-27): `shell-main` without `scroll-pt` (all 11
 * docked screens); without `scroll-pb` (Approvals); the decision row unregistered (Approvals); the
 * dock's region padding without `var(--spacing-card)` (all 11 docked screens); the dock, the WSP
 * run header or the WSP runs block unregistered (11, 1 and 1 screens here).
 */
import { expect, test, type Page, type Route } from '@playwright/test';
import { answer, WSP_RUN_ID, WSP_RUN_ROW } from '../src/testing/stubBackend';
import approvalsFixture from './fixtures/approvals.json' with { type: 'json' };
import backendFixture from './fixtures/backend.json' with { type: 'json' };
import treeFixture from './fixtures/tree.json' with { type: 'json' };
import { fontsSettled } from './fonts';
import { LONG, PHONE_PAGES } from '../src/config/phonePages';

const recorded = backendFixture as Record<string, unknown>;

/** Enough WSP runs that the list scrolls under its pinned block (the stub serves one). */
const WSP_ROWS = Array.from({ length: 30 }, (_, n) => ({
  ...WSP_RUN_ROW,
  run_id: n === 0 ? WSP_RUN_ROW.run_id : `wsp_${String(n).padStart(12, '0')}`,
}));

/** A transcript long enough to scroll, each step with a reasoning toggle to focus. */
const TRANSCRIPT = {
  run_id: WSP_RUN_ID,
  transcript_sha256: 'c'.repeat(64),
  partial: false,
  entries: Array.from({ length: 12 }, (_, n) => ({
    role: 'assistant',
    content: Array.from({ length: 12 }, (__, i) => `step ${n + 1} line ${i}`).join('\n'),
    reasoning: { text: `why ${n + 1}`, source: 'reasoning_content' },
  })),
};

function bodyFor(method: string, url: URL): { status: number; body: unknown } {
  const path = decodeURIComponent(url.pathname);
  if (method === 'GET' && path === '/api/v1/tree') return { status: 200, body: treeFixture };
  if (method === 'GET' && path === '/api/v1/approvals')
    return { status: 200, body: approvalsFixture };
  if (method === 'GET' && path === '/api/v1/wsp/runs')
    return { status: 200, body: { runs: WSP_ROWS } };
  if (method === 'GET' && path === `/api/v1/wsp/runs/${WSP_RUN_ID}/transcript`)
    return { status: 200, body: TRANSCRIPT };
  const exact = recorded[`${path}${url.search}`] ?? recorded[path];
  if (method === 'GET' && exact !== undefined) return { status: 200, body: exact };
  return answer(method, path);
}

async function serve(page: Page): Promise<void> {
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', (route: Route) => {
    const { status, body } = bodyFor(route.request().method(), new URL(route.request().url()));
    return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  });
}

interface Reading {
  readonly label: string;
  /** The focused element is inside shell-main (the walk has not left the screen). */
  readonly inMain: boolean;
  /** The focused element is itself inside a pinned element. */
  readonly inPinned: boolean;
  readonly top: number;
  readonly bottom: number;
  /** The clear band: below every pinned top, above every pinned bottom, inside every scroller. */
  readonly clearTop: number;
  readonly clearBottom: number;
  readonly covers: string;
}

/** Where the focused element sits, and the band of its scrollers that nothing pinned covers. */
const reading = (page: Page) =>
  page.evaluate((): Reading => {
    const active = document.activeElement as HTMLElement;
    const main = document.querySelector('[data-testid="shell-main"]') as HTMLElement;
    const describe = (element: Element): string =>
      `<${element.tagName.toLowerCase()} ${element.getAttribute('data-testid') ?? ''}> "${(
        element.textContent ?? ''
      )
        .trim()
        .slice(0, 40)}"`;
    const pinned = (element: Element): { top: boolean; bottom: boolean } => {
      const style = getComputedStyle(element);
      if (style.position !== 'sticky' && style.position !== 'fixed')
        return { top: false, bottom: false };
      return { top: style.top !== 'auto', bottom: style.bottom !== 'auto' };
    };
    const box = active.getBoundingClientRect();
    let clearTop = -Infinity;
    let clearBottom = Infinity;
    const covers: string[] = [];
    let inPinned = false;
    for (let up = active.parentElement; up !== null; up = up.parentElement) {
      const own = pinned(up);
      if (own.top || own.bottom) inPinned = true;
    }
    if (pinned(active).top || pinned(active).bottom) inPinned = true;
    // Each scroller the element is inside bounds it, and so does what is pinned within it.
    for (let up = active.parentElement; up !== null; up = up.parentElement) {
      const style = getComputedStyle(up);
      if (!/auto|scroll/.test(style.overflowY) || up.scrollHeight <= up.clientHeight) continue;
      const frame = up.getBoundingClientRect();
      // A sticky element sticks inside the scroller's padding, so that is where it is stuck.
      const inset = { top: parseFloat(style.paddingTop), bottom: parseFloat(style.paddingBottom) };
      clearTop = Math.max(clearTop, frame.top + up.clientTop);
      clearBottom = Math.min(clearBottom, frame.top + up.clientTop + up.clientHeight);
      for (const candidate of up.querySelectorAll('*')) {
        if (candidate.contains(active)) continue;
        const own = pinned(candidate);
        if (!own.top && !own.bottom) continue;
        // Pinned to this scroller, not to a nested one.
        let nearest: Element | null = candidate.parentElement;
        while (nearest !== null && nearest !== up) {
          const s = getComputedStyle(nearest);
          if (/auto|scroll|hidden/.test(s.overflowY)) break;
          nearest = nearest.parentElement;
        }
        if (nearest !== up) continue;
        const at = candidate.getBoundingClientRect();
        if (at.height === 0) continue;
        // It covers only where it is stuck at its edge, overlapping the band.
        if (
          own.top &&
          at.top <=
            frame.top + up.clientTop + inset.top + parseFloat(getComputedStyle(candidate).top) + 1
        ) {
          if (at.bottom > clearTop) covers.push(`top ${describe(candidate)} to ${at.bottom}`);
          clearTop = Math.max(clearTop, at.bottom);
        }
        if (
          own.bottom &&
          at.bottom >=
            frame.top +
              up.clientTop +
              up.clientHeight -
              inset.bottom -
              parseFloat(getComputedStyle(candidate).bottom) -
              1
        ) {
          if (at.top < clearBottom) covers.push(`bottom ${describe(candidate)} from ${at.top}`);
          clearBottom = Math.min(clearBottom, at.top);
        }
      }
    }
    return {
      label: describe(active),
      inMain: main.contains(active),
      inPinned,
      top: box.top,
      bottom: box.bottom,
      clearTop,
      clearBottom,
      covers: covers.join('; '),
    };
  });

const settle = (page: Page) =>
  page.evaluate(
    () =>
      new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => done(null)))),
  );

/** A control reached backwards starts below everything pinned at the top of its scrollers. */
function checkBelowTop(now: Reading): void {
  expect(
    now.top,
    `${now.label} starts below the pinned top (${now.covers})`,
  ).toBeGreaterThanOrEqual(now.clearTop - 1);
}

/** A control reached forwards ends above everything pinned at the bottom (one that fits). */
function checkAboveBottom(now: Reading): void {
  if (now.bottom - now.top > now.clearBottom - now.clearTop) return;
  expect(
    now.bottom,
    `${now.label} ends above the pinned bottom (${now.covers})`,
  ).toBeLessThanOrEqual(now.clearBottom + 1);
}

const FOCUSABLE =
  'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex="0"]';

test.use({ viewport: { width: 1280, height: 560 } });

// The phone audit's ids carry a long suffix; here the screens open at the ids the stub serves.
for (const [screen, hash] of Object.entries(PHONE_PAGES).map(
  ([id, at]) => [id, at.split(LONG).join('')] as const,
)) {
  test(`${screen} (${hash}): a control reached by the keyboard lands clear of what is pinned`, async ({
    page,
  }) => {
    await serve(page);
    await page.goto(`/${hash}`);
    await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
    if (screen !== 'approvals')
      await expect(page.getByTestId('approvals-banner-dock')).toBeVisible();
    await page.waitForLoadState('networkidle');
    await page.evaluate(() => {
      for (const details of document.querySelectorAll('details')) details.open = true;
    });
    await fontsSettled(page);
    const main = page.getByTestId('shell-main');
    // Precondition: the screen scrolls, so a pinned element can cover something.
    expect(
      await main.evaluate((element) => element.scrollHeight > element.clientHeight),
      'the screen is taller than the window',
    ).toBe(true);

    // Backwards from the end.
    const count = await main.evaluate((element, selector) => {
      element.scrollTop = element.scrollHeight;
      const all = element.querySelectorAll<HTMLElement>(selector);
      all[all.length - 1]!.focus();
      return all.length;
    }, FOCUSABLE);
    let checked = 0;
    for (let n = 0; n < Math.min(count, 60); n += 1) {
      await page.keyboard.press('Shift+Tab');
      await settle(page);
      const now = await reading(page);
      if (!now.inMain) break;
      if (now.inPinned) continue;
      checkBelowTop(now);
      checked += 1;
    }
    expect(checked, 'the backward walk checked real controls').toBeGreaterThan(0);

    // Forwards from the top.
    await main.evaluate((element, selector) => {
      element.scrollTop = 0;
      element.querySelector<HTMLElement>(selector)!.focus();
    }, FOCUSABLE);
    checked = 0;
    for (let n = 0; n < Math.min(count, 60); n += 1) {
      await page.keyboard.press('Tab');
      await settle(page);
      const now = await reading(page);
      if (!now.inMain) break;
      checkAboveBottom(now);
      checked += 1;
    }
    expect(checked, 'the forward walk checked real controls').toBeGreaterThan(0);
  });
}
