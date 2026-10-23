/**
 * The page scrolls; the sidebar does not (operator, 2026-10-20: *"The left navigation bar in
 * miForge should not scroll. The pages should scroll independently from it."*).
 *
 * Runs in both theme projects. Each test measures **geometry in a real browser**, because
 * the defect is one jsdom cannot see: before the fix the document scrolled and carried the
 * sidebar up with it (its bounding box moved by exactly the scroll distance).
 *
 * Negative control: restoring the pre-fix `AppShell.tsx`, `Sidebar.tsx` and `index.css`
 * turns the desktop and phone geometry tests red (recorded in the commit).
 *
 * The long page is New run: with no backend under `vite preview` every one of its regions
 * renders its refusal, and the form is several viewports tall. Each test first asserts the
 * content really overflows, so a shorter page can never make the test pass vacuously.
 */
import { expect, test, type Locator, type Page, type Route } from '@playwright/test';
import approvals from './fixtures/approvals.json' with { type: 'json' };

const json = (route: Route, body: unknown) =>
  route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });

/** Serve the approval list so the sidebar's pending badge is drawn. */
async function serveApprovals(page: Page): Promise<number> {
  const rows = (approvals as { approvals: { status: string }[] }).approvals;
  const pending = rows.filter((row) => row.status === 'pending').length;
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/v1/approvals', (route) =>
    json(route, { approvals: rows, count: rows.length, pending, limit: 200 }),
  );
  return pending;
}

async function box(locator: Locator) {
  const found = await locator.boundingBox();
  if (found === null) throw new Error('element has no box');
  return found;
}

const scrollTop = (locator: Locator) => locator.evaluate((element) => element.scrollTop);

/**
 * The scroll offset once a wheel's scroll has finished. A wheel scroll can animate, and a
 * reading taken mid-flight would not match a box measured a moment later.
 */
async function settledScrollTop(locator: Locator): Promise<number> {
  let last = -1;
  await expect
    .poll(async () => {
      const now = await scrollTop(locator);
      const settled = now === last;
      last = now;
      return settled;
    })
    .toBe(true);
  return last;
}

/** Where the document itself is scrolled to, and how far it could be. */
const documentScroll = (page: Page) =>
  page.evaluate(() => ({
    x: window.scrollX,
    y: window.scrollY,
    overflowY: document.documentElement.scrollHeight - window.innerHeight,
    overflowX: document.documentElement.scrollWidth - window.innerWidth,
  }));

async function openLongPage(page: Page): Promise<void> {
  await page.goto('/#/runs/new');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  // The strip is empty until the first host read lands (here, as "readings unavailable"),
  // and then grows by a line. Wait for it, so the boxes measured below are its final ones.
  await expect(
    page.getByTestId('shell-resource-strip').locator('[data-testid^="resource-"]').first(),
  ).toBeVisible();
  const main = page.getByTestId('shell-main');
  // Precondition: the screen is taller than the window, or nothing below means anything.
  await expect
    .poll(() => main.evaluate((element) => element.scrollHeight - element.clientHeight))
    .toBeGreaterThan(400);
}

/** Wheel over the centre of `over` and wait for `scroller` to have moved. */
async function wheel(page: Page, over: Locator, deltaY: number): Promise<void> {
  const target = await box(over);
  await page.mouse.move(target.x + target.width / 2, target.y + target.height / 2);
  await page.mouse.wheel(0, deltaY);
}

async function assertMainScrollsAlone(page: Page): Promise<void> {
  const sidebar = page.getByTestId('shell-sidebar');
  const strip = page.getByTestId('shell-resource-strip');
  const main = page.getByTestId('shell-main');
  const head = page.getByTestId('shell-page-head');
  const viewport = page.viewportSize();
  if (viewport === null) throw new Error('no viewport');

  const sidebarBefore = await box(sidebar);
  const stripBefore = await box(strip);
  const headBefore = await box(head);
  // The sidebar is the window's height, not the page's.
  expect(sidebarBefore.y).toBe(0);
  expect(sidebarBefore.height).toBe(viewport.height);

  await wheel(page, main, 500);
  await expect.poll(() => scrollTop(main)).toBeGreaterThan(200);
  const moved = await settledScrollTop(main);

  // The page content moved, by exactly what the content area scrolled...
  expect((await box(head)).y).toBeCloseTo(headBefore.y - moved, 0);
  // ...and nothing in the shell did.
  expect(await box(sidebar)).toEqual(sidebarBefore);
  expect(await box(strip)).toEqual(stripBefore);
  // The document is not what scrolled, and cannot scroll.
  const doc = await documentScroll(page);
  expect(doc).toEqual({ x: 0, y: 0, overflowY: 0, overflowX: 0 });
}

test.describe('desktop', () => {
  test('scrolling a long page moves the page and not the sidebar', async ({ page }) => {
    await openLongPage(page);
    await assertMainScrollsAlone(page);
  });

  test('the badge and every nav entry stay in place while the page scrolls', async ({ page }) => {
    const pending = await serveApprovals(page);
    await openLongPage(page);
    const badge = page.getByTestId('nav-badge-approvals');
    await expect(badge).toHaveText(String(pending));
    const manual = page.getByTestId('sidebar-manual-link');
    const before = [await box(badge), await box(manual)];

    await wheel(page, page.getByTestId('shell-main'), 2_000);
    await expect.poll(() => scrollTop(page.getByTestId('shell-main'))).toBeGreaterThan(1_000);

    await expect(badge).toBeInViewport();
    await expect(manual).toBeInViewport();
    expect([await box(badge), await box(manual)]).toEqual(before);
    await expect(page.getByTestId('sidebar-footer')).toBeInViewport({ ratio: 1 });
  });

  test('a wheel over the sidebar does not scroll the page', async ({ page }) => {
    await openLongPage(page);
    await wheel(page, page.getByTestId('shell-sidebar'), 600);
    // Give a stray scroll time to land before asserting it did not.
    await page.waitForTimeout(300);
    expect(await scrollTop(page.getByTestId('shell-main'))).toBe(0);
    expect((await documentScroll(page)).y).toBe(0);
  });

  test('a new screen opens at the top, not where the last one was left', async ({ page }) => {
    await openLongPage(page);
    const main = page.getByTestId('shell-main');
    // Without a backend only New run is long, and a short screen would clamp a stale offset
    // to 0 by itself - which would make this test pass with no reset at all. A spacer after
    // the screen (outside React's tree, so it survives navigation) keeps every screen long,
    // so only the reset can bring the offset back to the top.
    await main.evaluate((element) => {
      const spacer = document.createElement('div');
      spacer.style.height = '5000px';
      element.appendChild(spacer);
    });
    const nav = page.getByRole('navigation', { name: 'Screens' });

    for (const [label, heading] of [
      ['Domain packs', 'Domain packs'],
      ['New run', 'New run'],
    ] as const) {
      await wheel(page, main, 1_500);
      await expect.poll(() => scrollTop(main)).toBeGreaterThan(500);
      await nav.getByRole('button', { name: label }).click();
      await expect(page.getByRole('heading', { level: 1 })).toHaveText(heading);
      expect(await scrollTop(main)).toBe(0);
    }
  });

  test('tabbing from the sidebar into the page scrolls the page, not the sidebar', async ({
    page,
  }) => {
    await openLongPage(page);
    const sidebar = page.getByTestId('shell-sidebar');
    const before = await box(sidebar);
    // The first control is the first nav entry: the focus order is sidebar, then page.
    await page.keyboard.press('Tab');
    await expect(
      page.getByRole('navigation', { name: 'Screens' }).getByRole('button').first(),
    ).toBeFocused();
    // Walk well into the form; each focused control is brought into view by the browser.
    for (let step = 0; step < 40; step += 1) await page.keyboard.press('Tab');
    await expect.poll(() => scrollTop(page.getByTestId('shell-main'))).toBeGreaterThan(0);
    expect(await box(sidebar)).toEqual(before);
    expect((await documentScroll(page)).y).toBe(0);
  });
});

test.describe('a window shorter than the sidebar', () => {
  test.use({ viewport: { width: 1280, height: 320 } });

  test('the list of screens scrolls within the sidebar; the footer stays pinned', async ({
    page,
  }) => {
    const pending = await serveApprovals(page);
    await openLongPage(page);
    const sidebar = page.getByTestId('shell-sidebar');
    const screens = page.getByRole('navigation', { name: 'Screens' });
    const footer = page.getByTestId('sidebar-footer');
    const main = page.getByTestId('shell-main');
    await expect(page.getByTestId('nav-badge-approvals')).toHaveText(String(pending));
    // Precondition: the list of screens outgrows this window.
    expect(await screens.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(
      true,
    );
    const before = await box(sidebar);
    expect(before.height).toBe(320);
    // The footer is at the sidebar's bottom edge, whole, before any scroll.
    const footerBefore = await box(footer);
    expect(footerBefore.y + footerBefore.height).toBeCloseTo(320, 0);
    await expect(page.getByTestId('sidebar-manual-link')).toBeInViewport({ ratio: 1 });
    await expect(page.getByTestId('sidebar-version')).toBeInViewport({ ratio: 1 });

    await wheel(page, screens, 400);
    await expect.poll(() => scrollTop(screens)).toBeGreaterThan(0);
    // Only the list moved: the sidebar, the footer, the page and the document did not.
    expect(await box(sidebar)).toEqual(before);
    expect(await box(footer)).toEqual(footerBefore);
    expect(await scrollTop(sidebar)).toBe(0);
    expect(await scrollTop(main)).toBe(0);
    expect((await documentScroll(page)).y).toBe(0);
    // The lowest screen is reachable inside the list, above the footer.
    const badge = await box(page.getByTestId('nav-badge-approvals'));
    expect(badge.y + badge.height).toBeLessThanOrEqual(footerBefore.y);
    await expect(page.getByTestId('nav-badge-approvals')).toBeInViewport();
  });
});

test.describe('phone width', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('the page scrolls inside the content area, with no sideways page scroll', async ({
    page,
  }) => {
    await openLongPage(page);
    await assertMainScrollsAlone(page);
  });
});
