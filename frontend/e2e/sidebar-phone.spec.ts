/**
 * The sidebar starts collapsed on a phone (operator, 2026-11-27, train F4).
 *
 * Below Tailwind's `sm` breakpoint (640 px) the sidebar loads as the 64 px icon rail, whether or
 * not an "expanded" choice is remembered, and the chevron still expands it. At `sm` and wider
 * the remembered choice decides, as before. Loading at a phone width leaves the stored choice
 * as it was. The phone overflow audit (`phone-overflow.spec.ts`) loads every screen at 390 px
 * with no stored choice and asserts the rail there too.
 */
import { expect, test, type Page } from '@playwright/test';

/** `Sidebar.tsx`'s SIDEBAR_COLLAPSED_KEY (the app module cannot load here). */
const SIDEBAR_KEY = 'miforge.sidebar.collapsed';
const PHONE = { width: 390, height: 844 };
const DESKTOP = { width: 1280, height: 800 };

/** Opens Runs at `size` with the stored choice `stored` (`null`: nothing stored). */
async function open(page: Page, size: { width: number; height: number }, stored: string | null) {
  await page.setViewportSize(size);
  await page.addInitScript(
    ([key, value]) => {
      // Once per page load, so a reload in the test keeps what the app wrote meanwhile.
      if (window.sessionStorage.getItem('sidebar-seeded') !== null) return;
      window.sessionStorage.setItem('sidebar-seeded', '1');
      if (value === null) window.localStorage.removeItem(key as string);
      else window.localStorage.setItem(key as string, value);
    },
    [SIDEBAR_KEY, stored] as const,
  );
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.goto('/#/runs');
  await expect(page.getByTestId('shell-root')).toBeVisible();
  return page.getByTestId('shell-sidebar');
}

const storedChoice = (page: Page) =>
  page.evaluate((key) => window.localStorage.getItem(key), SIDEBAR_KEY);

test('a phone with no stored choice starts collapsed, and the arrow still expands it', async ({
  page,
}) => {
  const sidebar = await open(page, PHONE, null);
  await expect(sidebar).toHaveAttribute('data-collapsed', 'true');
  expect((await sidebar.boundingBox())?.width).toBe(64);
  await page.getByRole('button', { name: 'Expand the sidebar' }).click();
  await expect(sidebar).toHaveAttribute('data-collapsed', 'false');
  await expect.poll(async () => (await sidebar.boundingBox())?.width).toBe(224);
});

test('a phone with a stored "expanded" still starts collapsed and keeps the stored choice', async ({
  page,
}) => {
  const sidebar = await open(page, PHONE, 'false');
  await expect(sidebar).toHaveAttribute('data-collapsed', 'true');
  expect(await storedChoice(page)).toBe('false');
  await page.getByRole('button', { name: 'Expand the sidebar' }).click();
  await expect(sidebar).toHaveAttribute('data-collapsed', 'false');
});

test('one pixel under sm is a phone; sm itself keeps the stored choice', async ({ page }) => {
  const sidebar = await open(page, { width: 639, height: 844 }, 'false');
  await expect(sidebar).toHaveAttribute('data-collapsed', 'true');
  await page.setViewportSize({ width: 640, height: 844 });
  await page.reload();
  await expect(page.getByTestId('shell-root')).toBeVisible();
  await expect(sidebar).toHaveAttribute('data-collapsed', 'false');
});

test('a desktop keeps a stored "expanded"', async ({ page }) => {
  const sidebar = await open(page, DESKTOP, 'false');
  await expect(sidebar).toHaveAttribute('data-collapsed', 'false');
  expect((await sidebar.boundingBox())?.width).toBe(224);
});

test('a desktop keeps a stored "collapsed", and starts expanded with none', async ({ page }) => {
  const sidebar = await open(page, DESKTOP, 'true');
  await expect(sidebar).toHaveAttribute('data-collapsed', 'true');
  await page.evaluate((key) => window.localStorage.removeItem(key), SIDEBAR_KEY);
  await page.reload();
  await expect(page.getByTestId('shell-root')).toBeVisible();
  await expect(sidebar).toHaveAttribute('data-collapsed', 'false');
});
