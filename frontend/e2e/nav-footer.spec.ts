/**
 * The sidebar footer every suite app shares (operator, 2026-10-20: *"Settings. Manual link, and
 * the version"*), measured in a real browser in both theme projects.
 *
 * Laid out as miStudio's and miLLM's: a top rule; one row with Settings and, to its right, the
 * Manual as a book icon; the version under it in small, dim monospace; the whole footer at the
 * bottom edge of the sidebar.
 */
import { expect, test, type Locator } from '@playwright/test';

async function box(locator: Locator) {
  const found = await locator.boundingBox();
  if (found === null) throw new Error('element has no box');
  return found;
}

test.beforeEach(async ({ page }) => {
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.goto('/#/runs');
  await expect(page.getByTestId('shell-root')).toBeVisible();
});

test('Settings, then the Manual book icon on the same row, then the version below', async ({
  page,
}) => {
  const sidebar = page.getByTestId('shell-sidebar');
  const footer = page.getByTestId('sidebar-footer');
  const settings = footer.getByRole('button', { name: 'Settings' });
  const manual = footer.getByRole('link', { name: 'Manual (opens in a new tab)' });
  const version = footer.getByTestId('sidebar-version');

  // The footer is the sidebar's last region, flush with its bottom edge.
  const aside = await box(sidebar);
  const foot = await box(footer);
  expect(foot.y + foot.height).toBeCloseTo(aside.y + aside.height, 0);
  expect(
    await footer.evaluate((node) => getComputedStyle(node).borderTopWidth),
    'the footer has a top rule',
  ).toBe('1px');

  // One row: vertically centred together, the book to the right of Settings.
  const row = await box(settings);
  const book = await box(manual);
  expect(Math.abs(row.y + row.height / 2 - (book.y + book.height / 2))).toBeLessThanOrEqual(2);
  expect(book.x).toBeGreaterThanOrEqual(row.x + row.width);
  expect(book.x + book.width).toBeLessThanOrEqual(aside.x + aside.width);
  await expect(manual.locator('svg')).toBeVisible();
  await expect(manual).toHaveAttribute('title', 'User Manual');
  await expect(manual).toHaveAttribute('target', '_blank');

  // The version, under the row, in small dim mono.
  const ver = await box(version);
  expect(ver.y).toBeGreaterThanOrEqual(row.y + row.height);
  await expect(version).toHaveText(/^v\d+\.\d+\.\d+$/);
  const style = await version.evaluate((node) => {
    const computed = getComputedStyle(node);
    const probe = document.createElement('span');
    probe.style.color = 'var(--mf-dim)';
    document.body.appendChild(probe);
    const dim = getComputedStyle(probe).color;
    probe.remove();
    return { family: computed.fontFamily, size: computed.fontSize, color: computed.color, dim };
  });
  expect(style.family).toMatch(/mono/i);
  expect(style.size).toBe('12px');
  expect(style.color).toBe(style.dim);
});

test('the footer Settings entry opens the Settings screen and shows as current', async ({
  page,
}) => {
  const settings = page.getByTestId('sidebar-footer').getByRole('button', { name: 'Settings' });
  await settings.click();
  await expect(page.getByTestId('shell-page-head').getByRole('heading', { level: 1 })).toHaveText(
    'Settings',
  );
  await expect(settings).toHaveAttribute('aria-current', 'page');
  await expect(page).toHaveURL(/#\/settings$/);
  // No backend here: the screen names its unavailable region rather than rendering blank.
  await expect(page.getByTestId('shell-content')).toContainText(/settings/i);
});

test('the theme toggle moved to the resource strip, out of the footer', async ({ page }) => {
  const toggle = page.getByTestId('theme-toggle');
  await expect(page.getByTestId('shell-resource-strip').getByTestId('theme-toggle')).toBeVisible();
  await expect(page.getByTestId('sidebar-footer').getByTestId('theme-toggle')).toHaveCount(0);
  await expect(toggle).toHaveText(/Switch to (light|dark)/);
});
