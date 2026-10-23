/**
 * Help in a real browser (ENH-004, 2026-10-20): reached through the nav, linked from a screen's
 * page head, a topic in the URL that survives a reload, and manual links that open a new tab.
 * No backend: the Help screen needs none, so it renders in full.
 */
import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/socket.io/**', (route) => route.abort());
});

test('the last entry of the main nav opens Help', async ({ page }) => {
  await page.goto('/#/runs');
  const nav = page.getByRole('navigation', { name: 'Screens' });
  await expect(nav.getByRole('button').last()).toHaveText('Help');
  await nav.getByRole('button', { name: 'Help' }).click();
  await expect(page.getByTestId('shell-page-head').getByRole('heading', { level: 1 })).toHaveText(
    'Help',
  );
  await expect(page).toHaveURL(/#\/help$/);
});

test("a screen's page head opens its own topic, and a reload keeps it", async ({ page }) => {
  await page.goto('/#/checkpoints');
  const head = page.getByTestId('shell-page-head');
  const manual = head.getByTestId('manual-link-checkpoints');
  await expect(manual).toHaveAttribute(
    'href',
    'https://hitsainet.github.io/miForge/guides/checkpoints#the-checkpoints-screen',
  );
  await expect(manual).toHaveAttribute('target', '_blank');
  await head.getByTestId('help-open-checkpoints').click();
  await expect(page).toHaveURL(/#\/help\/checkpoints$/);
  const topic = page.getByTestId('help-topic-checkpoints');
  await expect(topic).toHaveAttribute('data-current', 'true');
  await expect(topic).toBeInViewport();

  await page.reload();
  await expect(page.getByTestId('help-topic-checkpoints')).toHaveAttribute('data-current', 'true');
  // Back returns to the screen the link was followed from.
  await page.goBack();
  await expect(page).toHaveURL(/#\/checkpoints$/);
});

test('every topic card links its manual section once', async ({ page }) => {
  await page.goto('/#/help/glossary');
  const cards = page.locator('[data-testid^="help-topic-"]');
  const count = await cards.count();
  expect(count).toBeGreaterThanOrEqual(12);
  for (let index = 0; index < count; index += 1) {
    const links = cards.nth(index).getByRole('link');
    await expect(links).toHaveCount(1);
    await expect(links.first()).toHaveAttribute(
      'href',
      /^https:\/\/hitsainet\.github\.io\/miForge\//,
    );
  }
  await expect(page.getByTestId('help-topic-glossary')).toContainText('Words used here');
});
