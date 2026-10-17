/**
 * The checkpoint tree, reached through the real nav (FR-009.21-.31; AC-009.17's screen half).
 *
 * **M2's end-to-end half.** Deleting the `checkpoints` row from `config/routes.ts` removes the
 * nav item and this spec fails on the missing button.
 *
 * With no backend the forest cannot be read, which is the case this spec is most useful for:
 * the screen must **name the seam and the failure** rather than rendering an empty tree that
 * reads as "there are no checkpoints".
 */
import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Checkpoints' })
    .click();
});

test('opens with the registrys title and subtitle', async ({ page }) => {
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Checkpoints');
  await expect(page.getByTestId('shell-page-head').locator('p')).toContainText(
    'Every saved step is a loadable model',
  );
});

test('states that the forest is composed in the browser, naming the absent endpoint', async ({
  page,
}) => {
  // The seam, on the screen rather than only in a comment: GET /api/v1/tree is not mounted
  // and `services.reindex.checkpoint_tree` has no route calling it.
  await expect(page.getByTestId('shell-content')).toContainText('/api/v1/tree');
  await expect(page.getByTestId('shell-content')).toContainText('not mounted');
});

test('names the failure rather than rendering an empty tree', async ({ page }) => {
  // An empty tree reads as "there are no checkpoints", which is a different and worse claim
  // than "the listing could not be read".
  await expect(page.getByTestId('shell-content')).toContainText(
    /unavailable|refused|unreadable|No response/i,
  );
});

test('offers no Export, no Load in miLLM and no Delete', async ({ page }) => {
  // FPRD section 3.9 and F-12: nothing defines export, no requirement obliges miForge to push
  // a checkpoint into a serving process, and no checkpoint lifecycle exists anywhere.
  const text = await page.getByTestId('shell-root').innerText();
  expect(text).not.toMatch(/export checkpoint/i);
  expect(text).not.toMatch(/load in millm/i);
  expect(text).not.toMatch(/\bdelete\b/i);
});
