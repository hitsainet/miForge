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
import forestFixture from './fixtures/tree.json' with { type: 'json' };

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

test('reads the forest from GET /api/v1/tree and places an adapter under its base', async ({
  page,
}) => {
  // The route's own answer (fixtures/tree.json, pinned by test_ui_picker_fixtures.py). Until
  // task 14.0 this screen joined two listings in the browser and said the route was not
  // mounted; it was, and nothing on the screen may say so now.
  await page.route('**/api/v1/tree**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(forestFixture),
    }),
  );
  await page.reload();
  await expect(page.getByTestId('tree-count')).toContainText('3 artifacts');
  const adapter = forestFixture.nodes.find((node) => node.kind === 'adapter');
  await expect(page.getByTestId(`tree-row-${adapter?.name}`)).toBeVisible();
  await expect(page.getByTestId('shell-content')).not.toContainText('not mounted');
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
  //
  // Over a POPULATED tree. Until 2026-11-11 this read the screen with no backend, so no row
  // was ever rendered and the check could not see a row action at all: a "Load in miServer"
  // button added to `TreeRow.tsx` left it green (015 control N18, first run). The rows are
  // where such an action would live, so the forest is served and its rows awaited first.
  await page.route('**/api/v1/tree**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(forestFixture),
    }),
  );
  await page.reload();
  await expect(page.getByTestId('tree-count')).toContainText('3 artifacts');
  for (const node of forestFixture.nodes) {
    await expect(page.getByTestId(`tree-row-${node.name}`)).toBeVisible();
  }
  const text = await page.getByTestId('shell-root').innerText();
  expect(text).not.toMatch(/export checkpoint/i);
  // 015 FR-015.21: the server is miServer now; an absence check on the old spelling alone
  // would pass against a "Load in miServer" button, so both spellings are refused.
  expect(text).not.toMatch(/load in (millm|miserver)/i);
  expect(text).not.toMatch(/\bdelete\b/i);
});
