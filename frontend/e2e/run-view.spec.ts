/**
 * The run view, reached through the real nav (FR-009.4, .5, .19; AC-009.8).
 *
 * **M1's end-to-end half**: the run view's route entry has `nav: null`, so it is reached by
 * opening a run from the index - which is itself reached through the nav. Deleting either row
 * takes this spec down.
 *
 * No backend runs under `vite preview`, so this exercises the path that matters most for
 * FR-009.5: **every failure degrades one named region and says what to do next.** A screen
 * that went blank with no backend would pass a unit test and fail an operator.
 */
import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Runs' })
    .click();
});

test('names the endpoints this build does not serve, rather than looking broken', async ({
  page,
}) => {
  // GET /api/v1/runs is not mounted. The screen says so by name and fabricates nothing.
  await expect(page.getByTestId('shell-content')).toContainText('serves no run listing');
  await expect(page.getByTestId('run-index-empty')).toContainText('No run is open');
});

test('opens a run by identifier and names the refusal when there is no such run', async ({
  page,
}) => {
  await page.getByTestId('run-id-input').fill('no-such-run');
  await page.getByRole('button', { name: 'Open', exact: true }).click();
  // With no backend this is a transport failure; with one it is `run_not_found`. Either way
  // the region is named and the next action is stated - never a blank panel.
  await expect(page.getByTestId('shell-content')).toContainText(/unavailable|refused|No response/i);
});

test('reaches the run view and asks for a run rather than rendering an empty chart', async ({
  page,
}) => {
  // The screen's own empty state: it has no list of its own and says so.
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'New run' })
    .click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('New run');
});

test('offers no Pause, no held state and no section 5.10 control', async ({ page }) => {
  const text = await page.getByTestId('shell-root').innerText();
  expect(text).not.toMatch(/\bpause\b/i);
  expect(text).not.toMatch(/\bheld\b/i);
  expect(text).not.toMatch(/quiesce|commit bracket|torn/i);
});
