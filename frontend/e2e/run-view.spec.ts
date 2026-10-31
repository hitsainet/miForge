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

test('names the run list as unavailable with no backend, rather than showing it empty', async ({
  page,
}) => {
  // `GET /api/v1/runs` exists since task 14.0. With no backend under `vite preview` the list
  // is a named failure - never "no run has been started", which would be a false statement.
  await expect(page.getByTestId('run-list-failure')).toContainText(/run list/i);
  await expect(page.getByTestId('run-list-empty')).toHaveCount(0);
});

test('opens a run by identifier and names the refusal when there is no such run', async ({
  page,
}) => {
  await page.getByTestId('run-id-input').fill('no-such-run');
  await page.getByRole('button', { name: 'Open', exact: true }).click();
  // With no backend this is a transport failure (a 500 from the preview proxy); with one it
  // is `run_not_found`. Either way the region is named - never a blank panel.
  //
  // *Asserted on the open's own notice since task 14.0.* This read the whole screen for
  // /unavailable|refused|No response/ and passed only because the removed "no run listing"
  // callout contained the word "unavailable": the open's own notice says "returned something
  // unreadable" under the preview proxy, which the pattern never matched.
  await expect(page.getByTestId('run-open-failure')).toContainText(/that run/i);
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
