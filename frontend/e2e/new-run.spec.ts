/**
 * The New run form, reached through the real nav (FR-009.33-.39; AC-009.25, .26, .29).
 *
 * The two things this spec exists to prove about the product's primary action:
 *
 * - **nothing is defaulted.** The body preview on an untouched form is `{}` - not a batch
 *   size, not a cadence, not the acquisition mode the registry marks `default: true`.
 * - **the copy is "Start run".** *"Attach and start"* is the mockup's pre-reframe label,
 *   describing section 5.10's mode, which this screen does not offer.
 *
 * With no backend the registries cannot be read, so the field panels report their refusals by
 * name - which is also the FR-009.5 case: a form whose selects were silently empty would look
 * like a form with no options rather than one that could not reach its registries.
 */
import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'New run' })
    .click();
});

test('reads Start run, and never Attach and start', async ({ page }) => {
  await expect(page.getByTestId('start-run')).toHaveText('Start run');
  expect(await page.getByTestId('shell-root').innerText()).not.toMatch(/attach and start/i);
});

test('submits an empty body for an untouched form', async ({ page }) => {
  // M19, end to end. The preview shows exactly what will be sent.
  await expect(page.getByTestId('submitted-body')).toContainText('{}');
});

test('carries only what was entered, inside F6s envelope', async ({ page }) => {
  await page.getByTestId('optimizer-batchSize').fill('8');
  await expect(page.getByTestId('submitted-body')).toContainText('"batch_size": 8');
  await expect(page.getByTestId('submitted-body')).toContainText('run_configuration');
  const body = await page.getByTestId('submitted-body').innerText();
  expect(body).not.toContain('learning_rate');
  // The envelope's other parts are absent entirely, not sent as empty objects.
  expect(body).not.toContain('model_source');
  expect(body).not.toContain('cadences');
});

test('explains the form in plain words, with no route or requirement id', async ({ page }) => {
  // The route is pinned by persistence.spec.ts, which answers `**/api/v1/runs/define`.
  await expect(page.getByTestId('shell-content')).toContainText('Nothing is filled in for you');
  const text = await page.getByTestId('shell-content').innerText();
  expect(text).not.toMatch(/POST \/api\/v1|\bR-\d{2}\.\d+|\bFR-\d{3}|\bADR-\d+|§\s?\d/);
});

test('names the failure when the validator cannot be reached, never claiming a pass', async ({
  page,
}) => {
  // No backend runs under `vite preview`. The honest rendering of "I could not ask" is not
  // "the answer is yes" - which is the pass-through failure this screen must never show.
  await page.getByTestId('check-definition').click();
  await expect(page.getByTestId('shell-content')).toContainText(
    /unavailable|refused|unreadable|No response/i,
  );
  await expect(page.getByTestId('resolved-summary')).toHaveCount(0);
});

test('offers no stop condition by default and says a run needs one', async ({ page }) => {
  await expect(page.getByTestId('stop-conditions')).toContainText('No stop condition added yet.');
  await expect(page.getByTestId('stop-conditions')).toContainText('A run needs at least one');
});

test('never disables the submit button on the budget previews verdict', async ({ page }) => {
  // M20, end to end: the preview is not a permission and the gate is the backend's.
  await expect(page.getByTestId('start-run')).toBeEnabled();
  await expect(page.getByTestId('budget-preview')).toContainText(
    'It does not block or allow anything on this form',
  );
});
