/**
 * A reload keeps the screen and the entries (F9 task 13.7; AC-009.31-.35, .39).
 *
 * The operator's report, as a test: *"regardless of which tab/page I'm on, when I refresh the
 * browser it clears my entries and resets back to the Runs tab."* Each case reaches its screen
 * **through the real nav**, enters something, and calls `page.reload()` - the real browser
 * reload, with the real `localStorage` and the real URL, against the built bundle.
 *
 * The API is stubbed at the network where a case needs an answer (`vite preview` serves no
 * backend), exactly as `onboard.spec.ts` does; elsewhere the screens render their refusals.
 */
import { expect, test, type Page, type Route } from '@playwright/test';
import { hashIs } from './hash';

const nav = (page: Page, name: string) =>
  page.getByRole('navigation', { name: 'Screens' }).getByRole('button', { name }).click();

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

test('the New run form keeps its screen and its entries across a reload', async ({ page }) => {
  await page.goto('/');
  await nav(page, 'New run');
  await page.getByTestId('optimizer-batchSize').fill('8');
  await page.getByTestId('run-id-field').fill('md-notary-7');
  await expect(page).toHaveURL(/#\/runs\/new$/);

  await page.reload();

  await expect(page.getByRole('heading', { level: 1 })).toHaveText('New run');
  await expect(page.getByTestId('optimizer-batchSize')).toHaveValue('8');
  await expect(page.getByTestId('run-id-field')).toHaveValue('md-notary-7');
  // Announced, so a restored value is never read as a default.
  await expect(page.getByTestId('draft-restored')).toBeVisible();
  await expect(page.getByTestId('submitted-body')).toContainText('"batch_size": 8');
});

test('a submitted run definition is not restored on the next reload', async ({ page }) => {
  const posted: unknown[] = [];
  await page.route('**/api/v1/runs/define', async (route) => {
    posted.push(route.request().postDataJSON());
    await json(route, { run_id: 'run-e2e' }, 202);
  });
  await page.goto('/');
  await nav(page, 'New run');
  await page.getByTestId('optimizer-batchSize').fill('8');
  await page.getByTestId('start-run').click();

  // Landed on the new run, and the URL says so.
  await expect(page).toHaveURL(/#\/runs\/run-e2e$/);
  expect(posted).toEqual([{ run_configuration: { batch_size: 8 } }]);

  await nav(page, 'New run');
  await expect(page.getByTestId('optimizer-batchSize')).toHaveValue('');
  await page.reload();
  await expect(page.getByTestId('optimizer-batchSize')).toHaveValue('');
  await expect(page.getByTestId('draft-restored')).toHaveCount(0);
  await expect(page.getByTestId('submitted-body')).toContainText('{}');
});

test('Clear every field survives a reload as empty', async ({ page }) => {
  await page.goto('/');
  await nav(page, 'New run');
  await page.getByTestId('optimizer-batchSize').fill('8');
  await page.getByTestId('clear-draft').click();
  await page.reload();
  await expect(page.getByTestId('optimizer-batchSize')).toHaveValue('');
});

test('every nav screen survives a reload, and back returns to the previous one', async ({
  page,
}) => {
  await page.goto('/');
  for (const [label, title, hash] of [
    ['Checkpoints', 'Checkpoints', '#/checkpoints'],
    ['Domain packs', 'Domain packs', '#/packs'],
    ['Runs', 'Runs', '#/runs'],
  ] as const) {
    await nav(page, label);
    await expect(page).toHaveURL(hashIs(hash));
    await page.reload();
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(title);
  }
  await page.goBack();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Domain packs');
  await page.goForward();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Runs');
});

test('a run view is a link: it opens from the URL and survives a reload', async ({ page }) => {
  await page.goto('/#/runs/md-notary-2026-09-11');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('md-notary-2026-09-11');
  await page.reload();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('md-notary-2026-09-11');
});

test('an unknown URL lands on Runs and is rewritten', async ({ page }) => {
  await page.goto('/#/selectors');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Runs');
  await expect(page).toHaveURL(/#\/runs$/);
});

test('the run box and the runs opened from it survive a reload', async ({ page }) => {
  await page.route('**/api/v1/runs/run-kept', (route) =>
    json(route, {
      run_id: 'run-kept',
      status: 'stopped',
      run_type: 'change',
      last_step: 40,
      cancel_requested_at: null,
      failure_code: null,
      heartbeat: {
        heartbeat_at: '2026-10-17T00:00:00+00:00',
        age_seconds: 12,
        longest_gap_seconds: 180,
        lapse_threshold_seconds: 900,
        lapsed: false,
      },
    }),
  );
  await page.route('**/api/v1/runs/run-kept/scores', (route) => json(route, { scores: [] }));
  await page.goto('/');
  await page.getByTestId('run-id-input').fill('run-kept');
  await page.getByRole('button', { name: 'Open', exact: true }).click();
  await expect(page.getByTestId('run-index-list')).toContainText('run-kept');
  await page.getByTestId('run-id-input').fill('half-typed');

  await page.reload();
  await expect(page.getByTestId('run-index-list')).toContainText('run-kept');
  await expect(page.getByTestId('run-id-input')).toHaveValue('half-typed');
});

test('an onboarding draft resumes on reload: same draft, same step, unsaved answers kept', async ({
  page,
}) => {
  const posts: string[] = [];
  const draft = {
    draft_id: 'draft-e2e-9',
    name: null,
    version: '1',
    domain: null,
    current_step: 1,
    step_count: 7,
    step_states: { '1': 'current' },
    slots: { retention_set: null },
    created_at: '2026-10-17T00:00:00Z',
    updated_at: '2026-10-17T00:00:00Z',
    last_dry_run: null,
  };
  await page.route('**/api/**', (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/v1/pack-drafts' && request.method() === 'POST') {
      posts.push(path);
      return json(route, draft, 201);
    }
    if (path === '/api/v1/pack-drafts/draft-e2e-9') return json(route, draft);
    return json(route, { error: { code: 'not_stubbed', message: path } }, 404);
  });
  await page.goto('/');
  await nav(page, 'Onboard a domain');
  await page.getByTestId('draft-name').fill('notary-md');
  // What was typed is on screen - it was not, before task 13.5.
  await expect(page.getByTestId('draft-name')).toHaveValue('notary-md');
  await expect(page).toHaveURL(/#\/packs\/onboard\?draftId=draft-e2e-9&step=1$/);

  await page.reload();
  await expect(page.getByTestId('onboard-panel')).toContainText('1. Describe the domain');
  await expect(page.getByTestId('draft-name')).toHaveValue('notary-md');
  expect(posts).toHaveLength(1);
});
