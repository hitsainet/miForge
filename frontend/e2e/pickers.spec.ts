/**
 * Choose from lists, start a run, find it in the Runs list, open it, reload (task 14.8;
 * AC-009.49).
 *
 * The operator's request, as a test: *"Selection boxes should be dropdowns that list what is
 * available."* Every choice below is made by opening a picker and clicking a row - nothing
 * that names an existing thing is typed.
 *
 * ## The backend is a fake, and its bodies are not hand-written
 *
 * `vite preview` serves no backend and CI's frontend job runs none (ADR-016), so the API is
 * served at the network by `page.route`. **Every GET body comes from `fixtures/backend.json`,
 * which `backend/tests/integration/test_ui_picker_fixtures.py` generates by seeding a real
 * database, parts, a pack and a model directory and calling the mounted app's routes - and
 * then pins.** So this spec reads what the routes really answer, and a route that changes
 * shape turns that backend test red. The fake adds only state: a run defined here is
 * prepended to the run list, newest first, and served by id.
 *
 * The body the form posts is compared with `fixtures/define-body.json`, which the same backend
 * file validates through the real `RunDefinition` model - so "the form sent this" and "the
 * backend accepts this" are one assertion each, about one document.
 *
 * `MIFORGE_E2E_LIVE=1` turns the fake off, so the same spec runs through `vite preview`'s
 * proxy against a real, seeded backend on :8000 (recorded in the 009 FTASKS, not run in CI).
 */
import { expect, test, type Page, type Route } from '@playwright/test';
import backend from './fixtures/backend.json' with { type: 'json' };
import defineBody from './fixtures/define-body.json' with { type: 'json' };
import defineCreateBody from './fixtures/define-body-create.json' with { type: 'json' };
import estimated from '../src/components/panels/newrun/fixtures/budget-estimate.json' with { type: 'json' };

const LIVE = process.env['MIFORGE_E2E_LIVE'] === '1';
const FIXTURE = backend as Record<string, unknown>;
const SEEDED = '/api/v1/runs/run-seeded-1';

interface RunRow {
  readonly run_id: string;
  readonly status: string;
  readonly run_type: string;
  readonly [key: string]: unknown;
}

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

/** The fake: the generated bodies, plus the runs this spec defines. Returns what it saw. */
async function serve(page: Page): Promise<{
  posted: unknown[];
  listQueries: string[];
  estimates: unknown[];
  validated: unknown[];
}> {
  const posted: unknown[] = [];
  const validated: unknown[] = [];
  const estimates: unknown[] = [];
  const listQueries: string[] = [];
  const defined: RunRow[] = [];
  const seededPage = FIXTURE['/api/v1/runs?limit=25&offset=0'] as {
    runs: RunRow[];
    [key: string]: unknown;
  };

  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    const key = `${url.pathname}${url.search}`;

    if (method === 'POST' && url.pathname === '/api/v1/runs/define') {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      posted.push(body);
      const runId =
        typeof body['run_id'] === 'string' ? body['run_id'] : `run-e2e-${posted.length}`;
      defined.unshift({
        ...(seededPage.runs[0] as RunRow),
        run_id: runId,
        status: 'pending',
        run_type: String(body['run_type']),
        created_at: new Date().toISOString(),
        finished_at: null,
        last_step: null,
        stop_reason: null,
        failure_code: null,
      });
      // `run_launch.launch`'s answer carries more; the form reads `run_id`.
      return json(route, { run_id: runId, status: 'pending' }, 202);
    }

    if (method === 'POST' && url.pathname === '/api/v1/runs/validate') {
      // The check's own answer is not generated: it carries the free disk and a generated
      // seed. Its tokenizer block echoes what was posted, in the shape
      // backend/tests/integration/test_tokenizer_picker.py pins on the real route.
      const body = route.request().postDataJSON() as Record<string, unknown>;
      validated.push(body);
      const source = (body['model_source'] ?? {}) as Record<string, unknown>;
      const tokenizer =
        typeof source['tokenizer_path'] === 'string'
          ? { source: 'directory', path: source['tokenizer_path'], vocab_size: 512 }
          : typeof source['tokenizer_repo'] === 'string'
            ? {
                source: 'hub',
                repo: source['tokenizer_repo'],
                revision: source['tokenizer_revision'] ?? 'main',
                commit: source['tokenizer_revision'],
                vocab_size: 512,
              }
            : null;
      return json(route, {
        ok: true,
        may_be_more_to_fix: false,
        steps_checked: ['structural', 'budget'],
        failures: [],
        skipped: [],
        resolved: { budget: { estimate_bytes: 1_000_000, free_bytes: null, tokenizer } },
      });
    }

    if (method === 'POST' && url.pathname === '/api/v1/budget/estimate') {
      // The body is generated from the real route (a model directory, two cards) and its shape
      // pinned by backend/tests/unit/test_budget_estimate_from_model_source.py.
      estimates.push(route.request().postDataJSON());
      return json(route, estimated);
    }

    if (method === 'GET' && url.pathname === '/api/v1/runs') {
      listQueries.push(url.search);
      const status = url.searchParams.get('status');
      const runType = url.searchParams.get('run_type');
      const limit = Number(url.searchParams.get('limit') ?? '50');
      const offset = Number(url.searchParams.get('offset') ?? '0');
      const all = [...defined, ...seededPage.runs].filter(
        (row) =>
          (status === null || row.status === status) &&
          (runType === null || row.run_type === runType),
      );
      return json(route, {
        ...seededPage,
        runs: all.slice(offset, offset + limit),
        total: all.length,
        limit,
        offset,
      });
    }

    const run = /^\/api\/v1\/runs\/(run-e2e-\d+)(\/.*)?$/.exec(url.pathname);
    if (method === 'GET' && run !== null) {
      const [, runId, rest = ''] = run;
      const row = defined.find((entry) => entry.run_id === runId);
      const template = FIXTURE[`${SEEDED}${rest}`] as Record<string, unknown> | undefined;
      if (row === undefined || template === undefined)
        return json(route, { error: { code: 'run_not_found', message: runId, details: {} } }, 404);
      return json(
        route,
        rest === ''
          ? { ...template, run_id: runId, status: 'pending', last_step: null, stop_reason: null }
          : { ...template, run_id: runId },
      );
    }

    if (method === 'GET' && key in FIXTURE) return json(route, FIXTURE[key]);
    return json(route, { error: { code: 'not_stubbed', message: key, details: {} } }, 404);
  });
  return { posted, listQueries, estimates, validated };
}

const nav = (page: Page, name: string) =>
  page.getByRole('navigation', { name: 'Screens' }).getByRole('button', { name }).click();

/** Open a picker and click the row that submits `value` - as an operator does. */
async function choose(page: Page, testId: string, value: string): Promise<void> {
  await page.getByTestId(testId).click();
  await page.getByTestId(`${testId}-listbox`).locator(`[data-value="${value}"]`).click();
  await expect(page.getByTestId(testId)).toHaveAttribute('data-value', value);
}

/**
 * Choose by the row's LABEL and return the value it submitted. For the model directory: a live
 * backend's root is wherever it was started, so only the path under the root is fixed.
 */
async function chooseLabel(page: Page, testId: string, label: string): Promise<string> {
  await page.getByTestId(testId).click();
  const row = page
    .getByTestId(`${testId}-listbox`)
    .getByRole('option')
    .filter({ has: page.getByText(label, { exact: true }) });
  const value = (await row.getAttribute('data-value')) ?? '';
  await row.click();
  await expect(page.getByTestId(testId)).toHaveAttribute('data-value', value);
  return value;
}

test('choose a model, a pack and a stop condition from lists, start, find the run, open it, reload', async ({
  page,
}) => {
  const fake = LIVE ? null : await serve(page);
  await page.goto('/');
  await nav(page, 'New run');

  // The lists are the routes' answers: the seeded model directory, pack and conditions.
  await choose(page, 'mode-select', 'load');
  const path = await chooseLabel(page, 'path-input', 'bases/tiny');
  expect(path).toMatch(/\/checkpoints\/bases\/tiny$/);
  await choose(page, 'run-type-select', 'change');
  await choose(page, 'pack-select', 'notary-md@1');
  // The label is shown; the identifier is what is held.
  await expect(page.getByTestId('pack-select')).toHaveValue('notary-md @ 1');
  await choose(page, 'signal-source-select', 'corpus');
  // Numbers are the operator's to choose, so they are typed - they name nothing that exists.
  await page.getByTestId('optimizer-learningRate').fill('0.00001');
  await page.getByTestId('optimizer-warmupSteps').fill('0');
  await page.getByTestId('optimizer-batchSize').fill('1');
  await page.getByRole('button', { name: 'Add a stop condition' }).click();
  await choose(page, 'stop-condition-0', 'step_count');
  await page.getByTestId('stop-param-0-total').fill('40');
  await choose(page, 'gpu-request', 'auto');
  // A NEW run's name is typed: it names nothing that exists yet.
  await page.getByTestId('run-id-field').fill('run-e2e-1');

  await page.getByTestId('start-run').click();

  // Landed on the new run's view.
  await expect(page).toHaveURL(/#\/runs\/run-e2e-1$/);
  if (fake !== null) expect(fake.posted).toEqual([defineBody]);

  // The run is in the Runs list, newest first, above the seeded one.
  await nav(page, 'Runs');
  const rows = page.getByTestId('run-list-rows').locator('[data-testid^="run-list-row-"]');
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(0)).toHaveAttribute('data-testid', 'run-list-row-run-e2e-1');
  await expect(rows.nth(1)).toHaveAttribute('data-testid', 'run-list-row-run-seeded-1');
  await expect(rows.nth(1)).toContainText('stopped: step_count');

  // A filter is chosen from the list's own vocabulary and sent as the value.
  await choose(page, 'run-filter-status', 'pending');
  await expect(rows).toHaveCount(1);
  if (fake !== null) expect(fake.listQueries.at(-1)).toContain('status=pending');

  // Reload: the screen and the filter are kept, and the list is re-read with it.
  await page.reload();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Runs');
  await expect(page.getByTestId('run-filter-status')).toHaveAttribute('data-value', 'pending');
  await expect(rows).toHaveCount(1);

  // Open it from the list.
  await rows.nth(0).click();
  await expect(page).toHaveURL(/#\/runs\/run-e2e-1$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('run-e2e-1');

  // Reload the run view: still this run.
  await page.reload();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('run-e2e-1');

  // The submitted draft is gone: the form is empty again, its pickers unchosen.
  await nav(page, 'New run');
  await expect(page.getByTestId('pack-select')).toHaveAttribute('data-value', '');
  await expect(page.getByTestId('submitted-body')).toContainText('{}');
});

test('a draft made of choices comes back after a reload, each picker holding its value', async ({
  page,
}) => {
  if (!LIVE) await serve(page);
  await page.goto('/');
  await nav(page, 'New run');
  await choose(page, 'pack-select', 'notary-md@1');
  const path = await chooseLabel(page, 'path-input', 'bases/tiny');

  await page.reload();

  await expect(page.getByTestId('draft-restored')).toBeVisible();
  await expect(page.getByTestId('pack-select')).toHaveAttribute('data-value', 'notary-md@1');
  await expect(page.getByTestId('pack-select')).toHaveValue('notary-md @ 1');
  await expect(page.getByTestId('path-input')).toHaveAttribute('data-value', path);
});

test('a picker is operable by keyboard alone', async ({ page }) => {
  if (!LIVE) await serve(page);
  await page.goto('/');
  await nav(page, 'New run');
  const picker = page.getByTestId('run-type-select');
  // Disabled while its list loads; a keyboard user tabbing in early meets that too.
  await expect(picker).toBeEnabled();
  await picker.focus();
  await page.keyboard.press('ArrowDown');
  await expect(picker).toHaveAttribute('aria-expanded', 'true');
  await page.keyboard.type('change');
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await expect(picker).toHaveAttribute('data-value', 'change');
  await expect(page.getByTestId('submitted-body')).toContainText('"run_type": "change"');
});

test('the Runs page lists every run in a dropdown, and choosing one opens it', async ({ page }) => {
  // Operator, 2026-10-18: "I want the available runs to be listed in a DROPDOWN. I don't want
  // to have to know the id from memory."
  if (!LIVE) await serve(page);
  await page.goto('/');
  await nav(page, 'Runs');
  const picker = page.getByTestId('run-id-input');
  await expect(picker).toBeEnabled();
  await picker.click();
  const row = page.getByTestId('run-id-input-listbox').locator('[data-value="run-seeded-1"]');
  // The label carries what tells runs apart: the id, its status, its type, when it started.
  await expect(row).toContainText('run-seeded-1 — completed · change · created');
  // Type to filter, then choose.
  await picker.fill('seeded');
  await row.click();
  await expect(page).toHaveURL(/#\/runs\/run-seeded-1$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('run-seeded-1');
});

test('estimate GPU memory for the chosen model: the model source is sent and every card is judged', async ({
  page,
}) => {
  // Found live 2026-10-20: the form sent no model source and no size, so every estimate was
  // refused budget_spec_incomplete. The server now reads the size from the model source.
  const fake = LIVE ? null : await serve(page);
  await page.goto('/');
  await nav(page, 'New run');

  await choose(page, 'mode-select', 'load');
  const path = await chooseLabel(page, 'path-input', 'bases/tiny');
  await choose(page, 'run-type-select', 'change');
  await page.getByTestId('optimizer-learningRate').fill('0.0001');
  await page.getByTestId('optimizer-batchSize').fill('2');
  await page.getByRole('button', { name: 'Estimate GPU memory' }).click();

  await expect(page.getByTestId('budget-source')).toContainText(
    'Size read from the model directory',
  );
  await expect(page.getByTestId('budget-estimate')).toContainText('5 terms');
  await expect(page.getByTestId('budget-preview')).not.toContainText('budget_spec_incomplete');
  if (fake !== null) {
    expect(fake.estimates).toEqual([
      {
        run_type: 'change',
        config: { learning_rate: 0.0001, warmup_steps: 0, batch_size: 2 },
        model_source: { mode: 'load', path },
      },
    ]);
    // Every card the fake's route listed, each with its own verdict.
    await expect(page.getByTestId('budget-cards').locator('li')).toHaveCount(
      estimated.cards.length,
    );
  }
});

test('a create run takes its tokenizer from a list: the check, the plan and the start all carry it', async ({
  page,
}) => {
  // Operator, 2026-10-20: a create run started from this form was refused tokenizer_required
  // every time, because the form had nowhere to name a tokenizer.
  const fake = LIVE ? null : await serve(page);
  await page.goto('/');
  await nav(page, 'New run');

  // Not asked for until the mode is create.
  await expect(page.getByTestId('tokenizer-path-input')).toHaveCount(0);
  await choose(page, 'mode-select', 'create');
  const config = await chooseLabel(page, 'config-path-input', 'bases/tiny/config.json');
  expect(config).toMatch(/\/checkpoints\/bases\/tiny\/config\.json$/);
  // The tokenizer list is the route's answer: a tokenizer-only folder and the model folder.
  const tokenizer = await chooseLabel(page, 'tokenizer-path-input', 'tokenizers/tiny-tok');
  expect(tokenizer).toMatch(/\/checkpoints\/tokenizers\/tiny-tok$/);
  await choose(page, 'run-type-select', 'create');
  await choose(page, 'pack-select', 'notary-md@1');
  await choose(page, 'signal-source-select', 'corpus');
  await page.getByTestId('optimizer-learningRate').fill('0.0001');
  await page.getByTestId('optimizer-warmupSteps').fill('0');
  await page.getByTestId('optimizer-batchSize').fill('1');
  await page.getByRole('button', { name: 'Add a stop condition' }).click();
  await choose(page, 'stop-condition-0', 'step_count');
  await page.getByTestId('stop-param-0-total').fill('40');
  await choose(page, 'gpu-request', 'auto');
  await page.getByTestId('run-id-field').fill('run-e2e-create');

  await page.getByTestId('check-definition').click();
  await expect(page.getByTestId('resolved-tokenizer')).toContainText(tokenizer);

  await page.getByTestId('start-run').click();
  await expect(page).toHaveURL(/#\/runs\/run-e2e-create$/);
  if (fake !== null) {
    expect(fake.validated).toEqual([defineCreateBody]);
    expect(fake.posted).toEqual([defineCreateBody]);
  }
});

test('a create run can take its tokenizer from the Hub: Check resolves the commit and the run is pinned to it', async ({
  page,
}) => {
  test.skip(LIVE, 'the live backend would reach the real Hub');
  const fake = await serve(page);
  const checked = FIXTURE['/api/v1/hub/revision?repo=org%2Ftiny-tok&revision=main'] as {
    commit: string;
  };
  await page.goto('/');
  await nav(page, 'New run');

  await choose(page, 'mode-select', 'create');
  await chooseLabel(page, 'config-path-input', 'bases/tiny/config.json');
  await page.getByTestId('tokenizer-repo-input').fill('org/tiny-tok');
  await page.getByTestId('tokenizer-revision-input').fill('main');
  await page.getByTestId('tokenizer-hub-check-button').click();
  const answer = page.getByTestId('tokenizer-hub-check-answer');
  await expect(answer).toContainText(`points at commit ${checked.commit} now`);
  await expect(answer).toContainText(
    'Tokenizer files at its root: tokenizer.json, tokenizer_config.json.',
  );
  await page.getByTestId('tokenizer-hub-check-pin').click();
  await expect(page.getByTestId('tokenizer-revision-input')).toHaveValue(checked.commit);
  await choose(page, 'run-type-select', 'create');
  await page.getByTestId('run-id-field').fill('run-e2e-hub');

  await page.getByTestId('start-run').click();
  await expect(page).toHaveURL(/#\/runs\/run-e2e-hub$/);
  expect(fake.posted).toEqual([
    {
      run_id: 'run-e2e-hub',
      model_source: {
        mode: 'create',
        config_path: '/data/checkpoints/bases/tiny/config.json',
        tokenizer_repo: 'org/tiny-tok',
        tokenizer_revision: checked.commit,
      },
      run_type: 'create',
    },
  ]);
});
