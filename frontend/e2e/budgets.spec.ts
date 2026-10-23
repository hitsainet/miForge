/**
 * Section 8's budgets, **measured with their samples** (FTASKS 6.11, 10.6; ADR-024).
 *
 * *A number without its topology is not a measurement.* So each figure below is recorded with
 * the sample it was taken over, the browser it was taken in, and the fact that it is a
 * **jsdom-free, real-Chromium, built-bundle** measurement against a stubbed API - not a dev
 * server and not a unit test. The API is stubbed at the network rather than mocked in the
 * app, so the parse, the compose and the paint are all in the number.
 */
/* eslint-disable no-console --
 * The console lines ARE the deliverable here. ADR-024 requires a figure to carry its sample
 * and its topology, and a measurement that only ever appears in an assertion is a measurement
 * nobody can read: the assertion says "under the budget", the log says what the number was.
 * Narrowed to this file rather than relaxed in the config, because every other console
 * statement in this codebase would be debugging left behind.
 */
import { expect, test } from '@playwright/test';
import forestFixture from './fixtures/tree.json' with { type: 'json' };

/**
 * ⚠ THE VISIBILITY WAITS BELOW CARRY AN EXPLICIT `timeout`, AND IT MUST STAY WELL ABOVE THE
 * BUDGET THEY PRECEDE - added 2026-10-16 after a CI failure that named the wrong cause.
 *
 * Playwright's default `expect` timeout is **5000 ms**, and two of these budgets are also
 * 5000 ms. So when first paint crept up to the budget on CI's slower runner, the *visibility
 * assertion* timed out first and the report read `expect(getByTestId('curve-decimation'))
 * .toBeVisible()` failed - an element problem - when the real finding was **"first paint
 * exceeded its budget on this hardware."** One colour scheme failed and the other passed on
 * retry, so it also read as flakiness rather than as a measurement.
 *
 * A budget test must fail **as a budget**. With the waits decoupled, the only thing that can
 * fail these tests for being slow is the explicit `expect(elapsed).toBeLessThan(...)`, which
 * prints the measured figure beside its sample and browser. **This does not loosen any
 * budget** - every one is unchanged - it stops a breach being reported as a missing element.
 *
 * The inverse mistake is the expensive one: do NOT raise a budget to make CI green. The
 * `RUN_VIEW_FIRST_PAINT_MS=` / `TREE_FIRST_PAINT_MS=` / `LONG_RUN_FIRST_PAINT_MS=` lines are
 * the record, and a budget fitted to the hardware measures nothing.
 */
//: Comfortably above every budget below, so a breach fails as a budget rather than as a
//: missing element. NOT itself a budget - see the warning above.
const PAINT_WAIT_MS = 60_000;

const STEPS = 1000;
const NODES = 500;
/** AC-009.23's case. 50,000 steps against a 2,000-point budget is a ratio of 25. */
const LONG_RUN_STEPS = 50_000;

function steps(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    step: n - i,
    loss: 2 - (n - i) / (n * 2),
    grad_norm: null,
    learning_rate: null,
    wall_time_ms: 400,
    batch_source_hashes: null,
  }));
}

function scores(n: number) {
  return Array.from({ length: Math.floor(n / 10) }, (_, i) => ({
    step: (i + 1) * 10,
    kind: i % 2 === 0 ? 'test' : 'retention',
    set: 'held-out@3',
    score: 0.5 + i / (n * 2),
    n: 320,
    floor: i % 2 === 0 ? null : 0.72,
    floor_layer: 'pack',
    breached: false,
    provider: 'in-process',
    provider_kind: 'in_process',
    provider_model: 'LFM2-1.2B',
    aggregation: 'mean',
    out_of_cadence: false,
    provider_discontinuity: i === 20,
    message: 'x',
  }));
}

test('records the run-view first paint for a 1,000-step run', async ({ page }) => {
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    const body = url.includes('/steps')
      ? { run_id: 'run-a', steps: steps(STEPS), count: STEPS, limit: STEPS, next_before_step: null }
      : url.includes('/scores')
        ? {
            run_id: 'run-a',
            scores: scores(STEPS),
            test_every: 10,
            retention_every: 20,
            exposure_window_steps: 19,
            provider_discontinuities: [210],
          }
        : url.includes('/batches')
          ? { run_id: 'run-a', batches: [], count: 0, limit: 50, next_before: null }
          : url.includes('/checkpoints')
            ? { run_id: 'run-a', checkpoints: [], latest_step: 900, steps_on_disk: [100, 500, 900] }
            : url.endsWith('/api/v1/runs/run-a')
              ? {
                  run_id: 'run-a',
                  status: 'running',
                  run_type: 'change',
                  last_step: STEPS,
                  cancel_requested_at: null,
                  failure_code: null,
                  heartbeat: {
                    heartbeat_at: new Date().toISOString(),
                    age_seconds: 5,
                    longest_gap_seconds: 120,
                    lapse_threshold_seconds: 900,
                    lapsed: false,
                  },
                }
              : {};
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  });

  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Runs' })
    .click();
  await page.getByTestId('run-id-input').fill('run-a');
  const started = Date.now();
  await page.getByRole('button', { name: 'Open', exact: true }).click();
  await page.getByRole('button', { name: 'Open this run' }).click();
  await expect(page.getByTestId('run-curves')).toBeVisible({ timeout: PAINT_WAIT_MS });
  await expect(page.getByTestId('curve-decimation')).toBeVisible({ timeout: PAINT_WAIT_MS });
  const elapsed = Date.now() - started;
  console.log(
    `RUN_VIEW_FIRST_PAINT_MS=${elapsed} sample=${STEPS}_steps/${Math.floor(STEPS / 10)}_scores browser=chromium bundle=production`,
  );
  expect(elapsed).toBeLessThan(5000);
});

test('records the tree first paint for a 500-node forest', async ({ page }) => {
  // `GET /api/v1/tree`'s own node shape, cloned from `fixtures/tree.json` - which
  // `test_ui_picker_fixtures.py` pins to what the route answers for a forest written by the
  // real writers. A chain of NODES checkpoints plus the fixture's adapter on the first.
  const [rootNode, adapterNode, childNode] = [
    forestFixture.nodes.find((node) => node.kind === 'checkpoint' && node.parent?.state === 'root'),
    forestFixture.nodes.find((node) => node.kind === 'adapter'),
    forestFixture.nodes.find((node) => node.kind === 'checkpoint' && node.parent?.state === 'ok'),
  ];
  const name = (i: number) => `LFM2-1.2B:run-a:${String((i + 1) * 100).padStart(6, '0')}`;
  const nodes = [
    ...Array.from({ length: NODES }, (_, i) => ({
      ...(i === 0 ? rootNode : childNode),
      name: name(i),
      step: (i + 1) * 100,
      parent: i === 0 ? { target: null, state: 'root' } : { target: name(i - 1), state: 'ok' },
    })),
    { ...adapterNode, base: { target: name(0), state: 'ok' } },
  ];
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    const body = url.includes('/api/v1/tree')
      ? { ...forestFixture, nodes, page: { limit: 2000, offset: 0, total: nodes.length } }
      : {};
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  });

  await page.goto('/');
  const started = Date.now();
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Checkpoints' })
    .click();
  await expect(page.getByTestId('tree')).toBeVisible({ timeout: PAINT_WAIT_MS });
  await expect(page.getByTestId('tree-count')).toContainText('501 artifacts');
  const elapsed = Date.now() - started;
  console.log(
    `TREE_FIRST_PAINT_MS=${elapsed} sample=${NODES}_checkpoints+1_adapter browser=chromium bundle=production`,
  );
  expect(elapsed).toBeLessThan(5000);
});

test('renders a 50,000-step run with the decimation and the sample named', async ({ page }) => {
  // AC-009.23. The assertion is not "it rendered" but **that it said what it did**: a
  // decimated curve presented as the raw series is a number quoted without its sample.
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    const body = url.includes('/steps')
      ? {
          run_id: 'run-long',
          steps: steps(LONG_RUN_STEPS),
          count: LONG_RUN_STEPS,
          limit: LONG_RUN_STEPS,
          next_before_step: null,
        }
      : url.includes('/scores')
        ? {
            run_id: 'run-long',
            scores: [],
            test_every: 10,
            retention_every: 20,
            exposure_window_steps: 19,
            provider_discontinuities: [],
          }
        : url.includes('/batches')
          ? { run_id: 'run-long', batches: [], count: 0, limit: 50, next_before: null }
          : url.includes('/checkpoints')
            ? { run_id: 'run-long', checkpoints: [], latest_step: null, steps_on_disk: [] }
            : {
                run_id: 'run-long',
                status: 'running',
                run_type: 'change',
                last_step: LONG_RUN_STEPS,
                cancel_requested_at: null,
                failure_code: null,
                heartbeat: {
                  heartbeat_at: new Date().toISOString(),
                  age_seconds: 5,
                  longest_gap_seconds: 120,
                  lapse_threshold_seconds: 900,
                  lapsed: false,
                },
              };
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  });

  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Runs' })
    .click();
  await page.getByTestId('run-id-input').fill('run-long');
  const started = Date.now();
  await page.getByRole('button', { name: 'Open', exact: true }).click();
  await page.getByRole('button', { name: 'Open this run' }).click();
  await expect(page.getByTestId('run-curves')).toBeVisible({ timeout: PAINT_WAIT_MS });
  // The decimation named, with the underlying sample: "1 point per 25 steps, 50,000 steps".
  await expect(page.getByTestId('curve-decimation')).toContainText('1 point per 25 steps');
  await expect(page.getByTestId('curve-decimation')).toContainText('50,000 steps');
  const elapsed = Date.now() - started;
  console.log(
    `LONG_RUN_FIRST_PAINT_MS=${elapsed} sample=${LONG_RUN_STEPS}_steps browser=chromium bundle=production`,
  );
  expect(elapsed).toBeLessThan(10_000);
});
