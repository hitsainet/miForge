/**
 * The run view's chart legend keeps the stat tiles' order: Loss, Test, Retention.
 *
 * Recharts 3 (suite baseline, 2026-10-21) sorts legend items alphabetically unless told
 * otherwise, and the upgrade silently turned the legend into Loss, Retention, Test - found by
 * a before/after screenshot comparison, not by any test. `RunCurves` now states the order
 * (`legendOrder`); this spec is what fails when that wiring is removed, because jsdom renders
 * no chart (it has no layout, so `ResponsiveContainer` has zero size) and a unit test cannot
 * see it.
 *
 * The run is served at the network from the generated fixture (`fixtures/backend.json`), with
 * a step page added so the chart has a curve to draw.
 */
import { expect, test, type Route } from '@playwright/test';
import backend from './fixtures/backend.json' with { type: 'json' };

const FIXTURE = backend as Record<string, unknown>;
const RUN = 'run-seeded-1';

const STEPS = Array.from({ length: 40 }, (_, index) => {
  const step = 40 - index;
  return {
    step,
    loss: 3 * Math.exp(-step / 15) + 0.4,
    grad_norm: 1,
    learning_rate: 2e-4,
    wall_time_ms: 150,
    batch_source_hashes: null,
    quiesce_hold_ms: null,
    commit_duration_ms: null,
  };
});

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

test('the legend reads Loss, Test, Retention - the order of the tiles, not the alphabet', async ({
  page,
}) => {
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === `/api/v1/runs/${RUN}/steps`) {
      return json(route, {
        run_id: RUN,
        steps: STEPS,
        count: 40,
        limit: 100,
        next_before_step: null,
      });
    }
    const body = FIXTURE[url.pathname + url.search] ?? FIXTURE[url.pathname];
    if (route.request().method() === 'GET' && body !== undefined) return json(route, body);
    return json(route, { error: { code: 'stub_absent', message: 'absent', details: {} } }, 503);
  });

  await page.goto(`/#/runs/${RUN}`);
  const items = page.getByTestId('run-curves').locator('.recharts-legend-item-text');
  await expect(items).toHaveText(['Loss', 'Test', 'Retention']);
});
