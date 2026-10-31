/**
 * Permanent deletion of archived runs, in a real browser (operator request 2026-10-20:
 * "I should be able to select/de-select all or select/de-select individual items for
 * permanent deletion").
 *
 * The backend is served at the network from the generated fixture's own run-row shape
 * (`fixtures/backend.json`), and it is **stateful**: a purge removes the rows it was asked
 * to remove, so the re-read after the confirmation shows exactly what is left.
 *
 * Pinned: the header checkbox selects every selectable archived run and deselects them all
 * again, is indeterminate in between, the count is live, a blocked run shows its blocker and
 * cannot be selected, and confirming sends **exactly** the remaining selected ids - after
 * which exactly those rows are gone.
 */
import { expect, test, type Page, type Route } from '@playwright/test';
import backend from './fixtures/backend.json' with { type: 'json' };

interface RunRow {
  readonly run_id: string;
  readonly archived_at: string | null;
  readonly [key: string]: unknown;
}

const SEEDED = (backend as Record<string, unknown>)['/api/v1/runs?limit=25&offset=0'] as {
  runs: RunRow[];
  [key: string]: unknown;
};
const TEMPLATE = SEEDED.runs[0] as RunRow;
const row = (run_id: string, archived: boolean): RunRow => ({
  ...TEMPLATE,
  run_id,
  status: 'completed',
  archived_at: archived ? '2026-10-20T09:00:00+00:00' : null,
});
const BLOCKED = 'old-4';
const BLOCKER = 'Run child-1 (not archived) resumes, loads or branches from this run.';

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

function plan(id: string) {
  const blocked = id === BLOCKED;
  return {
    kind: 'run',
    id,
    found: true,
    name: `run ${id} (change, completed)`,
    status: 'completed',
    archived: true,
    purgeable: !blocked,
    not_archived: null,
    blockers: blocked ? [{ code: 'run_reference', message: BLOCKER, run_id: 'child-1' }] : [],
    removes: {
      rows: { mf_runs: 1, mf_steps: 40 },
      row_total: 41,
      files: 6,
      bytes: 3 * 1024 * 1024,
      directories: [],
    },
    kept: [],
  };
}

/** Serve the run list, the purge preview and the purge. Returns every purge request body. */
async function serve(page: Page): Promise<Array<{ ids: string[]; reason?: string }>> {
  let rows: RunRow[] = [
    row('live-1', false),
    row('old-1', true),
    row('old-2', true),
    row('old-3', true),
    row(BLOCKED, true),
  ];
  const purges: Array<{ ids: string[]; reason?: string }> = [];
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === 'GET' && url.pathname === '/api/v1/runs') {
      const include = url.searchParams.get('include_archived') === 'true';
      const listed = include ? rows : rows.filter((r) => r.archived_at === null);
      return json(route, {
        ...SEEDED,
        include_archived: include,
        runs: listed,
        total: listed.length,
        limit: 25,
        offset: 0,
      });
    }
    if (request.method() === 'POST' && url.pathname === '/api/v1/runs/purge/plan') {
      const body = request.postDataJSON() as { ids: string[] };
      return json(route, { kind: 'run', items: body.ids.map(plan) });
    }
    if (request.method() === 'POST' && url.pathname === '/api/v1/runs/purge') {
      const body = request.postDataJSON() as { ids: string[]; reason?: string };
      purges.push(body);
      rows = rows.filter((r) => !body.ids.includes(r.run_id));
      return json(route, {
        kind: 'run',
        purged: body.ids.length,
        blocked: 0,
        bytes_removed: body.ids.length * 3 * 1024 * 1024,
        results: body.ids.map((id) => ({
          outcome: 'purged',
          id,
          name: `run ${id} (change, completed)`,
          tombstone_id: `purge-${id}`,
          bytes_removed: 3 * 1024 * 1024,
          files_removed: 6,
          rows_removed: { mf_runs: 1 },
          files_left: [],
        })),
      });
    }
    return json(route, { error: { code: 'not_stubbed', message: url.pathname, details: {} } }, 404);
  });
  return purges;
}

test('select all, deselect one, confirm: exactly the remaining selected runs are purged', async ({
  page,
}) => {
  const purges = await serve(page);
  await page.goto('/#/runs');
  await expect(page.getByTestId('run-list-row-live-1')).toBeVisible();
  // No purge UI until archived runs are shown.
  await expect(page.getByTestId('run-purge-header')).toHaveCount(0);

  await page.getByTestId('run-show-archived').check();
  const all = page.getByTestId('run-purge-select-all');
  const count = page.getByTestId('run-purge-count');
  await expect(page.getByTestId('run-purge-check-old-1')).toBeEnabled();
  await expect(page.getByTestId('run-purge-check-live-1')).toHaveCount(0);

  // The blocked run says why and cannot be selected.
  await expect(page.getByTestId(`run-purge-blocked-${BLOCKED}`)).toContainText('child-1');
  await expect(page.getByTestId(`run-purge-check-${BLOCKED}`)).toBeDisabled();

  // Select all toggles: on, off, on.
  await all.check();
  await expect(count).toHaveText('3 selected');
  await expect(page.getByTestId(`run-purge-check-${BLOCKED}`)).not.toBeChecked();
  await all.uncheck();
  await expect(count).toHaveText('0 selected');
  await expect(page.getByTestId('run-purge-delete')).toBeDisabled();
  await all.check();
  await expect(count).toHaveText('3 selected');
  await expect(page.getByTestId('run-purge-select-all-label')).toHaveText('Deselect all');

  // Deselect one: the header goes indeterminate.
  await page.getByTestId('run-purge-check-old-2').uncheck();
  await expect(count).toHaveText('2 selected');
  expect(await all.evaluate((element) => (element as HTMLInputElement).indeterminate)).toBe(true);

  // Confirm: every selected run listed with what goes; nothing sent until confirmed.
  await page.getByTestId('run-purge-delete').click();
  const dialog = page.getByTestId('run-purge-dialog');
  await expect(dialog.getByTestId('run-purge-going-old-1')).toContainText('41 database rows');
  await expect(dialog.getByTestId('run-purge-going-old-3')).toContainText('3.0 MB');
  await expect(dialog.getByTestId('run-purge-going-old-2')).toHaveCount(0);
  expect(purges).toHaveLength(0);
  await dialog.getByTestId('run-purge-reason').fill('e2e clutter');
  await dialog.getByTestId('run-purge-confirm').click();

  await expect(dialog.getByTestId('run-purge-outcome')).toContainText('Deleted 2');
  expect(purges).toEqual([{ ids: ['old-1', 'old-3'], reason: 'e2e clutter' }]);
  await dialog.getByTestId('run-purge-close').click();

  // Exactly those rows are gone; the deselected and the blocked runs remain.
  await expect(page.getByTestId('run-list-row-old-1')).toHaveCount(0);
  await expect(page.getByTestId('run-list-row-old-3')).toHaveCount(0);
  await expect(page.getByTestId('run-list-row-old-2')).toBeVisible();
  await expect(page.getByTestId(`run-list-row-${BLOCKED}`)).toBeVisible();
  await expect(page.getByTestId('run-list-row-live-1')).toBeVisible();
  await expect(count).toHaveText('0 selected');
});
