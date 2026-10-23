/**
 * An agent's request appears in Approvals and the operator approves it (F12 task 5.5;
 * FR-012.17).
 *
 * `vite preview` serves no backend, so the API is served at the network by `page.route`. The
 * approval list is **not hand-written**: it is `fixtures/approvals.json`, which
 * `backend/tests/integration/test_ui_approval_fixtures.py` generates by sending one agent
 * request of each action through the mounted app's gate and then pins - so this spec renders
 * what the gate stores. The fake adds only state: an approved request becomes `executed`.
 *
 * The spec reaches the screen through the real nav, so deleting the `approvals` row from
 * `ROUTES` takes it down as well as `routes.test.ts` (ADR-016 #1).
 */
import { expect, test, type Page, type Route } from '@playwright/test';
import approvals from './fixtures/approvals.json' with { type: 'json' };

interface Row {
  readonly approval_id: string;
  readonly action: string;
  readonly status: string;
  readonly [key: string]: unknown;
}

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

async function serve(
  page: Page,
): Promise<{ decisions: { path: string; headers: Record<string, string> }[] }> {
  const decisions: { path: string; headers: Record<string, string> }[] = [];
  let rows: Row[] = (approvals as { approvals: Row[] }).approvals.map((row) => ({ ...row }));

  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    if (method === 'GET' && url.pathname === '/api/v1/approvals') {
      return json(route, {
        approvals: rows,
        count: rows.length,
        pending: rows.filter((row) => row.status === 'pending').length,
        limit: 200,
      });
    }
    const decide = /^\/api\/v1\/approvals\/([^/]+)\/(approve|reject)$/.exec(url.pathname);
    if (method === 'POST' && decide !== null) {
      decisions.push({ path: url.pathname, headers: route.request().headers() });
      const [, id, verb] = decide;
      rows = rows.map((row) =>
        row.approval_id === id
          ? {
              ...row,
              status: verb === 'approve' ? 'executed' : 'rejected',
              decided_at: '2026-10-19T12:05:00+00:00',
              result: verb === 'approve' ? { run_id: 'run-f12', status: 'pending' } : null,
            }
          : row,
      );
      return json(
        route,
        rows.find((row) => row.approval_id === id),
      );
    }
    return json(route, { error: { code: 'not_stubbed', message: url.pathname, details: {} } }, 404);
  });
  return { decisions };
}

test('an agent request appears in Approvals and is approved by the operator', async ({ page }) => {
  const { decisions } = await serve(page);
  await page.goto('/');
  const nav = page.getByRole('navigation', { name: 'Screens' });
  // The badge counts the pending requests on every screen, before Approvals is opened.
  await expect(page.getByTestId('nav-badge-approvals')).toHaveText('4');
  await nav.getByRole('button', { name: /Approvals/ }).click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Approvals');
  await expect(page.getByTestId('approvals-count')).toHaveText('4 pending of 4 shown');

  const gpu = (approvals as { approvals: Row[] }).approvals.find(
    (row) => row.action === 'gpu_start',
  );
  expect(gpu).toBeDefined();
  const id = gpu?.approval_id ?? '';
  const card = page.getByTestId(`approval-${id}`);
  await expect(card).toContainText('Step budget');
  await expect(card).toContainText('agent:forge-mcp');
  await card.getByRole('button', { name: 'Approve and start run' }).click();

  await expect(page.getByTestId(`approval-outcome-${id}`)).toContainText('Open run run-f12');
  expect(decisions.map((d) => d.path)).toEqual([`/api/v1/approvals/${id}/approve`]);
  expect(Object.keys(decisions[0]?.headers ?? {})).not.toContain('x-miforge-agent');
  await expect(page.getByTestId('nav-badge-approvals')).toHaveText('3');

  await page
    .getByTestId(`approval-outcome-${id}`)
    .getByRole('button', { name: /Open run/ })
    .click();
  await expect(page).toHaveURL(/#\/runs\/run-f12$/);
});
