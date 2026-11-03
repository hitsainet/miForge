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
  extra: readonly Row[] = [],
): Promise<{ decisions: { path: string; headers: Record<string, string>; body: unknown }[] }> {
  const decisions: { path: string; headers: Record<string, string>; body: unknown }[] = [];
  let rows: Row[] = [
    ...(approvals as { approvals: Row[] }).approvals.map((row) => ({ ...row })),
    ...extra,
  ];

  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    if (method === 'GET' && url.pathname === '/api/v1/approvals') {
      const status = url.searchParams.get('status');
      const listed = rows.filter((row) => status === null || row.status === status);
      return json(route, {
        approvals: listed,
        count: listed.length,
        pending: listed.filter((row) => row.status === 'pending').length,
        limit: Number(url.searchParams.get('limit') ?? 200),
      });
    }
    if (method === 'POST' && url.pathname === '/api/v1/approvals/approve-batch') {
      const body = route.request().postDataJSON() as { approval_ids: string[] };
      decisions.push({ path: url.pathname, headers: route.request().headers(), body });
      rows = rows.map((row) =>
        body.approval_ids.includes(row.approval_id)
          ? {
              ...row,
              status: 'executed',
              decided_by: 'operator',
              decided_at: '2026-10-19T12:05:00+00:00',
              batch_id: 'bat_e2e',
              result: { key: 'x' },
            }
          : row,
      );
      const results = body.approval_ids.map((id) => ({
        ...rows.find((row) => row.approval_id === id),
        ok: true,
      }));
      return json(route, { batch_id: 'bat_e2e', count: results.length, results });
    }
    const decide = /^\/api\/v1\/approvals\/([^/]+)\/(approve|reject)$/.exec(url.pathname);
    if (method === 'POST' && decide !== null) {
      decisions.push({ path: url.pathname, headers: route.request().headers(), body: null });
      const [, id, verb] = decide;
      rows = rows.map((row) =>
        row.approval_id === id
          ? {
              ...row,
              status: verb === 'approve' ? 'executed' : 'rejected',
              decided_by: 'operator',
              decided_at: '2026-10-19T12:05:00+00:00',
              reason:
                verb === 'reject'
                  ? ((route.request().postDataJSON() as { reason?: string } | null)?.reason ?? null)
                  : null,
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
  // The badge counts the waiting requests on every screen, before Approvals is opened.
  await expect(page.getByTestId('nav-badge-approvals')).toHaveText('4');
  await expect(page.getByTestId('nav-badge-approvals')).toHaveAttribute('title', '4 waiting');
  await nav.getByRole('button', { name: /Approvals/ }).click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Approvals');
  await expect(page.getByRole('heading', { level: 2, name: 'Waiting for you (4)' })).toBeVisible();
  // The banner is not drawn on Approvals itself, which shows the same cards.
  await expect(page.getByTestId('approvals-banner')).toHaveCount(0);

  const gpu = (approvals as { approvals: Row[] }).approvals.find(
    (row) => row.action === 'gpu_start',
  );
  expect(gpu).toBeDefined();
  const id = gpu?.approval_id ?? '';
  const card = page.getByTestId(`approval-${id}`);
  await expect(card).toContainText('Step budget');
  await expect(card).toContainText('asked by agent forge-mcp');
  await card.getByRole('button', { name: 'Approve and start run' }).click();

  const history = page.getByTestId(`history-${id}`);
  await expect(history).toContainText('Approved and run');
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

test('the banner shows waiting requests on another screen, and a reject lands in History', async ({
  page,
}) => {
  const { decisions } = await serve(page);
  await page.goto('/#/checkpoints');
  // ENH-008 A2: removing the banner mount in Shell takes this down.
  const banner = page.getByRole('region', { name: 'Agent requests waiting for your approval' });
  await expect(banner).toBeVisible();
  await expect(banner.getByTestId('approvals-banner-heading')).toHaveText(
    '4 agent requests waiting for your approval',
  );
  await expect(banner.getByTestId('approvals-banner-cards').getByRole('listitem')).toHaveCount(3);
  await expect(banner.getByTestId('approvals-banner-more')).toHaveText(
    'and 1 more waiting — Open Approvals',
  );

  const first = banner.getByTestId('approvals-banner-cards').getByRole('listitem').first();
  const id = ((await first.getAttribute('data-testid')) ?? '').replace(/^approval-/, '');
  await first.getByRole('button', { name: 'Reject' }).click();
  await first.getByPlaceholder('Reason (optional)').fill('live check');
  await first.getByRole('button', { name: 'Confirm reject' }).click();
  await expect(banner.getByTestId('approvals-banner-heading')).toHaveText(
    '3 agent requests waiting for your approval',
  );
  expect(decisions.map((d) => d.path)).toEqual([`/api/v1/approvals/${id}/reject`]);

  await banner.getByRole('link', { name: 'Open Approvals' }).first().click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Approvals');
  const row = page.getByTestId(`history-${id}`);
  await expect(row).toContainText('Rejected');
  await expect(row).toContainText('operator');
  await expect(row).toContainText('live check');
});

test('two waiting settings changes are approved with one action, and History shows each (D79)', async ({
  page,
}) => {
  const stored = (approvals as { approvals: Row[] }).approvals.find(
    (row) => row.action === 'config_write',
  );
  expect(stored).toBeDefined();
  const second: Row = {
    ...(stored as Row),
    approval_id: 'apr_fixture_keep',
    summary: {
      ...((stored as Row)['summary'] as Record<string, unknown>),
      key: 'checkpoint_lifecycle_keep_last_n',
      label: 'Checkpoints to keep',
      old: 10,
      new: 3,
    },
  };
  const { decisions } = await serve(page, [second]);
  await page.goto('/#/checkpoints');
  const banner = page.getByRole('region', { name: 'Agent requests waiting for your approval' });
  // ENH-006 D79: removing the mount in the banner takes this down.
  await banner.getByRole('button', { name: 'Approve all settings changes (2)' }).click();
  const confirm = page.getByTestId('approve-all-confirm-banner');
  await expect(confirm).toContainText(String((stored?.['summary'] as { label: string }).label));
  await expect(confirm).toContainText('10 → 3');
  await confirm.getByRole('button', { name: 'Approve these 2 settings changes' }).click();
  await expect(banner.getByRole('button', { name: /Approve all settings changes/ })).toHaveCount(0);
  expect(decisions).toHaveLength(1);
  expect(decisions[0]?.path).toBe('/api/v1/approvals/approve-batch');
  expect(decisions[0]?.body).toEqual({
    approval_ids: [stored?.approval_id, 'apr_fixture_keep'],
  });
  expect(Object.keys(decisions[0]?.headers ?? {})).not.toContain('x-miforge-agent');

  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: /Approvals/ })
    .click();
  for (const id of [stored?.approval_id ?? '', 'apr_fixture_keep']) {
    await expect(page.getByTestId(`history-${id}`)).toContainText('Approved and run');
    await expect(page.getByTestId(`history-decided-by-${id}`)).toContainText('bat_e2e');
  }
});
