/**
 * The Editor endpoints screen in a real browser (2026-11-20).
 *
 * The backend is served at the network and keeps the endpoints it is given, as the real one does.
 * The screen is reached through the nav; an external endpoint is registered with its key, and the
 * spec asserts that the key went out once in the POST body and then exists nowhere in the page:
 * not in the field, not in the DOM, not in localStorage, not after a reload. A refusal is shown
 * with the backend's code and message; a change sends only the PATCHABLE fields; a removal waits
 * for its confirmation.
 */
import { expect, test, type Page, type Route } from '@playwright/test';

const KEY = 'sk-SENTINEL-e2e-key-0b5c7d9e';
const LOCAL = {
  id: 'miserver-gemma4-26b',
  kind: 'openai_compatible',
  base_url: 'http://miserver-backend.miserver.svc:8000/v1',
  model: 'gemma-4-26b',
  locality: 'local',
  serving: 'miserver',
  output_cap_field: 'max_completion_tokens',
  price_in_per_mtok: null,
  price_out_per_mtok: null,
  price_cache_write_per_mtok: null,
  price_cache_read_per_mtok: null,
  has_key: false,
  key_fingerprint: null,
  created_at: '2026-11-20T15:00:00+00:00',
  created_by: 'agent:forge-mcp',
  action_protocol: 'structured',
  context_window_tokens: 32768,
};

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

interface World {
  endpoints: Record<string, unknown>[];
  posts: unknown[];
  patches: unknown[];
  deletes: string[];
}

async function serve(page: Page): Promise<World> {
  const world: World = { endpoints: [{ ...LOCAL }], posts: [], patches: [], deletes: [] };
  await page.route('**/socket.io/**', (route) => route.abort());
  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    if (path === '/api/v1/approvals')
      return json(route, { approvals: [], count: 0, pending: 0, limit: 500 });
    if (path === '/api/v1/wsp/endpoints' && method === 'GET')
      return json(route, { endpoints: world.endpoints });
    if (path === '/api/v1/wsp/endpoints' && method === 'POST') {
      const body = request.postDataJSON() as Record<string, unknown>;
      world.posts.push(body);
      if (body['price_in_per_mtok'] === undefined)
        return json(
          route,
          {
            error: {
              code: 'wsp_endpoint_pricing_required',
              message: 'An external endpoint needs price_in_per_mtok and price_out_per_mtok.',
              details: {},
            },
          },
          422,
        );
      const row = {
        ...LOCAL,
        id: body['id'],
        kind: body['kind'],
        base_url: body['base_url'],
        model: body['model'],
        serving: 'generic',
        locality: 'external',
        output_cap_field: 'max_tokens',
        price_in_per_mtok: '3.000000',
        price_out_per_mtok: '15.000000',
        has_key: true,
        key_fingerprint: 'sha256:0b5c',
        created_by: 'operator',
        action_protocol: 'tool_calls',
        context_window_tokens: null,
      };
      world.endpoints.push(row);
      return json(route, row, 201);
    }
    const one = /^\/api\/v1\/wsp\/endpoints\/([^/]+)$/.exec(path);
    if (one !== null && method === 'PATCH') {
      world.patches.push(request.postDataJSON());
      return json(
        route,
        world.endpoints.find((row) => row['id'] === one[1]),
      );
    }
    if (one !== null && method === 'DELETE') {
      world.deletes.push(one[1] ?? '');
      world.endpoints = world.endpoints.filter((row) => row['id'] !== one[1]);
      return json(route, { id: one[1], removed_at: '2026-11-20T16:00:00+00:00' });
    }
    return json(route, { error: { code: 'not_stubbed', message: path, details: {} } }, 404);
  });
  return world;
}

async function storage(page: Page): Promise<string> {
  return page.evaluate(() => JSON.stringify({ ...window.localStorage }));
}

test('register an external endpoint: the key goes out once and is then nowhere', async ({
  page,
}) => {
  const world = await serve(page);
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Editor endpoints', exact: true })
    .click();
  await expect(page).toHaveURL(/#\/wsp\/endpoints$/);
  await expect(page.getByTestId(`endpoint-row-${LOCAL.id}`)).toContainText('gemma-4-26b');
  await expect(page.getByTestId(`endpoint-key-${LOCAL.id}`)).toHaveText('no key');

  await page.getByTestId('endpoint-add-id').fill('claude-ed');
  await page.getByTestId('endpoint-add-model').fill('claude-editor');
  await page.getByTestId('endpoint-add-base-url').fill('https://api.anthropic.com');
  await page.getByTestId('endpoint-add-kind').selectOption('anthropic');
  await expect(page.getByTestId('endpoint-add-api-key')).toHaveAttribute('type', 'password');
  await page.getByTestId('endpoint-add-api-key').fill(KEY);

  // First without prices: the backend's refusal, word for word; the key is already gone.
  await page.getByTestId('endpoint-add-submit').click();
  const refused = page.getByTestId('endpoint-add-failure');
  await expect(refused.getByTestId('failure-code')).toHaveText('wsp_endpoint_pricing_required');
  await expect(refused).toContainText(
    'An external endpoint needs price_in_per_mtok and price_out_per_mtok.',
  );
  await expect(page.getByTestId('endpoint-add-api-key')).toHaveValue('');
  await expect(page.getByTestId('endpoint-add-model')).toHaveValue('claude-editor');

  await page.getByTestId('endpoint-add-price-in-per-mtok').fill('3');
  await page.getByTestId('endpoint-add-price-out-per-mtok').fill('15');
  await page.getByTestId('endpoint-add-api-key').fill(KEY);
  await page.getByTestId('endpoint-add-submit').click();
  await expect(page.getByTestId('endpoint-registered')).toContainText('claude-ed');
  await expect(page.getByTestId('endpoint-row-claude-ed')).toContainText('fingerprint sha256:0b5c');

  expect(world.posts).toHaveLength(2);
  expect(world.posts[1]).toMatchObject({ id: 'claude-ed', api_key: KEY, price_in_per_mtok: '3' });
  expect(world.posts[1]).not.toHaveProperty('locality');
  await expect(page.getByTestId('endpoint-add-api-key')).toHaveValue('');
  expect(await page.content()).not.toContain(KEY);
  expect(await storage(page)).not.toContain(KEY);

  await page.reload();
  await expect(page.getByTestId('endpoint-row-claude-ed')).toBeVisible();
  expect(await page.content()).not.toContain(KEY);
  expect(await storage(page)).not.toContain(KEY);
});

test('change the protocol, then remove after confirming', async ({ page }) => {
  const world = await serve(page);
  await page.goto('/#/wsp/endpoints');
  const id = LOCAL.id;
  await page.getByTestId(`endpoint-change-open-${id}`).click();
  await page.getByTestId(`endpoint-change-protocol-${id}`).selectOption('tool_calls');
  await page.getByTestId(`endpoint-change-save-${id}`).click();
  await expect.poll(() => world.patches).toEqual([{ action_protocol: 'tool_calls' }]);

  await page.getByTestId(`endpoint-remove-${id}`).click();
  await expect(page.getByTestId(`endpoint-confirm-remove-${id}`)).toBeVisible();
  expect(world.deletes).toEqual([]);
  await page.getByTestId(`endpoint-remove-confirm-${id}`).click();
  await expect(page.getByTestId('endpoints-empty')).toBeVisible();
  expect(world.deletes).toEqual([id]);
});

test('hosted models (feature 020): the history mode and the effort go out once, and an old row says register again', async ({
  page,
}) => {
  const world = await serve(page);
  // A hosted Anthropic row from before 020: sealed without cache prices, so runs are refused.
  world.endpoints.push({
    ...LOCAL,
    id: 'claude-old',
    kind: 'anthropic',
    base_url: 'https://api.anthropic.com',
    model: 'claude-editor',
    locality: 'external',
    serving: 'generic',
    has_key: true,
    key_fingerprint: 'sha256:0b5c',
    price_in_per_mtok: '3.000000',
    price_out_per_mtok: '15.000000',
    context_window_tokens: 200000,
    thinking: 'default',
    history_mode: 'append_only',
    runs_refused: {
      code: 'wsp_endpoint_cache_prices_required',
      message:
        'This endpoint has no cache prices, which are sealed with its key: register it again with both cache prices to run it.',
    },
  });
  await page.goto('/#/wsp/endpoints');
  const refused = page.getByTestId('endpoint-runs-refused-claude-old');
  await expect(refused).toContainText('Cache prices required: register this endpoint again');
  await expect(refused).toContainText('wsp_endpoint_cache_prices_required');
  await expect(page.getByTestId('endpoint-facts-claude-old')).toContainText(
    'append_only: nothing already sent changes',
  );
  await expect(page.getByTestId(`endpoint-runs-refused-${LOCAL.id}`)).toHaveCount(0);

  await page.getByTestId('endpoint-add-id').fill('claude-ed');
  await page.getByTestId('endpoint-add-model').fill('claude-editor');
  await page.getByTestId('endpoint-add-base-url').fill('https://api.anthropic.com');
  await page.getByTestId('endpoint-add-kind').selectOption('anthropic');
  await page.getByTestId('endpoint-add-history-mode').selectOption('append_only');
  await page.getByTestId('endpoint-add-thinking').selectOption('high');
  await page.getByTestId('endpoint-add-context-window').fill('200000');
  await page.getByTestId('endpoint-add-price-in-per-mtok').fill('3');
  await page.getByTestId('endpoint-add-price-out-per-mtok').fill('15');
  await page.getByTestId('endpoint-add-price-cache-write-per-mtok').fill('3.75');
  await page.getByTestId('endpoint-add-price-cache-read-per-mtok').fill('0.3');
  await page.getByTestId('endpoint-add-api-key').fill(KEY);
  await page.getByTestId('endpoint-add-submit').click();
  await expect(page.getByTestId('endpoint-registered')).toContainText('claude-ed');
  expect(world.posts).toHaveLength(1);
  expect(world.posts[0]).toMatchObject({
    history_mode: 'append_only',
    thinking: 'high',
    context_window_tokens: 200000,
    price_cache_write_per_mtok: '3.75',
    price_cache_read_per_mtok: '0.3',
  });
});
