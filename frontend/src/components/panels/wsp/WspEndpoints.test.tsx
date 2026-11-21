/**
 * The Editor endpoints screen, through the real App, its nav and `fetch` (2026-11-20).
 *
 * The operator registered `miserver-gemma4-26b` with `curl`: the backend's refusal named
 * "miForge's own screens" and none existed. Every test here reaches the screen through the shell
 * (the nav entry or the URL), and the network is stubbed at `fetch`, so a deleted route row, nav
 * entry or API call is red here. The key-not-displayed guarantee is asserted four ways: the field
 * is a password input, it is emptied as soon as it is sent, it never reaches browser storage, and
 * a listing that carried a key or a ciphertext would not put either on the screen.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ApprovalCard } from '@/components/approvals/ApprovalCard';
import { ApprovalHistory } from '@/components/approvals/ApprovalHistory';
import { NavigationProvider, useNavigation } from '@/contexts/NavigationContext';
import type { Approval } from '@/api/approvals';
import { WSP_ENDPOINT, answer } from '@/testing/stubBackend';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';
import { publicEndpoint } from '@/api/wspEndpoints';
import { EMPTY_ENDPOINT_DRAFT, registrationBody } from './WspEndpointsPage';

const KEY = 'sk-SENTINEL-operator-key-7f3a9c';
const CIPHER = 'CIPHERTEXT-SENTINEL-b64';
const PATH = '/api/v1/wsp/endpoints';
const ID = WSP_ENDPOINT.id;

interface Seen {
  readonly method: string;
  readonly path: string;
  readonly body: unknown;
}

let calls: Seen[] = [];
let overrides: Record<string, { status: number; body: unknown }> = {};
let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  calls = [];
  overrides = {};
  window.localStorage.clear();
  useApprovalsStore.setState(INITIAL_APPROVALS_STATE);
  window.history.replaceState(null, '', '/');
  originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
    const method = (init?.method ?? 'GET').toUpperCase();
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined;
    calls.push({ method, path: url.pathname, body });
    const { status, body: answerBody } =
      overrides[`${method} ${url.pathname}`] ?? answer(method, url.pathname);
    return Promise.resolve(
      new Response(JSON.stringify(answerBody), {
        status,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  }) as typeof globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  window.history.replaceState(null, '', '/');
});

const inTheme = (element: React.ReactElement) => (
  <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
);

const writes = (method: string, path = PATH): Seen[] =>
  calls.filter((call) => call.method === method && call.path === path);

const refusal = (code: string, message: string, details: Record<string, unknown> = {}) => ({
  status: code.endsWith('_refused') && code.startsWith('wsp_endpoint_agent') ? 403 : 422,
  body: { error: { code, message, details } },
});

async function openThroughTheNav(): Promise<void> {
  render(inTheme(<App />));
  await userEvent.click(
    within(screen.getByRole('navigation', { name: 'Screens' })).getByRole('button', {
      name: 'Editor endpoints',
    }),
  );
  await screen.findByTestId(`endpoint-row-${ID}`);
}

function storage(): string {
  const out: string[] = [];
  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index) ?? '';
    out.push(key, window.localStorage.getItem(key) ?? '');
  }
  return out.join('\n');
}

function type(testId: string, value: string): void {
  fireEvent.change(screen.getByTestId(testId), { target: { value } });
}

// ---------------------------------------------------------------------------------- list

describe('[Editor endpoints] the list', () => {
  it('should be reached through the nav, read the endpoint route and render each row', async () => {
    await openThroughTheNav();
    expect(window.location.hash).toBe('#/wsp/endpoints');
    expect(screen.getByTestId('shell-page-head').querySelector('h1')?.textContent).toBe(
      'Editor endpoints',
    );
    expect(writes('GET').length).toBeGreaterThanOrEqual(1);
    const facts = screen.getByTestId(`endpoint-facts-${ID}`).textContent ?? '';
    for (const text of [
      'gemma-4-26b',
      WSP_ENDPOINT.base_url,
      'openai_compatible',
      'miserver',
      'local',
      'structured',
      '32,768 tokens',
      'max_completion_tokens',
    ]) {
      expect(facts).toContain(text);
    }
    expect(screen.getByTestId(`endpoint-key-${ID}`).textContent).toBe('no key');
  });

  it('should be reachable from its URL', async () => {
    window.history.replaceState(null, '', '/#/wsp/endpoints');
    render(inTheme(<App />));
    expect(await screen.findByTestId(`endpoint-row-${ID}`)).toBeInTheDocument();
  });

  it('should show a sealed key by its fingerprint and NEVER a key or ciphertext, even if served', async () => {
    overrides[`GET ${PATH}`] = {
      status: 200,
      body: {
        endpoints: [
          {
            ...WSP_ENDPOINT,
            has_key: true,
            key_fingerprint: 'sha256:abcd1234',
            // A backend defect that served these must still not reach the screen.
            api_key: KEY,
            key_ciphertext: CIPHER,
          },
        ],
      },
    };
    await openThroughTheNav();
    expect(screen.getByTestId(`endpoint-key-${ID}`).textContent).toBe('key sealed');
    expect(screen.getByTestId(`endpoint-facts-${ID}`).textContent).toContain(
      'sealed, fingerprint sha256:abcd1234',
    );
    // The stored record is rendered in full: the guarantee is what it holds, not what is hidden.
    expect(screen.getByTestId(`endpoint-record-${ID}`).textContent).toContain('key_fingerprint');
    expect(document.body.textContent).not.toContain(KEY);
    expect(document.body.textContent).not.toContain(CIPHER);
    expect(document.body.innerHTML).not.toContain(KEY);
  });

  it('publicEndpoint should copy the public fields by name and drop everything else', () => {
    const row = publicEndpoint({ ...WSP_ENDPOINT, api_key: KEY, key_ciphertext: CIPHER });
    expect(Object.keys(row).sort()).toEqual(Object.keys(WSP_ENDPOINT).sort());
    expect(JSON.stringify(row)).not.toContain(KEY);
  });

  it('should render a refused list read with its code and message', async () => {
    overrides[`GET ${PATH}`] = refusal('wsp_phase_not_enabled', 'WSP is not enabled here.');
    window.history.replaceState(null, '', '/#/wsp/endpoints');
    render(inTheme(<App />));
    const notice = await screen.findByTestId('endpoints-failure');
    expect(within(notice).getByTestId('failure-code').textContent).toBe('wsp_phase_not_enabled');
    expect(notice.textContent).toContain('WSP is not enabled here.');
  });
});

// ----------------------------------------------------------------------------------- add

describe('[Editor endpoints] add', () => {
  it('should send every field once, with the key, and then hold no key anywhere', async () => {
    overrides[`POST ${PATH}`] = {
      status: 201,
      body: {
        ...WSP_ENDPOINT,
        id: 'claude-ed',
        locality: 'external',
        has_key: true,
        key_fingerprint: 'sha256:ffff',
      },
    };
    await openThroughTheNav();
    const key = screen.getByTestId('endpoint-add-api-key') as HTMLInputElement;
    expect(key.type).toBe('password');
    expect(key.autocomplete).toBe('new-password');
    type('endpoint-add-id', 'claude-ed');
    type('endpoint-add-model', 'claude-editor');
    type('endpoint-add-base-url', 'https://api.anthropic.com');
    type('endpoint-add-kind', 'anthropic');
    type('endpoint-add-serving', 'generic');
    type('endpoint-add-action-protocol', 'tool_calls');
    type('endpoint-add-output-cap-field', 'max_tokens');
    type('endpoint-add-context-window', '200000');
    type('endpoint-add-price-in-per-mtok', '3');
    type('endpoint-add-price-out-per-mtok', '15');
    type('endpoint-add-price-cache-write-per-mtok', '3.75');
    type('endpoint-add-price-cache-read-per-mtok', '0.3');
    type('endpoint-add-api-key', KEY);
    expect(screen.getByTestId('endpoint-add-locality').textContent).toMatch(/looks external/);
    // Typed, the key is not in storage: the draft never holds it.
    expect(storage()).not.toContain(KEY);
    expect(storage()).toContain('claude-editor');
    await userEvent.click(screen.getByTestId('endpoint-add-submit'));
    await screen.findByTestId('endpoint-registered');
    expect(writes('POST')).toEqual([
      {
        method: 'POST',
        path: PATH,
        body: {
          id: 'claude-ed',
          kind: 'anthropic',
          base_url: 'https://api.anthropic.com',
          model: 'claude-editor',
          serving: 'generic',
          output_cap_field: 'max_tokens',
          action_protocol: 'tool_calls',
          context_window_tokens: 200000,
          price_in_per_mtok: '3',
          price_out_per_mtok: '15',
          price_cache_write_per_mtok: '3.75',
          price_cache_read_per_mtok: '0.3',
          api_key: KEY,
        },
      },
    ]);
    // Sent once, then gone: the field, the page, storage.
    expect((screen.getByTestId('endpoint-add-api-key') as HTMLInputElement).value).toBe('');
    expect(document.body.innerHTML).not.toContain(KEY);
    expect(storage()).not.toContain(KEY);
    // A success clears the draft and re-reads the list.
    expect((screen.getByTestId('endpoint-add-model') as HTMLInputElement).value).toBe('');
    await waitFor(() => expect(writes('GET').length).toBeGreaterThanOrEqual(2));
  });

  it('should send a local endpoint without a key, prices or a locality', async () => {
    overrides[`POST ${PATH}`] = { status: 201, body: WSP_ENDPOINT };
    await openThroughTheNav();
    type('endpoint-add-id', 'local-ed');
    type('endpoint-add-model', 'qwen');
    type('endpoint-add-base-url', 'http://miserver-backend.miserver.svc:8000/v1');
    type('endpoint-add-kind', 'openai_compatible');
    type('endpoint-add-serving', 'miserver');
    expect(screen.getByTestId('endpoint-add-locality').textContent).toMatch(/looks local/);
    await userEvent.click(screen.getByTestId('endpoint-add-submit'));
    await screen.findByTestId('endpoint-registered');
    expect(writes('POST').map((call) => call.body)).toEqual([
      {
        id: 'local-ed',
        kind: 'openai_compatible',
        base_url: 'http://miserver-backend.miserver.svc:8000/v1',
        model: 'qwen',
        serving: 'miserver',
      },
    ]);
  });

  it('should render the backend refusal faithfully, keep the entries and still drop the key', async () => {
    overrides[`POST ${PATH}`] = refusal(
      'wsp_endpoint_pricing_required',
      'An external endpoint needs price_in_per_mtok and price_out_per_mtok.',
      { field: 'price_in_per_mtok' },
    );
    await openThroughTheNav();
    type('endpoint-add-model', 'claude-editor');
    type('endpoint-add-base-url', 'https://api.anthropic.com');
    type('endpoint-add-kind', 'anthropic');
    type('endpoint-add-api-key', KEY);
    await userEvent.click(screen.getByTestId('endpoint-add-submit'));
    const notice = await screen.findByTestId('endpoint-add-failure');
    expect(within(notice).getByTestId('failure-code').textContent).toBe(
      'wsp_endpoint_pricing_required',
    );
    expect(notice.textContent).toContain(
      'An external endpoint needs price_in_per_mtok and price_out_per_mtok.',
    );
    expect(within(notice).getByTestId('failure-details').textContent).toContain(
      'price_in_per_mtok',
    );
    expect((screen.getByTestId('endpoint-add-model') as HTMLInputElement).value).toBe(
      'claude-editor',
    );
    expect((screen.getByTestId('endpoint-add-api-key') as HTMLInputElement).value).toBe('');
    expect(screen.getByTestId('endpoint-add-key-cleared')).toBeInTheDocument();
    expect(document.body.innerHTML).not.toContain(KEY);
    expect(storage()).not.toContain(KEY);
  });

  it('should send thinking and its allowance when chosen, and neither when left empty', () => {
    const chosen = registrationBody(
      {
        ...EMPTY_ENDPOINT_DRAFT,
        kind: 'openai_compatible',
        model: 'm',
        base_url: 'http://x',
        thinking: 'on',
        thinking_allowance: '6000',
      },
      '',
    );
    expect([chosen.thinking, chosen.thinking_tokens]).toEqual(['on', 6000]);
    expect(
      registrationBody({ ...EMPTY_ENDPOINT_DRAFT, kind: 'anthropic', output_cap: 'off' }, '')
        .output_cap,
    ).toBe('off');
    const plain = registrationBody({ ...EMPTY_ENDPOINT_DRAFT, kind: 'anthropic', model: 'm' }, '');
    expect('thinking' in plain || 'thinking_tokens' in plain).toBe(false);
    // Paging (2026-11-21): sent when chosen, left to the backend's default (on) otherwise.
    expect(
      registrationBody({ ...EMPTY_ENDPOINT_DRAFT, kind: 'anthropic', paging: 'off' }, '').paging,
    ).toBe('off');
    expect('paging' in plain).toBe(false);
  });

  it('should send a context window that is not a number as typed, so the backend refuses it', () => {
    const body = registrationBody(
      { ...EMPTY_ENDPOINT_DRAFT, kind: 'anthropic', model: 'm', context_window: 'lots' },
      '',
    );
    expect(body.context_window_tokens).toBe('lots');
    expect('api_key' in body).toBe(false);
  });
});

// ----------------------------------------------------------- change, remove, protocol test

describe('[Editor endpoints] change, remove and the protocol test', () => {
  it('should PATCH only the changed fields, and render a refusal of the change', async () => {
    overrides[`PATCH ${PATH}/${ID}`] = { status: 200, body: { ...WSP_ENDPOINT } };
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    type(`endpoint-change-protocol-${ID}`, 'tool_calls');
    type(`endpoint-change-window-${ID}`, '');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    await waitFor(() => expect(writes('PATCH', `${PATH}/${ID}`)).toHaveLength(1));
    expect(writes('PATCH', `${PATH}/${ID}`)[0]?.body).toEqual({
      action_protocol: 'tool_calls',
      context_window_tokens: null,
    });

    overrides[`PATCH ${PATH}/${ID}`] = refusal(
      'wsp_endpoint_protocol_unsupported',
      'action_protocol structured is accepted only for an openai_compatible endpoint served by miServer.',
    );
    await waitFor(() => expect(screen.getByTestId(`endpoint-change-open-${ID}`)).toBeVisible());
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    type(`endpoint-change-window-${ID}`, '16384');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    const notice = await screen.findByTestId(`endpoint-change-failure-${ID}`);
    expect(within(notice).getByTestId('failure-code').textContent).toBe(
      'wsp_endpoint_protocol_unsupported',
    );
    expect(writes('PATCH', `${PATH}/${ID}`)[1]?.body).toEqual({ context_window_tokens: 16384 });
  });

  it('should show the thinking setting and PATCH only a changed thinking or allowance (2026-11-21)', async () => {
    overrides[`PATCH ${PATH}/${ID}`] = {
      status: 200,
      body: { ...WSP_ENDPOINT, thinking: 'on', thinking_tokens: 6000 },
    };
    await openThroughTheNav();
    const facts = screen.getByTestId(`endpoint-facts-${ID}`);
    expect(facts.textContent).toContain('off: the engine asks the editor not to think');
    expect(facts.textContent).toContain("4,096 tokens per turn (the engine's default)");
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    type(`endpoint-change-thinking-${ID}`, 'on');
    type(`endpoint-change-allowance-${ID}`, '6000');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    await waitFor(() => expect(writes('PATCH', `${PATH}/${ID}`)).toHaveLength(1));
    expect(writes('PATCH', `${PATH}/${ID}`)[0]?.body).toEqual({
      thinking: 'on',
      thinking_tokens: 6000,
    });

    overrides[`PATCH ${PATH}/${ID}`] = refusal(
      'wsp_endpoint_thinking_unsupported',
      'thinking on is accepted only for an openai_compatible endpoint served by miServer.',
    );
    await waitFor(() => expect(screen.getByTestId(`endpoint-change-open-${ID}`)).toBeVisible());
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    type(`endpoint-change-thinking-${ID}`, 'default');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    const notice = await screen.findByTestId(`endpoint-change-failure-${ID}`);
    expect(within(notice).getByTestId('failure-code').textContent).toBe(
      'wsp_endpoint_thinking_unsupported',
    );
    expect(writes('PATCH', `${PATH}/${ID}`)[1]?.body).toEqual({ thinking: 'default' });
  });

  it('should show an endpoint that thinks as the backend listed it', async () => {
    // Review 2026-11-21, F5: publicEndpoint pinned to 'off' survived every test, because every
    // listed endpoint was at the default. This one is not.
    overrides[`GET ${PATH}`] = {
      status: 200,
      body: { endpoints: [{ ...WSP_ENDPOINT, thinking: 'on', thinking_tokens: 6000 }] },
    };
    await openThroughTheNav();
    const facts = screen.getByTestId(`endpoint-facts-${ID}`);
    expect(facts.textContent).toContain('on: the editor reasons before each action');
    expect(facts.textContent).toContain('6,000 tokens per turn');
    expect(facts.textContent).not.toContain('used when thinking is on');
  });

  it('should show the output cap and PATCH it alone when changed (2026-11-21)', async () => {
    overrides[`PATCH ${PATH}/${ID}`] = {
      status: 200,
      body: { ...WSP_ENDPOINT, output_cap: 'off' },
    };
    await openThroughTheNav();
    expect(screen.getByTestId(`endpoint-facts-${ID}`).textContent).toContain(
      'on: each request names the per-turn cap',
    );
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    type(`endpoint-change-output-cap-${ID}`, 'off');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    await waitFor(() => expect(writes('PATCH', `${PATH}/${ID}`)).toHaveLength(1));
    expect(writes('PATCH', `${PATH}/${ID}`)[0]?.body).toEqual({ output_cap: 'off' });
  });

  it('should show paging and PATCH it alone when changed (2026-11-21)', async () => {
    overrides[`PATCH ${PATH}/${ID}`] = {
      status: 200,
      body: { ...WSP_ENDPOINT, paging: 'off' },
    };
    await openThroughTheNav();
    expect(screen.getByTestId(`endpoint-facts-${ID}`).textContent).toContain(
      'on: bulky tool results are paged out after a few turns',
    );
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    type(`endpoint-change-paging-${ID}`, 'off');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    await waitFor(() => expect(writes('PATCH', `${PATH}/${ID}`)).toHaveLength(1));
    expect(writes('PATCH', `${PATH}/${ID}`)[0]?.body).toEqual({ paging: 'off' });
  });

  it('should show an endpoint whose paging is off as the backend listed it', async () => {
    overrides[`GET ${PATH}`] = {
      status: 200,
      body: { endpoints: [{ ...WSP_ENDPOINT, paging: 'off' }] },
    };
    await openThroughTheNav();
    expect(screen.getByTestId(`endpoint-facts-${ID}`).textContent).toContain(
      'off: every tool result stays in the conversation',
    );
  });

  it('should say a thinking protocol test reports whether reasoning came back', async () => {
    overrides[`POST ${PATH}/${ID}/protocol-test`] = {
      status: 201,
      body: { run_id: 'wsp_00000000bbbb', status: 'queued', thinking: 'on' },
    };
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-test-${ID}`));
    const queued = await screen.findByTestId(`endpoint-test-queued-${ID}`);
    expect(queued.textContent).toContain('whether the server sent the reasoning back');
  });

  it('should send nothing when nothing changed', async () => {
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    expect(screen.getByTestId(`endpoint-change-nothing-${ID}`)).toBeInTheDocument();
    expect(writes('PATCH', `${PATH}/${ID}`)).toEqual([]);
  });

  it('should remove only after the confirmation, and render a refused removal', async () => {
    overrides[`DELETE ${PATH}/${ID}`] = refusal('wsp_endpoint_unknown', 'No such editor endpoint.');
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-remove-${ID}`));
    expect(screen.getByTestId(`endpoint-confirm-remove-${ID}`)).toBeInTheDocument();
    expect(writes('DELETE', `${PATH}/${ID}`)).toEqual([]);
    await userEvent.click(screen.getByTestId(`endpoint-remove-cancel-${ID}`));
    expect(screen.queryByTestId(`endpoint-confirm-remove-${ID}`)).toBeNull();
    expect(writes('DELETE', `${PATH}/${ID}`)).toEqual([]);

    await userEvent.click(screen.getByTestId(`endpoint-remove-${ID}`));
    await userEvent.click(screen.getByTestId(`endpoint-remove-confirm-${ID}`));
    const notice = await screen.findByTestId(`endpoint-failure-${ID}`);
    expect(within(notice).getByTestId('failure-code').textContent).toBe('wsp_endpoint_unknown');
    expect(writes('DELETE', `${PATH}/${ID}`)).toHaveLength(1);

    overrides[`DELETE ${PATH}/${ID}`] = {
      status: 200,
      body: { id: ID, removed_at: '2026-11-20T16:00:00+00:00' },
    };
    overrides[`GET ${PATH}`] = { status: 200, body: { endpoints: [] } };
    await userEvent.click(screen.getByTestId(`endpoint-remove-confirm-${ID}`));
    expect(await screen.findByTestId('endpoints-empty')).toBeInTheDocument();
    expect(writes('DELETE', `${PATH}/${ID}`)).toHaveLength(2);
  });

  it('should queue the protocol test, show its run and open it on the WSP run screen', async () => {
    const run = 'wsp_00000000aaaa';
    overrides[`POST ${PATH}/${ID}/protocol-test`] = {
      status: 201,
      body: { run_id: run, status: 'queued' },
    };
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-test-${ID}`));
    const queued = await screen.findByTestId(`endpoint-test-queued-${ID}`);
    expect(queued.textContent).toContain(run);
    expect(writes('POST', `${PATH}/${ID}/protocol-test`)).toEqual([
      { method: 'POST', path: `${PATH}/${ID}/protocol-test`, body: {} },
    ]);
    await userEvent.click(screen.getByTestId(`endpoint-test-open-${ID}`));
    await waitFor(() => expect(window.location.hash).toBe(`#/wsp/runs/${run}`));
  });

  it('should render a refused protocol test with its code', async () => {
    overrides[`POST ${PATH}/${ID}/protocol-test`] = {
      status: 503,
      body: {
        error: { code: 'provider_unavailable', message: 'miwsp_engine is off.', details: {} },
      },
    };
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-test-${ID}`));
    const notice = await screen.findByTestId(`endpoint-failure-${ID}`);
    expect(within(notice).getByTestId('failure-code').textContent).toBe('provider_unavailable');
  });
});

// -------------------------------------------------------- an agent's proposal in Approvals

const PROPOSAL: Approval = {
  approval_id: 'apr_endpoint',
  action: 'wsp_endpoint_register',
  status: 'pending',
  summary: {
    route: 'POST /api/v1/wsp/endpoints',
    endpoint_id: ID,
    model: 'gemma-4-26b',
    base_url: WSP_ENDPOINT.base_url,
    action_protocol: 'structured',
    kind: 'openai_compatible',
    serving: 'miserver',
    locality: 'local',
    context_window_tokens: 32768,
    output_cap_field: 'max_completion_tokens',
    key_and_prices: 'none: an agent cannot send an API key or set prices',
  },
  requested_by: 'agent:forge-mcp',
  decided_by: null,
  reason: null,
  result: null,
  error: null,
  created_at: '2026-11-20T15:00:00Z',
  decided_at: null,
  expires_at: '2026-11-21T15:00:00Z',
  payload: { body: { id: ID } },
};

describe('[Editor endpoints] an agent proposal in Approvals', () => {
  it('should name the action and show what will be registered, id first', () => {
    render(inTheme(<ApprovalCard approval={PROPOSAL} />));
    expect(screen.getByText('Register a local editor endpoint')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Approve and register endpoint' })).toBeVisible();
    const summary = screen.getByTestId('approval-summary-apr_endpoint');
    const labels = [...summary.querySelectorAll(':scope > div > dt')].map((dt) => dt.textContent);
    expect(labels.slice(0, 4)).toEqual(['Endpoint', 'Model', 'Base URL', 'Action protocol']);
    for (const text of [ID, 'gemma-4-26b', WSP_ENDPOINT.base_url, 'structured']) {
      expect(summary.textContent).toContain(text);
    }
  });

  it('should open Editor endpoints from an executed proposal in History', async () => {
    function Where(): React.ReactElement {
      return <span data-testid="where">{useNavigation().screen}</span>;
    }
    render(
      inTheme(
        <NavigationProvider initialScreen="approvals">
          <ApprovalHistory
            rows={[{ ...PROPOSAL, status: 'executed', result: { ...WSP_ENDPOINT } }]}
          />
          <Where />
        </NavigationProvider>,
      ),
    );
    await userEvent.click(screen.getByTestId('approval-open-endpoints-apr_endpoint'));
    expect(screen.getByTestId('where')).toHaveTextContent('wsp-endpoints');
  });
});
