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
import {
  EMPTY_ENDPOINT_DRAFT,
  ifReadFailsRows,
  registrationBody,
  thinkingText,
  windowFacts,
} from './WspEndpointsPage';

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

describe('[Editor endpoints] the window and the output ceiling (feature 019)', () => {
  it('should show the window as the backend serves it, and the ceiling', async () => {
    await openThroughTheNav();
    const facts = screen.getByTestId(`endpoint-facts-${ID}`).textContent ?? '';
    expect(facts).toContain('32,768 tokens (stated)');
    expect(facts).toMatch(/Read at registration32,768 tokens, /);
    expect(facts).toContain(`16,384 tokens, run ${WSP_ENDPOINT.latest_run_window.run_id}`);
    expect(facts).toContain('Output ceiling4,096 tokens');
    // The false text of 018 (FR-019.16's pre-existing defect) is gone.
    expect(document.body.textContent).not.toContain('the window the server reports');
    expect(screen.getByTestId(`endpoint-window-mismatch-${ID}`).textContent).toContain(
      '16,384 tokens at the latest run',
    );
  });

  it('should open the latest run that read the window', async () => {
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-latest-run-${ID}`));
    await waitFor(() =>
      expect(window.location.hash).toBe(`#/wsp/runs/${WSP_ENDPOINT.latest_run_window.run_id}`),
    );
  });

  it('should say runs assume 8,192 when nothing is stated or read, and warn of no mismatch', async () => {
    overrides[`GET ${PATH}`] = {
      status: 200,
      body: {
        endpoints: [
          {
            ...WSP_ENDPOINT,
            context_window_tokens: null,
            output_ceiling_tokens: null,
            output_ceiling_effective: 32000,
            registration_window: { tokens: null, read_at: '2026-11-20T15:00:01+00:00' },
            latest_run_window: null,
            window_now: { tokens: 8192, source: 'assumed' },
            window_warnings: [],
          },
        ],
      },
    };
    await openThroughTheNav();
    const facts = screen.getByTestId(`endpoint-facts-${ID}`).textContent ?? '';
    expect(facts).toContain('Context windownot stated');
    expect(facts).toMatch(/Read at registrationnone reported, /);
    expect(facts).toContain('none reported: runs assume 8,192 tokens');
    expect(facts).toContain('Output ceiling32,000 tokens (the default)');
    expect(screen.queryByTestId(`endpoint-window-mismatch-${ID}`)).toBeNull();
    expect(screen.queryByTestId(`endpoint-latest-run-${ID}`)).toBeNull();
  });

  it('should send the ceiling in the registration body once, and not when empty', async () => {
    overrides[`POST ${PATH}`] = { status: 201, body: WSP_ENDPOINT };
    await openThroughTheNav();
    type('endpoint-add-id', 'local-ed');
    type('endpoint-add-model', 'qwen');
    type('endpoint-add-base-url', 'http://10.0.0.5:8000/v1');
    type('endpoint-add-kind', 'openai_compatible');
    type('endpoint-add-output-ceiling', '4096');
    await userEvent.click(screen.getByTestId('endpoint-add-submit'));
    await screen.findByTestId('endpoint-registered');
    expect(writes('POST')).toHaveLength(1);
    expect(writes('POST')[0]?.body).toEqual({
      id: 'local-ed',
      kind: 'openai_compatible',
      base_url: 'http://10.0.0.5:8000/v1',
      model: 'qwen',
      output_ceiling_tokens: 4096,
    });
    expect(
      registrationBody(
        { ...EMPTY_ENDPOINT_DRAFT, kind: 'anthropic', base_url: 'x', model: 'm' },
        '',
      ),
    ).not.toHaveProperty('output_ceiling_tokens');
  });

  it('should PATCH the ceiling alone when changed, and null when cleared', async () => {
    overrides[`PATCH ${PATH}/${ID}`] = { status: 200, body: WSP_ENDPOINT };
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    const field = screen.getByTestId(`endpoint-change-ceiling-${ID}`) as HTMLInputElement;
    expect(field.value).toBe('4096');
    type(`endpoint-change-ceiling-${ID}`, '8192');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    await waitFor(() => expect(writes('PATCH', `${PATH}/${ID}`)).toHaveLength(1));
    expect(writes('PATCH', `${PATH}/${ID}`)[0]?.body).toEqual({ output_ceiling_tokens: 8192 });
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    type(`endpoint-change-ceiling-${ID}`, '');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    await waitFor(() => expect(writes('PATCH', `${PATH}/${ID}`)).toHaveLength(2));
    expect(writes('PATCH', `${PATH}/${ID}`)[1]?.body).toEqual({ output_ceiling_tokens: null });
  });

  it('should keep the window facts from a row an older backend served without them', () => {
    const old = publicEndpoint({ ...WSP_ENDPOINT, registration_window: undefined });
    expect(old.registration_window).toEqual({ tokens: null, read_at: null });
    expect(old.window_now).toEqual({ tokens: 32768, source: 'stated', run_id: null, at: null });
    expect(old.window_if_read_fails).toEqual({
      tokens: 32768,
      source: 'stated',
      no_answer: 'stated',
    });
    // A backend from before D1 sends no `no_answer`: it reads null and the detail keeps one row.
    const before = publicEndpoint({
      ...WSP_ENDPOINT,
      window_if_read_fails: { tokens: 8192, source: 'assumed' },
    });
    expect(before.window_if_read_fails).toEqual({
      tokens: 8192,
      source: 'assumed',
      no_answer: null,
    });
    expect(ifReadFailsRows(before)).toEqual([
      ['If the run cannot read the server', '8,192 tokens (assumed)'],
    ]);
  });

  it("should expect the latest run's reported window next, not the registration read (F-P4-1)", async () => {
    // The live case (2026-11-25, cpu-lfm25-1b2): registration read 32,768, latest run 32,768.
    overrides[`GET ${PATH}`] = {
      status: 200,
      body: {
        endpoints: [
          {
            ...WSP_ENDPOINT,
            context_window_tokens: null,
            registration_window: { tokens: 32768, read_at: '2026-11-25T20:19:26Z' },
            latest_run_window: {
              tokens: 32768,
              source: 'reported',
              run_id: 'wsp_e3ca0fefe0a0',
              at: '2026-11-25T20:30:00Z',
            },
            window_now: {
              tokens: 32768,
              source: 'reported',
              run_id: 'wsp_e3ca0fefe0a0',
              at: '2026-11-25T20:30:00Z',
            },
            window_if_read_fails: { tokens: 8192, source: 'assumed' },
            window_warnings: [],
          },
        ],
      },
    };
    await openThroughTheNav();
    const facts = screen.getByTestId(`endpoint-facts-${ID}`).textContent ?? '';
    expect(facts).toContain(
      'Next run expects32,768 tokens, as the server reported for run wsp_e3ca0fefe0a0',
    );
    expect(facts).toContain('each run reads it again when it starts');
    expect(facts).toContain('If the run cannot read the server8,192 tokens (assumed)');
    expect(facts).not.toContain('none reported: runs assume');
  });

  it('should say a miServer endpoint stating no window waits for its model, and assume nothing (2026-11-27)', () => {
    // Run wsp_4fbf5a18758e: pushed before miServer finished loading, it assumed 8,192 and failed.
    const endpoint = publicEndpoint({
      ...WSP_ENDPOINT,
      serving: 'miserver',
      context_window_tokens: null,
      latest_run_window: null,
      // The backend sends tokens 0 ("none": the contract types tokens an integer).
      window_now: { tokens: 0, source: 'waits', run_id: null, at: null },
      window_if_read_fails: { tokens: 0, source: 'waits' },
    });
    expect(endpoint.window_now).toEqual({ tokens: null, source: 'waits', run_id: null, at: null });
    expect(endpoint.window_if_read_fails).toEqual({
      tokens: null,
      source: 'waits',
      no_answer: null,
    });
    const facts = Object.fromEntries(windowFacts(endpoint));
    expect(facts['Next run expects']).toBe(
      'none reported yet: each run waits in the queue until miServer reports a window (a model is loaded); none is assumed',
    );
    expect(facts['If the run cannot read the server']).toBe(
      'the run waits in the queue for miServer to report a window, then fails editor_not_resident; no window is assumed',
    );
    expect(Object.values(facts).join(' ')).not.toContain('8,192');
  });

  it('should give a generic endpoint stating no window both cases: answers without one, and no answer (D1, 2026-11-27)', async () => {
    overrides[`GET ${PATH}`] = {
      status: 200,
      body: {
        endpoints: [
          {
            ...WSP_ENDPOINT,
            serving: 'generic',
            context_window_tokens: null,
            latest_run_window: null,
            window_now: { tokens: 8192, source: 'assumed', run_id: null, at: null },
            window_if_read_fails: { tokens: 8192, source: 'assumed', no_answer: 'waits' },
            window_warnings: [],
          },
        ],
      },
    };
    await openThroughTheNav();
    const facts = screen.getByTestId(`endpoint-facts-${ID}`).textContent ?? '';
    expect(facts).toContain('If the server answers without a window8,192 tokens (assumed)');
    expect(facts).toContain(
      'If the server does not answerthe run waits in the queue until the server answers, and fails editor_not_resident if it has not answered within the wait (15 minutes by default); no window is assumed',
    );
    expect(facts).not.toContain('If the run cannot read the server');
  });

  it('should keep one failed-read row when the no-answer case is the same (stated, miServer, hosted)', () => {
    for (const [fails, text] of [
      [{ tokens: 65536, source: 'stated', no_answer: 'stated' }, '65,536 tokens (stated)'],
      [
        { tokens: 0, source: 'waits', no_answer: 'waits' },
        'the run waits in the queue for miServer to report a window, then fails editor_not_resident; no window is assumed',
      ],
      [{ tokens: 8192, source: 'assumed', no_answer: 'not_read' }, '8,192 tokens (assumed)'],
    ] as const) {
      const endpoint = publicEndpoint({ ...WSP_ENDPOINT, window_if_read_fails: fails });
      expect(ifReadFailsRows(endpoint)).toEqual([['If the run cannot read the server', text]]);
    }
  });

  it('should say stated for both when a window is stated', () => {
    const endpoint = publicEndpoint({
      ...WSP_ENDPOINT,
      context_window_tokens: 65536,
      window_now: { tokens: 65536, source: 'stated', run_id: null, at: null },
      window_if_read_fails: { tokens: 65536, source: 'stated' },
    });
    const facts = Object.fromEntries(windowFacts(endpoint));
    expect(facts['Next run expects']).toBe('65,536 tokens (stated)');
    expect(facts['If the run cannot read the server']).toBe('65,536 tokens (stated)');
  });
});

describe('[Editor endpoints] the protocol test (feature 019 P2)', () => {
  const listed = (protocolTest: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
    status: 200,
    body: {
      endpoints: [
        {
          ...WSP_ENDPOINT,
          ...extra,
          protocol_test: { ...WSP_ENDPOINT.protocol_test, ...protocolTest },
        },
      ],
    },
  });

  it('should say passed with the run, the time and the tested values, and not required on miServer', async () => {
    await openThroughTheNav();
    const facts = screen.getByTestId(`endpoint-facts-${ID}`).textContent ?? '';
    expect(facts).toMatch(/Protocol testpassed, run wsp_0000000000aa, /);
    expect(facts).toContain('(tested structured on gemma-4-26b at http://miserver-backend');
    expect(facts).toContain('not required on a miServer endpoint');
  });

  it('should open the latest protocol test run', async () => {
    await openThroughTheNav();
    await userEvent.click(screen.getByTestId(`endpoint-protocol-test-run-${ID}`));
    await waitFor(() => expect(window.location.hash).toBe('#/wsp/runs/wsp_0000000000aa'));
  });

  it('should say failed on a generic endpoint, and that a pass is required', async () => {
    overrides[`GET ${PATH}`] = listed(
      { result: 'failed', current: false, required: true },
      { serving: 'generic' },
    );
    await openThroughTheNav();
    const facts = screen.getByTestId(`endpoint-facts-${ID}`).textContent ?? '';
    expect(facts).toMatch(/Protocol testfailed, run wsp_0000000000aa/);
    expect(facts).toContain('required before any run');
  });

  it('should say cleared when the protocol changed since a pass', async () => {
    overrides[`GET ${PATH}`] = listed(
      { current: false, cleared: true, required: true },
      { serving: 'generic' },
    );
    await openThroughTheNav();
    expect(screen.getByTestId(`endpoint-facts-${ID}`).textContent).toContain(
      'cleared: the protocol changed since the test',
    );
  });

  it('should say not tested, with no run link, when no test ran', async () => {
    overrides[`GET ${PATH}`] = listed(
      {
        result: null,
        run_id: null,
        at: null,
        tested: { base_url: null, model: null, action_protocol: null },
        current: false,
        required: true,
      },
      { serving: 'generic' },
    );
    await openThroughTheNav();
    expect(screen.getByTestId(`endpoint-facts-${ID}`).textContent).toMatch(
      /Protocol testnot tested; required before any run/,
    );
    expect(screen.queryByTestId(`endpoint-protocol-test-run-${ID}`)).toBeNull();
  });
});

// ---------------------------------------------------------------- hosted models (feature 020)

/** A hosted Anthropic row as the backend serves it after 020 (window, four prices, an effort). */
const HOSTED = {
  ...WSP_ENDPOINT,
  id: 'claude-ed',
  kind: 'anthropic',
  base_url: 'https://api.anthropic.com',
  model: 'claude-editor',
  locality: 'external',
  serving: 'generic',
  output_cap_field: 'max_tokens',
  price_in_per_mtok: '3.000000',
  price_out_per_mtok: '15.000000',
  price_cache_write_per_mtok: '3.750000',
  price_cache_read_per_mtok: '0.300000',
  has_key: true,
  key_fingerprint: 'sha256:ffff',
  action_protocol: 'tool_calls',
  context_window_tokens: 200000,
  thinking: 'medium',
  history_mode: 'append_only',
  paging: 'on',
  latest_run_window: null,
  window_now: { tokens: 200000, source: 'stated' },
  window_warnings: [],
  runs_refused: null,
} as const;

/** A control's label and hint, as the operator reads them (`Field` ties both to the control). */
function labelAndHint(testId: string): { label: string; hint: string } {
  const control = screen.getByTestId(testId);
  const label = document.querySelector(`label[for="${control.id}"]`)?.textContent ?? '';
  const hintId = control.getAttribute('aria-describedby') ?? '';
  return { label, hint: document.getElementById(hintId)?.textContent ?? '' };
}

const optionsOf = (testId: string): string[] =>
  Array.from((screen.getByTestId(testId) as HTMLSelectElement).options)
    .map((option) => option.value)
    .filter((value) => value !== '');

describe('[Editor endpoints] hosted models (feature 020, U-1)', () => {
  it('should send the history mode and the effort once in the body, offering an external Anthropic host its efforts', async () => {
    overrides[`POST ${PATH}`] = { status: 201, body: HOSTED };
    await openThroughTheNav();
    type('endpoint-add-id', 'claude-ed');
    type('endpoint-add-model', 'claude-editor');
    type('endpoint-add-base-url', 'https://api.anthropic.com');
    type('endpoint-add-kind', 'anthropic');
    // The efforts in place of on/off, and the history mode's two values (FR-020.22).
    expect(optionsOf('endpoint-add-thinking')).toEqual(['default', 'low', 'medium', 'high']);
    expect(optionsOf('endpoint-add-history-mode')).toEqual(['paging', 'append_only']);
    type('endpoint-add-history-mode', 'append_only');
    type('endpoint-add-thinking', 'medium');
    type('endpoint-add-context-window', '200000');
    type('endpoint-add-price-in-per-mtok', '3');
    type('endpoint-add-price-out-per-mtok', '15');
    type('endpoint-add-price-cache-write-per-mtok', '3.75');
    type('endpoint-add-price-cache-read-per-mtok', '0.3');
    type('endpoint-add-api-key', KEY);
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
          context_window_tokens: 200000,
          thinking: 'medium',
          history_mode: 'append_only',
          price_in_per_mtok: '3',
          price_out_per_mtok: '15',
          price_cache_write_per_mtok: '3.75',
          price_cache_read_per_mtok: '0.3',
          api_key: KEY,
        },
      },
    ]);
  });

  it("should send a local endpoint's append_only once, and no history mode when left empty", async () => {
    overrides[`POST ${PATH}`] = { status: 201, body: WSP_ENDPOINT };
    await openThroughTheNav();
    type('endpoint-add-id', 'local-ed');
    type('endpoint-add-model', 'qwen');
    type('endpoint-add-base-url', 'http://10.0.0.5:8000/v1');
    type('endpoint-add-kind', 'openai_compatible');
    // A local host keeps 018's choices; the efforts are an external endpoint's.
    expect(optionsOf('endpoint-add-thinking')).toEqual(['off', 'on', 'default']);
    type('endpoint-add-history-mode', 'append_only');
    await userEvent.click(screen.getByTestId('endpoint-add-submit'));
    await screen.findByTestId('endpoint-registered');
    expect(writes('POST').map((call) => call.body)).toEqual([
      {
        id: 'local-ed',
        kind: 'openai_compatible',
        base_url: 'http://10.0.0.5:8000/v1',
        model: 'qwen',
        history_mode: 'append_only',
      },
    ]);
    expect(
      registrationBody(
        { ...EMPTY_ENDPOINT_DRAFT, kind: 'anthropic', base_url: 'x', model: 'm' },
        '',
      ),
    ).not.toHaveProperty('history_mode');
  });

  it('should mark the window and both cache prices required for an external Anthropic host, and explain the refused paging', async () => {
    await openThroughTheNav();
    type('endpoint-add-base-url', 'https://api.anthropic.com');
    type('endpoint-add-kind', 'anthropic');
    const window = labelAndHint('endpoint-add-context-window');
    expect(window.label).toBe('Context window (tokens, required)');
    expect(window.hint).toContain('Required for an external endpoint');
    expect(window.hint).toContain('where the session hands off');
    for (const name of ['cache-write', 'cache-read']) {
      const price = labelAndHint(`endpoint-add-price-${name}-per-mtok`);
      expect(price.label).toContain('required');
      expect(price.hint).toContain('Required for an external Anthropic endpoint');
    }
    const history = labelAndHint('endpoint-add-history-mode');
    expect(history.hint).toContain('Empty: append_only');
    expect(history.hint).toContain('cannot be paging');
    expect(screen.getByTestId('endpoint-add-locality').textContent).toContain('both cache prices');

    // An external OpenAI-compatible host: the window still required, the cache prices not.
    type('endpoint-add-kind', 'openai_compatible');
    expect(labelAndHint('endpoint-add-context-window').label).toContain('required');
    expect(labelAndHint('endpoint-add-price-cache-read-per-mtok').label).not.toContain('required');
    expect(optionsOf('endpoint-add-thinking')).toEqual(['off', 'default', 'low', 'medium', 'high']);
    expect(labelAndHint('endpoint-add-history-mode').hint).not.toContain('cannot be paging');

    // A local host: 019's window hint, nothing required, paging its default.
    type('endpoint-add-base-url', 'http://10.0.0.5:8000/v1');
    const local = labelAndHint('endpoint-add-context-window');
    expect(local.label).toBe('Context window (tokens)');
    expect(local.hint).toContain('Empty: each run reads the server');
    expect(labelAndHint('endpoint-add-history-mode').hint).toContain('Empty: paging');
  });

  it('should show both runs_refused states with their codes and messages, and none when runs start', async () => {
    const window = {
      code: 'endpoint_window_required',
      message:
        "Set the endpoint's context window to run it: a hosted endpoint must state its window.",
    };
    const prices = {
      code: 'wsp_endpoint_cache_prices_required',
      message:
        'This endpoint has no cache prices, which are sealed with its key: register it again with both cache prices to run it.',
    };
    overrides[`GET ${PATH}`] = {
      status: 200,
      body: {
        endpoints: [
          { ...HOSTED, id: 'old-a', context_window_tokens: null, runs_refused: window },
          { ...HOSTED, id: 'old-b', price_cache_read_per_mtok: null, runs_refused: prices },
          WSP_ENDPOINT,
        ],
      },
    };
    render(inTheme(<App />));
    await userEvent.click(
      within(screen.getByRole('navigation', { name: 'Screens' })).getByRole('button', {
        name: 'Editor endpoints',
      }),
    );
    const a = await screen.findByTestId('endpoint-runs-refused-old-a');
    expect(a.textContent).toContain('Window required: set it to run');
    expect(a.textContent).toContain('endpoint_window_required');
    expect(a.textContent).toContain(window.message);
    const b = screen.getByTestId('endpoint-runs-refused-old-b');
    expect(b.textContent).toContain('Cache prices required: register this endpoint again');
    expect(b.textContent).toContain('wsp_endpoint_cache_prices_required');
    expect(b.textContent).toContain(prices.message);
    expect(screen.queryByTestId(`endpoint-runs-refused-${ID}`)).toBeNull();
  });

  it("should show a hosted row's history mode and effort in the detail, and its paging as not applicable", async () => {
    overrides[`GET ${PATH}`] = { status: 200, body: { endpoints: [HOSTED] } };
    render(inTheme(<App />));
    await userEvent.click(
      within(screen.getByRole('navigation', { name: 'Screens' })).getByRole('button', {
        name: 'Editor endpoints',
      }),
    );
    const facts = (await screen.findByTestId('endpoint-facts-claude-ed')).textContent ?? '';
    expect(facts).toContain('History modeappend_only: nothing already sent changes');
    expect(facts).toContain('Thinkingmedium: adaptive thinking at effort medium');
    expect(facts).toContain('Pagingnot applicable: the history mode is append_only');
    // A local row reads as before 020.
    expect(thinkingText('default')).toBe("default: nothing is sent; the server's template decides");
    expect(thinkingText('default', { kind: 'anthropic', locality: 'external' })).toContain(
      "adaptive thinking at the model's own effort",
    );
    expect(thinkingText('high', { kind: 'openai_compatible', locality: 'external' })).toBe(
      'high: reasoning_effort high is sent with each request',
    );
  });

  it('should PATCH the history mode alone when changed', async () => {
    overrides[`PATCH ${PATH}/${ID}`] = {
      status: 200,
      body: { ...WSP_ENDPOINT, history_mode: 'append_only' },
    };
    await openThroughTheNav();
    expect(screen.getByTestId(`endpoint-facts-${ID}`).textContent).toContain(
      'History modepaging: bulky tool results are replaced by page stubs',
    );
    await userEvent.click(screen.getByTestId(`endpoint-change-open-${ID}`));
    expect(
      (screen.getByTestId(`endpoint-change-history-mode-${ID}`) as HTMLSelectElement).value,
    ).toBe('paging');
    type(`endpoint-change-history-mode-${ID}`, 'append_only');
    await userEvent.click(screen.getByTestId(`endpoint-change-save-${ID}`));
    await waitFor(() => expect(writes('PATCH', `${PATH}/${ID}`)).toHaveLength(1));
    expect(writes('PATCH', `${PATH}/${ID}`)[0]?.body).toEqual({ history_mode: 'append_only' });
  });

  it('should offer a hosted Anthropic row only default and the efforts, and PATCH an effort alone', async () => {
    overrides[`GET ${PATH}`] = { status: 200, body: { endpoints: [HOSTED] } };
    overrides[`PATCH ${PATH}/claude-ed`] = { status: 200, body: { ...HOSTED, thinking: 'high' } };
    render(inTheme(<App />));
    await userEvent.click(
      within(screen.getByRole('navigation', { name: 'Screens' })).getByRole('button', {
        name: 'Editor endpoints',
      }),
    );
    await userEvent.click(await screen.findByTestId('endpoint-change-open-claude-ed'));
    expect(optionsOf('endpoint-change-thinking-claude-ed')).toEqual([
      'default',
      'low',
      'medium',
      'high',
    ]);
    expect(labelAndHint('endpoint-change-window-claude-ed').label).toContain('required');
    expect(labelAndHint('endpoint-change-history-mode-claude-ed').hint).toContain(
      'cannot be paging',
    );
    type('endpoint-change-thinking-claude-ed', 'high');
    await userEvent.click(screen.getByTestId('endpoint-change-save-claude-ed'));
    await waitFor(() => expect(writes('PATCH', `${PATH}/claude-ed`)).toHaveLength(1));
    expect(writes('PATCH', `${PATH}/claude-ed`)[0]?.body).toEqual({ thinking: 'high' });
  });

  it('publicEndpoint should copy history_mode and runs_refused, and read an older row as paging with none', () => {
    const row = publicEndpoint({
      ...HOSTED,
      runs_refused: { code: 'endpoint_window_required', message: 'm', extra: 'dropped' },
    });
    expect(row.history_mode).toBe('append_only');
    expect(row.runs_refused).toEqual({ code: 'endpoint_window_required', message: 'm' });
    const old = publicEndpoint({
      ...WSP_ENDPOINT,
      history_mode: undefined,
      runs_refused: undefined,
    });
    expect(old.history_mode).toBe('paging');
    expect(old.runs_refused).toBeNull();
    expect(publicEndpoint({ ...WSP_ENDPOINT, runs_refused: 'yes' }).runs_refused).toBeNull();
  });
});
