/**
 * The WSP runs screens, through the real App and its nav (fix 2026-11-20).
 *
 * Production, 2026-11-20: the operator looked in Runs for the WSP runs an agent had started and
 * found none (nothing on the frontend read `/wsp/`), and Approvals' "Open run" link for an
 * executed WSP session opened `#/runs/wsp_…`, the TRAINING run view, which read "This run's
 * record was refused - run_not_found". Each test here goes through the shell, so deleting a
 * route row, its nav entry or a link's routing is red here, not only in a component test.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { NavigationProvider, useNavigation } from '@/contexts/NavigationContext';
import { ApprovalHistory } from '@/components/approvals/ApprovalHistory';
import type { Approval } from '@/api/approvals';
import { WSP_RUN_DOCUMENT, WSP_RUN_ID, answer } from '@/testing/stubBackend';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';
import { WspRunPage } from './WspRunPage';

interface Seen {
  readonly method: string;
  readonly path: string;
  readonly search: string;
}

let calls: Seen[] = [];
let overrides: Record<string, { status: number; body: unknown }> = {};
let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  calls = [];
  overrides = {};
  useApprovalsStore.setState(INITIAL_APPROVALS_STATE);
  window.history.replaceState(null, '', '/');
  originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
    const method = (init?.method ?? 'GET').toUpperCase();
    calls.push({ method, path: url.pathname, search: url.search });
    const { status, body } = overrides[url.pathname] ?? answer(method, url.pathname);
    return Promise.resolve(
      new Response(JSON.stringify(body), {
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

function heading(): string {
  return screen.getByTestId('shell-page-head').querySelector('h1')?.textContent ?? '';
}

describe('[WSP runs] the list, reached through the nav', () => {
  it('should call the WSP list route once and render the run with why it stopped', async () => {
    render(inTheme(<App />));
    await userEvent.click(
      within(screen.getByRole('navigation', { name: 'Screens' })).getByRole('button', {
        name: 'WSP runs',
      }),
    );
    const row = await screen.findByTestId(`wsp-run-row-${WSP_RUN_ID}`);
    expect(heading()).toBe('WSP runs');
    expect(window.location.hash).toBe('#/wsp/runs');
    const listed = calls.filter((c) => c.path === '/api/v1/wsp/runs');
    expect(listed).toEqual([{ method: 'GET', path: '/api/v1/wsp/runs', search: '?limit=200' }]);
    // What the operator could not see: the cause, beside the stop reason and the outcome.
    expect(within(row).getByTestId('wsp-run-stop')).toHaveTextContent(
      'stopped: gateway_refused · outcome: no_change',
    );
    expect(within(row).getByTestId('wsp-run-cause')).toHaveTextContent('editor_context_exceeded');
    expect(within(row).getByTestId('wsp-run-provider')).toHaveTextContent('weightspace 0.3.0');
    expect(within(row).getByTestId('wsp-run-kind')).toHaveTextContent('repair · session');
    expect(within(row).getByTestId('wsp-run-usage')).toHaveTextContent(
      '1 request · 39,000 tokens (7,000 in, 32,000 out)',
    );
    expect(within(row).getByTestId('wsp-run-forward')).toHaveTextContent('0 forward passes');
    expect(within(row).getByTestId('wsp-run-caller-ref')).toHaveTextContent('probe-17');
    expect(within(row).getByTestId('wsp-run-finished')).toHaveTextContent(/^finished /);
    // Read only: nothing but GETs.
    expect(calls.every((c) => c.method === 'GET')).toBe(true);
  });

  it('should open one run, with its document and its transcript, and keep WSP runs current', async () => {
    render(inTheme(<App initialScreen="wsp-runs" />));
    await userEvent.click(await screen.findByTestId(`wsp-run-row-${WSP_RUN_ID}`));
    await screen.findByTestId('wsp-run-summary');
    expect(heading()).toBe(WSP_RUN_ID);
    expect(window.location.hash).toBe(`#/wsp/runs/${WSP_RUN_ID}`);
    // The run, its progress and its calls; then, the record being terminal, the transcript once.
    await screen.findAllByTestId('wsp-transcript-entry');
    expect(calls.filter((c) => c.path.startsWith(`/api/v1/wsp/runs/${WSP_RUN_ID}`))).toEqual([
      { method: 'GET', path: `/api/v1/wsp/runs/${WSP_RUN_ID}`, search: '' },
      { method: 'GET', path: `/api/v1/wsp/runs/${WSP_RUN_ID}/progress`, search: '' },
      { method: 'GET', path: `/api/v1/wsp/runs/${WSP_RUN_ID}/calls`, search: '' },
      { method: 'GET', path: `/api/v1/wsp/runs/${WSP_RUN_ID}/transcript`, search: '' },
    ]);
    expect(screen.getByTestId('wsp-run-failure')).toHaveTextContent('editor_context_exceeded');
    expect(screen.getByTestId('wsp-run-no-candidate')).toHaveTextContent(
      'left the weights as they were',
    );
    expect(await screen.findAllByTestId('wsp-transcript-entry')).toHaveLength(1);
    const nav = screen.getByRole('navigation', { name: 'Screens' });
    expect(within(nav).getByRole('button', { name: 'WSP runs' })).toHaveAttribute(
      'aria-current',
      'page',
    );
  });
});

describe('[WSP runs] a WSP id never reaches the training run view', () => {
  it('should send #/runs/wsp_… to the WSP run screen without asking the training routes', async () => {
    window.history.replaceState(null, '', `/#/runs/${WSP_RUN_ID}`);
    render(inTheme(<App />));
    await screen.findByTestId('wsp-run-summary');
    expect(heading()).toBe(WSP_RUN_ID);
    expect(window.location.hash).toBe(`#/wsp/runs/${WSP_RUN_ID}`);
    expect(calls.some((c) => c.path.startsWith(`/api/v1/runs/${WSP_RUN_ID}`))).toBe(false);
    expect(document.body.textContent ?? '').not.toContain('was refused');
  });

  it('should say an unknown id is no training run, and point to WSP runs, not a raw refusal', async () => {
    // The production answer for an id no training run has (read 2026-11-20 from the route).
    overrides['/api/v1/runs/run-gone'] = {
      status: 404,
      body: {
        error: {
          code: 'run_not_found',
          message: "No run with id 'run-gone'. Check the identifier.",
          details: { run_id: 'run-gone' },
        },
      },
    };
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-gone' }} />));
    const notice = await screen.findByTestId('run-not-training');
    expect(notice).toHaveTextContent('No training run is named run-gone');
    expect(notice).toHaveTextContent('run_not_found');
    expect(document.body.textContent ?? '').not.toContain('was refused');
    await userEvent.click(screen.getByTestId('run-not-training-wsp'));
    expect(heading()).toBe('WSP runs');
  });

  it('should open a WSP id pasted into Runs on the WSP run screen', async () => {
    render(inTheme(<App initialScreen="runs" />));
    const box = await screen.findByTestId('run-id-input');
    fireEvent.change(box, { target: { value: WSP_RUN_ID } });
    fireEvent.submit(box.closest('form')!);
    await screen.findByTestId('wsp-run-summary');
    expect(heading()).toBe(WSP_RUN_ID);
    expect(calls.some((c) => c.path.startsWith(`/api/v1/runs/${WSP_RUN_ID}`))).toBe(false);
  });

  it('should open an executed WSP session from Approvals history on the WSP run screen', async () => {
    const approval = {
      approval_id: 'appr-wsp',
      action: 'wsp_session',
      status: 'executed',
      summary: {},
      requested_by: 'agent:forge-mcp',
      decided_by: 'operator',
      reason: null,
      result: { run_id: WSP_RUN_ID, status: 'queued' },
      error: null,
      created_at: '2026-11-20T15:00:00Z',
      decided_at: '2026-11-20T15:00:01Z',
      expires_at: '2026-11-21T15:00:00Z',
    } as unknown as Approval;
    function Where(): React.ReactNode {
      const navigation = useNavigation();
      return (
        <span data-testid="where">
          {navigation.screen}:{navigation.params['runId'] ?? ''}
        </span>
      );
    }
    render(
      <NavigationProvider initialScreen="approvals">
        <ApprovalHistory rows={[approval]} />
        <Where />
      </NavigationProvider>,
    );
    const link = screen.getByTestId('approval-open-run-appr-wsp');
    expect(link).toHaveTextContent(`Open WSP run ${WSP_RUN_ID}`);
    await userEvent.click(link);
    expect(screen.getByTestId('where')).toHaveTextContent(`wsp-run:${WSP_RUN_ID}`);
  });
});

describe('[WSP run] the transcript is data', () => {
  function mount(entries: unknown[], document: unknown = WSP_RUN_DOCUMENT): void {
    render(
      <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
        <WspRunPage
          fetchRun={() =>
            Promise.resolve({ ok: true, value: document as typeof WSP_RUN_DOCUMENT as never })
          }
          fetchTranscript={() =>
            Promise.resolve({
              ok: true,
              value: {
                run_id: WSP_RUN_ID,
                transcript_sha256: 'c'.repeat(64),
                entries,
                partial: true,
              },
            })
          }
        />
      </NavigationProvider>,
    );
  }

  it('should render hostile content as text, never as markup', async () => {
    mount([
      { role: 'assistant', content: '<img src=x onerror="window.pwned=1"><script>x()</script>' },
      { role: 'tool', content: '{"html": "<b>bold</b>"}' },
      { raw: '<iframe src="//evil">' },
    ]);
    const entries = await screen.findAllByTestId('wsp-transcript-entry');
    expect(entries).toHaveLength(3);
    const list = screen.getByTestId('wsp-transcript');
    expect(list.querySelector('img, script, iframe, b')).toBeNull();
    expect(entries[0]).toHaveTextContent(
      '<img src=x onerror="window.pwned=1"><script>x()</script>',
    );
    expect(entries[1]).toHaveTextContent('"html": "<b>bold</b>"');
    expect(entries[2]).toHaveTextContent('<iframe src="//evil">');
    expect((window as unknown as { pwned?: number }).pwned).toBeUndefined();
  });

  it("should show the editor's reasoning apart from its content, as text (2026-11-21)", async () => {
    mount([
      {
        role: 'assistant',
        content: '{"tool":"submit","arguments":{"answer":"done"}}',
        reasoning: { text: 'I think <b>the bias</b> moved.', source: 'reasoning_content' },
      },
      { role: 'assistant', content: 'no reasoning here' },
    ]);
    const [thought, plain] = await screen.findAllByTestId('wsp-transcript-entry');
    const reasoning = within(thought as HTMLElement).getByTestId('wsp-transcript-reasoning');
    expect(reasoning).toHaveTextContent('reasoning_content; recorded, never run');
    expect(reasoning).toHaveTextContent('I think <b>the bias</b> moved.');
    expect(reasoning.querySelector('b')).toBeNull();
    expect(thought?.querySelector('pre')?.textContent).not.toContain('the bias');
    expect(within(plain as HTMLElement).queryByTestId('wsp-transcript-reasoning')).toBeNull();
  });

  it('should show the thinking frozen in the run request', async () => {
    mount([], {
      ...WSP_RUN_DOCUMENT,
      request: { ...WSP_RUN_DOCUMENT.request, thinking: 'on', thinking_tokens: 4096 },
    });
    const value = await screen.findByTestId('wsp-run-thinking');
    expect(value).toHaveTextContent('on, 4,096 reasoning tokens per turn');
  });

  it('should show an output cap frozen off', async () => {
    mount([], { ...WSP_RUN_DOCUMENT, request: { ...WSP_RUN_DOCUMENT.request, output_cap: 'off' } });
    expect(await screen.findByTestId('wsp-run-output-cap')).toHaveTextContent(
      'off (no per-turn cap named)',
    );
  });

  it('should show the paging frozen in the run request (2026-11-21)', async () => {
    mount([], { ...WSP_RUN_DOCUMENT, request: { ...WSP_RUN_DOCUMENT.request, paging: 'off' } });
    expect(await screen.findByTestId('wsp-run-paging')).toHaveTextContent(
      'off (every result kept in the conversation)',
    );
  });

  it('should show an engine run with no paging frozen as paging on', async () => {
    mount([], {
      ...WSP_RUN_DOCUMENT,
      request: {
        ...WSP_RUN_DOCUMENT.request,
        provider: { id: 'miwsp_engine', version: '0.1.0' },
      },
    });
    expect(await screen.findByTestId('wsp-run-paging')).toHaveTextContent(
      'on (bulky results paged out; experiment ledger kept)',
    );
  });

  it('should say a weightspace run does not page', async () => {
    mount([]);
    expect(await screen.findByTestId('wsp-run-paging')).toHaveTextContent(
      'not applicable (this provider does not page)',
    );
  });

  it("should say an unset thinking is the server's default", async () => {
    mount([]);
    expect(await screen.findByTestId('wsp-run-thinking')).toHaveTextContent(
      "not set (the server's default)",
    );
  });

  it('should lay out the engine JSON turns and show a malformed one as written', async () => {
    mount([
      { role: 'assistant', content: '{"action":"python","code":"print(W.sum())"}' },
      { role: 'assistant', content: '{"action": "python", "code": ' },
    ]);
    const [good, bad] = await screen.findAllByTestId('wsp-transcript-entry');
    expect(good).toHaveTextContent('(JSON, formatted)');
    expect(good?.querySelector('pre')?.textContent).toBe(
      '{\n  "action": "python",\n  "code": "print(W.sum())"\n}',
    );
    expect(bad).not.toHaveTextContent('(JSON, formatted)');
    expect(bad?.querySelector('pre')?.textContent).toBe('{"action": "python", "code": ');
  });

  it('should show the candidate and its verdict when the run has them', async () => {
    mount([], {
      ...WSP_RUN_DOCUMENT,
      outcome: 'candidate',
      failure: null,
      candidate_id: 'wspc_0123456789ab',
      patch_sha256: 'e'.repeat(64),
      content_sha256: 'f'.repeat(64),
      verdict: { value: 'improved', verdict_id: 'v1' },
    });
    expect(await screen.findByTestId('wsp-run-candidate')).toHaveTextContent('wspc_0123456789ab');
    expect(screen.getByTestId('wsp-run-verdict')).toHaveTextContent('Verdict: improved');
    expect(screen.queryByTestId('wsp-run-failure')).toBeNull();
  });

  it('should keep the run on screen when only the transcript cannot be read', async () => {
    render(
      <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
        <WspRunPage
          fetchRun={() => Promise.resolve({ ok: true, value: WSP_RUN_DOCUMENT as never })}
          fetchTranscript={() =>
            Promise.resolve({
              ok: false,
              failure: {
                kind: 'malformed',
                status: 502,
                reason: 'body is not a JSON object',
                body: '<html>Bad gateway</html>',
              },
            })
          }
        />
      </NavigationProvider>,
    );
    await waitFor(() => expect(screen.getByTestId('wsp-run-summary')).toBeTruthy());
    // The transcript is read after the record, on a later tick: waited for, not raced (it failed
    // under a loaded full run on 2026-11-20 while passing alone).
    await waitFor(() =>
      expect(screen.getByTestId('wsp-run-transcript')).toHaveTextContent(
        "This run's transcript returned something unreadable",
      ),
    );
  });
});
