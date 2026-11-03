/**
 * The Approvals screen (FR-012.17, .18; ENH-008 D6, A4).
 *
 * The summaries rendered here are the ones the backend STORED for a real agent request: they
 * come from `e2e/fixtures/approvals.json`, which `backend/tests/integration/test_ui_approval_fixtures.py`
 * generates by sending agent calls through the mounted app and pins - so the screen is tested
 * against what the gate actually writes, not against a hand-written shape. The fake backend
 * adds only state: a decided request changes status, as the real one does.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import fixture from '../../../../e2e/fixtures/approvals.json';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';
import { APPROVE_LABEL } from '@/components/approvals/ApprovalCard';
import { HISTORY_COLUMNS } from '@/components/approvals/ApprovalHistory';
import { formatLocal } from '@/components/common/time';
import { ApprovalsPanel, HISTORY_EMPTY, WAITING_EMPTY } from './ApprovalsPanel';
import type { Approval, ApprovalListing } from '@/api/approvals';

const LISTING = fixture as unknown as ApprovalListing;
const DECIDED_AT = '2026-10-19T12:05:00+00:00';

let calls: { method: string; path: string; body: unknown; headers: Record<string, string> }[] = [];
let rows: Approval[] = [];

function respond(body: unknown, status = 200): Promise<Response> {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  );
}

beforeEach(() => {
  calls = [];
  rows = LISTING.approvals.map((row) => ({ ...row }));
  useApprovalsStore.setState({ ...INITIAL_APPROVALS_STATE, inFlight: new Set<string>() });
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://local');
    const method = (init?.method ?? 'GET').toUpperCase();
    const body = init?.body === undefined ? undefined : JSON.parse(String(init.body));
    calls.push({
      method,
      path: url.pathname,
      body,
      headers: (init?.headers ?? {}) as Record<string, string>,
    });
    const decide = /^\/api\/v1\/approvals\/([^/]+)\/(approve|reject)$/.exec(url.pathname);
    if (method === 'POST' && decide !== null) {
      const [, id, verb] = decide;
      rows = rows.map((row) =>
        row.approval_id !== id
          ? row
          : verb === 'approve'
            ? {
                ...row,
                status: 'executed',
                decided_by: 'operator',
                decided_at: DECIDED_AT,
                result: { run_id: 'run-approved' },
              }
            : {
                ...row,
                status: 'rejected',
                decided_by: 'operator',
                decided_at: DECIDED_AT,
                reason: (body as { reason: string }).reason,
              },
      );
      return respond(rows.find((row) => row.approval_id === id));
    }
    const status = url.searchParams.get('status');
    const limit = Number(url.searchParams.get('limit') ?? 200);
    const listed = rows.filter((row) => status === null || row.status === status).slice(0, limit);
    return respond({
      approvals: listed,
      count: listed.length,
      pending: listed.filter((row) => row.status === 'pending').length,
      limit,
    });
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function settle(): Promise<void> {
  for (let i = 0; i < 6; i += 1) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

function renderPanel(): void {
  render(
    <NavigationProvider>
      <ApprovalsPanel />
    </NavigationProvider>,
  );
}

function byAction(action: string): Approval {
  const row = LISTING.approvals.find((a) => a.action === action);
  if (row === undefined) throw new Error(`the fixture holds no ${action} approval`);
  return row;
}

describe('[ApprovalsPanel]', () => {
  it('should disable both decision buttons while a decision is in flight', async () => {
    renderPanel();
    await settle();
    const row = byAction('delete');
    act(() => useApprovalsStore.setState({ inFlight: new Set([row.approval_id]) }));
    expect((screen.getByTestId(`approve-${row.approval_id}`) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect((screen.getByTestId(`reject-${row.approval_id}`) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it('should list every waiting request under "Waiting for you (n)", and an empty History', async () => {
    renderPanel();
    await settle();
    expect(LISTING.approvals.length).toBeGreaterThanOrEqual(4);
    expect(
      screen.getByRole('heading', { level: 2, name: `Waiting for you (${LISTING.pending})` }),
    ).toBeInTheDocument();
    const waiting = screen.getByTestId('approvals-waiting');
    for (const approval of LISTING.approvals) {
      expect(within(waiting).getByTestId(`approval-${approval.approval_id}`)).toBeTruthy();
    }
    expect(screen.getByRole('heading', { level: 2, name: 'History' })).toBeInTheDocument();
    expect(screen.getByTestId('approvals-history-empty')).toHaveTextContent(HISTORY_EMPTY);
  });

  it('should say nothing is waiting when nothing is', async () => {
    rows = [];
    renderPanel();
    await settle();
    expect(screen.getByRole('heading', { level: 2, name: 'Waiting for you (0)' })).toBeTruthy();
    expect(screen.getByTestId('approvals-empty')).toHaveTextContent(WAITING_EMPTY);
  });

  it('should show each card in the suite order: action, asked by, when, facts, arguments, id', async () => {
    renderPanel();
    await settle();
    const row = byAction('gpu_start');
    const card = screen.getByTestId(`approval-${row.approval_id}`);
    const text = card.textContent ?? '';
    const order = [
      'Start GPU work',
      'Waiting',
      'asked by agent forge-mcp',
      `asked ${formatLocal(row.created_at).text}`,
      'expire',
      'Step budget',
      'All arguments the agent sent',
      row.approval_id,
      APPROVE_LABEL['gpu_start'] ?? '',
      'Reject',
    ];
    let at = -1;
    for (const part of order) {
      const next = text.indexOf(part, at + 1);
      expect(next, `"${part}" after position ${at}`).toBeGreaterThan(at);
      at = next;
    }
    expect(card.querySelector('time')).toHaveAttribute(
      'title',
      new Date(row.created_at).toISOString(),
    );
    const args = screen.getByTestId(`approval-arguments-${row.approval_id}`);
    expect(args.tagName).toBe('DETAILS');
    expect(args).not.toHaveAttribute('open');
  });

  it('should name what a GPU start will do: run, run type, model, GPU, step budget', async () => {
    renderPanel();
    await settle();
    const row = byAction('gpu_start');
    const summary = screen.getByTestId(`approval-summary-${row.approval_id}`);
    for (const label of ['Run', 'Run type', 'Model', 'GPU', 'Step budget']) {
      expect(within(summary).getByText(label)).toBeTruthy();
    }
    expect(within(summary).getByText(String(row.summary['step_budget']))).toBeTruthy();
  });

  it('should name the miLLM target for an attach, the target for a delete, old and new for a setting', async () => {
    renderPanel();
    await settle();
    const attach = screen.getByTestId(
      `approval-summary-${byAction('hotupdate_attach').approval_id}`,
    );
    expect(within(attach).getByText('miLLM target')).toBeTruthy();
    const del = screen.getByTestId(`approval-summary-${byAction('delete').approval_id}`);
    expect(within(del).getByText('Target')).toBeTruthy();
    const setting = byAction('config_write');
    const config = screen.getByTestId(`approval-summary-${setting.approval_id}`);
    expect(within(config).getByText('Current value')).toBeTruthy();
    expect(within(config).getByText('New value')).toBeTruthy();
  });

  it('should name each approve button by its action, never a bare "Approve"', async () => {
    renderPanel();
    await settle();
    for (const approval of LISTING.approvals) {
      const button = screen.getByTestId(`approve-${approval.approval_id}`);
      expect(button.textContent).toBe(APPROVE_LABEL[approval.action]);
    }
  });

  it('should POST approve ONCE with no agent header, and show the run it started in History', async () => {
    renderPanel();
    await settle();
    const row = byAction('gpu_start');
    const button = screen.getByTestId(`approve-${row.approval_id}`);
    fireEvent.click(button);
    fireEvent.click(button);
    await settle();
    const posts = calls.filter((c) => c.method === 'POST');
    expect(posts).toEqual([
      {
        method: 'POST',
        path: `/api/v1/approvals/${row.approval_id}/approve`,
        body: undefined,
        headers: {},
      },
    ]);
    const history = screen.getByTestId(`history-${row.approval_id}`);
    expect(within(history).getByText('Approved and run')).toBeTruthy();
    expect(screen.getByTestId(`approval-outcome-${row.approval_id}`).textContent).toContain(
      'Open run run-approved',
    );
    expect(screen.queryByTestId(`approval-${row.approval_id}`)).toBeNull();
  });

  it('A4: a rejected request appears in History with the reason typed in the field', async () => {
    renderPanel();
    await settle();
    const row = byAction('config_write');
    const id = row.approval_id;
    fireEvent.click(screen.getByTestId(`reject-${id}`));
    const field = screen.getByTestId(`reject-reason-${id}`) as HTMLInputElement;
    expect(field.placeholder).toBe('Reason (optional)');
    expect(field.maxLength).toBe(1000);
    fireEvent.change(field, { target: { value: 'x' } });
    fireEvent.click(screen.getByTestId(`confirm-reject-${id}`));
    await settle();

    const posts = calls.filter((c) => c.method === 'POST');
    expect(posts).toHaveLength(1);
    expect(posts[0]?.path).toBe(`/api/v1/approvals/${id}/reject`);
    expect(posts[0]?.body).toEqual({ reason: 'x' });

    expect(screen.queryByTestId(`approval-${id}`)).toBeNull();
    const history = screen.getByTestId('approvals-history');
    const headers = [...history.querySelectorAll('th')].map((th) => th.textContent);
    expect(headers).toEqual([...HISTORY_COLUMNS]);
    const cells = [...screen.getByTestId(`history-${id}`).querySelectorAll('td')].map((td) =>
      (td.textContent ?? '').replace(/^[^:]+: /, ''),
    );
    expect(cells).toEqual([
      'Rejected',
      'Change a setting',
      'agent forge-mcp',
      formatLocal(row.created_at).text,
      'operator',
      formatLocal(DECIDED_AT).text,
      'x',
      '—',
      id,
    ]);
    const decided = screen.getByTestId(`history-${id}`).querySelectorAll('time')[1];
    expect(decided).toHaveAttribute('title', '2026-10-19T12:05:00.000Z');
  });

  it('should send the default reason when the field is left blank, and Cancel sends nothing', async () => {
    renderPanel();
    await settle();
    const id = byAction('delete').approval_id;
    fireEvent.click(screen.getByTestId(`reject-${id}`));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByTestId(`reject-reason-${id}`)).toBeNull();
    expect(calls.filter((c) => c.method === 'POST')).toHaveLength(0);
    fireEvent.click(screen.getByTestId(`reject-${id}`));
    fireEvent.click(screen.getByTestId(`confirm-reject-${id}`));
    await settle();
    expect(calls.filter((c) => c.method === 'POST')[0]?.body).toEqual({
      reason: 'Rejected by the operator.',
    });
  });

  it('should show an expired refusal inside the card that asked', async () => {
    renderPanel();
    await settle();
    const id = byAction('delete').approval_id;
    vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
      if ((init?.method ?? 'GET') === 'POST') {
        return respond(
          { error: { code: 'approval_expired', message: 'Approval expired.', details: {} } },
          409,
        );
      }
      const listed = rows.filter((row) => row.status === 'pending');
      void input;
      return respond({ approvals: listed, count: listed.length, pending: 0, limit: 200 });
    });
    fireEvent.click(screen.getByTestId(`approve-${id}`));
    await settle();
    expect(
      within(screen.getByTestId(`approval-${id}`)).getByTestId(`approval-error-${id}`),
    ).toHaveTextContent('This request expired. Ask the agent to try again.');
    // §2.7: focus stays on the card whose decision failed.
    expect(screen.getByTestId(`approval-${id}`).contains(document.activeElement)).toBe(true);
  });

  it('should show "nobody (expired)" and the expiry as the result of an expired request', async () => {
    const expiredRow = { ...byAction('delete'), status: 'expired', decided_at: DECIDED_AT };
    rows = [expiredRow as Approval];
    renderPanel();
    await settle();
    const history = screen.getByTestId(`history-${expiredRow.approval_id}`);
    expect(history).toHaveTextContent('nobody (expired)');
    expect(history).toHaveTextContent('expired without a decision');
    expect(within(history).getByText('Expired')).toBeTruthy();
  });

  it('should load older History when the page is full, with "Show older"', async () => {
    rows = Array.from(
      { length: 250 },
      (_, i) =>
        ({
          ...byAction('delete'),
          approval_id: `old-${i}`,
          status: 'rejected',
          decided_at: new Date(Date.UTC(2026, 9, 19, 0, 0, i)).toISOString(),
        }) as Approval,
    );
    renderPanel();
    await settle();
    expect(screen.getByTestId('approvals-history').querySelectorAll('tbody tr')).toHaveLength(200);
    fireEvent.click(screen.getByRole('button', { name: 'Show older' }));
    await settle();
    expect(screen.getByTestId('approvals-history').querySelectorAll('tbody tr')).toHaveLength(250);
    expect(screen.queryByRole('button', { name: 'Show older' })).toBeNull();
  });
});
