/**
 * The Approvals screen (FR-012.17, .18).
 *
 * The summaries rendered here are the ones the backend STORED for a real agent request: they
 * come from `e2e/fixtures/approvals.json`, which `backend/tests/integration/test_ui_approval_fixtures.py`
 * generates by sending agent calls through the mounted app and pins - so the screen is tested
 * against what the gate actually writes, not against a hand-written shape.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import fixture from '../../../../e2e/fixtures/approvals.json';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { APPROVE_LABEL, ApprovalsPanel } from './ApprovalsPanel';
import type { Approval, ApprovalListing } from '@/api/approvals';

const LISTING = fixture as unknown as ApprovalListing;

let calls: { method: string; path: string; headers: Record<string, string> }[] = [];
let listing: ApprovalListing = LISTING;

beforeEach(() => {
  calls = [];
  listing = LISTING;
  useApprovalsStore.setState({
    listing: null,
    failure: null,
    decisionFailure: null,
    inFlight: new Set<string>(),
  });
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://local');
    const method = (init?.method ?? 'GET').toUpperCase();
    calls.push({
      method,
      path: url.pathname,
      headers: (init?.headers ?? {}) as Record<string, string>,
    });
    if (method === 'POST' && url.pathname.endsWith('/approve')) {
      const id = url.pathname.split('/')[4] ?? '';
      const row = listing.approvals.find((a) => a.approval_id === id) as Approval;
      const executed = { ...row, status: 'executed', result: { run_id: 'run-approved' } };
      listing = {
        ...listing,
        pending: listing.pending - 1,
        approvals: listing.approvals.map((a) => (a.approval_id === id ? executed : a)),
      } as ApprovalListing;
      return Promise.resolve(new Response(JSON.stringify(executed), { status: 200 }));
    }
    return Promise.resolve(
      new Response(JSON.stringify(listing), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function settle(): Promise<void> {
  for (let i = 0; i < 4; i += 1) {
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
    useApprovalsStore.setState({ inFlight: new Set([row.approval_id]) });
    await settle();
    expect((screen.getByTestId(`approve-${row.approval_id}`) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect((screen.getByTestId(`reject-${row.approval_id}`) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it('should list every pending request from the fixture, with its count', async () => {
    renderPanel();
    await settle();
    expect(LISTING.approvals.length).toBeGreaterThanOrEqual(4);
    expect(screen.getByTestId('approvals-count').textContent).toBe(
      `${LISTING.pending} pending of ${LISTING.approvals.length} shown`,
    );
    for (const approval of LISTING.approvals) {
      expect(screen.getByTestId(`approval-${approval.approval_id}`)).toBeTruthy();
    }
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

  it('should POST approve ONCE with no agent header, and show the run it started', async () => {
    renderPanel();
    await settle();
    const row = byAction('gpu_start');
    const button = screen.getByTestId(`approve-${row.approval_id}`);
    fireEvent.click(button);
    fireEvent.click(button);
    await settle();
    const posts = calls.filter((c) => c.method === 'POST');
    expect(posts).toEqual([
      { method: 'POST', path: `/api/v1/approvals/${row.approval_id}/approve`, headers: {} },
    ]);
    expect(screen.getByTestId(`approval-outcome-${row.approval_id}`).textContent).toContain(
      'Open run run-approved',
    );
  });
});
