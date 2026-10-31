/**
 * Approve all settings changes (ENH-006 D79): on the banner and on the Approvals screen, only
 * from two waiting settings changes; a confirm that lists each (label, from → to, who asked);
 * ONE `POST /approvals/approve-batch` carrying exactly the listed ids; History shows each
 * request with its batch.
 *
 * The settings rows are the `config_write` request the gate really stored
 * (`e2e/fixtures/approvals.json`, pinned by `test_ui_approval_fixtures.py`), copied with a
 * different id, key and values - so the shape is the backend's, not a hand-written one.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import fixture from '../../../e2e/fixtures/approvals.json';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';
import { ApprovalsPanel } from '@/components/panels/approvals/ApprovalsPanel';
import type { Approval, ApprovalListing } from '@/api/approvals';
import { ApprovalsBanner } from './ApprovalsBanner';
import { approveAllLabel, confirmAllLabel } from './ApproveAllSettings';

const LISTING = fixture as unknown as ApprovalListing;
const DECIDED_AT = '2026-10-19T12:05:00+00:00';

function stored(action: string): Approval {
  const row = LISTING.approvals.find((a) => a.action === action);
  if (row === undefined) throw new Error(`the fixture holds no ${action} approval`);
  return row;
}

/** The stored setting request, as another request for another key. */
function setting(id: string, key: string, label: string, from: unknown, to: unknown): Approval {
  const base = stored('config_write');
  return {
    ...base,
    approval_id: id,
    summary: { ...base.summary, key, label, old: from, new: to },
    payload: { key, body: { value: to, reason: null } },
  };
}

const PRUNE = { ...stored('config_write') };
const KEEP = setting('apr_keep', 'checkpoint_lifecycle_keep_last_n', 'Checkpoints to keep', 10, 3);
const LOG = setting('apr_log', 'log_level', 'Log level', 'INFO', 'DEBUG');
const GPU = stored('gpu_start');

let rows: Approval[] = [];
let posts: { path: string; body: unknown; headers: Record<string, string> }[] = [];
let refuseBatch = false;

function respond(body: unknown, status = 200): Promise<Response> {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  );
}

beforeEach(() => {
  rows = [];
  posts = [];
  refuseBatch = false;
  window.sessionStorage.clear();
  useApprovalsStore.setState({ ...INITIAL_APPROVALS_STATE, inFlight: new Set<string>() });
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://local');
    const method = (init?.method ?? 'GET').toUpperCase();
    if (method === 'POST') {
      const body = init?.body === undefined ? undefined : JSON.parse(String(init.body));
      posts.push({
        path: url.pathname,
        body,
        headers: (init?.headers ?? {}) as Record<string, string>,
      });
      if (url.pathname === '/api/v1/approvals/approve-batch') {
        if (refuseBatch) {
          return respond(
            {
              error: {
                code: 'approval_batch_refused',
                message: 'Refused: apr_keep (not_pending (rejected))',
                details: {},
              },
            },
            409,
          );
        }
        const ids = (body as { approval_ids: string[] }).approval_ids;
        rows = rows.map((row) =>
          ids.includes(row.approval_id)
            ? {
                ...row,
                status: 'executed',
                decided_by: 'operator',
                decided_at: DECIDED_AT,
                batch_id: 'bat_1',
                result: { key: row.summary.key },
              }
            : row,
        );
        const results = ids.map((id) => ({
          ...rows.find((row) => row.approval_id === id),
          ok: true,
        }));
        return respond({ batch_id: 'bat_1', count: results.length, results });
      }
      return respond({}, 404);
    }
    const status = url.searchParams.get('status');
    const listed = rows.filter((row) => status === null || row.status === status);
    return respond({ approvals: listed, count: listed.length, pending: 0, limit: 200 });
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function settle(): Promise<void> {
  for (let i = 0; i < 8; i += 1) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

async function loadWith(waiting: Approval[]): Promise<void> {
  rows = waiting.map((row) => ({ ...row }));
  await act(async () => {
    await useApprovalsStore.getState().load();
  });
}

function renderBanner(): void {
  render(
    <NavigationProvider>
      <ApprovalsBanner />
    </NavigationProvider>,
  );
}

function renderPanel(): void {
  render(
    <NavigationProvider>
      <ApprovalsPanel />
    </NavigationProvider>,
  );
}

describe('[ApproveAllSettings]', () => {
  it('should show no approve-all button for zero or one waiting settings change', async () => {
    await loadWith([GPU]);
    renderBanner();
    expect(screen.queryByTestId('approve-all-button-banner')).toBeNull();
    cleanup();
    await loadWith([PRUNE, GPU]);
    renderBanner();
    expect(screen.getByTestId(`approve-${PRUNE.approval_id}`)).toBeInTheDocument();
    expect(screen.queryByTestId('approve-all-button-banner')).toBeNull();
    expect(screen.queryByRole('button', { name: /Approve all/ })).toBeNull();
  });

  it('should name the action and count only the settings changes, in the banner', async () => {
    await loadWith([PRUNE, GPU, KEEP]);
    renderBanner();
    const button = screen.getByTestId('approve-all-button-banner');
    expect(button).toHaveTextContent(approveAllLabel(2));
    expect(button).toHaveTextContent('Approve all settings changes (2)');
    expect(
      within(screen.getByTestId('approvals-banner')).getByRole('button', {
        name: 'Approve all settings changes (2)',
      }),
    ).toBe(button);
  });

  it('should count every waiting settings change, not only the three cards the banner shows', async () => {
    const more = [1, 2, 3].map((n) => setting(`apr_s${n}`, `k${n}`, `Setting ${n}`, n, n + 1));
    await loadWith([PRUNE, KEEP, LOG, ...more]);
    renderBanner();
    expect(screen.getByTestId('approve-all-button-banner')).toHaveTextContent(
      'Approve all settings changes (6)',
    );
  });

  it('should list every request in the confirm: label, from → to, who asked', async () => {
    await loadWith([PRUNE, GPU, KEEP]);
    renderBanner();
    fireEvent.click(screen.getByTestId('approve-all-button-banner'));
    const confirm = screen.getByTestId('approve-all-confirm-banner');
    const items = within(confirm).getAllByRole('listitem');
    expect(items.map((item) => item.getAttribute('data-testid'))).toEqual([
      `approve-all-item-${PRUNE.approval_id}`,
      'approve-all-item-apr_keep',
    ]);
    // The label the gate stored for the real request, its from → to, and who asked.
    expect(items[0]).toHaveTextContent(String(PRUNE.summary.label));
    expect(items[0]).toHaveTextContent('false → true');
    expect(items[0]).toHaveTextContent('asked by agent forge-mcp');
    expect(items[1]).toHaveTextContent('Checkpoints to keep');
    expect(items[1]).toHaveTextContent('10 → 3');
    expect(within(confirm).queryByText(/GPU/)).toBeNull();
    expect(posts).toEqual([]);
  });

  it('should send ONE approve-batch with exactly the listed ids, no agent header, then drop the cards', async () => {
    await loadWith([PRUNE, GPU, KEEP]);
    renderBanner();
    fireEvent.click(screen.getByTestId('approve-all-button-banner'));
    const confirmButton = screen.getByTestId('approve-all-confirm-button-banner');
    expect(confirmButton).toHaveTextContent(confirmAllLabel(2));
    fireEvent.click(confirmButton);
    fireEvent.click(confirmButton);
    await settle();
    expect(posts).toEqual([
      {
        path: '/api/v1/approvals/approve-batch',
        body: { approval_ids: [PRUNE.approval_id, 'apr_keep'] },
        headers: { 'Content-Type': 'application/json' },
      },
    ]);
    expect(Object.keys(posts[0]!.headers).map((h) => h.toLowerCase())).not.toContain(
      'x-miforge-agent',
    );
    expect(screen.queryByTestId(`approval-${PRUNE.approval_id}`)).toBeNull();
    expect(screen.queryByTestId('approval-apr_keep')).toBeNull();
    expect(screen.getByTestId(`approval-${GPU.approval_id}`)).toBeInTheDocument();
    expect(screen.queryByTestId('approve-all-button-banner')).toBeNull();
  });

  it('should never approve a request that arrived while the confirm was open', async () => {
    await loadWith([PRUNE, KEEP]);
    renderBanner();
    fireEvent.click(screen.getByTestId('approve-all-button-banner'));
    await loadWith([PRUNE, KEEP, LOG]);
    expect(
      within(screen.getByTestId('approve-all-confirm-banner')).getAllByRole('listitem'),
    ).toHaveLength(2);
    fireEvent.click(screen.getByTestId('approve-all-confirm-button-banner'));
    await settle();
    expect(posts.map((post) => post.body)).toEqual([
      { approval_ids: [PRUNE.approval_id, 'apr_keep'] },
    ]);
  });

  it('should send nothing on Cancel', async () => {
    await loadWith([PRUNE, KEEP]);
    renderBanner();
    fireEvent.click(screen.getByTestId('approve-all-button-banner'));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await settle();
    expect(posts).toEqual([]);
    expect(screen.getByTestId('approve-all-button-banner')).toBeInTheDocument();
  });

  it('should say nothing was approved when the backend refuses the batch', async () => {
    refuseBatch = true;
    await loadWith([PRUNE, KEEP]);
    renderPanel();
    await settle();
    fireEvent.click(screen.getByTestId('approve-all-button-page'));
    fireEvent.click(screen.getByTestId('approve-all-confirm-button-page'));
    await settle();
    expect(screen.getByTestId('approve-all-failure-page')).toHaveTextContent(
      'Nothing was approved. approval_batch_refused',
    );
    expect(screen.getByTestId(`approval-${PRUNE.approval_id}`)).toBeInTheDocument();
  });

  it('should be on the Approvals screen too, and History shows each request with its batch', async () => {
    await loadWith([PRUNE, GPU, KEEP]);
    renderPanel();
    await settle();
    fireEvent.click(screen.getByTestId('approve-all-button-page'));
    fireEvent.click(screen.getByTestId('approve-all-confirm-button-page'));
    await settle();
    expect(screen.getByTestId('approve-all-outcome-page')).toHaveTextContent(
      'Approved 2 settings changes together.',
    );
    for (const id of [PRUNE.approval_id, 'apr_keep']) {
      expect(screen.getByTestId(`history-${id}`)).toBeInTheDocument();
      expect(screen.getByTestId(`history-decided-by-${id}`)).toHaveTextContent(
        'operatorwith other settings changes, in one approval bat_1',
      );
    }
    expect(screen.getByTestId(`approval-${GPU.approval_id}`)).toBeInTheDocument();
  });

  it('should render a hostile label as text', async () => {
    const hostile = setting('apr_h', 'log_level', '<img src=x onerror=alert(1)>', '<b>a</b>', 'b');
    await loadWith([hostile, KEEP]);
    renderBanner();
    fireEvent.click(screen.getByTestId('approve-all-button-banner'));
    const confirm = screen.getByTestId('approve-all-confirm-banner');
    expect(confirm.querySelector('img')).toBeNull();
    expect(confirm.querySelector('b')).toBeNull();
    expect(confirm).toHaveTextContent('<img src=x onerror=alert(1)>');
  });
});
