/**
 * The approval store (F12; FTDD section 7; ENH-008 D6, D7).
 *
 * The button is disabled while its request is in flight too, but that is the view's guard; this
 * pins the store's own, so either can be removed without the other going unnoticed. The rest
 * pins the two lists: Waiting is the pending rows, soonest to expire first; History is every
 * other row, most recent decision first, and grows by 200 up to the API's 500.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Approval } from '@/api/approvals';
import {
  APPROVALS_POLL_MS,
  INITIAL_APPROVALS_STATE,
  historyOf,
  useApprovalsStore,
} from './approvalsStore';

let posts = 0;
let gets: string[] = [];
let release: () => void = () => undefined;
let rows: Approval[] = [];

function row(id: string, status: Approval['status'], extra: Partial<Approval> = {}): Approval {
  return {
    approval_id: id,
    action: 'config_write',
    status,
    summary: {},
    requested_by: 'agent:forge-mcp',
    decided_by: status === 'pending' || status === 'expired' ? null : 'operator',
    reason: null,
    result: null,
    error: null,
    created_at: '2026-10-19T12:00:00+00:00',
    decided_at: null,
    expires_at: '2026-10-20T12:00:00+00:00',
    ...extra,
  };
}

beforeEach(() => {
  posts = 0;
  gets = [];
  rows = [];
  useApprovalsStore.setState({ ...INITIAL_APPROVALS_STATE, inFlight: new Set<string>() });
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    if ((init?.method ?? 'GET') === 'POST') {
      posts += 1;
      return new Promise<Response>((resolve) => {
        release = () =>
          resolve(new Response(JSON.stringify({ approval_id: 'apr_1', status: 'executed' })));
      });
    }
    const url = new URL(String(input), 'http://local');
    gets.push(`${url.pathname}${url.search}`);
    const status = url.searchParams.get('status');
    const limit = Number(url.searchParams.get('limit') ?? 200);
    const listed = rows.filter((r) => status === null || r.status === status).slice(0, limit);
    return Promise.resolve(
      new Response(JSON.stringify({ approvals: listed, count: listed.length, pending: 0, limit })),
    );
  });
});

afterEach(() => vi.unstubAllGlobals());

describe('[approvalsStore]', () => {
  it('should poll every 15 seconds (ENH-008 D7)', () => {
    expect(APPROVALS_POLL_MS).toBe(15_000);
  });

  it('should send one approve for two concurrent requests to decide the same approval', async () => {
    const first = useApprovalsStore.getState().approve('apr_1');
    const second = useApprovalsStore.getState().approve('apr_1');
    expect(useApprovalsStore.getState().inFlight.has('apr_1')).toBe(true);
    expect(await second).toBeNull();
    release();
    await first;
    expect(posts).toBe(1);
    expect(useApprovalsStore.getState().inFlight.has('apr_1')).toBe(false);
  });

  it('should send one approve-batch for two concurrent approve-alls, every id in flight meanwhile (D79)', async () => {
    const first = useApprovalsStore.getState().approveSettingsBatch(['apr_1', 'apr_2']);
    const second = useApprovalsStore.getState().approveSettingsBatch(['apr_1', 'apr_2']);
    expect(useApprovalsStore.getState().batchInFlight).toBe(true);
    expect([...useApprovalsStore.getState().inFlight]).toEqual(['apr_1', 'apr_2']);
    expect(await second).toBeNull();
    release();
    await first;
    expect(posts).toBe(1);
    expect(useApprovalsStore.getState().batchInFlight).toBe(false);
    expect(useApprovalsStore.getState().inFlight.size).toBe(0);
  });

  it('should read Waiting as the pending rows, soonest to expire first', async () => {
    rows = [
      row('late', 'pending', { expires_at: '2026-10-20T15:00:00+00:00' }),
      row('soon', 'pending', { expires_at: '2026-10-19T13:00:00+00:00' }),
      row('done', 'executed'),
    ];
    await useApprovalsStore.getState().load();
    expect(gets).toEqual(['/api/v1/approvals?status=pending&limit=500']);
    expect(useApprovalsStore.getState().waiting?.map((r) => r.approval_id)).toEqual([
      'soon',
      'late',
    ]);
  });

  it('should read History as every row that is not pending, the latest decision first', async () => {
    rows = [
      row('waiting', 'pending'),
      row('older', 'rejected', { decided_at: '2026-10-19T12:10:00+00:00' }),
      row('newer', 'executed', { decided_at: '2026-10-19T12:30:00+00:00' }),
      row('running', 'executing', { decided_at: '2026-10-19T12:20:00+00:00' }),
    ];
    await useApprovalsStore.getState().loadHistory();
    expect(gets).toEqual(['/api/v1/approvals?limit=200']);
    expect(useApprovalsStore.getState().history?.map((r) => r.approval_id)).toEqual([
      'newer',
      'running',
      'older',
    ]);
    expect(useApprovalsStore.getState().historyMore).toBe(false);
  });

  it('should keep a pending row out of History whatever its order (the History filter)', () => {
    expect(historyOf([row('a', 'pending'), row('b', 'expired')]).map((r) => r.approval_id)).toEqual(
      ['b'],
    );
  });

  it('should offer older rows when a page is full, and stop at the API limit of 500', async () => {
    rows = Array.from({ length: 600 }, (_, i) =>
      row(`r${i}`, 'rejected', {
        decided_at: new Date(Date.UTC(2026, 9, 19, 0, 0, i)).toISOString(),
      }),
    );
    await useApprovalsStore.getState().loadHistory();
    expect(useApprovalsStore.getState().history).toHaveLength(200);
    expect(useApprovalsStore.getState().historyMore).toBe(true);
    await useApprovalsStore.getState().showOlder();
    expect(useApprovalsStore.getState().history).toHaveLength(400);
    await useApprovalsStore.getState().showOlder();
    expect(useApprovalsStore.getState().history).toHaveLength(500);
    expect(useApprovalsStore.getState().historyMore).toBe(false);
    expect(gets).toEqual([
      '/api/v1/approvals?limit=200',
      '/api/v1/approvals?limit=400',
      '/api/v1/approvals?limit=500',
    ]);
  });

  it('should keep a refused decision on its own request id', async () => {
    vi.stubGlobal('fetch', (_input: RequestInfo | URL, init?: RequestInit) =>
      Promise.resolve(
        (init?.method ?? 'GET') === 'POST'
          ? new Response(
              JSON.stringify({
                error: { code: 'approval_expired', message: 'expired', details: {} },
              }),
              { status: 409, headers: { 'Content-Type': 'application/json' } },
            )
          : new Response(JSON.stringify({ approvals: [], count: 0, pending: 0, limit: 200 })),
      ),
    );
    expect(await useApprovalsStore.getState().approve('apr_9')).toBeNull();
    const failure = useApprovalsStore.getState().decisionFailures['apr_9'];
    expect(failure?.kind === 'refusal' ? failure.code : null).toBe('approval_expired');
  });
});
