/**
 * The store sends each decision once, however many times it is asked (F12; FTDD section 7).
 *
 * The button is disabled while its request is in flight too, but that is the view's guard; this
 * pins the store's own, so either can be removed without the other going unnoticed.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useApprovalsStore } from './approvalsStore';

let posts = 0;
let release: () => void = () => undefined;

beforeEach(() => {
  posts = 0;
  useApprovalsStore.setState({
    listing: null,
    failure: null,
    decisionFailure: null,
    inFlight: new Set<string>(),
  });
  vi.stubGlobal('fetch', (_input: RequestInfo | URL, init?: RequestInit) => {
    if ((init?.method ?? 'GET') === 'POST') {
      posts += 1;
      return new Promise<Response>((resolve) => {
        release = () =>
          resolve(new Response(JSON.stringify({ approval_id: 'apr_1', status: 'executed' })));
      });
    }
    return Promise.resolve(
      new Response(JSON.stringify({ approvals: [], count: 0, pending: 0, limit: 200 })),
    );
  });
});

afterEach(() => vi.unstubAllGlobals());

describe('[approvalsStore]', () => {
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
});
