/**
 * The approval client sends the operator's decision - with NO agent header (FR-012.17).
 *
 * The backend refuses an agent's approve with `403 agent_cannot_decide`; the frontend's calls
 * must therefore never carry `X-Miforge-Agent`. Asserted on the request that leaves.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { approveApproval, listApprovals, rejectApproval } from './approvals';

let calls: { url: string; init: RequestInit | undefined }[] = [];

beforeEach(() => {
  calls = [];
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return Promise.resolve(
      new Response(JSON.stringify({ approval_id: 'apr_1', status: 'executed' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  });
});

afterEach(() => vi.unstubAllGlobals());

function headerNames(init: RequestInit | undefined): string[] {
  return Object.keys((init?.headers ?? {}) as Record<string, string>).map((h) => h.toLowerCase());
}

describe('[approvals client]', () => {
  it('should POST approve once, to the approval, with no agent header', async () => {
    await approveApproval('apr_1');
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe('/api/v1/approvals/apr_1/approve');
    expect(calls[0]?.init?.method).toBe('POST');
    expect(headerNames(calls[0]?.init)).not.toContain('x-miforge-agent');
  });

  it('should POST reject with no agent header', async () => {
    await rejectApproval('apr_2');
    expect(calls[0]?.url).toBe('/api/v1/approvals/apr_2/reject');
    expect(headerNames(calls[0]?.init)).not.toContain('x-miforge-agent');
  });

  it('should read the list from the approvals route', async () => {
    await listApprovals();
    expect(calls[0]?.url).toBe('/api/v1/approvals');
    expect(calls[0]?.init?.method).toBe('GET');
  });
});
