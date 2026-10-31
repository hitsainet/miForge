/**
 * The approval client sends the operator's decision - with NO agent header (FR-012.17).
 *
 * The backend refuses an agent's approve with `403 agent_cannot_decide`; the frontend's calls
 * must therefore never carry `X-Miforge-Agent`. Asserted on the request that leaves.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  approveApproval,
  approveBatch,
  isSettingsChange,
  listApprovals,
  rejectApproval,
} from './approvals';
import type { Approval } from './approvals';

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

  it('should send the typed reason, trimmed and at most 1000 characters (ENH-008 D8c)', async () => {
    await rejectApproval('apr_2', '  not now  ');
    expect(JSON.parse(String(calls[0]?.init?.body))).toEqual({ reason: 'not now' });
    await rejectApproval('apr_2', 'y'.repeat(1500));
    expect(JSON.parse(String(calls[1]?.init?.body)).reason).toHaveLength(1000);
  });

  it('should send "Rejected by the operator." when the reason is left blank', async () => {
    await rejectApproval('apr_2', '   ');
    await rejectApproval('apr_3');
    for (const call of calls) {
      expect(JSON.parse(String(call.init?.body))).toEqual({ reason: 'Rejected by the operator.' });
    }
  });

  it('should ask for one status and a limit when given them', async () => {
    await listApprovals({ status: 'pending', limit: 500 });
    expect(calls[0]?.url).toBe('/api/v1/approvals?status=pending&limit=500');
  });

  it('should read the list from the approvals route', async () => {
    await listApprovals();
    expect(calls[0]?.url).toBe('/api/v1/approvals');
    expect(calls[0]?.init?.method).toBe('GET');
  });

  it('should POST approve-batch once with the ids in order and no agent header (D79)', async () => {
    await approveBatch(['apr_1', 'apr_2']);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe('/api/v1/approvals/approve-batch');
    expect(calls[0]?.init?.method).toBe('POST');
    expect(JSON.parse(String(calls[0]?.init?.body))).toEqual({ approval_ids: ['apr_1', 'apr_2'] });
    expect(headerNames(calls[0]?.init)).not.toContain('x-miforge-agent');
  });

  it('should count only waiting config_write requests as settings changes', () => {
    const row = (action: string, status: string) => ({ action, status }) as unknown as Approval;
    expect(isSettingsChange(row('config_write', 'pending'))).toBe(true);
    expect(isSettingsChange(row('config_write', 'executed'))).toBe(false);
    for (const action of ['gpu_start', 'hotupdate_attach', 'delete']) {
      expect(isSettingsChange(row(action, 'pending'))).toBe(false);
    }
  });
});
