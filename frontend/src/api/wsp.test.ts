/**
 * The WSP runs client's routes, by URL, method and call count (fix 2026-11-20).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { WSP_RUNS_LIMIT, fetchWspRun, fetchWspTranscript, isWspRunId, listWspRuns } from './wsp';

let calls: { url: string; method: string }[] = [];

beforeEach(() => {
  calls = [];
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), method: init?.method ?? 'GET' });
    return Promise.resolve(
      new Response(JSON.stringify({ runs: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  });
});

afterEach(() => vi.unstubAllGlobals());

describe('[wsp client]', () => {
  it('should GET the WSP run list once, at the route maximum', async () => {
    await listWspRuns();
    expect(calls).toEqual([{ url: `/api/v1/wsp/runs?limit=${WSP_RUNS_LIMIT}`, method: 'GET' }]);
    expect(WSP_RUNS_LIMIT).toBe(200);
  });

  it('should GET one run and its transcript, the id encoded', async () => {
    await fetchWspRun('wsp_0123456789ab');
    await fetchWspTranscript('wsp_0123456789ab');
    expect(calls).toEqual([
      { url: '/api/v1/wsp/runs/wsp_0123456789ab', method: 'GET' },
      { url: '/api/v1/wsp/runs/wsp_0123456789ab/transcript', method: 'GET' },
    ]);
  });

  it('should tell a WSP run id from a training run id', () => {
    expect(isWspRunId('wsp_7a1a707d2cf8')).toBe(true);
    for (const id of ['run-a', 'wsp_7A1A707D2CF8', 'wsp_7a1a707d2cf', 'xwsp_7a1a707d2cf8', '']) {
      expect(isWspRunId(id), id).toBe(false);
    }
  });
});
