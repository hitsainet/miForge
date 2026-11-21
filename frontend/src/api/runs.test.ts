/**
 * The New run form's two calls go to the routes the backend declares.
 *
 * The form used to print `POST /api/v1/runs/define` on screen, and a test read that text as
 * the proof of the route. The copy review of 2026-10-20 took the route off the screen, so the
 * route is pinned here, on the client the form calls, by URL, method, body and call count.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { defineRun, validateRun } from './runs';

let calls: { url: string; init: RequestInit | undefined }[] = [];

beforeEach(() => {
  calls = [];
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return Promise.resolve(
      new Response(JSON.stringify({ run_id: 'run-a', ok: true }), {
        status: 202,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  });
});

afterEach(() => vi.unstubAllGlobals());

describe('[runs client] the New run form routes', () => {
  it('should POST the whole definition to /define, once - never to the acquisition route', async () => {
    const definition = { run_type: 'change', run_configuration: { batch_size: 8 } };
    await defineRun(definition);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe('/api/v1/runs/define');
    expect(calls[0]?.init?.method).toBe('POST');
    expect(JSON.parse(String(calls[0]?.init?.body))).toEqual(definition);
  });

  it('should POST the same definition to /validate for the check', async () => {
    const definition = { run_type: 'change' };
    await validateRun(definition);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe('/api/v1/runs/validate');
    expect(calls[0]?.init?.method).toBe('POST');
    expect(JSON.parse(String(calls[0]?.init?.body))).toEqual(definition);
  });
});
