/**
 * The API client (foundation task 7.4; ADR-013, ADR-010, ADR-022).
 *
 * `fetch` is stubbed rather than a server started: these assertions are about the
 * client's decisions - which body shape becomes which failure kind, and that no request
 * ever leaves for a host that is not miForge.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_BASE, apiGet, apiPost, requestOrThrow } from './client';
import { ApiError } from './errors';

function stubFetch(status: number, body: string, contentType = 'application/json'): typeof fetch {
  // A null-body status (204, 205, 304) may not carry a body at all - not even an empty
  // string - or the Response constructor refuses it.
  const payload = body === '' ? null : body;
  return vi.fn(
    async () => new Response(payload, { status, headers: { 'Content-Type': contentType } }),
  ) as unknown as typeof fetch;
}

const originalFetch = globalThis.fetch;

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('[api client]', () => {
  it('should return a plain JSON success body unwrapped', () => {
    // ADR-013: success bodies are plain JSON. A uniform wrapper makes every client
    // unwrap twice and tells it nothing.
    globalThis.fetch = stubFetch(200, JSON.stringify({ runs: [], total: 0 }));
    return apiGet<{ runs: unknown[]; total: number }>('/runs').then((result) => {
      expect(result).toEqual({ ok: true, value: { runs: [], total: 0 } });
    });
  });

  it('should not treat a success body with an error field as a failure', () => {
    // The envelope wraps failures only. A resource with a field called `error` is not a
    // refusal, and a client that looked for the key on 2xx would invent one.
    globalThis.fetch = stubFetch(200, JSON.stringify({ error: 'the last step errored' }));
    return apiGet<unknown>('/runs/abc').then((result) => {
      expect(result.ok).toBe(true);
    });
  });

  it('should decode a refusal body into a typed failure, keeping the code', () => {
    globalThis.fetch = stubFetch(
      422,
      JSON.stringify({
        error: { code: 'run_has_no_stop_condition', message: 'Add one.', details: {} },
      }),
    );
    return apiPost<unknown>('/runs', {}).then((result) => {
      expect(result.ok).toBe(false);
      if (result.ok) throw new Error('unreachable');
      expect(result.failure).toMatchObject({
        kind: 'refusal',
        code: 'run_has_no_stop_condition',
        status: 422,
      });
    });
  });

  it('should report a proxy HTML error page as malformed, with the body kept', () => {
    globalThis.fetch = stubFetch(502, '<html>502</html>', 'text/html');
    return apiGet<unknown>('/runs').then((result) => {
      expect(result.ok).toBe(false);
      if (result.ok) throw new Error('unreachable');
      expect(result.failure).toMatchObject({ kind: 'malformed', status: 502 });
    });
  });

  it('should report a thrown fetch as a transport failure, never as a refusal', () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    }) as unknown as typeof fetch;
    return apiGet<unknown>('/runs').then((result) => {
      expect(result.ok).toBe(false);
      if (result.ok) throw new Error('unreachable');
      expect(result.failure).toMatchObject({ kind: 'transport', status: 0 });
    });
  });

  it('should treat an empty success body as undefined rather than a parse failure', () => {
    globalThis.fetch = stubFetch(204, '');
    return apiPost<undefined>('/runs/abc/cancel').then((result) => {
      expect(result).toEqual({ ok: true, value: undefined });
    });
  });

  it('should report a 2xx body that is not JSON as malformed', () => {
    globalThis.fetch = stubFetch(200, 'not json at all', 'text/plain');
    return apiGet<unknown>('/runs').then((result) => {
      expect(result.ok).toBe(false);
      if (result.ok) throw new Error('unreachable');
      expect(result.failure.kind).toBe('malformed');
    });
  });

  it('should request only relative paths under /api, never a sibling host', () => {
    // ADR-022 fences the core path off every sibling endpoint, and a browser `fetch` is
    // on that path. There is no base-URL setting here to point elsewhere; this asserts
    // the URL that is actually built.
    const spy = vi.fn<typeof fetch>(async () => new Response('{}', { status: 200 }));
    globalThis.fetch = spy as unknown as typeof fetch;
    return apiGet('/runs', { query: { domain: 'null', limit: 50 } }).then(() => {
      const url = String(spy.mock.calls[0]?.[0]);
      expect(url.startsWith(`${API_BASE}/`)).toBe(true);
      expect(url).toBe('/api/runs?domain=null&limit=50');
      expect(/^https?:\/\//.test(url)).toBe(false);
    });
  });

  it('should omit an undefined query value rather than send the string "undefined"', () => {
    const spy = vi.fn<typeof fetch>(async () => new Response('{}', { status: 200 }));
    globalThis.fetch = spy as unknown as typeof fetch;
    return apiGet('/tree', { query: { domain: undefined, kind: 'checkpoint' } }).then(() => {
      expect(String(spy.mock.calls[0]?.[0])).toBe('/api/tree?kind=checkpoint');
    });
  });

  it('should throw an ApiError carrying the failure from requestOrThrow', () => {
    globalThis.fetch = stubFetch(
      409,
      JSON.stringify({
        error: { code: 'midataworks_not_configured', message: 'set its base URL', details: {} },
      }),
    );
    return requestOrThrow('GET', '/packs').then(
      () => {
        throw new Error('should have thrown');
      },
      (error: unknown) => {
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).failure.kind).toBe('service_not_configured');
      },
    );
  });
});
