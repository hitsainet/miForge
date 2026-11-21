/**
 * A checkpoint response with no manifest object is a named failure, never a value the view
 * renders. Found 2026-10-18 by CI: the run view's manifest panel received `undefined`, its
 * render threw `Cannot read properties of undefined (reading 'kind')`, and the whole app
 * unmounted to a blank page after the curves had painted.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchCheckpointManifest } from './checkpoints';

const originalFetch = globalThis.fetch;

function respond(body: unknown): void {
  globalThis.fetch = vi.fn(
    async () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
  ) as unknown as typeof fetch;
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('[fetchCheckpointManifest]', () => {
  it.each([
    ['a listing body', { run_id: 'run-a', checkpoints: [], latest_step: 900 }],
    ['a null manifest', { run_id: 'run-a', step: 900, manifest: null }],
    ['an array manifest', { run_id: 'run-a', step: 900, manifest: [] }],
    ['a string manifest', { run_id: 'run-a', step: 900, manifest: 'nope' }],
  ])('should return malformed for %s', async (_name, body) => {
    respond(body);
    const result = await fetchCheckpointManifest('run-a', 900);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.failure.kind).toBe('malformed');
    }
  });

  it('should pass a real manifest response through unchanged', async () => {
    const body = {
      run_id: 'run-a',
      step: 900,
      manifest: { kind: 'miforge.checkpoint-manifest', step: 900 },
      manifest_sha256: 'a'.repeat(64),
      provider_sources: {},
    };
    respond(body);
    expect(await fetchCheckpointManifest('run-a', 900)).toEqual({ ok: true, value: body });
  });
});
