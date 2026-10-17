/**
 * The store convention (foundation task 7.4; ADR-007, ADR-013).
 *
 * The assertion worth reading is "a failed refresh keeps the last good value". A run view
 * that blanks on one failed poll of an hours-long run has thrown away the thing the
 * operator was watching, and the same rule appears one layer up as F9's M21 - "a 422
 * keeps the form".
 */

import { describe, expect, it, vi } from 'vitest';
import { createRecordStore } from './createRecordStore';
import { decodeErrorEnvelope } from '@/api/errors';
import type { ApiResult } from '@/api/client';

interface Run {
  readonly id: string;
  readonly step: number;
}

const ok = (step: number): ApiResult<Run> => ({ ok: true, value: { id: 'r1', step } });
const refused = (): ApiResult<Run> => ({
  ok: false,
  failure: decodeErrorEnvelope(409, {
    error: { code: 'index_unavailable', message: 'reindex rebuilds it.', details: {} },
  }),
});

describe('[createRecordStore]', () => {
  it('should start idle with no value and no failure', () => {
    const store = createRecordStore<Run>(async () => ok(1));
    expect(store.getState()).toMatchObject({ phase: 'idle', value: null, failure: null });
  });

  it('should become ready with the value and a load timestamp', async () => {
    const store = createRecordStore<Run>(
      async () => ok(7),
      () => 1_700_000_000_000,
    );
    await store.getState().load();
    expect(store.getState()).toMatchObject({
      phase: 'ready',
      value: { id: 'r1', step: 7 },
      failure: null,
      loadedAtMs: 1_700_000_000_000,
    });
  });

  it('should keep the last good value when a later read fails', async () => {
    const fetcher = vi
      .fn<() => Promise<ApiResult<Run>>>()
      .mockResolvedValueOnce(ok(100))
      .mockResolvedValueOnce(refused());
    const store = createRecordStore<Run>(fetcher);

    await store.getState().load();
    await store.getState().load();

    const state = store.getState();
    expect(state.phase).toBe('failed');
    expect(state.value).toEqual({ id: 'r1', step: 100 });
    expect(state.failure).toMatchObject({ kind: 'refusal', code: 'index_unavailable' });
  });

  it('should keep the failure as an ApiFailure so the code survives', async () => {
    // Stringifying it here would discard the code before any screen saw it, and the code
    // is what `<service>_not_configured` rendering by name depends on.
    const store = createRecordStore<Run>(async () => refused());
    await store.getState().load();
    expect(store.getState().failure).toMatchObject({ code: 'index_unavailable' });
  });

  it('should clear a stale failure on a successful read', async () => {
    const fetcher = vi
      .fn<() => Promise<ApiResult<Run>>>()
      .mockResolvedValueOnce(refused())
      .mockResolvedValueOnce(ok(3));
    const store = createRecordStore<Run>(fetcher);
    await store.getState().load();
    await store.getState().load();
    expect(store.getState()).toMatchObject({ phase: 'ready', failure: null });
  });

  it('should show loading only when there is nothing to show yet', async () => {
    let release: (result: ApiResult<Run>) => void = () => {};
    const store = createRecordStore<Run>(
      () => new Promise<ApiResult<Run>>((resolve) => (release = resolve)),
    );

    const first = store.getState().load();
    expect(store.getState().phase).toBe('loading');
    release(ok(1));
    await first;

    const second = store.getState().load();
    // A refresh behind an existing value stays ready: flipping a live run view to a
    // spinner every poll is a worse screen than one that updates in place.
    expect(store.getState().phase).toBe('ready');
    release(ok(2));
    await second;
  });

  it('should apply an in-band update without a read', () => {
    const store = createRecordStore<Run>(
      async () => ok(1),
      () => 42,
    );
    store.getState().apply({ id: 'r1', step: 590 });
    expect(store.getState()).toMatchObject({
      phase: 'ready',
      value: { id: 'r1', step: 590 },
      loadedAtMs: 42,
    });
  });

  it('should derive its next poll from the transport and never return zero', () => {
    const store = createRecordStore<Run>(async () => ok(1));
    expect(store.getState().nextPollMs()).toBeGreaterThan(0);
    store.getState().setTransport('disconnected');
    const disconnected = store.getState().nextPollMs();
    store.getState().setTransport('connected');
    expect(disconnected).toBeLessThan(store.getState().nextPollMs());
  });

  it('should reset to the initial state', async () => {
    const store = createRecordStore<Run>(async () => ok(1));
    await store.getState().load();
    store.getState().reset();
    expect(store.getState()).toMatchObject({ phase: 'idle', value: null, loadedAtMs: null });
  });
});
