/**
 * The Zustand store convention (foundation task 7.4; ADR-011, ADR-013, ADR-007).
 *
 * **One store per screen** (F9 FTDD 7: "no global store, no context for data";
 * `contexts/` is for the shell's concerns). This factory is the shape they share, so a
 * feature writes its fetch and its reducers and inherits the four behaviours that are
 * easy to get wrong:
 *
 * 1. **Four explicit phases** - `idle`, `loading`, `ready`, `failed` - rather than
 *    `{data, loading, error}`, where `data === undefined && !loading && !error` is a
 *    fifth, unnamed state that every screen then renders as empty. FR-009.5 requires a
 *    loading, empty, error and refusal state for every surface; a union makes the set
 *    enumerable and a missing branch a type error under `strict`.
 * 2. **A failure is kept as an `ApiFailure`, never as a string.** The `code` is what the
 *    UI switches on (ADR-010's `<service>_not_configured` must render by name), and
 *    stringifying it here would discard it before any screen saw it.
 * 3. **A refusal does not clear the last good value.** `ready` data survives a failed
 *    refresh and both are held at once, because a run view that blanks on one failed poll
 *    of an hours-long run has thrown away the thing the operator was watching. F9's M21
 *    is the same rule for a form: "a 422 keeps the form".
 * 4. **`transport` is stored but never consulted for anything but display.** See
 *    `stores/transport.ts`.
 */

import { create, type StoreApi, type UseBoundStore } from 'zustand';
import type { ApiFailure } from '@/api/errors';
import type { ApiResult } from '@/api/client';
import { pollIntervalMs, type TransportState } from './transport';

export type RecordPhase = 'idle' | 'loading' | 'ready' | 'failed';

export interface RecordState<T> {
  readonly phase: RecordPhase;
  /** The last successfully loaded value. Survives a later failure (behaviour 3). */
  readonly value: T | null;
  /** The most recent failure. Survives alongside `value`. */
  readonly failure: ApiFailure | null;
  /** `Date.now()` of the last successful read, for "as of" copy. */
  readonly loadedAtMs: number | null;
  readonly transport: TransportState;
}

export interface RecordActions<T> {
  /** Read the record. Safe to call concurrently; the last response wins. */
  readonly load: () => Promise<void>;
  /** Replace the value from a socket event, or any other in-band update. */
  readonly apply: (next: T) => void;
  readonly setTransport: (transport: TransportState) => void;
  /** Milliseconds until the next read. Always positive - see `transport.ts`. */
  readonly nextPollMs: () => number;
  readonly reset: () => void;
}

export type RecordStore<T> = UseBoundStore<StoreApi<RecordState<T> & RecordActions<T>>>;

export function initialRecordState<T>(): RecordState<T> {
  return {
    phase: 'idle',
    value: null,
    failure: null,
    loadedAtMs: null,
    transport: 'connecting',
  };
}

/**
 * Build a store around one `fetcher`.
 *
 * @param fetcher returns an `ApiResult`, so a refusal arrives here as data rather than as
 *   a thrown exception. A fetcher that throws is a programming error and is allowed to
 *   propagate: swallowing it would make a bug look like a refusal with no code.
 * @param now injectable clock, so a test asserts a timestamp instead of tolerating one.
 */
export function createRecordStore<T>(
  fetcher: () => Promise<ApiResult<T>>,
  now: () => number = Date.now,
): RecordStore<T> {
  return create<RecordState<T> & RecordActions<T>>((set, get) => ({
    ...initialRecordState<T>(),

    load: async () => {
      // `loading` only when there is nothing to show. A refresh behind an existing value
      // stays `ready`, because flipping a live run view to a spinner every poll is a
      // worse screen than one that updates in place.
      set((state) => ({ phase: state.value === null ? 'loading' : state.phase }));
      const result = await fetcher();
      if (result.ok) {
        set({ phase: 'ready', value: result.value, failure: null, loadedAtMs: now() });
      } else {
        set((state) => ({
          phase: 'failed',
          failure: result.failure,
          // Behaviour 3: the last good value is kept on purpose.
          value: state.value,
        }));
      }
    },

    apply: (next) => {
      set({ phase: 'ready', value: next, loadedAtMs: now() });
    },

    setTransport: (transport) => set({ transport }),

    nextPollMs: () => pollIntervalMs(get().transport),

    reset: () => set(initialRecordState<T>()),
  }));
}
