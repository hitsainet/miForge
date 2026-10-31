/**
 * Poll a record store on the interval its transport dictates (foundation tasks 6.6, 7.4;
 * ADR-007, ADR-013).
 *
 * The rule and its reasoning are in `stores/transport.ts`. This hook is only the timer,
 * and it is a separate file for one reason: **the decision must be testable without a
 * React tree.** `pollIntervalMs` is a pure function with its own test; if the interval
 * were computed inside the effect, the mutation control for "the socket drop falls back
 * to polling" would have to go through `renderHook` and fake timers, and a control that
 * is awkward to run is a control that does not get run.
 *
 * **It schedules the next read after the previous one resolves**, not on a fixed
 * `setInterval`. A 15 s interval over a request that takes 20 s queues requests behind
 * each other forever; this cannot.
 */

import { useEffect } from 'react';
import { pollIntervalMs, type TransportState } from '@/stores/transport';

export interface RecordPollingOptions {
  /** Read the record. Must not throw for a refusal - the store holds the failure. */
  readonly load: () => Promise<void>;
  readonly transport: TransportState;
  /**
   * Stop polling. For a run the record says is terminal: there is nothing left to
   * observe, and ADR-007's heartbeat "refuses to move a terminal row" on the other side.
   *
   * **It is the record that decides this, never the socket.** Passing a transport-derived
   * value here would reintroduce exactly the dependency `stores/transport.ts` exists to
   * forbid.
   */
  readonly paused?: boolean;
}

export function useRecordPolling({ load, transport, paused = false }: RecordPollingOptions): void {
  useEffect(() => {
    if (paused) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async (): Promise<void> => {
      await load();
      if (cancelled) return;
      timer = setTimeout(() => void tick(), pollIntervalMs(transport));
    };

    void tick();

    return () => {
      cancelled = true;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [load, transport, paused]);
}
