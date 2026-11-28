/**
 * The run's progress channel (FR-009.4; ADR-007).
 *
 * **Two things this hook may do and one it may not.** It may update the series between polls,
 * and it may report the transport so the operator knows the latency changed. It may **not**
 * decide whether the run is live: that comes from `deriveLiveness`, which takes the record.
 * The connection state is reported, never consulted - which is the only honest use of it.
 *
 * *Progress reported only over the socket is invisible to the janitor* - that is how miStudio
 * reaped a live 5.8-hour job as "worker lost" (ADR-007). So the socket is additive to the
 * poll and never a substitute for it: `useRecordPolling` keeps running while this is
 * connected, and the poll interval merely lengthens.
 *
 * **The event names are the emitters', not this feature's.** `RUN_EVENT_NAMES` in
 * `stores/runStore.ts` lists the six, each traced to the feature that emits it. An earlier
 * draft of F9's own documents invented four names (`step | score | save | signal`) and they
 * are withdrawn: a screen subscribing to names the emitter does not emit is **silently dead
 * and looks connected**, which is the exact failure direction FR-009.4 exists to prevent.
 */
import { useEffect } from 'react';
import { io, type Socket } from 'socket.io-client';
import { RUN_EVENT_NAMES, type RunSocketEvent } from '@/stores/runStore';
import type { TransportState } from '@/stores/transport';

/** `miforge/run/<id>`, the room ADR-007 names. */
export function runRoom(runId: string): string {
  return `miforge/run/${runId}`;
}

export interface RunSocketOptions {
  readonly runId: string | null;
  readonly onEvent: (event: RunSocketEvent) => void;
  readonly onTransport: (transport: TransportState) => void;
  /** Injected in tests. A hook that can only be exercised against a real server is untested. */
  readonly connect?: (room: string) => Socket;
}

function defaultConnect(): Socket {
  // Same origin. ADR-022 fences the core path off every sibling endpoint, and a browser
  // `fetch` or socket is on that path: there is exactly one host the UI may talk to.
  return io({ path: '/socket.io', transports: ['websocket', 'polling'] });
}

export function useRunSocket({
  runId,
  onEvent,
  onTransport,
  connect = defaultConnect,
}: RunSocketOptions): void {
  useEffect(() => {
    if (runId === null) return;
    const room = runRoom(runId);
    const socket = connect(room);

    const join = (): void => {
      onTransport('connected');
      socket.emit('join', { room });
    };
    const drop = (): void => onTransport('disconnected');

    socket.on('connect', join);
    socket.on('disconnect', drop);
    socket.on('connect_error', drop);

    const handlers = RUN_EVENT_NAMES.map((name) => {
      const handler = (payload: unknown): void => {
        onEvent({ name, payload: asRecord(payload) });
      };
      socket.on(name, handler);
      return [name, handler] as const;
    });

    return () => {
      socket.off('connect', join);
      socket.off('disconnect', drop);
      socket.off('connect_error', drop);
      for (const [name, handler] of handlers) socket.off(name, handler);
      socket.disconnect();
    };
  }, [runId, onEvent, onTransport, connect]);
}

function asRecord(payload: unknown): Readonly<Record<string, unknown>> {
  return payload !== null && typeof payload === 'object' && !Array.isArray(payload)
    ? (payload as Readonly<Record<string, unknown>>)
    : {};
}
