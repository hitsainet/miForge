/**
 * The socket/record transport rule (foundation tasks 6.6 and 7.4; ADR-007, ADR-013).
 *
 * ADR-013: *"Live progress over Socket.IO, with stores falling back to polling the run
 * record when the socket drops - the socket is a convenience, the database is the truth."*
 *
 * **The rule, stated so it cannot be read as an optimisation.** The record is polled
 * **always**. A live socket does not switch polling off; it lets the interval relax. A
 * dropped socket shortens it. There is no state of this module in which the interval is
 * absent.
 *
 * Two failures this exact shape exists to prevent, one on each side:
 *
 * - **Trusting the socket.** A store that polls only while disconnected shows a dead run
 *   as live for as long as the socket stays open, because a connected socket that has
 *   stopped emitting looks exactly like a run that is working. F9's FR-009.4 calls this
 *   out: *"a screen that believes the socket shows a stalled run as healthy"*, and no
 *   janitor catches it - the bug is in the browser.
 * - **Trusting only the socket for progress.** ADR-007 records the inverse, already paid
 *   for: miStudio reaped a live 5.8-hour job as "worker lost" because progress went over
 *   the socket and never into the record.
 *
 * **What this module does not do.** It does not decide liveness. `deriveLiveness` is
 * F9's (its FTID 3.5, rule 1: takes the record and the clock, nothing else) and takes no
 * transport argument at all. This module answers one question - how often to ask the
 * database - and `transportState` is for *display*, which F9's FTID 3.5 rule 3 calls
 * "the only honest use of the socket's connection state: telling the operator the latency
 * changed, not the liveness."
 */

/** The socket's connection state, as the store observes it. */
export type TransportState = 'connected' | 'disconnected' | 'connecting';

/**
 * Poll intervals, in milliseconds.
 *
 * `connectedMs` is long because a connected socket is already delivering steps; the poll
 * is the floor under it, not the source. `disconnectedMs` is the fallback cadence, and it
 * is **not** aggressive: a training step is seconds at minimum and a run is hours, so a
 * five-second poll is already far finer than the thing it measures, and anything tighter
 * only loads the API.
 *
 * Neither value is a liveness threshold and neither may be used as one. F6 establishes
 * that the 60 s heartbeat throttle is *"a minimum interval between writes, not a maximum
 * gap"*, and a threshold derived from a poll interval would mark a healthy run stalled
 * during every test pass and every save - the defect F-15 withdrew from three of F9's
 * four documents and left standing in the fourth.
 */
export const POLL_INTERVALS = {
  connectedMs: 15_000,
  disconnectedMs: 5_000,
  connectingMs: 5_000,
} as const;

/**
 * How long to wait before the next record read, given the socket's state.
 *
 * **Always a positive number.** Returning `null`, `0` or `Infinity` for any state is the
 * mutation this function's test exists to catch, because each of the three is a way of
 * saying "stop reading the database", and that is the one thing the rule above forbids.
 */
export function pollIntervalMs(transport: TransportState): number {
  switch (transport) {
    case 'connected':
      return POLL_INTERVALS.connectedMs;
    case 'disconnected':
      return POLL_INTERVALS.disconnectedMs;
    case 'connecting':
      return POLL_INTERVALS.connectingMs;
  }
}

/**
 * Human copy for the transport chip. Says what it means for the *data*, not for the
 * socket - "updates every 5 s" is actionable, "websocket closed" is not (ADR-021:
 * messages say what to do next, not how the user should feel).
 */
export function describeTransport(transport: TransportState): string {
  const seconds = Math.round(pollIntervalMs(transport) / 1000);
  switch (transport) {
    case 'connected':
      return 'Live';
    case 'connecting':
      return `Connecting · reading the record every ${seconds} s`;
    case 'disconnected':
      return `Live updates unavailable · reading the record every ${seconds} s`;
  }
}
