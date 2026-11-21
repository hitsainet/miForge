/**
 * The socket-drop fallback (foundation tasks 6.6, 7.4; ADR-007, ADR-013).
 *
 * ADR-013: *"stores falling back to polling the run record when the socket drops - the
 * socket is a convenience, the database is the truth."*
 *
 * **The assertion that matters is the third one**: the record is polled in *every*
 * transport state, including `connected`. A store that polls only while disconnected
 * shows a dead run as live for as long as the socket stays open, because a connected
 * socket that has stopped emitting is indistinguishable from a run that is working - and
 * no janitor catches that one, because the bug is in the browser.
 */

import { describe, expect, it } from 'vitest';
import {
  describeTransport,
  POLL_INTERVALS,
  pollIntervalMs,
  type TransportState,
} from './transport';

const ALL_STATES: readonly TransportState[] = ['connected', 'connecting', 'disconnected'];

describe('[transport polling]', () => {
  it('should poll faster when the socket is down than when it is up', () => {
    expect(pollIntervalMs('disconnected')).toBeLessThan(pollIntervalMs('connected'));
  });

  it('should poll the record in every transport state, including connected', () => {
    // This is the rule, not an optimisation. A live socket relaxes the interval; it never
    // switches polling off.
    for (const state of ALL_STATES) {
      const interval = pollIntervalMs(state);
      expect(Number.isFinite(interval), `${state} must have a finite interval`).toBe(true);
      expect(interval, `${state} must still poll`).toBeGreaterThan(0);
    }
  });

  it('should use the fallback cadence while connecting, not the relaxed one', () => {
    // "Connecting" has delivered nothing yet, so it is the disconnected case until it is
    // not. Treating it as connected is the optimistic reading and it hides a socket that
    // never finishes connecting.
    expect(pollIntervalMs('connecting')).toBe(POLL_INTERVALS.disconnectedMs);
  });

  it('should keep every interval far finer than the thing it measures', () => {
    // A training step is seconds at minimum and a run is hours. Anything tighter than
    // this only loads the API; anything looser and the fallback is not a fallback.
    for (const state of ALL_STATES) {
      expect(pollIntervalMs(state)).toBeLessThanOrEqual(30_000);
      expect(pollIntervalMs(state)).toBeGreaterThanOrEqual(1_000);
    }
  });

  it('should not expose any interval that could be mistaken for a liveness threshold', () => {
    // F6: the 60 s heartbeat throttle is "a minimum interval between writes, not a
    // maximum gap", and a threshold derived from it marks a healthy run stalled during
    // every test pass and every save. No value here is 60 s, so nobody can reach for one.
    for (const state of ALL_STATES) {
      expect(pollIntervalMs(state)).not.toBe(60_000);
    }
  });

  it('should describe the transport in terms of the data, not the socket', () => {
    expect(describeTransport('connected')).toBe('Live');
    expect(describeTransport('disconnected')).toContain('reading the record every 5 s');
    // Says what it means for the data and what happens next (ADR-021), not
    // "websocket closed".
    expect(describeTransport('disconnected')).not.toMatch(/socket|websocket/i);
  });
});
