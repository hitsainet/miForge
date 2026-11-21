/**
 * The current time, re-read every `CLOCK_MS`, for a screen that shows a live age or elapsed
 * clock ("updated 4 s ago", "running for 3 m 12 s"; WSP live progress, 2026-11-20).
 *
 * It is `usePolledRead` over the clock, not a timer of its own: the app has one way to re-read
 * something on an interval, and this is a read like any other (of `Date.now()`, which never
 * fails). It reads nothing from the API.
 */
import { useState } from 'react';
import type { ApiResult } from '@/api/client';
import { usePolledRead } from './usePolledRead';

export const CLOCK_MS = 1_000;

const readClock = (): Promise<ApiResult<number>> =>
  Promise.resolve({ ok: true, value: Date.now() });

export function useClock(): number {
  const [mounted] = useState(() => Date.now());
  return usePolledRead(readClock, CLOCK_MS).value ?? mounted;
}
