/**
 * Read an endpoint now and again every `intervalMs`, keeping the latest answer.
 *
 * For the shell's two live reads - host resources and the attachment chip - which are on
 * every screen and describe the present, not a record. Like `useRecordPolling` it schedules
 * the next read after the previous one resolves, so a slow endpoint cannot queue requests.
 * A failed read keeps the last good value beside the failure, so the caller can say both
 * "unreachable" and how old the last reading is.
 *
 * **A changed interval applies from the next read** (WSP live progress, 2026-11-20): it is held
 * in a ref, so a screen that relaxes or tightens its cadence (the WSP runs list: 5 s while a run
 * is live, 30 s otherwise) does not restart the loop and send an extra read at the change. A
 * changed `load` still restarts it, with a read at once: that is a different question.
 */
import { useEffect, useRef, useState } from 'react';
import type { ApiResult } from '@/api/client';
import type { ApiFailure } from '@/api/errors';

export interface PolledRead<T> {
  readonly value: T | null;
  readonly failure: ApiFailure | null;
}

export function usePolledRead<T>(
  load: () => Promise<ApiResult<T>>,
  intervalMs: number,
): PolledRead<T> {
  const [state, setState] = useState<PolledRead<T>>({ value: null, failure: null });
  const every = useRef(intervalMs);
  useEffect(() => {
    every.current = intervalMs;
  }, [intervalMs]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async (): Promise<void> => {
      const result = await load();
      if (cancelled) return;
      setState((previous) =>
        result.ok
          ? { value: result.value, failure: null }
          : { value: previous.value, failure: result.failure },
      );
      timer = setTimeout(() => void tick(), every.current);
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [load]);

  return state;
}
