/**
 * Read an endpoint now and again every `intervalMs`, keeping the latest answer.
 *
 * For the shell's two live reads - host resources and the attachment chip - which are on
 * every screen and describe the present, not a record. Like `useRecordPolling` it schedules
 * the next read after the previous one resolves, so a slow endpoint cannot queue requests.
 * A failed read keeps the last good value beside the failure, so the caller can say both
 * "unreachable" and how old the last reading is.
 */
import { useEffect, useState } from 'react';
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
      timer = setTimeout(() => void tick(), intervalMs);
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [load, intervalMs]);

  return state;
}
