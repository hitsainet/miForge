/**
 * Read a list once and hand it to a `Picker` (F9 task 14.0).
 *
 * The five loaders the New run form had each repeated the same eleven lines - a
 * `useState` for the rows, one for the failure, an effect with a cancel flag - and each
 * rendered the three states its own way. This is those lines once, returning the
 * `PickerListing` union so a call site cannot render "loading" as "empty".
 *
 * `load` is called again when its identity changes, so a filtered read (the parts of one
 * kind) passes a `useCallback` over its filter.
 */
import { useEffect, useState } from 'react';
import type { ApiResult } from '@/api/client';
import type { ApiFailure } from '@/api/errors';
import type { PickerListing, PickerOption } from '@/components/common/Picker';

export interface Listing<T> {
  /** The decoded body, once read. */
  readonly value: T | null;
  readonly failure: ApiFailure | null;
  /** The picker's view of it. */
  readonly listing: PickerListing;
}

export function useListing<T>(
  load: () => Promise<ApiResult<T>>,
  toOptions: (value: T) => readonly PickerOption[],
): Listing<T> {
  const [value, setValue] = useState<T | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  useEffect(() => {
    let cancelled = false;
    setValue(null);
    setFailure(null);
    void load().then((result) => {
      if (cancelled) return;
      if (result.ok) setValue(result.value);
      else setFailure(result.failure);
    });
    return () => {
      cancelled = true;
    };
  }, [load]);

  if (failure !== null) return { value, failure, listing: { kind: 'failed', failure } };
  if (value === null) return { value, failure, listing: { kind: 'loading' } };
  try {
    return { value, failure, listing: { kind: 'ready', options: toOptions(value) } };
  } catch (cause) {
    // A 200 whose body is not the shape this list reads - `{}` from a proxy, a route that
    // changed. Named as malformed rather than shown as an empty list, because an empty list
    // reads as a fact ("there are no packs") and this is not one.
    const malformed: ApiFailure = {
      kind: 'malformed',
      status: 200,
      reason: `the response is not the list this field reads (${String(cause)})`,
      body: JSON.stringify(value).slice(0, 2000),
    };
    return { value: null, failure: malformed, listing: { kind: 'failed', failure: malformed } };
  }
}
