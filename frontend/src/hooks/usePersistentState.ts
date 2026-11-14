/**
 * `useState` whose value survives a reload (F9 task 13.3; operator request 2026-10-17).
 *
 * **The one way a form keeps unsaved entries.** Every persisted form uses this hook, and
 * `components/persistence.test.tsx` proves it by walking every screen in `ROUTES`, filling
 * every editable control, remounting, and requiring each value back - so a form that keeps its
 * entries some other way, or not at all, fails that test without anybody listing it.
 *
 * Storage, the envelope, versioning, the age limit, the secret rule and the failure policy are
 * all `stores/draftStorage.ts`'s; this hook only binds them to React state.
 *
 * - The stored draft is read **once**, in the `useState` initialiser, so the first paint
 *   already shows it (no flash of an empty form).
 * - Every change is written; a value equal to `initial` **removes** the entry instead, so a
 *   form put back to empty leaves nothing behind.
 * - `clear()` sets `initial` and removes the entry **synchronously**, so a submit that
 *   navigates away in the same handler cannot leave a draft behind it.
 * - `restoredAt` is when the restored draft was saved, or `null`: forms show it, so restored
 *   entries are never mistaken for defaults (M19's concern is values nobody chose).
 * - Changing `key` (a different onboarding draft) re-reads for the new key.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { readDraft, removeDraft, writeDraft } from '@/stores/draftStorage';

export interface PersistentStateOptions<T> {
  /** Bump when the shape changes; an entry of another version is discarded. */
  readonly version: number;
  /** The form's shape check. Return `null` for anything it cannot use; it is then discarded. */
  readonly decode: (data: unknown) => T | null;
}

export interface PersistentState<T> {
  readonly value: T;
  readonly set: (next: T | ((current: T) => T)) => void;
  /** Back to `initial`, and the stored draft removed. For a successful submit and for reset. */
  readonly clear: () => void;
  /** When the draft on screen was saved, if it was restored from storage. */
  readonly restoredAt: number | null;
}

function sameJson(a: unknown, b: unknown): boolean {
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return false;
  }
}

export function usePersistentState<T>(
  key: string | null,
  initial: T,
  { version, decode }: PersistentStateOptions<T>,
): PersistentState<T> {
  const load = (forKey: string | null): { value: T; restoredAt: number | null } => {
    if (forKey === null) return { value: initial, restoredAt: null };
    const stored = readDraft(forKey, version, decode);
    return stored === null
      ? { value: initial, restoredAt: null }
      : { value: stored.data, restoredAt: stored.savedAt };
  };

  const [state, setState] = useState(() => ({ key, ...load(key) }));

  // A different key is a different draft: re-read during render, React's documented pattern
  // for state derived from a prop, so the old draft is never written under the new key.
  let current = state;
  if (state.key !== key) {
    current = { key, ...load(key) };
    setState(current);
  }

  const initialRef = useRef(initial);
  initialRef.current = initial;

  // State -> storage. Removing this effect leaves nothing to restore.
  useEffect(() => {
    if (current.key === null) return;
    if (sameJson(current.value, initialRef.current)) removeDraft(current.key);
    else writeDraft(current.key, version, current.value);
  }, [current.key, current.value, version]);

  const set = useCallback((next: T | ((value: T) => T)) => {
    setState((previous) => ({
      ...previous,
      value: typeof next === 'function' ? (next as (value: T) => T)(previous.value) : next,
    }));
  }, []);

  const keyRef = useRef(key);
  keyRef.current = key;

  const clear = useCallback(() => {
    // Removed HERE, not in the state updater: a submit that navigates away unmounts this
    // component in the same batch, and a queued update of an unmounted component never runs.
    if (keyRef.current !== null) removeDraft(keyRef.current);
    setState((previous) => ({ key: previous.key, value: initialRef.current, restoredAt: null }));
  }, []);

  return { value: current.value, set, clear, restoredAt: current.restoredAt };
}
