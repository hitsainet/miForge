/**
 * One remembered on/off choice - "Show archived runs/parts" (operator decision 2026-10-20) -
 * kept across a reload like every other entry on a screen (AC-009.31). A stored value that
 * is not a boolean decodes to the default.
 */
import { usePersistentState } from './usePersistentState';

export function usePersistentFlag(key: string): readonly [boolean, (next: boolean) => void] {
  const state = usePersistentState<boolean>(key, false, {
    version: 1,
    decode: (data) => (typeof data === 'boolean' ? data : null),
  });
  return [state.value, (next: boolean) => state.set(next)] as const;
}
