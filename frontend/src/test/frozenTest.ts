/**
 * `frozenTest(id, name, fn)`: a Vitest test that exercises a frozen identifier (feature 015,
 * FTID §3.7). It registers `vitest:<path>::<name>` for the entry, which
 * `frozenIdentifiers.test.ts` compares with the census export's `exercise` ids.
 */
import { it } from 'vitest';

export const FROZEN_REGISTRY = new Map<string, string>();

export function frozenTest(entryId: string, file: string, name: string, fn: () => void): void {
  const node = `vitest:${file}::${name}`;
  const existing = FROZEN_REGISTRY.get(entryId);
  if (existing !== undefined && existing !== node) {
    throw new Error(`${entryId} is already exercised by ${existing}`);
  }
  FROZEN_REGISTRY.set(entryId, node);
  it(name, fn);
}
