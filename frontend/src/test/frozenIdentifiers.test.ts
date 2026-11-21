/**
 * Every `vitest:` exercise the census export names is a registered `frozenTest` (feature 015,
 * FTID §3.7). miForge's frozen identifiers are all exercised by backend tests at P0, so the set
 * is empty - and the loader must still have read entries, or the comparison would pass on an
 * empty file.
 */
import { describe, expect, it } from 'vitest';
import { FROZEN_REGISTRY } from './frozenTest';
import { loadExport } from './legacyIdentifiers';

describe('frozen identifiers (frontend)', () => {
  it('every vitest exercise id names a registered frozen test', () => {
    const exported = loadExport();
    expect(exported.entries.length).toBeGreaterThan(0);
    const wanted = new Map(
      exported.entries
        .filter((e) => e.exercise?.startsWith('vitest:'))
        .map((e) => [e.id, e.exercise as string]),
    );
    expect(new Map(FROZEN_REGISTRY)).toStrictEqual(wanted);
    for (const exercise of wanted.values()) {
      expect(exercise.startsWith('vitest:frontend/')).toBe(true);
    }
  });
});
