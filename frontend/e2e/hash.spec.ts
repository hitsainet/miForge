/** {@link hashIs} compares the hash as a string, never as a pattern (CodeQL alert 27). */
import { expect, test } from '@playwright/test';
import { hashIs } from './hash';

const at = (hash: string) => new URL(`http://localhost:4173/${hash}`);

test.describe('hashIs', () => {
  test('admits the exact hash', () => {
    for (const hash of ['#/runs', '#/checkpoints', '#/packs', '#/runs/new']) {
      expect(hashIs(hash)(at(hash)), hash).toBe(true);
    }
  });

  test('a regex metacharacter in the hash is literal, not a pattern', () => {
    // Each pair matched under the old `/`-only escape: `.` is any character, `+` and `?`
    // are quantifiers, and `\` escaped the character after it.
    for (const [expected, actual] of [
      ['#/runs.v1', '#/runsXv1'],
      ['#/a+b', '#/aab'],
      ['#/runs?', '#/run'],
      ['#/a\\.b', '#/a.b'],
    ] as const) {
      expect(hashIs(expected)(at(actual)), `${expected} vs ${actual}`).toBe(false);
      expect(hashIs(expected)(at(expected)), expected).toBe(true);
    }
  });

  test('a longer hash with the same suffix does not match', () => {
    expect(hashIs('#/runs')(at('#/x/runs'))).toBe(false);
    expect(hashIs('#/runs')(at('#/runs/new'))).toBe(false);
  });
});
