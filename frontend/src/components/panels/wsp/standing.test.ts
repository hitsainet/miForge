/**
 * FU-7b: the run page's standing rule against the case table the backend's rule is held to.
 */
import { describe, expect, it } from 'vitest';
import table from './standing.cases.json';
import { verdictStanding } from './standing';

describe('[wsp] verdict standing', () => {
  it('should read the case table, all of it', () => {
    expect(table.cases.length).toBeGreaterThanOrEqual(10);
  });

  for (const c of table.cases) {
    it(`should give ${c.standing} for: ${c.case}`, () => {
      const got = verdictStanding(c.document as Readonly<Record<string, unknown>>);
      expect(got.standing).toBe(c.standing);
      expect(got.confirmedBy).toEqual(c.confirmed_by);
    });
  }
});
