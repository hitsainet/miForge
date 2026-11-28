/**
 * A verdict's standing (FU-7b, operator decision 2026-11-24): an `improved` verdict stands only
 * after a fresh held-out confirmation. It is `improved` when at least one PRIMARY held-out set
 * reads `improved` in it and was scored for the first time by it (`heldout_uses` 1); otherwise it
 * is `improved_unconfirmed`, not to be acted on until a held-out set never scored before confirms
 * it. Every other value stands as it is.
 *
 * The same rule as `backend/src/wsp/standing.py`, read from the run document's verdict alone (no
 * contract field); both are held to one case table, `standing.cases.json`.
 */

export const UNCONFIRMED = 'improved_unconfirmed';

export interface VerdictStanding {
  readonly standing: string;
  readonly confirmedBy: readonly string[];
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function confirmedBy(verdict: Readonly<Record<string, unknown>>): string[] {
  if (verdict['value'] !== 'improved') return [];
  const sets = Array.isArray(verdict['per_set']) ? (verdict['per_set'] as unknown[]) : [];
  return sets
    .filter(
      (entry): entry is Readonly<Record<string, unknown>> =>
        isRecord(entry) &&
        entry['primary'] === true &&
        entry['value'] === 'improved' &&
        entry['heldout_uses'] === 1,
    )
    .map((entry) => String(entry['name']));
}

export function verdictStanding(verdict: Readonly<Record<string, unknown>>): VerdictStanding {
  const value = String(verdict['value']);
  const by = confirmedBy(verdict);
  if (value !== 'improved') return { standing: value, confirmedBy: by };
  return { standing: by.length > 0 ? 'improved' : UNCONFIRMED, confirmedBy: by };
}
