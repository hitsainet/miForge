/**
 * A count with its noun, singular for exactly one (train F4, 2026-11-27).
 *
 * The checkpoint tree read "3 artifacts: 2 checkpoints, 1 adapters." because each count line
 * picked its own noun, and most picked the plural unconditionally. One helper so a count line
 * cannot drift back: `countOf(1, 'adapter')` is `1 adapter`, `countOf(0, 'adapter')` is
 * `0 adapters` (English uses the plural for zero).
 */

/** The noun alone: `singular` for exactly 1, else `plural` (default: `singular` + "s"). */
export function nounFor(count: number, singular: string, plural = `${singular}s`): string {
  return count === 1 ? singular : plural;
}

/**
 * `"<count> <noun>"`. `shown` is how the number is written (e.g. `formatCount(count)`); the
 * noun is always chosen from `count` itself, never from the written form.
 */
export function countOf(
  count: number,
  singular: string,
  plural = `${singular}s`,
  shown: string = String(count),
): string {
  return `${shown} ${nounFor(count, singular, plural)}`;
}
