/**
 * F3's `SelectionHealth`, rendered **whole** (F7 task 12.7, 10.1a; FR-009.15).
 *
 * `Accepted 3, rejected 1, undecided 1 of 5` - all three counts and their denominator, from
 * the record. **This component computes no rate of its own.** Two reasons, and both are
 * recorded failures rather than preferences:
 *
 * - *"A single place that treats `undecided` as a reject makes the health record lie about an
 *   unattended queue"* (F2's review). `Accepted 41 of 64` on a batch with 12 pending is that
 *   lie, and the step log is one of the places that could tell it.
 * - A second place holding the counts is a second answer. `acceptance_rate` arrives beside
 *   its denominator precisely so a bare percentage is not expressible (ADR-012), and
 *   recomputing it here would reintroduce the thing the record's shape prevents.
 */
import type { ReactNode } from 'react';
import type { SelectionHealth } from '@/types/packs';
import { formatCount } from './NumberWithSample';

export interface SelectionCountsProps {
  readonly health: Pick<SelectionHealth, 'n' | 'accepted' | 'rejected' | 'undecided'>;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

/** The one sentence form, so the step log and the dry-run panel cannot diverge. */
export function selectionCountsText(
  health: Pick<SelectionHealth, 'n' | 'accepted' | 'rejected' | 'undecided'>,
): string {
  return (
    `Accepted ${formatCount(health.accepted)}, ` +
    `rejected ${formatCount(health.rejected)}, ` +
    `undecided ${formatCount(health.undecided)} ` +
    `of ${formatCount(health.n)}`
  );
}

export function SelectionCounts({
  health,
  className = '',
  ...rest
}: SelectionCountsProps): ReactNode {
  return (
    <span
      className={`tabular-nums ${className}`.trim()}
      data-counts={`${health.accepted}/${health.rejected}/${health.undecided}/${health.n}`}
      {...rest}
    >
      {selectionCountsText(health)}
    </span>
  );
}
