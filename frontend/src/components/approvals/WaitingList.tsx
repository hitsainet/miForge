/**
 * A list of waiting cards (ENH-008 §2.1, §2.7), used by the banner and the Approvals screen.
 *
 * Focus: a failed decision keeps focus on its own card (the card does that); a decision the
 * server accepted moves focus to the next card - the one that followed it, else the one before
 * it - once the re-read list arrives.
 */
import { useEffect, useRef, type ReactNode } from 'react';
import type { Approval } from '@/api/approvals';
import { ApprovalCard } from './ApprovalCard';

export interface WaitingListProps {
  readonly approvals: readonly Approval[];
  readonly label: string;
  readonly 'data-testid'?: string;
}

export function WaitingList({ approvals, label, ...rest }: WaitingListProps): ReactNode {
  const cards = useRef(new Map<string, HTMLElement>());
  // The candidates to focus once the decided card has left the list; a ref, not state, because
  // nothing renders from it.
  const focusNext = useRef<readonly string[] | null>(null);

  useEffect(() => {
    const candidates = focusNext.current;
    if (candidates === null) return;
    const present = new Set(approvals.map((row) => row.approval_id));
    const target = candidates.find((id) => present.has(id));
    if (target === undefined) return;
    cards.current.get(target)?.focus();
    focusNext.current = null;
  }, [approvals]);

  const onDecided = (id: string): void => {
    const ids = approvals.map((row) => row.approval_id);
    const index = ids.indexOf(id);
    if (index < 0) return;
    const candidates = [...ids.slice(index + 1), ...ids.slice(0, index).reverse()];
    // The store re-reads the list before the decision settles, so the next card is usually
    // mounted already; otherwise the effect focuses it when the list arrives.
    if (!cards.current.has(id)) {
      const target = candidates.find((candidate) => cards.current.has(candidate));
      if (target !== undefined) cards.current.get(target)?.focus();
      return;
    }
    focusNext.current = candidates;
  };

  return (
    <ul className="flex flex-col gap-3" aria-label={label} {...rest}>
      {approvals.map((approval) => (
        <ApprovalCard
          key={approval.approval_id}
          approval={approval}
          onDecided={onDecided}
          cardRef={(element) => {
            if (element === null) cards.current.delete(approval.approval_id);
            else cards.current.set(approval.approval_id, element);
          }}
        />
      ))}
    </ul>
  );
}
