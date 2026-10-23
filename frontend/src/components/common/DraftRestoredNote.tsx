/**
 * "These entries were restored" (F9 task 13.0, D-13d).
 *
 * A form that comes back from a reload with values in it must say so. Otherwise a restored
 * learning rate reads exactly like a default, and M19's rule - *a value the operator never
 * chose must not appear as theirs* - is broken by the persistence that was meant to help them.
 * Shown only when `restoredAt` is set; offers the form's own reset.
 */
import type { ReactNode } from 'react';
import { Callout } from './Callout';
import { Button } from './Button';
import { formatEt } from './time';

export interface DraftRestoredNoteProps {
  readonly restoredAt: number | null;
  readonly onDiscard: () => void;
  readonly className?: string;
}

export function DraftRestoredNote({
  restoredAt,
  onDiscard,
  className = '',
}: DraftRestoredNoteProps): ReactNode {
  if (restoredAt === null) return null;
  return (
    <Callout tone="info" title="Unsaved entries restored" className={className}>
      <span data-testid="draft-restored">
        These are the entries you left on this screen, saved in this browser at{' '}
        {formatEt(new Date(restoredAt).toISOString())}. Nothing has been submitted.{' '}
      </span>
      <Button onClick={onDiscard} data-testid="draft-discard">
        Discard them
      </Button>
    </Callout>
  );
}
