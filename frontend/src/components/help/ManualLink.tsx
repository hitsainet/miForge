/**
 * A link to a manual page or section, in a new tab (ENH-004).
 *
 * Origin: miDataworks (Onegaishimas/miDataworks) `frontend/src/components/help/ManualLink.tsx`
 * @ 32cf406. Mode: adapt (`docs/REUSE.md`): the label is the caller's, and the sidebar's own
 * "Manual" link stays the only link whose name starts "Manual" (`App.test.tsx` counts it).
 */
import type { ReactNode } from 'react';
import { BookOpen } from 'lucide-react';
import { manualUrl } from '@/config/manual';

export interface ManualLinkProps {
  /** `page#heading-slug`, relative to the manual's base. */
  readonly page: string;
  readonly label: string;
  readonly 'data-testid'?: string;
}

export function ManualLink({ page, label, 'data-testid': testId }: ManualLinkProps): ReactNode {
  return (
    <a
      href={manualUrl(page)}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-hint text-dim underline-offset-2 hover:text-text hover:underline"
      data-testid={testId}
    >
      <BookOpen aria-hidden="true" size={14} />
      {label}
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}
