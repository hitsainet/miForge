/**
 * A screen's link to its Help topic, with the matching manual section beside it (ENH-004 D4).
 *
 * Origin: miDataworks (Onegaishimas/miDataworks) `frontend/src/components/help/HelpLink.tsx`
 * @ 32cf406. Mode: adapt (`docs/REUSE.md`). Changed: the topic travels in the URL
 * (`#/help/<topic>`, D2) through the navigation, not through a store; outside a navigation
 * provider (a component test of one screen) it renders nothing rather than an inert button.
 *
 * Exactly one manual link per HelpLink (`test/manualLinks.test.tsx`). An unknown topic renders
 * nothing: a Help button that opens no topic would be worse than none.
 */
import type { ReactNode } from 'react';
import { CircleHelp } from 'lucide-react';
import { helpTopic } from '@/config/help';
import { useOptionalNavigation } from '@/contexts/NavigationContext';
import { ManualLink } from './ManualLink';

export interface HelpLinkProps {
  readonly topic: string;
  /** The button's words. "Help" in a page head; "Help with this step" in a flow. */
  readonly label?: string;
}

export function HelpLink({ topic, label = 'Help' }: HelpLinkProps): ReactNode {
  const navigation = useOptionalNavigation();
  const entry = helpTopic(topic);
  if (navigation === null || entry === undefined) return null;
  return (
    <div className="flex flex-wrap items-center gap-3" data-testid={`help-link-${topic}`}>
      <button
        type="button"
        className="inline-flex items-center gap-1 text-hint text-dim underline-offset-2 hover:text-text hover:underline"
        onClick={() => navigation.go('help', { topic })}
        data-testid={`help-open-${topic}`}
      >
        <CircleHelp aria-hidden="true" size={14} />
        {label}
        <span className="sr-only">: {entry.title}</span>
      </button>
      <ManualLink page={entry.manual} label="In the manual" data-testid={`manual-link-${topic}`} />
    </div>
  );
}
