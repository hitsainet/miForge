/**
 * The one orange panel on the packs screen (F7 task 12.3).
 *
 * It carries the mockup's copy, and it is the **only** accent surface on that screen -
 * ADR-012 permits "the callout banner tint" and `Callout tone="accent"` is that one surface.
 * *If a screen starts to look orange, something is wrong.*
 *
 * The sentence is load-bearing rather than decorative: it is the reason `PartReference`
 * refuses to invent a pack segment for an unowned part (F7's OQ-9), and the reason this
 * product ships no example packs rather than seeding ones that look real.
 */
import type { ReactNode } from 'react';
import { Callout } from '@/components/common/Callout';

export const NO_BUILT_IN_DOMAIN_TITLE = 'miForge has no built-in domain.';

export function NoBuiltInDomainBanner({
  className = '',
}: {
  readonly className?: string;
}): ReactNode {
  return (
    <Callout
      tone="accent"
      title={NO_BUILT_IN_DOMAIN_TITLE}
      className={className}
      data-testid="no-built-in-domain"
    >
      A pack is how you describe one: a prompt set, optionally a corpus and preference pairs, a
      selector and grader configuration, a test set, and a retention set with its floor. Packs hold
      references to parts, never copies — two installations with &ldquo;the same pack&rdquo; must
      not end up with two datasets drifting apart.
    </Callout>
  );
}
