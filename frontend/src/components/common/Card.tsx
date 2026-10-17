/**
 * The card (foundation task 7.3; ADR-012's card grammar).
 *
 * > Cards 12 px radius, 1 px `line` border, 20 px padding.
 *
 * The three numbers are Tailwind tokens generated from `config/tokens.ts`
 * (`rounded-card`, `border-line`, `p-card`), so the grammar has one definition and
 * `Card.test.tsx` can assert against the same constants the ADR quotes.
 *
 * **The border is `line`, never accent.** ADR-012 forbids the accent on borders, and a
 * card is the single most repeated surface in the product: an accent border here is how
 * "the screen starts to look orange".
 */

import type { ReactNode } from 'react';

export interface CardProps {
  readonly children: ReactNode;
  readonly className?: string;
  /** Inline styles for the one thing utilities cannot express: the tree's depth indent. */
  readonly style?: React.CSSProperties;
  /** Render as a different element - `li` inside a list, `article` for a region. */
  readonly as?: 'div' | 'li' | 'article' | 'section';
  readonly 'aria-label'?: string;
  readonly 'data-testid'?: string;
}

/** The base utilities, exported so the tests assert the grammar rather than re-type it. */
export const CARD_CLASSES = 'rounded-card border border-line bg-card p-card';

export function Card({
  children,
  className = '',
  style,
  as: Element = 'div',
  ...rest
}: CardProps): ReactNode {
  return (
    <Element className={`${CARD_CLASSES} ${className}`.trim()} style={style} {...rest}>
      {children}
    </Element>
  );
}
