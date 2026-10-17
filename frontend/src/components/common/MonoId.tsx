/**
 * Identifiers, hashes and step logs in JetBrains Mono (foundation task 7.3; ADR-012).
 *
 * > Inter for text, **JetBrains Mono for identifiers, hashes and step logs.**
 *
 * **One component so the font is applied in one place.** F9's FTID 2.1 puts it exactly
 * that way: *"Identifiers, hashes and qualified names render through `MonoId`, so
 * JetBrains Mono is applied in one place rather than remembered at 30 call sites."* The
 * alternative is `font-mono` sprinkled across every screen, which is the state in which
 * one screen renders a 64-character digest in Inter and nobody notices until it is
 * mistaken for prose.
 *
 * **`truncate` is opt-in and always keeps the full value in the title.** A shortened hash
 * with no way to read the rest is a value the operator cannot verify, and verifying a
 * digest against a manifest is a thing this product asks them to do.
 */

import type { ReactNode } from 'react';

export type MonoIdVariant = 'inline' | 'block';

export interface MonoIdProps {
  /** The identifier itself. Rendered verbatim; never reformatted or case-folded. */
  readonly value: string;
  /**
   * Show the first and last `n` characters only, with the whole value in the tooltip and
   * in the accessible name. Off by default: a truncated identifier is a decision, and
   * the default should not be the one that hides information.
   */
  readonly truncateTo?: number;
  readonly variant?: MonoIdVariant;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

/** Exported so the test asserts the font utility rather than re-typing the class list. */
export const MONO_CLASSES = 'font-mono text-ident';

export function MonoId({
  value,
  truncateTo,
  variant = 'inline',
  className = '',
  ...rest
}: MonoIdProps): ReactNode {
  const shortened = shortenIdentifier(value, truncateTo);
  const Element = variant === 'block' ? 'pre' : 'span';
  return (
    <Element
      className={`${MONO_CLASSES} ${variant === 'block' ? 'whitespace-pre-wrap' : ''} ${className}`.trim()}
      title={shortened === value ? undefined : value}
      aria-label={shortened === value ? undefined : value}
      {...rest}
    >
      {shortened}
    </Element>
  );
}

/**
 * `7c1e4f…a0b1` from a long digest; the value unchanged when it is already short enough.
 *
 * Returns the input untouched when `truncateTo` is absent **or** when truncating would
 * not actually shorten it - an ellipsis that replaces two characters costs a reader more
 * than it saves.
 */
export function shortenIdentifier(value: string, truncateTo?: number): string {
  if (truncateTo === undefined || truncateTo <= 0) return value;
  if (value.length <= truncateTo * 2 + 1) return value;
  return `${value.slice(0, truncateTo)}…${value.slice(-truncateTo)}`;
}
