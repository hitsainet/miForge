/**
 * Buttons (foundation task 7.3; ADR-012).
 *
 * Two variants and no third. `primary` is the one permitted accent surface here;
 * `ghost` is a `line` border over the ambient background.
 *
 * **Three things this component fixes from the mockup, each from its own review.**
 *
 * 1. **`Primary` accepted no `onClick` at all**, so *every* primary action in the mockup
 *    was inert - "Attach and start", "Branch from here", "Save and continue", "Install
 *    plugin", "New pack" (handoff 6, "Found in review (2026-09-19)"). Here the props
 *    extend the real button attributes, so there is no handler to forget.
 * 2. **The hover was two inline `onMouseOver`/`onMouseOut` handlers** writing
 *    `style.background`. That is a hover that does not survive keyboard focus and cannot
 *    be themed; `hover:bg-accent-hover` is one class and works in both themes.
 * 3. **No focus styles anywhere** (handoff 6). The ring comes from `index.css`'s
 *    `:focus-visible` rule, which covers every `button` - including any written later -
 *    rather than from a class each component has to remember.
 *
 * **Copy, not decoration (ADR-012).** A button names its action: "Load and start",
 * "Create and start", "Branch from here". Never "Submit", "OK", "Go". That is enforced by
 * review of the call site, not here - but `aria-label` is deliberately not offered as a
 * substitute for visible text, because a button whose visible label is a glyph and whose
 * action lives only in its `aria-label` fails ADR-012 for every sighted operator.
 */

import type { ButtonHTMLAttributes, ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';

export type ButtonVariant = 'primary' | 'ghost';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  readonly variant?: ButtonVariant;
  readonly icon?: LucideIcon;
  /** Full width, for a form's terminal action. */
  readonly full?: boolean;
  readonly children: ReactNode;
}

const BASE =
  'inline-flex items-center justify-center gap-2 rounded-control font-medium ' +
  'transition-colors disabled:pointer-events-none';

const VARIANTS: Record<ButtonVariant, string> = {
  // accent-ok:primary-button - ADR-012 names primary buttons as a permitted surface.
  // White text on `accent` in both themes: the orange is the same hex in light and dark
  // (config/tokens.ts) precisely so this pairing does not need a second answer.
  primary: 'bg-accent text-white hover:bg-accent-hover px-4 py-3 text-body font-semibold',
  ghost: 'border border-line text-text hover:bg-panel px-3 py-2 text-body',
};

export function Button({
  variant = 'ghost',
  icon: Icon,
  full = false,
  children,
  className = '',
  type = 'button',
  ...rest
}: ButtonProps): ReactNode {
  return (
    <button
      type={type}
      className={`${BASE} ${VARIANTS[variant]} ${full ? 'w-full' : ''} ${className}`.trim()}
      {...rest}
    >
      {Icon ? <Icon size={16} aria-hidden="true" /> : null}
      {children}
    </button>
  );
}
