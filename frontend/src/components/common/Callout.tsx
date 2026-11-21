/**
 * The callout banner (foundation task 7.3; ADR-012).
 *
 * ADR-012's seventh permitted accent surface is *"the callout banner tint"* - the
 * orange-tinted card the mockup uses for *"miForge has no built-in domain."* The handoff
 * calls that copy intentional and says to keep it or an equivalent, so the banner is a
 * shell primitive rather than one screen's markup.
 *
 * **Four tones, and only `accent` tints with the accent.** `warning` is amber,
 * `error` red, `info` neutral `panel`. ADR-012 permits the accent on *the callout banner
 * tint* - singular, the product statement - and an orange warning would make warnings and
 * identity the same colour, which is how an operator stops reading either.
 *
 * **Copy rule, from ADR-012 and the handoff 5:** *"Empty and warning states say what to
 * do next, not how the user feels."* So `title` is a statement and `children` says what
 * follows from it. There is no `variant="success"`: a banner congratulating the operator
 * is not information.
 */

import type { ReactNode } from 'react';
import { AlertTriangle, Info, OctagonAlert, type LucideIcon } from 'lucide-react';

export type CalloutTone = 'accent' | 'warning' | 'error' | 'info';

interface ToneStyle {
  readonly container: string;
  readonly glyph: string;
  readonly icon: LucideIcon;
  /** `role="alert"` for the two that demand attention now; a region otherwise. */
  readonly assertive: boolean;
}

const TONES: Record<CalloutTone, ToneStyle> = {
  // accent-ok:callout-tint - ADR-012's seventh surface. The border is the accent at a
  // low alpha, matching the mockup's `rgba(249,115,22,0.35)`; it is the one place a
  // tinted edge is permitted, and it is the banner's tint rather than a card border.
  accent: {
    container: 'bg-accent-tint border-accent/35',
    glyph: 'text-accent',
    icon: Info,
    assertive: false,
  },
  warning: {
    container: 'bg-panel border-line',
    glyph: 'text-amber',
    icon: AlertTriangle,
    assertive: true,
  },
  error: {
    container: 'bg-panel border-line',
    glyph: 'text-red',
    icon: OctagonAlert,
    assertive: true,
  },
  info: {
    container: 'bg-panel border-line',
    glyph: 'text-dim',
    icon: Info,
    assertive: false,
  },
};

export interface CalloutProps {
  readonly tone: CalloutTone;
  /** A statement, in sentence case. Optional: a one-line banner needs no heading. */
  readonly title?: string;
  /** What follows from it, or what to do next. */
  readonly children: ReactNode;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function Callout({
  tone,
  title,
  children,
  className = '',
  ...rest
}: CalloutProps): ReactNode {
  const style = TONES[tone];
  const Glyph = style.icon;
  return (
    <div
      className={`flex items-start gap-3 rounded-card border p-card ${style.container} ${className}`.trim()}
      role={style.assertive ? 'alert' : 'region'}
      data-tone={tone}
      {...rest}
    >
      <Glyph size={18} className={`mt-0.5 shrink-0 ${style.glyph}`} aria-hidden="true" />
      <div className="min-w-0 text-body">
        {title === undefined ? null : <div className="mb-1 font-semibold">{title}</div>}
        <div className="text-dim">{children}</div>
      </div>
    </div>
  );
}
