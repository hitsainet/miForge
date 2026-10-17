/**
 * The page head (foundation task 7.3; ADR-012, FR-009.2, handoff 3).
 *
 * > Every page opens with a title and a one-line subtitle in `dim`. Right-aligned actions
 * > sit on the same line.
 *
 * **The subtitle is required, not optional.** ADR-012 says *every* page has one, and
 * F9's FTID 3.1 puts the field in the route table for exactly this reason: *"`subtitle`
 * lives here, not in the page component, because ADR-012 requires every page to open with
 * a one-line subtitle and a field in a table is checkable"*. A `subtitle?: string` here
 * would let a screen ship without one and nothing would say so, so the prop is required
 * and `PageHead.test.tsx` asserts it renders.
 *
 * **The title is an `h1` and there is exactly one per screen.** The sidebar's product name
 * is deliberately not a heading: two `h1`s make the document outline say the page is
 * about miForge rather than about this run.
 *
 * **Accent appears nowhere here.** The subtitle is `dim` and the title is `text`.
 * ADR-012 forbids the accent on body text, and a page title is the most tempting place in
 * the product to break that - which is why the mutation control for the accent rule is
 * run against this file.
 */

import type { ReactNode } from 'react';

export interface PageHeadProps {
  readonly title: string;
  /** One line. Required - see the module docstring. */
  readonly subtitle: string;
  /** Right-aligned actions, on the same line as the title. */
  readonly actions?: ReactNode;
}

export function PageHead({ title, subtitle, actions }: PageHeadProps): ReactNode {
  return (
    <div className="mb-6 flex items-start justify-between gap-6" data-testid="shell-page-head">
      <div className="min-w-0">
        {/* 24 px semibold (handoff 2). `text-title` is generated from TYPE. */}
        {/* `text-title` IS 24 px semibold: the utility is generated from TYPE in
            `tailwind.config.ts`, so the handoff's figure has one definition rather than a
            class and an inline style that could disagree. */}
        <h1 className="text-title font-semibold">{title}</h1>
        <p className="mt-1 text-body text-dim">{subtitle}</p>
      </div>
      {actions === undefined ? null : (
        <div className="flex shrink-0 items-center gap-2">{actions}</div>
      )}
    </div>
  );
}
