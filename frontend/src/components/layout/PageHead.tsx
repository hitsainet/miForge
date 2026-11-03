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
import { ArrowLeft } from 'lucide-react';
import { HelpLink } from '@/components/help/HelpLink';
import { HELP_FOR_SCREEN } from '@/config/help';
import { useOptionalNavigation, type ScreenId } from '@/contexts/NavigationContext';

export interface PageHeadProps {
  readonly title: string;
  /** One line. Required - see the module docstring. */
  readonly subtitle: string;
  /** Right-aligned actions, on the same line as the title. */
  readonly actions?: ReactNode;
  /**
   * The Help topic this head links (ENH-004 D4). **Absent: the current screen's topic**
   * (`HELP_FOR_SCREEN`), so every screen links its Help from this one place. `null`: no link
   * (the Help screen itself).
   */
  readonly help?: string | null;
  /**
   * A detail screen's way back to its list (operator request 2026-10-21). It always goes to
   * `screen`, never "back in history": a run opened from a bookmark or a log line has no
   * history to go back to, and the list is where every run starts from.
   */
  readonly back?: { readonly screen: ScreenId; readonly label: string };
}

/** The topic a head links: the explicit one, else the screen's. */
function useHelpTopic(help: string | null | undefined): string | null {
  const navigation = useOptionalNavigation();
  if (help !== undefined) return help;
  return navigation === null ? null : HELP_FOR_SCREEN[navigation.screen];
}

export function PageHead({ title, subtitle, actions, help, back }: PageHeadProps): ReactNode {
  const topic = useHelpTopic(help);
  const navigation = useOptionalNavigation();
  return (
    <div className="mb-6 flex items-start justify-between gap-6" data-testid="shell-page-head">
      <div className="min-w-0">
        {back === undefined || navigation === null ? null : (
          <button
            type="button"
            onClick={() => navigation.go(back.screen)}
            className="mb-2 inline-flex items-center gap-1 text-hint text-dim hover:underline"
            data-testid="page-head-back"
          >
            <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
            {back.label}
          </button>
        )}
        {/* 24 px semibold (handoff 2). `text-title` is generated from TYPE. */}
        {/* `text-title` IS 24 px semibold: the utility's size is the `@theme` token in
            `index.css`, held equal to TYPE by `tailwindTheme.test.ts`, so the handoff's figure
            has one definition rather than a class and an inline style that could disagree. */}
        <h1 className="text-title font-semibold">{title}</h1>
        <p className="mt-1 text-body text-dim">{subtitle}</p>
        {/* A div, not a p: the shell's end-to-end spec pins exactly one <p> per head. */}
        {topic === null ? null : (
          <div className="mt-1" data-testid="page-head-help">
            <HelpLink topic={topic} />
          </div>
        )}
      </div>
      {actions === undefined ? null : (
        <div className="flex shrink-0 items-center gap-2">{actions}</div>
      )}
    </div>
  );
}
