/**
 * The suite shell (foundation task 7.3; ADR-012, FR-009.2).
 *
 * ```
 * ┌──────────┬──────────────────────────────────────────────┐
 * │ logo     │ resource strip                    [trailing] │
 * │ miForge  ├──────────────────────────────────────────────┤
 * │ nav      │ page head: title / subtitle      [actions]   │
 * │  …       │ content, max-width ~72rem, left aligned      │
 * │ v0.1.0   │                                              │
 * └──────────┴──────────────────────────────────────────────┘
 * ```
 *
 * Four regions, each its own component and each with its own `data-testid` so the
 * Playwright smoke spec can assert *per region* (foundation 7.6) rather than asserting
 * that a page rendered.
 *
 * **Three things this component does not do, each on purpose.**
 *
 * 1. **It does not own the nav.** F9's route table is the live registry (FR-009.1); a
 *    second list of screens here would be the thing ADR-016 #1's reachability test is not
 *    looking at. The nav arrives as a prop.
 * 2. **It does not route.** ADR-011 lists no router and F9's F-6 records that adding
 *    `react-router-dom` is an ADR amendment the operator owns. `AppShell` renders
 *    `children` and reports navigation intent through `onNavigate`, so whichever way that
 *    is settled, the shell is unaffected.
 * 3. **It knows nothing about any sibling.** No attachment chip, no miLLM status. ADR-022
 *    fences 5.10 off the core path and the shell is on every screen; the handoff's 6c
 *    moves the attachment chip to the hot-update mode. The `resourceTrailing` slot is
 *    where F11 puts it.
 *
 * **Desktop only.** PPRD 7 and the mockup's 6 both say so; there is no responsive work
 * here and the content column is a `max-width`, not a breakpoint ladder.
 */

import type { ReactNode } from 'react';
import { SHELL } from '@/config/tokens';
import { Sidebar } from './Sidebar';
import { ResourceStrip } from './ResourceStrip';
import type { ResourceReading, ShellNavItem } from './types';

export interface AppShellProps {
  readonly nav: readonly ShellNavItem[];
  readonly activeId: string | null;
  readonly onNavigate: (id: string) => void;
  readonly version: string;
  readonly resources: readonly ResourceReading[];
  /** The resource strip's right-hand slot. Empty on the core path. */
  readonly resourceTrailing?: ReactNode;
  /** The screen: its own `PageHead` followed by its panels. */
  readonly children: ReactNode;
}

export function AppShell({
  nav,
  activeId,
  onNavigate,
  version,
  resources,
  resourceTrailing,
  children,
}: AppShellProps): ReactNode {
  return (
    <div className="flex min-h-screen bg-bg text-body text-text" data-testid="shell-root">
      <Sidebar nav={nav} activeId={activeId} onNavigate={onNavigate} version={version} />
      <main className="min-w-0 flex-1" data-testid="shell-main">
        <ResourceStrip readings={resources} trailing={resourceTrailing} />
        <div
          className="px-8 py-6"
          style={{ maxWidth: SHELL.contentMaxWidthPx }}
          data-testid="shell-content"
        >
          {children}
        </div>
      </main>
    </div>
  );
}
