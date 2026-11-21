/**
 * The suite shell (foundation task 7.3; ADR-012, FR-009.2).
 *
 * ```
 * ┌──────────┬──────────────────────────────────────────────┐
 * │ logo     │ resource strip                    [trailing] │
 * │ miForge  ├──────────────────────────────────────────────┤
 * │ nav      │ page head: title / subtitle      [actions]   │
 * │  …       │ content, max-width ~72rem, left aligned      │
 * │ ──────── │                                              │
 * │ Settings │                                              │
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
 *
 * **The page scrolls; the sidebar does not** (operator, 2026-10-20: *"The left navigation bar
 * should not scroll. The pages should scroll independently from it."*). The shell is exactly
 * one viewport tall (`.mf-shell`, `100dvh`) and never lets the document scroll. Three
 * regions own their own overflow:
 *
 * - the **sidebar** is the viewport's height; its **list of screens** scrolls within itself
 *   when it outgrows a short window, while the logo and the footer (Settings, the Manual, the
 *   version) stay pinned at the sidebar's top and bottom;
 * - the **resource strip** sits above the scroll container, so it stays put;
 * - **`shell-main`** is the scroll container for the screen (vertical, and horizontal when a
 *   screen is wider than a narrow window, so the document never scrolls sideways either).
 *
 * Because the window no longer scrolls, anything that wants "the page's scroll position" must
 * ask `shell-main`. The one such need today is resetting it when the screen changes, which is
 * `scrollKey` below: without it a new screen would open part-way down, at the scroll offset
 * the previous screen was left at. `e2e/layout.spec.ts` pins all of this in both themes.
 */

import { useLayoutEffect, useRef, type ReactNode } from 'react';
import { SHELL } from '@/config/tokens';
import { Sidebar } from './Sidebar';
import { ResourceStrip } from './ResourceStrip';
import { ThemeToggle } from './ThemeToggle';
import type { ResourceReading, ShellNavItem } from './types';

export interface AppShellProps {
  readonly nav: readonly ShellNavItem[];
  readonly activeId: string | null;
  readonly onNavigate: (id: string) => void;
  readonly version: string;
  readonly resources: readonly ResourceReading[];
  /** The resource strip's right-hand slot. Empty on the core path. */
  readonly resourceTrailing?: ReactNode;
  /**
   * Identifies the screen being shown. When it changes the content area is scrolled back to
   * the top - the window no longer scrolls, so nothing else would. Leave it stable for
   * in-screen changes (a filter, a tab) that should keep the operator's place.
   */
  readonly scrollKey?: string;
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
  scrollKey,
  children,
}: AppShellProps): ReactNode {
  const main = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    const element = main.current;
    if (element === null) return;
    // `scrollTo` is absent in jsdom; the property write works everywhere.
    element.scrollTop = 0;
    element.scrollLeft = 0;
  }, [scrollKey]);

  return (
    <div className="mf-shell flex bg-bg text-body text-text" data-testid="shell-root">
      <Sidebar nav={nav} activeId={activeId} onNavigate={onNavigate} version={version} />
      <div className="flex min-w-0 flex-1 flex-col" data-testid="shell-column">
        {/* The theme toggle sits at the strip's right end, where miStudio and miLLM keep theirs
            in the header; the sidebar's footer is Settings, the Manual and the version only. */}
        <ResourceStrip
          readings={resources}
          trailing={
            <>
              {resourceTrailing}
              <ThemeToggle />
            </>
          }
        />
        <main ref={main} className="min-h-0 flex-1 overflow-auto" data-testid="shell-main">
          <div
            className="px-8 py-6"
            style={{ maxWidth: SHELL.contentMaxWidthPx }}
            data-testid="shell-content"
          >
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
