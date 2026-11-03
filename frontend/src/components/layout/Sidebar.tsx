/**
 * The 224 px sidebar (foundation task 7.3; ADR-012, FR-009.2).
 *
 * > Sidebar 224 px, `panel` background, right border. Logo tile is solid accent with a
 * > white Flame. Subtitle under the name reads "AI Model Foundry."
 *
 * The width is `SHELL.sidebarWidthPx`, not a literal and not `w-56`: ADR-012 and
 * FR-009.2 both quote **224**, and `Sidebar.test.tsx` compares the rendered width to the
 * same constant the documents name. `w-56` is the same 224 px, but checking it requires
 * knowing Tailwind's scale, and that is one indirection between a document and a test.
 *
 * **It is the viewport's height and never scrolls with the page** (operator, 2026-10-20). The
 * logo sits at the top and the footer at the bottom; only the **list of screens** between them
 * scrolls (`overflow-y-auto`, contained so the wheel does not chain on) when it outgrows a short
 * window, so Settings, the Manual and the version are always on screen.
 *
 * **The footer is the suite's** (operator, 2026-10-20: *"Settings. Manual link, and the
 * version"*), laid out as miStudio's and miLLM's are: a top rule; one row with **Settings** and,
 * to its right, the **Manual** as a book icon that opens the published manual in a new tab;
 * under it the **version** in small, dim monospace. Their theme toggle lives in the header, and
 * miForge's is in the resource strip for the same reason.
 *
 * **It collapses to a 64 px icon rail, as miStudio's and miLLM's do** (operator, 2026-10-31: it
 * "cannot be collapsed"). The same round chevron on the right edge toggles it; collapsed, each
 * item shows its icon with the label as a tooltip, and the footer shows Settings alone. The
 * choice is remembered in `localStorage`, every access guarded as `ThemeContext` guards its own.
 *
 * **The nav comes in as a prop.** See `layout/types.ts` - F9's route table is the live
 * registry and the shell must not be a second one.
 */

import { useState, type ReactNode } from 'react';
import { BookOpen, ChevronLeft, ChevronRight, Flame } from 'lucide-react';
import { MANUAL_URL } from '@/config/manual';
import { SHELL } from '@/config/tokens';
import { IconTile } from '@/components/common/IconTile';
import { NavItem } from './NavItem';
import type { ShellNavItem } from './types';

/** Fixed copy from the handoff 3. Exported so the test quotes it once. */
export const PRODUCT_NAME = 'miForge';
export const PRODUCT_SUBTITLE = 'AI Model Foundry';
/** The sidebar's link to the published manual. Exported so the test quotes it once. */
export const MANUAL_LABEL = 'Manual';
/** The book icon's hover text, as miStudio and miLLM word it. */
export const MANUAL_TITLE = 'User Manual';

/** Where the collapsed choice is kept. Exported so the test quotes it once. */
export const SIDEBAR_COLLAPSED_KEY = 'miforge.sidebar.collapsed';

function readCollapsed(): boolean {
  try {
    return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === 'true';
  } catch {
    return false;
  }
}

function writeCollapsed(collapsed: boolean): void {
  try {
    window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(collapsed));
  } catch {
    // A preference that cannot be saved is a lost convenience, not a failure.
  }
}

export interface SidebarProps {
  readonly nav: readonly ShellNavItem[];
  readonly activeId: string | null;
  readonly onNavigate: (id: string) => void;
  /** Shown under the footer row in small, dim mono (`APP_VERSION`, from `package.json`). */
  readonly version: string;
}

export function Sidebar({ nav, activeId, onNavigate, version }: SidebarProps): ReactNode {
  const screens = nav.filter((item) => item.placement !== 'footer');
  const footer = nav.filter((item) => item.placement === 'footer');
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const toggle = (): void => {
    setCollapsed((was) => {
      writeCollapsed(!was);
      return !was;
    });
  };
  const toggleLabel = collapsed ? 'Expand the sidebar' : 'Collapse the sidebar';
  return (
    <aside
      className="relative z-30 flex h-full shrink-0 flex-col border-r border-line bg-panel transition-[width] duration-200"
      style={{ width: collapsed ? SHELL.sidebarCollapsedWidthPx : SHELL.sidebarWidthPx }}
      data-testid="shell-sidebar"
      data-collapsed={collapsed}
    >
      <div
        className={`flex shrink-0 items-center gap-3 py-5 ${collapsed ? 'justify-center px-2' : 'px-5'}`}
      >
        <IconTile icon={Flame} surface="logo-tile" />
        {collapsed ? null : (
          <div className="min-w-0">
            {/* The product name is written `miForge` everywhere it is displayed (handoff,
                head): lower-case m, capital F, never title-cased by a CSS transform. */}
            <div className="font-semibold leading-tight">{PRODUCT_NAME}</div>
            <div className="text-hint text-dim">{PRODUCT_SUBTITLE}</div>
          </div>
        )}
      </div>

      <nav
        className={`min-h-0 flex-1 overflow-y-auto overscroll-contain ${collapsed ? 'px-2' : 'px-3'}`}
        aria-label="Screens"
        data-testid="sidebar-screens"
      >
        {screens.map((item) => (
          <NavItem
            key={item.id}
            item={item}
            active={item.id === activeId}
            onSelect={onNavigate}
            collapsed={collapsed}
          />
        ))}
      </nav>

      <div
        className={`shrink-0 border-t border-line py-2 ${collapsed ? 'px-2' : 'px-3'}`}
        data-testid="sidebar-footer"
      >
        <div className="flex items-center gap-1">
          <div className="min-w-0 flex-1">
            {footer.map((item) => (
              <NavItem
                key={item.id}
                item={item}
                active={item.id === activeId}
                onSelect={onNavigate}
                collapsed={collapsed}
              />
            ))}
          </div>
          {/* The manual is a page OUTSIDE this app (GitHub Pages), so it is a plain link in a
              new tab and not a row of the route table. Neutral like an inactive nav item -
              ADR-012 lists no accent surface for it. Collapsed, the footer is Settings alone. */}
          {collapsed ? null : (
            <a
              href={MANUAL_URL}
              target="_blank"
              rel="noopener noreferrer"
              title={MANUAL_TITLE}
              aria-label={`${MANUAL_LABEL} (opens in a new tab)`}
              className="mb-0.5 shrink-0 rounded-control p-2 text-dim hover:bg-card hover:text-text"
              data-testid="sidebar-manual-link"
            >
              <BookOpen size={16} aria-hidden="true" />
            </a>
          )}
        </div>
        {collapsed ? null : (
          <div className="px-3 pt-1">
            <span className="font-mono text-hint text-dim" data-testid="sidebar-version">
              {version}
            </span>
          </div>
        )}
      </div>
      {/* Last, as in miStudio's markup: Tab reaches it after the footer, and it is positioned
          absolutely on the right edge, so its place in the markup does not move it. */}
      <button
        type="button"
        onClick={toggle}
        className="absolute top-20 -right-3 flex h-6 w-6 items-center justify-center rounded-full border border-line bg-card text-dim hover:text-text"
        aria-label={toggleLabel}
        aria-expanded={!collapsed}
        title={toggleLabel}
        data-testid="sidebar-toggle"
      >
        {collapsed ? (
          <ChevronRight size={12} aria-hidden="true" />
        ) : (
          <ChevronLeft size={12} aria-hidden="true" />
        )}
      </button>
    </aside>
  );
}
