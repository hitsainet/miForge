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
 * **The nav comes in as a prop.** See `layout/types.ts` - F9's route table is the live
 * registry and the shell must not be a second one.
 */

import type { ReactNode } from 'react';
import { Flame } from 'lucide-react';
import { SHELL } from '@/config/tokens';
import { IconTile } from '@/components/common/IconTile';
import { NavItem } from './NavItem';
import { ThemeToggle } from './ThemeToggle';
import type { ShellNavItem } from './types';

/** Fixed copy from the handoff 3. Exported so the test quotes it once. */
export const PRODUCT_NAME = 'miForge';
export const PRODUCT_SUBTITLE = 'AI Model Foundry';

export interface SidebarProps {
  readonly nav: readonly ShellNavItem[];
  readonly activeId: string | null;
  readonly onNavigate: (id: string) => void;
  /** Shown in the footer in mono, as the mockup does. */
  readonly version: string;
}

export function Sidebar({ nav, activeId, onNavigate, version }: SidebarProps): ReactNode {
  return (
    <aside
      className="flex shrink-0 flex-col border-r border-line bg-panel"
      style={{ width: SHELL.sidebarWidthPx }}
      data-testid="shell-sidebar"
    >
      <div className="flex items-center gap-3 px-5 py-5">
        <IconTile icon={Flame} surface="logo-tile" />
        <div className="min-w-0">
          {/* The product name is written `miForge` everywhere it is displayed (handoff,
              head): lower-case m, capital F, never title-cased by a CSS transform. */}
          <div className="font-semibold leading-tight">{PRODUCT_NAME}</div>
          <div className="text-hint text-dim">{PRODUCT_SUBTITLE}</div>
        </div>
      </div>

      <nav className="flex-1 px-3" aria-label="Screens">
        {nav.map((item) => (
          <NavItem key={item.id} item={item} active={item.id === activeId} onSelect={onNavigate} />
        ))}
      </nav>

      <div className="flex items-center justify-between gap-2 px-5 py-4">
        <span className="font-mono text-ident text-dim">{version}</span>
        <ThemeToggle />
      </div>
    </aside>
  );
}
