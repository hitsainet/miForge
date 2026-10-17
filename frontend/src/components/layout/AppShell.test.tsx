/**
 * The suite shell (foundation tasks 7.3, 7.5; ADR-012, FR-009.2).
 *
 * Two assertions here are not about rendering at all and are the reason the file exists:
 *
 * - **The shell carries no sibling reference.** ADR-022 fences 5.10 off the core path,
 *   the shell is on every screen, and the handoff's 6c moved the attachment chip into
 *   the hot-update mode. This scans the layout source for the strings, because a
 *   render test cannot see a chip nobody has added yet.
 * - **The shell declares no nav of its own.** F9's route table is the live registry
 *   (FR-009.1); a second list of screens is the condition ADR-016 #1's reachability test
 *   cannot see.
 */

import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Cpu, Flame, HardDrive, LayoutGrid } from 'lucide-react';
import { AppShell } from './AppShell';
import { PageHead } from './PageHead';
import { PRODUCT_SUBTITLE, Sidebar } from './Sidebar';
import { formatReading, ResourceStrip } from './ResourceStrip';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { SHELL } from '@/config/tokens';
import type { ResourceReading, ShellNavItem } from './types';

const NAV: readonly ShellNavItem[] = [
  { id: 'runs', label: 'Runs', icon: LayoutGrid },
  { id: 'checkpoints', label: 'Checkpoints', icon: Flame },
];

const RESOURCES: readonly ResourceReading[] = [
  { id: 'cpu', label: 'CPU', icon: Cpu, value: 41, unit: '%' },
  { id: 'vram', label: 'VRAM', icon: HardDrive, value: 14.2, unit: 'GB', of: 24 },
  { id: 'gpu', label: 'GPU', icon: Cpu, value: null, unit: '%' },
];

function renderShell(activeId: string | null = 'runs', onNavigate = vi.fn()) {
  return {
    onNavigate,
    ...render(
      <ThemeProvider initialTheme="dark">
        <AppShell
          nav={NAV}
          activeId={activeId}
          onNavigate={onNavigate}
          version="v0.1.0"
          resources={RESOURCES}
        >
          <PageHead title="Runs" subtitle="One line, in dim" />
        </AppShell>
      </ThemeProvider>,
    ),
  };
}

describe('[AppShell]', () => {
  it('should render all four shell regions', () => {
    renderShell();
    for (const id of [
      'shell-sidebar',
      'shell-resource-strip',
      'shell-page-head',
      'shell-content',
    ]) {
      expect(screen.getByTestId(id)).toBeInTheDocument();
    }
  });

  it('should give the sidebar the 224 px width both ADR-012 and FR-009.2 quote', () => {
    renderShell();
    expect(screen.getByTestId('shell-sidebar').style.width).toBe(`${SHELL.sidebarWidthPx}px`);
    expect(SHELL.sidebarWidthPx).toBe(224);
  });

  it('should hold the content column to the ~72rem the handoff specifies', () => {
    renderShell();
    expect(screen.getByTestId('shell-content').style.maxWidth).toBe(`${SHELL.contentMaxWidthPx}px`);
  });

  it('should leave the resource strip trailing slot empty on the core path', () => {
    renderShell();
    // F11's attachment chip goes in `resourceTrailing`. Nothing core puts anything there.
    const strip = screen.getByTestId('shell-resource-strip');
    expect(strip.textContent).not.toMatch(/attached|detached|stale|quiesced/i);
  });
});

describe('[Sidebar]', () => {
  it('should render the product name and its fixed subtitle', () => {
    renderShell();
    expect(screen.getByText('miForge')).toBeInTheDocument();
    expect(screen.getByText(PRODUCT_SUBTITLE)).toBeInTheDocument();
    expect(PRODUCT_SUBTITLE).toBe('AI Model Foundry');
  });

  it('should mark the active screen with aria-current, not only with a tint', () => {
    renderShell('checkpoints');
    expect(screen.getByRole('button', { name: /Checkpoints/ })).toHaveAttribute(
      'aria-current',
      'page',
    );
    expect(screen.getByRole('button', { name: /^Runs$/ })).not.toHaveAttribute('aria-current');
  });

  it('should report navigation by id rather than by label', async () => {
    const { onNavigate } = renderShell('runs');
    await userEvent.click(screen.getByRole('button', { name: /Checkpoints/ }));
    // The label is copy and will be reworded; the id is the contract with F9's table.
    expect(onNavigate).toHaveBeenCalledExactlyOnceWith('checkpoints');
  });

  it('should render every nav item it is given and no others', () => {
    render(
      <ThemeProvider initialTheme="dark">
        <Sidebar nav={NAV} activeId={null} onNavigate={vi.fn()} version="v0.1.0" />
      </ThemeProvider>,
    );
    const items = screen.getByRole('navigation', { name: 'Screens' }).querySelectorAll('button');
    expect(items).toHaveLength(NAV.length);
  });

  it('should declare no nav table of its own anywhere in components/layout', () => {
    // F9's FR-009.1: "A screen not in the table is not reachable and therefore does not
    // exist." Two tables is the condition the reachability test cannot see.
    const dir = path.resolve(__dirname);
    for (const entry of readdirSync(dir)) {
      if (!/\.tsx?$/.test(entry) || /\.test\./.test(entry)) continue;
      const source = readFileSync(path.join(dir, entry), 'utf8');
      const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
      expect(
        /ShellNavItem\[\]\s*=\s*\[/.test(code),
        `${entry} declares a nav array; the nav is a prop (layout/types.ts)`,
      ).toBe(false);
    }
  });
});

describe('[ResourceStrip]', () => {
  it('should name the scale on every number', () => {
    // ADR-012: "every number names its scale and sample". `41` is not a number; `41%` is.
    expect(formatReading(RESOURCES[0] as ResourceReading)).toBe('41%');
  });

  it('should name the total when a reading is a fraction of a budget', () => {
    // `14.2 GB` says nothing about how much is left; `14.2/24.0 GB` does.
    expect(formatReading(RESOURCES[1] as ResourceReading)).toBe('14.2/24.0 GB');
  });

  it('should render an unread value as not reported, never as zero', () => {
    // A GPU at 0 % and a GPU that could not be read are different facts, and this node's
    // driver reaches the "Driver/library version mismatch" state often enough for that to
    // matter (AGENT.md, Cautions).
    expect(formatReading(RESOURCES[2] as ResourceReading)).toBe('not reported');
    render(<ResourceStrip readings={RESOURCES} />);
    expect(screen.getByTestId('resource-gpu').textContent).toContain('not reported');
    expect(screen.getByTestId('resource-gpu').textContent).not.toMatch(/\b0\b/);
  });
});

describe('[shell standalone purity]', () => {
  it('should reference no sibling application in any layout component', () => {
    // ADR-022, and ADR-001's guard in spirit: the shell is on every screen, so it is the
    // most load-bearing place a removed dependency could grow back.
    const dir = path.resolve(__dirname);
    const forbidden = /\bmillm\b|\bmistudio\b|\bmidataworks\b|api\/weights|\/v1\/chat/i;
    for (const entry of readdirSync(dir)) {
      if (!/\.tsx?$/.test(entry) || /\.test\./.test(entry)) continue;
      const source = readFileSync(path.join(dir, entry), 'utf8');
      // Prose about why the chip is absent is allowed; code referencing a sibling is not.
      const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
      expect(forbidden.test(code), `${entry} references a sibling application`).toBe(false);
    }
  });
});
