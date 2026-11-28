/**
 * Keyboard focus (foundation task 7.5; FR-009.7, mockup handoff 6).
 *
 * > No keyboard focus styles; add the sibling apps' focus ring.
 *
 * Tests `src/index.css`'s focus layer, which has no component of its own to sit beside.
 *
 * **Why three kinds of assertion.** The ring itself is a stylesheet rule, and jsdom does
 * not apply the stylesheet, so a render test cannot see it - that half is asserted
 * against the CSS source, and the *computed* ring is asserted for real in
 * `e2e/shell.spec.ts`, in a browser, in both themes. What a render test *can* see is the
 * thing the ring is useless without: that every interactive surface in the shell is a
 * real focusable element in a sensible tab order. The mockup's nav items are buttons, but
 * its list rows are `div`s with `onClick`, which no ring can rescue.
 *
 * The third assertion is the one both siblings fail: `outline-none` with nothing put
 * back. miStudio writes `focus:outline-none` at 210 call sites; miLLM applies
 * `outline-none` to every `input`, `select` and `button` in its base layer and replaces
 * it only on inputs, so **its buttons have no focus indicator at all**. Copying the
 * siblings' colour is right; copying that is not, and a test is the only thing that keeps
 * the distinction.
 */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Flame, LayoutGrid } from 'lucide-react';
import { AppShell } from '@/components/layout/AppShell';
import { PageHead } from '@/components/layout/PageHead';
import { Button } from '@/components/common/Button';
import { Field, Input } from '@/components/common/Field';
import { ThemeProvider } from '@/contexts/ThemeContext';
import type { ShellNavItem } from '@/components/layout/types';

const SRC_ROOT = path.resolve(__dirname);
const CSS = readFileSync(path.join(SRC_ROOT, 'index.css'), 'utf8');

/** The `:focus-visible` rule blocks, with their declarations. */
function focusVisibleBlocks(): { selector: string; body: string }[] {
  const blocks: { selector: string; body: string }[] = [];
  const pattern = /([^{}]*:focus-visible[^{}]*)\{([^}]*)\}/g;
  for (const match of CSS.matchAll(pattern)) {
    blocks.push({ selector: (match[1] ?? '').trim(), body: (match[2] ?? '').trim() });
  }
  return blocks;
}

function listSourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...listSourceFiles(full));
    else if (/\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry)) out.push(full);
  }
  return out;
}

describe('[focus ring in index.css]', () => {
  it('should declare a focus ring at all', () => {
    expect(focusVisibleBlocks().length).toBeGreaterThan(0);
  });

  it('should use :focus-visible and never bare :focus for the ring', () => {
    // A ring on `:focus` fires on a mouse click too, which is the reason so many
    // codebases then add `outline-none` and end up with no keyboard affordance.
    const bareFocus = CSS.split('\n').filter(
      (line) => /:focus(?![-\w])/.test(line) && !/focus-visible/.test(line),
    );
    expect(bareFocus).toEqual([]);
  });

  it('should draw the ring with the accent and an offset in the page background', () => {
    const ring = focusVisibleBlocks().find((block) => /box-shadow/.test(block.body));
    expect(ring, 'no :focus-visible block draws a box-shadow').toBeDefined();
    expect(ring?.body).toContain('--mf-accent');
    // The offset ring is `--mf-bg` so the indicator reads against a `panel` sidebar and a
    // `card` row alike, without a per-surface variant.
    expect(ring?.body).toContain('--mf-bg');
  });

  it('should cover every interactive element, including a button', () => {
    const selectors = focusVisibleBlocks()
      .map((block) => block.selector)
      .join(' ');
    for (const element of ['a', 'button', 'input', 'select', 'textarea', '[tabindex]']) {
      expect(selectors, `the focus ring does not cover ${element}`).toContain(element);
    }
  });

  it('should never remove an outline without putting a ring back', () => {
    const blocksRemovingOutline = CSS.replace(/\/\*[\s\S]*?\*\//g, '')
      .split('}')
      .filter((block) => /outline:\s*none|@apply[^;]*outline-none/.test(block));
    expect(blocksRemovingOutline.length).toBeGreaterThan(0);
    for (const block of blocksRemovingOutline) {
      expect(
        /focus-visible/.test(block) && /box-shadow/.test(block),
        `a block removes the outline without a replacement ring: ${block.trim().slice(0, 120)}`,
      ).toBe(true);
    }
  });

  it('should have no focus:outline-none utility anywhere in the source', () => {
    // The sibling anti-pattern. A component that suppresses the outline locally escapes
    // the global rule above, and nothing would say so.
    const offenders: string[] = [];
    for (const file of listSourceFiles(SRC_ROOT)) {
      const source = readFileSync(file, 'utf8');
      const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
      if (/focus:outline-none|focus-visible:outline-none/.test(code)) {
        offenders.push(path.relative(SRC_ROOT, file));
      }
    }
    expect(offenders).toEqual([]);
  });
});

describe('[keyboard reachability]', () => {
  const NAV: readonly ShellNavItem[] = [
    { id: 'runs', label: 'Runs', icon: LayoutGrid },
    { id: 'checkpoints', label: 'Checkpoints', icon: Flame },
  ];

  it('should reach every shell action by Tab, in order, with no div doing a button job', async () => {
    render(
      <ThemeProvider initialTheme="dark">
        <AppShell nav={NAV} activeId="runs" onNavigate={vi.fn()} version="v0.1.0" resources={[]}>
          <PageHead
            title="Runs"
            subtitle="One line, in dim"
            actions={<Button variant="primary">Create and start</Button>}
          />
          <Field label="Steps" hint="Whole steps. The run stops here unless a condition fires.">
            {({ id, describedBy }) => <Input id={id} aria-describedby={describedBy} />}
          </Field>
        </AppShell>
      </ThemeProvider>,
    );

    // Assert the sequence against the elements themselves. Deriving an accessible name
    // from `textContent` silently yields '' for an input - and `?? ` does not fall
    // through on an empty string, so the first version of this test compared '' to
    // 'Steps' and reported the wrong failure.
    const expectedOrder: readonly Element[] = [
      screen.getByRole('button', { name: /^Runs$/ }),
      screen.getByRole('button', { name: /Checkpoints/ }),
      // The sidebar's link to the published manual: a real anchor, so Tab reaches it.
      screen.getByRole('link', { name: /^Manual/ }),
      // The sidebar's collapse toggle, after its footer, as miStudio's and miLLM's.
      screen.getByRole('button', { name: 'Collapse the sidebar' }),
      screen.getByRole('button', { name: /Switch to light theme/ }),
      screen.getByRole('button', { name: /Create and start/ }),
      screen.getByLabelText('Steps'),
    ];

    for (const [index, element] of expectedOrder.entries()) {
      await userEvent.tab();
      expect(document.activeElement, `tab stop ${index + 1}`).toBe(element);
    }
  });

  it('should label a field so its control is identifiable and its hint announced', async () => {
    // The mockup's `Field` renders the label as a `div` with no `htmlFor`, so no control
    // has a label and no hint is announced. An unlabelled select is operable but
    // unidentifiable, which fails FR-009.7 as surely as an unreachable one.
    render(
      <Field label="Save every" hint="Steps between checkpoints.">
        {({ id, describedBy }) => <Input id={id} aria-describedby={describedBy} />}
      </Field>,
    );
    const input = screen.getByLabelText('Save every');
    expect(input).toHaveAccessibleDescription('Steps between checkpoints.');
  });
});
