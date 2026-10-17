/**
 * The accent rule, as a test (foundation task 7.2; ADR-012).
 *
 * > **Accent discipline:** orange appears on the active nav item, primary buttons, the
 * > logo tile, list-row icon tiles, the loss curve, running progress bars, and the
 * > callout banner tint. **Not** on body text, borders, backgrounds, or any status pill
 * > except "Running." If a screen starts to look orange, something is wrong.
 *
 * **Why this is a test and not a convention.** A design rule nobody checks is a design
 * rule that erodes one reasonable exception at a time, and this project has already paid
 * for the general version of that mistake four times - a claim recorded as settled whose
 * cited source did not support it, caught not by re-reading but by checking. So the rule
 * is checked.
 *
 * Three independent checks, because each catches a different way of breaking it:
 *
 * 1. **The surface list is exactly the ADR's seven plus one documented addition.**
 *    Widening the rule has to be an edit to `tokens.ts` that fails this test, rather than
 *    a new line in an allowlist that passes.
 * 2. **A static scan of the source tree.** Catches a *new* file painting with the accent,
 *    which no render test can - there is no render test for a component nobody wrote yet.
 * 3. **Render assertions over the shell.** Catches the accent landing on a surface inside
 *    an already-allowlisted file, which the scan permits by construction.
 *
 * Plus a fourth, narrower one: the only accent in `index.css` is the focus ring.
 */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Flame } from 'lucide-react';
import {
  ACCENT_SURFACES,
  ADDED_ACCENT_SURFACES,
  ADR_012_ACCENT_SURFACES,
  type AccentSurface,
} from './tokens';
import {
  ACCENT_ALLOWLIST,
  ACCENT_MARKER_PREFIX,
  ACCENT_MARKER_WINDOW,
  ACCENT_SCAN_EXCLUSIONS,
} from './accentAllowlist';
import { NavItem } from '@/components/layout/NavItem';
import { PageHead } from '@/components/layout/PageHead';
import { Button } from '@/components/common/Button';
import { Card } from '@/components/common/Card';
import { StatePill } from '@/components/common/Pill';
import { IconTile } from '@/components/common/IconTile';

/* ------------------------------------------------------------------------------------ *
 * The scanner.
 * ------------------------------------------------------------------------------------ */

/** `frontend/`, found from this file rather than from `process.cwd()`. */
const FRONTEND_ROOT = path.resolve(__dirname, '..', '..');
const SRC_ROOT = path.join(FRONTEND_ROOT, 'src');

/**
 * The patterns that mean "this line paints with the accent".
 *
 * Deliberately **not** `/\baccent\b/`. That would also match `tone="accent"` and
 * `surface: 'logo-tile' | ...` - a call site *selecting* a permitted surface, which is
 * the correct use of the design system and not a paint. What is matched is a Tailwind
 * utility, a CSS variable reference and a direct read of the token object: the three ways
 * orange actually reaches a pixel.
 */
const PAINT_PATTERNS: readonly RegExp[] = [
  // bg-accent, text-accent, border-accent/35, hover:bg-accent-hover, bg-accent-tint …
  /(?:^|[^\w-])(?:[a-z-]+:)*(?:bg|text|border|ring|from|via|to|fill|stroke|shadow|outline|decoration|divide|caret|placeholder)-accent(?:-(?:hover|tint))?(?:\/\d+)?(?![\w-])/,
  // var(--mf-accent), '--mf-accent-tint'
  /--mf-accent(?:-(?:hover|tint))?(?![\w-])/,
  // DARK.accent, LIGHT.accentTint, a bare accentHover / accentTint reference
  /\b(?:DARK|LIGHT)\.accent(?:Hover|Tint)?\b/,
  /\baccent(?:Hover|Tint)\b/,
];

function listSourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      out.push(...listSourceFiles(full));
    } else if (/\.tsx?$/.test(entry)) {
      out.push(full);
    }
  }
  return out.sort();
}

/**
 * Blank out comments so prose about the rule is not mistaken for the rule being broken.
 *
 * The result is only ever used for *detection*, so a URL mangled at its `//` is harmless;
 * what matters is that it cannot create a false positive. Markers are read from the raw
 * line, which is why they survive this.
 */
function stripComments(lines: readonly string[]): string[] {
  let inBlock = false;
  return lines.map((line) => {
    let out = '';
    let i = 0;
    while (i < line.length) {
      if (inBlock) {
        const end = line.indexOf('*/', i);
        if (end === -1) return out;
        inBlock = false;
        i = end + 2;
        continue;
      }
      if (line.startsWith('//', i)) return out;
      if (line.startsWith('/*', i)) {
        inBlock = true;
        i += 2;
        continue;
      }
      out += line[i];
      i += 1;
    }
    return out;
  });
}

function paints(line: string): boolean {
  return PAINT_PATTERNS.some((pattern) => pattern.test(line));
}

/** Every `accent-ok:<surface>` marker within the window above (and on) a line. */
function markersNear(raw: readonly string[], index: number): string[] {
  const start = Math.max(0, index - ACCENT_MARKER_WINDOW);
  const found: string[] = [];
  for (let i = start; i <= index; i += 1) {
    const line = raw[i] ?? '';
    const pattern = new RegExp(`${ACCENT_MARKER_PREFIX}([a-z-]+)`, 'g');
    for (const match of line.matchAll(pattern)) {
      const surface = match[1];
      if (surface !== undefined) found.push(surface);
    }
  }
  return found;
}

interface PaintSite {
  readonly file: string;
  readonly line: number;
  readonly text: string;
  readonly markers: readonly string[];
}

function findPaintSites(): PaintSite[] {
  const excluded = new Set(ACCENT_SCAN_EXCLUSIONS.map((row) => row.file));
  const sites: PaintSite[] = [];
  for (const absolute of listSourceFiles(SRC_ROOT)) {
    const relative = path.relative(FRONTEND_ROOT, absolute).split(path.sep).join('/');
    if (excluded.has(relative)) continue;
    // A component's own test may legitimately assert the accent is present.
    if (/\.test\.tsx?$/.test(relative)) continue;

    const raw = readFileSync(absolute, 'utf8').split('\n');
    const code = stripComments(raw);
    code.forEach((line, index) => {
      if (!paints(line)) return;
      sites.push({
        file: relative,
        line: index + 1,
        text: (raw[index] ?? '').trim(),
        markers: markersNear(raw, index),
      });
    });
  }
  return sites;
}

/* ------------------------------------------------------------------------------------ *
 * 1. The surface list.
 * ------------------------------------------------------------------------------------ */

describe('[ADR-012 accent surfaces]', () => {
  it('should be exactly the seven ADR-012 names plus the one documented addition', () => {
    expect([...ADR_012_ACCENT_SURFACES]).toEqual([
      'active-nav',
      'primary-button',
      'logo-tile',
      'row-icon-tile',
      'loss-curve',
      'running-progress',
      'callout-tint',
    ]);
    // The addition is argued for in `tokens.ts` (ADDED_ACCENT_SURFACES) and the operator
    // may reverse it. If it grows, this assertion is where the growth has to be stated.
    expect([...ADDED_ACCENT_SURFACES]).toEqual(['focus-ring']);
    expect([...ACCENT_SURFACES]).toEqual([...ADR_012_ACCENT_SURFACES, ...ADDED_ACCENT_SURFACES]);
  });
});

/* ------------------------------------------------------------------------------------ *
 * 2. The static scan.
 * ------------------------------------------------------------------------------------ */

describe('[accent allowlist]', () => {
  const sites = findPaintSites();
  const allowed = new Map<string, Set<AccentSurface>>();
  for (const entry of ACCENT_ALLOWLIST) {
    const set = allowed.get(entry.file) ?? new Set<AccentSurface>();
    set.add(entry.surface);
    allowed.set(entry.file, set);
  }

  it('should find the accent somewhere, so a passing scan is not a scan of nothing', () => {
    // The construction-green guard. `review_chain.py`'s first version passed against a
    // tree containing zero documents; a scanner whose patterns silently stopped matching
    // would pass against a tree full of violations in exactly the same way.
    expect(sites.length).toBeGreaterThan(0);
  });

  it('should paint with the accent only in allowlisted files', () => {
    const offenders = sites
      .filter((site) => !allowed.has(site.file))
      .map((site) => `${site.file}:${site.line}  ${site.text}`);
    expect(offenders).toEqual([]);
  });

  it('should carry an accent-ok marker naming a permitted surface on every painting line', () => {
    const offenders: string[] = [];
    for (const site of sites) {
      const permitted = allowed.get(site.file);
      if (permitted === undefined) continue; // reported by the previous assertion
      const named = site.markers.filter((marker): marker is AccentSurface =>
        permitted.has(marker as AccentSurface),
      );
      if (named.length === 0) {
        offenders.push(
          `${site.file}:${site.line} has no '${ACCENT_MARKER_PREFIX}<surface>' marker ` +
            `naming one of [${[...permitted].join(', ')}]  ${site.text}`,
        );
      }
    }
    expect(offenders).toEqual([]);
  });

  it('should have no stale allowlist entry', () => {
    // The direction that keeps the list from becoming a rubber stamp.
    const exercised = new Set<string>();
    for (const site of sites) {
      for (const marker of site.markers) exercised.add(`${site.file}::${marker}`);
    }
    const stale = ACCENT_ALLOWLIST.filter(
      (entry) => !exercised.has(`${entry.file}::${entry.surface}`),
    ).map((entry) => `${entry.file} :: ${entry.surface}`);
    expect(stale).toEqual([]);
  });

  it('should use the accent in index.css only for the focus ring', () => {
    const css = readFileSync(path.join(SRC_ROOT, 'index.css'), 'utf8');

    // Checked per RULE BLOCK, not per line. A `box-shadow` with two rings is written over
    // three lines, so the line carrying `var(--mf-accent)` has no `box-shadow` on it - a
    // per-line check reported the focus ring itself as a violation. The question is which
    // *rule* paints with the accent, and that is a block-level question.
    //
    // `--mf-accent-hover: …` is a declaration of the token; `var(--mf-accent)` is a use.
    // Each occurrence is tested on its own rather than with a lookahead, because a
    // lookahead on an optional suffix backtracks into the shorter name and reads every
    // declaration as a use.
    const usesAccent = (text: string): boolean => {
      for (const match of text.matchAll(/--mf-accent[\w-]*/g)) {
        const after = text.slice((match.index ?? 0) + match[0].length);
        if (!/^\s*:/.test(after)) return true;
      }
      return false;
    };

    const stripped = css.replace(/\/\*[\s\S]*?\*\//g, '');
    const blocks: { selector: string; body: string }[] = [];
    for (const match of stripped.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      blocks.push({ selector: (match[1] ?? '').trim(), body: (match[2] ?? '').trim() });
    }

    const painting = blocks.filter((block) => usesAccent(block.body));
    // The construction-green guard: a pattern that stopped matching would pass here with
    // an empty list exactly as it would pass over a file full of violations.
    expect(painting.length).toBeGreaterThan(0);

    for (const block of painting) {
      expect(
        block.selector.includes(':focus-visible'),
        `index.css paints with the accent outside a :focus-visible rule: ${block.selector}`,
      ).toBe(true);
    }
  });
});

/* ------------------------------------------------------------------------------------ *
 * 3. Render assertions - the permitted surfaces, and the forbidden ones.
 * ------------------------------------------------------------------------------------ */

const ACCENT_CLASS =
  /(?:^|\s)(?:[a-z-]+:)*(?:bg|text|border)-accent(?:-(?:hover|tint))?(?:\/\d+)?(?=\s|$)/;

function classesOf(element: Element | null): string {
  return element?.className ?? '';
}

describe('[accent on rendered surfaces]', () => {
  const navItem = { id: 'runs', label: 'Runs', icon: Flame };

  it('should put the accent on the ACTIVE nav item', () => {
    render(<NavItem item={navItem} active onSelect={() => {}} />);
    expect(classesOf(screen.getByRole('button'))).toMatch(ACCENT_CLASS);
  });

  it('should put no accent on an inactive nav item', () => {
    render(<NavItem item={navItem} active={false} onSelect={() => {}} />);
    expect(classesOf(screen.getByRole('button'))).not.toMatch(ACCENT_CLASS);
  });

  it('should put the accent on a primary button and not on a ghost one', () => {
    const { unmount } = render(<Button variant="primary">Create and start</Button>);
    expect(classesOf(screen.getByRole('button'))).toMatch(ACCENT_CLASS);
    unmount();

    render(<Button variant="ghost">Branch from here</Button>);
    expect(classesOf(screen.getByRole('button'))).not.toMatch(ACCENT_CLASS);
  });

  it('should put the accent on the logo tile and on a row icon tile', () => {
    const { container } = render(
      <>
        <IconTile icon={Flame} surface="logo-tile" />
        <IconTile icon={Flame} surface="row-icon-tile" />
        <IconTile icon={Flame} surface="neutral" />
      </>,
    );
    const tiles = [...container.querySelectorAll('[data-surface]')];
    const bySurface = new Map(tiles.map((tile) => [tile.getAttribute('data-surface'), tile]));
    expect(classesOf(bySurface.get('logo-tile') ?? null)).toMatch(ACCENT_CLASS);
    expect(classesOf(bySurface.get('row-icon-tile') ?? null)).toMatch(ACCENT_CLASS);
    expect(classesOf(bySurface.get('neutral') ?? null)).not.toMatch(ACCENT_CLASS);
  });

  it('should put the accent on no status pill but Running', () => {
    const { container } = render(
      <>
        <StatePill state="running" data-testid="pill-running" />
        <StatePill state="completed" data-testid="pill-completed" />
        <StatePill state="queued" data-testid="pill-queued" />
        <StatePill state="failed" data-testid="pill-failed" />
        <StatePill state="something-new" data-testid="pill-unknown" />
      </>,
    );
    const tones = [...container.querySelectorAll('[data-tone]')].map((node) => ({
      label: node.textContent,
      tone: node.getAttribute('data-tone'),
    }));
    const accented = tones.filter((row) => row.tone === 'accent');
    expect(accented).toEqual([{ label: 'Running', tone: 'accent' }]);
  });

  it('should put no accent on a page title, its subtitle, or a card border', () => {
    const { container } = render(
      <>
        <PageHead title="md-notary-2026-09-11" subtitle="One line, in dim" />
        <Card>Body text</Card>
      </>,
    );
    // ADR-012 forbids the accent on body text, borders and backgrounds. The page head is
    // the most tempting place in the product to break the first of those, which is why
    // the mutation control for this rule is run against `PageHead.tsx`.
    for (const node of container.querySelectorAll('h1, p, div')) {
      expect(classesOf(node)).not.toMatch(ACCENT_CLASS);
    }
  });
});
