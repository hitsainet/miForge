/**
 * The tokens, and the one duplication in the frontend (foundation task 7.2; ADR-012).
 *
 * `config/tokens.ts` holds the values; `index.css` declares them as CSS custom properties
 * because a stylesheet cannot import a TypeScript module. **This test is what makes that
 * duplication safe**, and it is the same arrangement the repo already uses for the
 * vendered schemas: `cmp` agreeing today is not enforcement, so the agreement is a test.
 *
 * It also pins the three numbers the binding documents quote - the 224 px sidebar, the
 * 12 px card radius and the 20 px card padding - against the documents' own figures, so
 * changing one is a failing test rather than a screen that looks nearly right.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { cssVariableName, DARK, LIGHT, SHELL, TOKEN_NAMES, TYPE } from './tokens';

const CSS = readFileSync(path.resolve(__dirname, '..', 'index.css'), 'utf8');

/** Pull one block's declarations: `:root { … }` or `.dark { … }`. */
function blockDeclarations(selector: string): Map<string, string> {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(`${escaped}\\s*\\{([^}]*)\\}`).exec(CSS);
  if (match === null) throw new Error(`index.css has no \`${selector}\` block`);
  const declarations = new Map<string, string>();
  for (const line of (match[1] ?? '').split(';')) {
    const [name, ...rest] = line.split(':');
    if (name === undefined || rest.length === 0) continue;
    declarations.set(name.trim(), rest.join(':').trim());
  }
  return declarations;
}

describe('[design tokens]', () => {
  it('should declare every token in index.css for both themes', () => {
    const dark = blockDeclarations('.dark');
    const light = blockDeclarations(':root');
    for (const token of TOKEN_NAMES) {
      const variable = cssVariableName(token);
      expect(dark.has(variable), `.dark is missing ${variable}`).toBe(true);
      expect(light.has(variable), `:root is missing ${variable}`).toBe(true);
    }
  });

  it('should agree with index.css value for value, in both themes', () => {
    const dark = blockDeclarations('.dark');
    const light = blockDeclarations(':root');
    const normalise = (value: string): string => value.replace(/\s+/g, '').toLowerCase();
    for (const token of TOKEN_NAMES) {
      const variable = cssVariableName(token);
      expect(normalise(dark.get(variable) ?? ''), `${variable} in .dark`).toBe(
        normalise(DARK[token]),
      );
      expect(normalise(light.get(variable) ?? ''), `${variable} in :root`).toBe(
        normalise(LIGHT[token]),
      );
    }
  });

  it('should keep the dark palette byte-identical to the mockup', () => {
    // Quoted from `0xcc/prds/miforge-mockup.jsx`'s `const T = {...}`, which ADR-012 then
    // restates. The three apps share a shell and the shared part is the hex; a drift here
    // is miForge quietly leaving the suite's look.
    expect(DARK).toEqual({
      bg: '#0d1420',
      panel: '#131b28',
      card: '#18212f',
      line: '#243044',
      text: '#e6ebf2',
      dim: '#8b97a8',
      accent: '#F97316',
      accentHover: '#FB923C',
      accentTint: 'rgba(249,115,22,0.12)',
      green: '#22c55e',
      cyan: '#22d3ee',
      red: '#ef4444',
      amber: '#f59e0b',
    });
  });

  it('should keep the accent and the reserved state colours identical across themes', () => {
    // ADR-012 reserves cyan for miLLM state. A theme-dependent cyan would make "this
    // means miLLM" theme-dependent too.
    expect(LIGHT.accent).toBe(DARK.accent);
    expect(LIGHT.accentTint).toBe(DARK.accentTint);
  });

  it('should carry the measurements the binding documents quote', () => {
    // ADR-012 and FR-009.2 both say 224. The handoff's card grammar says 12 and 20.
    expect(SHELL.sidebarWidthPx).toBe(224);
    expect(SHELL.cardRadiusPx).toBe(12);
    expect(SHELL.cardPaddingPx).toBe(20);
    expect(SHELL.cardBorderPx).toBe(1);
    expect(SHELL.controlRadiusPx).toBe(8);
    expect(SHELL.iconTilePx).toBe(36);
    // The mockup's `c.depth * 28` for the checkpoint tree's indent.
    expect(SHELL.treeIndentPx).toBe(28);
  });

  it('should carry the handoff type scale', () => {
    expect(TYPE).toEqual({
      pageTitlePx: 24,
      sectionHeadingPx: 16,
      bodyPx: 14,
      hintPx: 12,
      monoPx: 12,
    });
  });

  it('should map a camelCase token to its kebab-case variable', () => {
    expect(cssVariableName('accentTint')).toBe('--mf-accent-tint');
    expect(cssVariableName('bg')).toBe('--mf-bg');
  });
});
