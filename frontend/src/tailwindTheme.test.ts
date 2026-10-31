/**
 * Tailwind CSS 4 keeps miForge looking as it did on Tailwind CSS 3.4 (`index.css`).
 *
 * On v3 the theme lived in `tailwind.config.ts`, which imported `config/tokens.ts`, so the
 * type scale and shell sizes had one definition. On v4 the theme is CSS (`@theme`), which
 * cannot import a TypeScript module - so, as `tokens.test.ts` does for the colour variables,
 * this test is what keeps the two in step. It compiles the real `index.css` through the
 * plugins `postcss.config.js` names and pins every token and every v3 behaviour the file
 * restores, so deleting or changing one of those lines turns it red. The before/after
 * screenshot and computed-style comparison of every screen in both themes is the wider check;
 * these are the pieces a test can hold.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import postcss from 'postcss';
import { beforeAll, describe, expect, it } from 'vitest';

import { FONTS, SHELL, TYPE } from './config/tokens';

const ROOT = path.resolve(__dirname, '..');
const SRC = path.join(ROOT, 'src');

// Utilities the pins below need, generated whether or not a component uses them today.
const PROBES = [
  'stack-y-2',
  'w-sidebar',
  'p-card',
  'pl-tree-indent',
  'h-icon-tile',
  'max-w-content',
  'rounded-card',
  'rounded-control',
  'text-title',
  'text-heading',
  'text-body',
  'text-hint',
  'text-ident',
  'text-sm',
  'hover:bg-accent-hover',
  'dark:bg-panel',
  'border-accent/35',
  'bg-green-500',
  'font-sans',
  'font-mono',
  // A family v4 added (v3 had none): with the v3 palette in place it generates nothing.
  'bg-mauve-500',
];

let css = '';
let theme: Record<string, string> = {};

beforeAll(async () => {
  const input = `@import './index.css';\n@source inline(${JSON.stringify(PROBES.join(' '))});\n`;
  // The plugins postcss.config.js names, loaded as Vite loads them, so the app's own PostCSS
  // wiring is what compiles here (an empty or v3-style config leaves the utilities ungenerated).
  const config = (await import(path.join(ROOT, 'postcss.config.js'))) as {
    default: { plugins: Record<string, object> };
  };
  const plugins = await Promise.all(
    Object.entries(config.default.plugins).map(async ([name, opts]) => {
      const mod = (await import(name)) as { default: (o: object) => postcss.AcceptedPlugin };
      return mod.default({ ...opts, base: ROOT, optimize: false });
    }),
  );
  const result = await postcss(plugins).process(input, {
    from: path.join(SRC, '__tailwindTheme.test.css'),
  });
  css = result.css;
  // The last declaration of each theme variable wins, as in the browser.
  theme = {};
  for (const m of css.matchAll(/(--[\w-]+):\s*([^;]+);/g)) theme[m[1] ?? ''] = (m[2] ?? '').trim();
}, 60_000);

/** The CSS rule bodies whose selector contains `selector`, joined. */
function rulesFor(selector: string): string {
  const out: string[] = [];
  let i = css.indexOf(selector);
  while (i !== -1) {
    const open = css.indexOf('{', i);
    const close = css.indexOf('}', open);
    out.push(css.slice(open + 1, close));
    i = css.indexOf(selector, close);
  }
  return out.join('\n');
}

const px = (n: number): string => `${n}px`;

describe('[tailwind theme] the tokens, held equal to config/tokens.ts', () => {
  it('maps every colour name to its --mf variable', () => {
    for (const name of [
      'bg',
      'panel',
      'card',
      'line',
      'text',
      'dim',
      'green',
      'cyan',
      'red',
      'amber',
    ])
      expect([name, theme[`--color-${name}`]]).toEqual([name, `var(--mf-${name})`]);
    expect(theme['--color-accent']).toBe('var(--mf-accent)');
    expect(theme['--color-accent-hover']).toBe('var(--mf-accent-hover)');
    expect(theme['--color-accent-tint']).toBe('var(--mf-accent-tint)');
    expect(rulesFor('.dark\\:bg-panel')).toContain('var(--color-panel)');
  });

  it('keeps the type scale TYPE names, with its line heights', () => {
    const scale: [string, number, string][] = [
      ['title', TYPE.pageTitlePx, '1.2'],
      ['heading', TYPE.sectionHeadingPx, '1.4'],
      ['body', TYPE.bodyPx, '1.5'],
      ['hint', TYPE.hintPx, '1.45'],
      ['ident', TYPE.monoPx, '1.45'],
    ];
    for (const [name, size, leading] of scale) {
      expect([name, theme[`--text-${name}`]]).toEqual([name, px(size)]);
      expect([name, theme[`--text-${name}--line-height`]]).toEqual([name, leading]);
      expect(rulesFor(`.text-${name} {`)).toContain(`var(--text-${name}--line-height)`);
    }
  });

  it('keeps the shell sizes SHELL names', () => {
    expect(theme['--spacing-sidebar']).toBe(px(SHELL.sidebarWidthPx));
    expect(theme['--spacing-card']).toBe(px(SHELL.cardPaddingPx));
    expect(theme['--spacing-tree-indent']).toBe(px(SHELL.treeIndentPx));
    expect(theme['--spacing-icon-tile']).toBe(px(SHELL.iconTilePx));
    expect(theme['--container-content']).toBe(px(SHELL.contentMaxWidthPx));
    expect(theme['--radius-card']).toBe(px(SHELL.cardRadiusPx));
    expect(theme['--radius-control']).toBe(px(SHELL.controlRadiusPx));
    expect(rulesFor('.w-sidebar {')).toContain('var(--spacing-sidebar)');
    expect(rulesFor('.p-card {')).toContain('var(--spacing-card)');
    expect(rulesFor('.max-w-content {')).toContain('var(--container-content)');
    expect(rulesFor('.rounded-control {')).toContain('var(--radius-control)');
  });

  it('keeps the font stacks FONTS names', () => {
    const norm = (s: string | undefined): string =>
      (s ?? '').replace(/["']/g, "'").replace(/\s+/g, ' ');
    expect(norm(theme['--font-sans'])).toBe(norm(FONTS.sans));
    expect(norm(theme['--font-mono'])).toBe(norm(FONTS.mono));
  });
});

describe('[tailwind theme] values carried from Tailwind CSS 3', () => {
  it('keeps the v3 hex palette, not v4 OKLCH', () => {
    expect(theme['--color-indigo-500']).toBe('#6366f1');
    expect(theme['--color-slate-950']).toBe('#020617');
    expect(theme['--color-white']).toBe('#fff');
    expect(
      Object.entries(theme).filter(([k, v]) => k.startsWith('--color-') && v.startsWith('oklch')),
    ).toEqual([]);
    expect(css).not.toContain('.bg-mauve-500');
  });

  it('gives green, cyan, red and amber no shades, as the v3 theme did', () => {
    // v3's theme replaced each family with one token, so `bg-green-500` generated nothing.
    expect(Object.keys(theme).filter((k) => /^--color-(green|cyan|red|amber)-/.test(k))).toEqual(
      [],
    );
    expect(css).not.toContain('.bg-green-500');
  });

  it('keeps the v3 absolute line heights for the default text sizes', () => {
    expect(theme['--text-sm--line-height']).toBe('1.25rem');
    expect(theme['--text-xs--line-height']).toBe('1rem');
  });
});

describe('[tailwind theme] variants and Preflight carried from Tailwind CSS 3', () => {
  it('dark: follows the dark class with the v3 selector', () => {
    expect(css).toMatch(
      /\.dark\\:bg-panel:is\(\.dark \*\)\s*\{\s*background-color: var\(--color-panel\)/,
    );
    expect(css).not.toContain('prefers-color-scheme: dark');
  });

  it('hover: applies on every device, as in v3', () => {
    expect(css).toContain('hover\\:bg-accent-hover');
    expect(css).not.toMatch(/@media \(hover: hover\)/);
  });

  it('restores the v3 default border colour, placeholder colour, cell padding and button cursor', () => {
    expect(css).toMatch(
      /::file-selector-button\s*\{\s*border-color: var\(--color-gray-200, currentcolor\)/,
    );
    expect(theme['--color-gray-200']).toBe('#e5e7eb');
    expect(rulesFor('textarea::placeholder')).toContain('color: var(--color-gray-400)');
    expect(rulesFor('option {')).toContain('padding: revert');
    expect(css).toMatch(/\btd,\s*th\s*\{\s*padding: 1px;/);
    expect(rulesFor("[role='button']:not(:disabled)")).toContain('cursor: pointer');
    // The file input's button keeps the browser's own look (the pack import dialog).
    const fileButton = rulesFor('::file-selector-button {');
    for (const declaration of [
      'padding: revert',
      'border: revert',
      'border-radius: revert',
      'background-color: revert',
      'color: revert',
      '-webkit-appearance: button',
    ])
      expect(fileButton).toContain(declaration);
  });

  it('stack-y-* gives every child after the first a top margin (v3 space-y-*)', () => {
    expect(css).toMatch(
      /\.stack-y-2 > :not\(\[hidden\]\) ~ :not\(\[hidden\]\)\s*\{\s*margin-top: calc\(var\(--spacing\) \* 2\);\s*margin-bottom: 0;/,
    );
  });

  it("draws the accent callout's 35% edge, which v3 silently dropped", () => {
    // On v3 a colour that is a CSS variable took no opacity modifier: `border-accent/35`
    // generated no rule and the callout fell back to the gray-200 default border, a light
    // grey edge in the dark theme. Callout.tsx asks for the mockup's rgba(249,115,22,0.35).
    expect(rulesFor('.border-accent\\/35')).toMatch(/border-color: .*var\(--color-accent\) 35%/);
  });
});

describe('[tailwind theme] source', () => {
  const files: string[] = [];
  const walk = (dir: string): void => {
    for (const name of readdirSync(dir)) {
      const p = path.join(dir, name);
      if (statSync(p).isDirectory()) walk(p);
      else if (
        /\.(tsx?|css)$/.test(name) &&
        !p.endsWith('tailwindTheme.test.ts') &&
        !p.endsWith('index.css')
      )
        files.push(p);
    }
  };
  walk(SRC);

  it('uses stack-y-* rather than the v4 space-y-*/space-x-* (different selector)', () => {
    expect(files.length).toBeGreaterThan(100);
    expect(files.some((f) => readFileSync(f, 'utf8').includes('stack-y-'))).toBe(true);
    const offenders = files.filter((f) =>
      /(^|[\s'"`:])-?space-[xy]-/m.test(readFileSync(f, 'utf8')),
    );
    expect(offenders.map((f) => path.relative(ROOT, f))).toEqual([]);
  });
});
