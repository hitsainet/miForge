/**
 * The corner rule, as a test (operator decision, 2026-10-31).
 *
 * > Every miSuite app uses two named corner sizes: **card = 12 px** and **control = 8 px**,
 * > as named theme tokens.
 *
 * miForge already has both tokens (`index.css` `@theme` `--radius-card` / `--radius-control`,
 * `tokens.ts` `SHELL.cardRadiusPx` / `controlRadiusPx`, pinned by `tailwindTheme.test.ts`).
 * That pins what the tokens ARE; this test pins that nothing goes around them. Tailwind CSS 4
 * still generates its default scale (`rounded-sm`, `rounded-lg`, `rounded-[6px]` …), so a
 * stray class compiles and renders - only a check stops the drift.
 *
 * Allowed: `rounded`/`rounded-<side>` with `card`, `control`, `full` or `none`, under any
 * responsive or state prefix (`md:rounded-none`). In CSS: `border-*-radius` only as
 * `var(--radius-card|control)`, plus the named exceptions below. In an inline style object,
 * `borderRadius` only as the same `var(...)` string.
 *
 * **Why an AST and not a line regex.** AGENT.md records source-scrape guards in this project
 * matching their own comments, and matching nothing on an unexpected layout. So the
 * TypeScript side reads the files with the TypeScript parser and looks only inside string
 * and template-literal tokens: a comment is trivia and is never visited, and a class in
 * `cn([...])`, a ternary or a `${}` template is a string token like any other. And the walk
 * carries a positive check - it must find the known `rounded-card` in `Card.tsx` - so a
 * walker pointed at the wrong root, or a pattern that stopped matching, goes red rather than
 * passing over nothing.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

const FRONTEND_ROOT = path.resolve(__dirname, '..', '..');
const SRC_ROOT = path.join(FRONTEND_ROOT, 'src');

const ALLOWED_SIZES = ['card', 'control', 'full', 'none'] as const;
const ALLOWED_DESCRIPTION =
  `rounded[-t|-r|-b|-l|-tl|-tr|-br|-bl|-s|-e|-ss|-se|-es|-ee]-(${ALLOWED_SIZES.join('|')}) ` +
  '(any prefix such as md: or hover:), or in CSS/inline styles var(--radius-card) / var(--radius-control)';

/** A Tailwind corner utility, prefixes included. `isRounded` and `roundedCorners` do not match. */
const ROUNDED_TOKEN = /(?<![\w-])((?:[\w-]+:)*)(rounded(?:-[\w[\]().%/#,-]+)?)(?![\w-])/g;
const ALLOWED_ROUNDED = new RegExp(
  `^rounded(?:-(?:t|r|b|l|tl|tr|br|bl|s|e|ss|se|es|ee))?-(?:${ALLOWED_SIZES.join('|')})$`,
);
const ALLOWED_RADIUS_VALUE = /^var\(--radius-(?:card|control)\)$/;

/**
 * The CSS radius declarations that are deliberately not a token. Each must still exist
 * (the stale check below), so the list cannot outlive its reason.
 */
const CSS_RADIUS_EXCEPTIONS: readonly {
  readonly selector: string;
  readonly value: string;
  readonly reason: string;
}[] = [
  {
    selector: '::file-selector-button',
    value: 'revert',
    reason:
      "Tailwind CSS 4 strips the browser's own file-input button look; `revert` restores the " +
      "browser's corners (as v3 did), which is not a miForge surface.",
  },
  {
    selector: ':focus-visible',
    value: 'inherit',
    reason:
      'The focus ring is a box-shadow drawn on the focused element; `inherit` makes it follow ' +
      "that element's own (token) corners instead of choosing one.",
  },
];

interface CornerUse {
  readonly file: string;
  readonly line: number;
  readonly token: string;
  readonly allowed: boolean;
}

function lineOf(text: string, offset: number): number {
  let line = 1;
  for (let i = 0; i < offset && i < text.length; i += 1) if (text[i] === '\n') line += 1;
  return line;
}

function scanText(file: string, source: string, start: number, text: string, out: CornerUse[]) {
  for (const match of text.matchAll(ROUNDED_TOKEN)) {
    const utility = match[2] ?? '';
    out.push({
      file,
      line: lineOf(source, start + (match.index ?? 0)),
      token: match[0],
      allowed: ALLOWED_ROUNDED.test(utility),
    });
  }
}

const STRING_KINDS = new Set<ts.SyntaxKind>([
  ts.SyntaxKind.StringLiteral,
  ts.SyntaxKind.NoSubstitutionTemplateLiteral,
  ts.SyntaxKind.TemplateHead,
  ts.SyntaxKind.TemplateMiddle,
  ts.SyntaxKind.TemplateTail,
]);

const RADIUS_PROPERTY = /^border(?:[A-Z][a-z]+)*Radius$/;

/** Every corner class in a TS/TSX file's string tokens, and every inline `borderRadius`. */
function scanTypeScript(file: string, source: string): CornerUse[] {
  const sf = ts.createSourceFile(
    file,
    source,
    ts.ScriptTarget.Latest,
    true,
    file.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const out: CornerUse[] = [];
  const visit = (node: ts.Node): void => {
    if (STRING_KINDS.has(node.kind)) {
      const start = node.getStart(sf);
      scanText(file, source, start, source.slice(start, node.getEnd()), out);
    }
    if (
      ts.isPropertyAssignment(node) &&
      (ts.isIdentifier(node.name) || ts.isStringLiteral(node.name)) &&
      RADIUS_PROPERTY.test(node.name.text)
    ) {
      const init = node.initializer;
      const value = ts.isStringLiteralLike(init) ? init.text : init.getText(sf);
      out.push({
        file,
        line: lineOf(source, node.getStart(sf)),
        token: `${node.name.text}: ${init.getText(sf)}`,
        allowed: ALLOWED_RADIUS_VALUE.test(value.trim()),
      });
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return out;
}

/** Every `border-*-radius` declaration and `@apply rounded-*` in a CSS file, comments removed. */
function scanCss(file: string, source: string): (CornerUse & { selector: string })[] {
  // Blank comments to spaces of the same length so offsets (and line numbers) survive.
  const code = source.replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, ' '));
  const out: (CornerUse & { selector: string })[] = [];
  for (const block of code.matchAll(/([^{}]*)\{([^{}]*)\}/g)) {
    const selector = (block[1] ?? '').trim();
    const bodyStart = (block.index ?? 0) + (block[1] ?? '').length + 1;
    const body = block[2] ?? '';
    for (const decl of body.matchAll(/(?<![\w-])(border(?:-[a-z]+)*-radius)\s*:\s*([^;]+)/g)) {
      const value = (decl[2] ?? '').trim();
      const excepted = CSS_RADIUS_EXCEPTIONS.some(
        (row) => selector.includes(row.selector) && value === row.value,
      );
      out.push({
        file,
        selector,
        line: lineOf(source, bodyStart + (decl.index ?? 0)),
        token: `${decl[1]}: ${value}`,
        allowed: excepted || ALLOWED_RADIUS_VALUE.test(value),
      });
    }
    for (const apply of body.matchAll(/@apply\s+([^;]+)/g)) {
      const scratch: CornerUse[] = [];
      scanText(file, source, bodyStart + (apply.index ?? 0), apply[0], scratch);
      for (const use of scratch) out.push({ ...use, selector });
    }
  }
  return out;
}

function listFiles(dir: string, pattern: RegExp): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...listFiles(full, pattern));
    else if (pattern.test(entry) && !/\.test\.tsx?$/.test(entry)) out.push(full);
  }
  return out.sort();
}

const relative = (absolute: string): string =>
  path.relative(FRONTEND_ROOT, absolute).split(path.sep).join('/');

function walk(root: string) {
  const tsFiles = listFiles(root, /\.tsx?$/);
  const cssFiles = listFiles(root, /\.css$/);
  const tsUses = tsFiles.flatMap((file) =>
    scanTypeScript(relative(file), readFileSync(file, 'utf8')),
  );
  const cssUses = cssFiles.flatMap((file) => scanCss(relative(file), readFileSync(file, 'utf8')));
  return { tsFiles, cssFiles, tsUses, cssUses };
}

const report = (uses: readonly CornerUse[]): string[] =>
  uses
    .filter((use) => !use.allowed)
    .map((use) => `${use.file}:${use.line}  ${use.token}  (allowed: ${ALLOWED_DESCRIPTION})`);

describe('[corner tokens: the scanner itself]', () => {
  it('should flag a stray class in a string, a template literal, an array and a ternary', () => {
    const source = [
      "const a = 'p-2 rounded-lg';",
      'const b = `border ${on ? "x" : "y"} rounded-md`;',
      "const c = cn(['rounded-[6px]', 'rounded']);",
      "const d = on ? 'rounded-t-sm' : 'rounded-none';",
      'const e = <div className="hover:rounded-xl" />;',
      'const f = { borderRadius: 8 };',
    ].join('\n');
    const bad = scanTypeScript('x.tsx', source).filter((use) => !use.allowed);
    expect(bad.map((use) => [use.line, use.token])).toEqual([
      [1, 'rounded-lg'],
      [2, 'rounded-md'],
      [3, 'rounded-[6px]'],
      [3, 'rounded'],
      [4, 'rounded-t-sm'],
      [5, 'hover:rounded-xl'],
      [6, 'borderRadius: 8'],
    ]);
  });

  it('should accept the tokens under any prefix and ignore comments and identifiers', () => {
    const source = [
      '// rounded-lg in a line comment',
      '/* rounded-xl in a block comment */',
      'const a = <div>{/* rounded-2xl */}</div>;',
      "const b = 'rounded-card md:rounded-none rounded-tl-control hover:rounded-full';",
      "const isRounded = roundedCorners; const s = { borderRadius: 'var(--radius-card)' };",
    ].join('\n');
    const uses = scanTypeScript('x.tsx', source);
    expect(uses.filter((use) => !use.allowed)).toEqual([]);
    expect(uses.map((use) => use.token)).toEqual([
      'rounded-card',
      'md:rounded-none',
      'rounded-tl-control',
      'hover:rounded-full',
      "borderRadius: 'var(--radius-card)'",
    ]);
  });

  it('should flag a raw CSS radius but not a token, a named exception or a comment', () => {
    const css = [
      '/* border-radius: 3px; */',
      '.a { border-radius: 4px; }',
      '.b { border-top-left-radius: var(--radius-control); }',
      '::file-selector-button { border-radius: revert; }',
      '.c:focus-visible { border-radius: inherit; }',
      '.d { border-radius: inherit; }',
      '.e { @apply rounded-lg; }',
    ].join('\n');
    const bad = scanCss('x.css', css).filter((use) => !use.allowed);
    expect(bad.map((use) => [use.line, use.token])).toEqual([
      [2, 'border-radius: 4px'],
      [6, 'border-radius: inherit'],
      [7, 'rounded-lg'],
    ]);
  });
});

describe('[corner tokens: the source tree]', () => {
  const { tsFiles, cssFiles, tsUses, cssUses } = walk(SRC_ROOT);

  it('should actually walk the tree and find the known token uses', () => {
    // The construction-green guard: a wrong root or a dead pattern finds nothing and would
    // otherwise pass over a tree full of violations.
    expect(tsFiles.length).toBeGreaterThan(50);
    expect(cssFiles.map(relative)).toContain('src/index.css');
    const cardUses = tsUses.filter((use) => use.token === 'rounded-card');
    expect(cardUses.map((use) => use.file)).toContain('src/components/common/Card.tsx');
    expect(cardUses.length).toBeGreaterThanOrEqual(5);
    expect(tsUses.filter((use) => use.token === 'rounded-control').length).toBeGreaterThanOrEqual(
      5,
    );
  });

  it('should use only the card/control/full/none corner classes in source', () => {
    expect(report(tsUses)).toEqual([]);
  });

  it('should declare a CSS border radius only as a token or a named exception', () => {
    expect(report(cssUses)).toEqual([]);
  });

  it('should have no stale CSS radius exception', () => {
    const stale = CSS_RADIUS_EXCEPTIONS.filter(
      (row) =>
        !cssUses.some(
          (use) => use.selector.includes(row.selector) && use.token.endsWith(row.value),
        ),
    ).map((row) => `${row.selector} :: ${row.value}`);
    expect(stale).toEqual([]);
  });
});
