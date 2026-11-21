/**
 * The brand guard, frontend half (feature 015; FR-015.10, FR-015.22; FTID §3.8).
 *
 * Every TypeScript string literal, template part and JSX text under `src/` (tests excluded),
 * and the text and attribute values of `index.html`, are scanned for miLLM, miStudio and
 * miSuite. What may keep an old name comes only from the census export: each register entry's
 * `pattern_js` and literals, the old environment and repository names still in transition,
 * and the active `allow` guards. The guard reports until the export's `closed` reaches P4
 * (`brand-guard.json`); from then it is enforced in the private repository and still reports
 * on the public mirror (suite R23, `brandGuardMode`). It is proven able to fail on planted
 * input from day one.
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import ts from 'typescript';
import { describe, expect, it, vi } from 'vitest';
import {
  REPO_ROOT,
  activeGuards,
  compileJs,
  loadExport,
  mirrorSnapshot,
  phaseIndex,
  withHeader,
  type Export,
} from './legacyIdentifiers';

const OLD = /millm|mistudio|misuite/gi;
const SRC = path.join(REPO_ROOT, 'frontend', 'src');
const INDEX_HTML = path.join(REPO_ROOT, 'frontend', 'index.html');
const PLANTS = JSON.parse(
  readFileSync(
    path.join(REPO_ROOT, 'backend', 'tests', 'support', 'fixtures', 'brand_plants.json'),
    'utf8',
  ),
) as {
  tsx_string: string;
  suite_phrase: { allowed_from_p5: string[]; always_denied: string[]; misuite: string };
};

export interface Item {
  readonly where: string;
  readonly text: string;
  readonly segment: string | null;
}

export interface Finding {
  readonly where: string;
  readonly token: string;
  readonly rules: readonly string[];
}

export function sourceFiles(root: string): string[] {
  return readdirSync(root).flatMap((name) => {
    const full = path.join(root, name);
    if (statSync(full).isDirectory()) return name === 'test' ? [] : sourceFiles(full);
    return /\.(ts|tsx)$/.test(name) && !/\.test\./.test(name) ? [full] : [];
  });
}

export function tsItems(text: string, where: string): Item[] {
  const sf = ts.createSourceFile(where, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const items: Item[] = [];
  const add = (node: ts.Node, value: string): void => {
    const line = sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1;
    items.push({ where: `${where}:${line}`, text: value, segment: node.getText(sf) });
  };
  const visit = (node: ts.Node): void => {
    if (
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      ts.isTemplateHead(node) ||
      ts.isTemplateMiddle(node) ||
      ts.isTemplateTail(node)
    ) {
      add(node, node.text);
    } else if (ts.isJsxText(node) && node.text.trim() !== '') {
      add(node, node.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return items;
}

/** Text nodes and attribute values of an HTML document, comments skipped. */
export function htmlItems(html: string, where: string): Item[] {
  const clean = html.replace(/<!--[\s\S]*?-->/g, '');
  const items: Item[] = [];
  for (const match of clean.matchAll(/\s[\w:-]+\s*=\s*("([^"]*)"|'([^']*)')/g)) {
    items.push({ where: `${where} attribute`, text: match[2] ?? match[3] ?? '', segment: null });
  }
  for (const match of clean.matchAll(/>([^<]+)</g)) {
    const text = (match[1] ?? '').trim();
    if (text !== '') items.push({ where: `${where} text`, text, segment: null });
  }
  return items;
}

interface Rules {
  readonly masks: RegExp[];
  readonly literals: string[];
  readonly bare: Set<string>;
  readonly substrings: string[];
  readonly deny: Array<[string, RegExp]>;
}

export function rulesFor(exported: Export, phase: string = exported.header.phase): Rules {
  const entries = exported.entries.filter((e) => e.status !== 'retired');
  const literals = entries.flatMap((e) => e.literals);
  const isBare = (lit: string): boolean => new RegExp(`^(${OLD.source})$`, 'i').test(lit);
  const active = activeGuards(exported, phase);
  return {
    masks: [
      ...entries.map((e) => compileJs(e, true)),
      ...active.filter((g) => g.effect === 'allow').map((g) => compileJs(g, true)),
    ],
    literals: literals.filter((lit) => !isBare(lit)),
    bare: new Set(literals.filter(isBare)),
    substrings: [
      ...exported.env.filter((r) => r.status === 'transition').map((r) => r.old),
      ...exported.repo.filter((r) => r.status === 'transition').map((r) => r.old),
    ],
    deny: active.filter((g) => g.effect === 'deny').map((g) => [g.id, compileJs(g, true)]),
  };
}

function blank(text: string, start: number, end: number): string {
  return text.slice(0, start) + ' '.repeat(end - start) + text.slice(end);
}

export function scan(items: readonly Item[], rules: Rules): Finding[] {
  const findings: Finding[] = [];
  for (const item of items) {
    let text = item.segment ?? item.text;
    if (rules.bare.has(item.text)) continue;
    for (const pattern of rules.masks) {
      for (const match of [...text.matchAll(pattern)]) {
        text = blank(text, match.index, match.index + match[0].length);
      }
    }
    for (const name of [...rules.literals, ...rules.substrings]) {
      let start = text.indexOf(name);
      while (start !== -1) {
        text = blank(text, start, start + name.length);
        start = text.indexOf(name, start + name.length);
      }
    }
    const denied = rules.deny.flatMap(([id, pattern]) =>
      [...text.matchAll(pattern)].map((m) => [id, m.index, m.index + m[0].length] as const),
    );
    for (const match of text.matchAll(OLD)) {
      const end = match.index + match[0].length;
      const ids = denied.filter(([, s, e]) => s < end && match.index < e).map(([id]) => id);
      findings.push({
        where: item.where,
        token: match[0],
        rules: [`token:${match[0].toLowerCase()}`, ...ids],
      });
    }
  }
  return findings;
}

/** CF-25 m5: every literal must match its entry's `pattern_js` (the JavaScript dialect). */
export function literalMisses(exported: Export): string[] {
  return exported.entries.flatMap((entry) =>
    entry.literals.filter((lit) => !compileJs(entry).test(lit)).map((lit) => `${entry.id}: ${lit}`),
  );
}

function configuredEnforced(root = REPO_ROOT): boolean {
  const config = JSON.parse(readFileSync(path.join(root, 'brand-guard.json'), 'utf8')) as {
    enforced: boolean;
  };
  return config.enforced;
}

/** Set to "1" to enforce whatever `brand-guard.json` says (IPD-8; both halves honour it). */
export const ENFORCE_ENV = 'MIFORGE_BRAND_GUARD_ENFORCE';

/**
 * Whether a finding FAILS ('enforced') or is printed ('report'). The same rule as the backend
 * half (`brand_scan.mode`): the switch enforces; `enforced: false` reports; `enforced: true`
 * enforces in the private repository and reports on the public mirror (suite R23, operator,
 * 2026-11-11), because the mirror's export withholds the private repository names and the old
 * host names. The mirror is the sync's commit shape, never a missing file.
 */
export function brandGuardMode(
  root = REPO_ROOT,
  env: Record<string, string | undefined> = process.env,
): 'report' | 'enforced' {
  if (env[ENFORCE_ENV] === '1') return 'enforced';
  if (!configuredEnforced(root)) return 'report';
  return mirrorSnapshot(root) ? 'report' : 'enforced';
}

export function expectedEnforced(exported: Export): boolean {
  return phaseIndex(exported.header.closed) >= phaseIndex('P4');
}

function report(
  name: string,
  findings: readonly Finding[],
  mode: 'report' | 'enforced' = brandGuardMode(),
): void {
  const lines = findings.map((f) => `${f.where}: ${f.token} [${f.rules.join(', ')}]`).sort();
  console.warn(
    `[brand guard: ${name}] ${lines.length} finding(s) (${mode} mode)\n` +
      lines.map((l) => `  ${l}`).join('\n'),
  );
  if (mode === 'enforced') expect(lines).toStrictEqual([]);
}

function frontendItems(root = SRC): Item[] {
  return sourceFiles(root).flatMap((file) =>
    tsItems(readFileSync(file, 'utf8'), path.relative(REPO_ROOT, file)),
  );
}

const plant = (text: string): Item[] => [{ where: 'planted.tsx:1', text, segment: null }];

describe('brand guard (frontend)', () => {
  const exported = loadExport();

  it('scans TS string literals and JSX text', () => {
    const items = frontendItems();
    expect(items.length).toBeGreaterThan(100);
    report('frontend/src', scan(items, rulesFor(exported)));
  });

  it('scans index.html text and attributes', () => {
    const items = htmlItems(readFileSync(INDEX_HTML, 'utf8'), 'frontend/index.html');
    expect(items.length).toBeGreaterThan(0);
    report('frontend/index.html', scan(items, rulesFor(exported)));
  });

  it('surface discovery yields files', () => {
    expect(sourceFiles(SRC).length).toBeGreaterThan(50);
    expect(sourceFiles(SRC).some((f) => f.endsWith('.test.tsx'))).toBe(false);
  });

  it('enforced mode fails on planted input', () => {
    const findings = scan(
      tsItems(`const label = "${PLANTS.tsx_string}";`, 'x.tsx'),
      rulesFor(exported),
    );
    expect(findings.map((f) => f.token)).toStrictEqual(['miLLM']);
    expect(scan(plant('<p>Open miLLM</p>'), rulesFor(exported)).length).toBe(1);
  });

  describe('suite_phrase', () => {
    const at = (phase: string, text: string): Finding[] =>
      scan(plant(text), rulesFor(withHeader(exported, { phase }), phase));

    // The public mirror's export withholds G-STUDIO-PATH (it names a workstation path): skip by name.
    const pathWithheld = exported.header.withheld.includes('G-STUDIO-PATH');

    it.skipIf(pathWithheld)('P5 allows the suite phrases and paths that P4 denies', () => {
      for (const allowed of PLANTS.suite_phrase.allowed_from_p5) {
        expect(at('P5', allowed)).toStrictEqual([]);
        expect(at('P4', allowed).length).toBeGreaterThan(0);
      }
      for (const denied of PLANTS.suite_phrase.always_denied) {
        expect(at('P5', denied).length).toBeGreaterThan(0);
        expect(at('P4', denied).length).toBeGreaterThan(0);
      }
    });

    it('miSuite is attributed to G-MISUITE only from P5', () => {
      const p5 = at('P5', PLANTS.suite_phrase.misuite);
      const p4 = at('P4', PLANTS.suite_phrase.misuite);
      expect(p5[0]?.rules).toContain('G-MISUITE');
      expect(p4[0]?.rules).toStrictEqual(['token:misuite']);
    });
  });

  it('compiles every pattern_js with js_flags', () => {
    for (const record of [...exported.entries, ...exported.guards]) {
      expect(() => compileJs(record)).not.toThrow();
    }
  });

  it('every literal matches pattern_js', () => {
    expect(literalMisses(exported)).toStrictEqual([]);
    const missing = {
      ...exported,
      entries: [{ ...exported.entries[0]!, id: 'DR-X', pattern_js: 'nothing', literals: ['x'] }],
    };
    expect(literalMisses(missing)).toStrictEqual(['DR-X: x']);
  });

  describe('where the guard enforces (suite R23)', () => {
    // The shapes of `backend/tests/support/mirror_snapshot.py`: only one parentless commit
    // carrying BOTH sync trailers is the mirror.
    const SHA = 'a68add3e85ae0cf25c16b885826f5fbfaeded0d5';
    const SOURCE = ['--trailer', `Source-Commit: ${SHA}`];
    const PASSED = ['--trailer', 'Private-CI: passed'];
    const shapes: Record<string, { commits: string[][]; mirror: boolean }> = {
      none: { commits: [], mirror: false },
      private: { commits: [[], []], mirror: false },
      'orphan-plain': { commits: [[]], mirror: false },
      'orphan-source-only': { commits: [SOURCE], mirror: false },
      'trailers-with-parent': { commits: [[], [...SOURCE, ...PASSED]], mirror: false },
      mirror: { commits: [[...SOURCE, ...PASSED]], mirror: true },
    };

    function tree(shape: string, enforced: boolean): string {
      const root = mkdtempSync(path.join(tmpdir(), `brand-${shape}-`));
      const git = (...args: string[]): void => {
        execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@t', ...args], {
          cwd: root,
          stdio: 'pipe',
        });
      };
      const commits = shapes[shape]!.commits;
      if (commits.length > 0) git('init', '-q');
      commits.forEach((trailers, index) => {
        writeFileSync(path.join(root, 'README.md'), `${index}\n`);
        git('add', '-A');
        git('commit', '-qm', 'miForge - filtered snapshot', ...trailers);
      });
      writeFileSync(path.join(root, 'brand-guard.json'), JSON.stringify({ enforced }));
      return root;
    }

    it.each(Object.keys(shapes))('%s: enforced privately, report on the mirror', (shape) => {
      const on = tree(shape, true);
      const off = tree(shape, false);
      try {
        expect(mirrorSnapshot(on)).toBe(shapes[shape]!.mirror);
        expect(brandGuardMode(on, {})).toBe(shapes[shape]!.mirror ? 'report' : 'enforced');
        expect(brandGuardMode(off, {})).toBe('report');
        expect(brandGuardMode(off, { [ENFORCE_ENV]: '1' })).toBe('enforced');
        expect(brandGuardMode(off, { [ENFORCE_ENV]: '0' })).toBe('report');
      } finally {
        rmSync(on, { recursive: true, force: true });
        rmSync(off, { recursive: true, force: true });
      }
    });

    it('the same finding fails privately and is reported on the mirror', () => {
      const findings = scan(plant('<p>Open miLLM</p>'), rulesFor(exported));
      expect(findings.length).toBe(1);
      const priv = tree('private', true);
      const mirror = tree('mirror', true);
      try {
        expect(() => report('planted', findings, brandGuardMode(priv, {}))).toThrow();
        expect(() => report('planted', findings, brandGuardMode(mirror, {}))).not.toThrow();
      } finally {
        rmSync(priv, { recursive: true, force: true });
        rmSync(mirror, { recursive: true, force: true });
      }
    });
  });

  it('the surface tests report in the mode of this tree', () => {
    // Wiring: `report()` with no mode asks `brandGuardMode()`. Forced by the switch, a finding
    // must fail; a `report()` that printed and passed in every mode would make the flip a no-op.
    const findings = scan(plant('<p>Open miLLM</p>'), rulesFor(exported));
    vi.stubEnv(ENFORCE_ENV, '1');
    try {
      expect(brandGuardMode()).toBe('enforced');
      expect(() => report('planted', findings)).toThrow();
      expect(() => report('planted', [])).not.toThrow();
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it('enforced keys to closed', () => {
    expect(configuredEnforced()).toBe(expectedEnforced(exported));
    expect(expectedEnforced(withHeader(exported, { phase: 'P4', closed: 'P3' }))).toBe(false);
    expect(expectedEnforced(withHeader(exported, { phase: 'P4', closed: 'P4' }))).toBe(true);
  });
});
