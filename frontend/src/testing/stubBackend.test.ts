/**
 * The test backend answers `/api/health` with the backend's own body (feature 015, FR-015.15).
 *
 * The stub used to hand-write `{status, dependencies: []}` - an array where the backend serves
 * an object - and a parser for that imagined shape (`src/api/health.ts`) sat unused. The body is
 * now the generated fixture (`backend/scripts/write_health_fixture.py`), and nothing may
 * re-grow a private health shape.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';
import healthFixture from './fixtures/health.json';
import { answer } from './stubBackend';

const SRC = path.resolve(__dirname, '..');

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return sources(full);
    return /\.(ts|tsx)$/.test(name) ? [full] : [];
  });
}

describe('stubBackend', () => {
  it('the stub serves the health fixture', () => {
    const served = answer('GET', '/api/health');
    expect(served.status).toBe(200);
    expect(served.body).toStrictEqual(healthFixture);
    expect(Array.isArray((served.body as { dependencies: unknown }).dependencies)).toBe(false);
    expect(Object.keys(healthFixture)).toContain('legacy_names');
  });

  it('nothing imports @/api/health', () => {
    const files = sources(SRC);
    expect(files.length).toBeGreaterThan(50);
    const importers: string[] = [];
    for (const file of files) {
      const text = readFileSync(file, 'utf8');
      const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
      sf.forEachChild((node) => {
        if (
          ts.isImportDeclaration(node) &&
          ts.isStringLiteral(node.moduleSpecifier) &&
          /(^@\/api\/health$|\/api\/health$|^\.\/health$)/.test(node.moduleSpecifier.text)
        ) {
          importers.push(path.relative(SRC, file));
        }
      });
    }
    expect(importers).toStrictEqual([]);
    expect(() => statSync(path.join(SRC, 'api', 'health.ts'))).toThrow();
  });
});
