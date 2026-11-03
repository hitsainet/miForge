/**
 * Every JavaScript dependency is pinned EXACTLY, and each lockfile agrees with its manifest.
 *
 * Suite dependency baseline, 2026-10-21: "pin exactly". A caret range let this frontend sit
 * on whatever the lockfile happened to hold - `^1.49.1` was quietly resolving Playwright
 * 1.64.0 - so the manifest stopped saying what ran. The image (`frontend/Dockerfile`) and CI
 * (`frontend-tests.yml`, `manual-ci.yml`, `deploy-manual.yml`) all install with `npm ci`, which
 * installs the LOCKFILE; so the two failure modes are a range in the manifest (the next lock
 * regeneration takes a different version with no commit naming it) and a lockfile whose root
 * entry or installed entry disagrees with the manifest (what ships is not what is declared).
 *
 * Both manifests are read: this frontend and the manual (`manual/`, Docusaurus), which has no
 * test runner of its own beyond `node --test` for its publish check.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const REPO = path.resolve(__dirname, '..', '..', '..');

const PROJECTS = ['frontend', 'manual'] as const;
const SECTIONS = ['dependencies', 'devDependencies'] as const;

interface Manifest {
  readonly dependencies?: Record<string, string>;
  readonly devDependencies?: Record<string, string>;
}

interface Lock {
  readonly packages: Record<string, Manifest & { readonly version?: string }>;
}

const read = <T>(project: string, file: string): T =>
  JSON.parse(readFileSync(path.join(REPO, project, file), 'utf8')) as T;

/** An exact semver: `1.2.3`, optionally with a prerelease. No range operator of any kind. */
export const isExact = (spec: string): boolean => /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(spec);

const declared = (manifest: Manifest): [string, string, string][] =>
  SECTIONS.flatMap((section) =>
    Object.entries(manifest[section] ?? {}).map(
      ([name, spec]) => [section, name, spec] as [string, string, string],
    ),
  );

describe('the exactness check itself', () => {
  it('accepts an exact version and refuses every range shape', () => {
    expect(isExact('19.3.1')).toBe(true);
    expect(isExact('1.0.0-rc.1')).toBe(true);
    for (const spec of ['^19.3.1', '~5.9.3', '>=1.0.0', '1.x', '*', 'latest', '1.2', '<2']) {
      expect(isExact(spec), spec).toBe(false);
    }
  });
});

describe.each(PROJECTS)('%s', (project) => {
  const manifest = read<Manifest>(project, 'package.json');
  const lock = read<Lock>(project, 'package-lock.json');

  it('declares at least one dependency (so the checks below are not vacuous)', () => {
    expect(declared(manifest).length).toBeGreaterThan(3);
  });

  it('pins every dependency and devDependency exactly', () => {
    const loose = declared(manifest).filter(([, , spec]) => !isExact(spec));
    expect(loose, 'pin with an exact version, as the suite baseline does').toEqual([]);
  });

  it("has a lockfile whose root entry declares the manifest's specs, unchanged", () => {
    const root = lock.packages[''] ?? {};
    for (const section of SECTIONS) {
      expect(root[section] ?? {}, `${project}/package-lock.json ${section}`).toEqual(
        manifest[section] ?? {},
      );
    }
  });

  it('has a lockfile that installs exactly the pinned version of each', () => {
    const drift = declared(manifest).flatMap(([, name, spec]) => {
      const installed = lock.packages[`node_modules/${name}`]?.version;
      return installed === spec ? [] : [{ name, pinned: spec, locked: installed ?? 'absent' }];
    });
    expect(drift).toEqual([]);
  });
});
