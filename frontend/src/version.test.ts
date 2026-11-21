/**
 * The version the sidebar footer shows comes from `package.json`, and agrees with the backend.
 *
 * The suite versioning rule: miForge is 0.1.0 until its feature MVP, and the operator decides
 * each bump. A bump is one field in `package.json` and one in `pyproject.toml`; this holds the
 * two equal so the footer never names a version the backend is not.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { APP_VERSION } from './App';

const ROOT = path.resolve(__dirname, '..');

function packageVersion(): string {
  const parsed = JSON.parse(readFileSync(path.join(ROOT, 'package.json'), 'utf8')) as {
    version: string;
  };
  return parsed.version;
}

function pyprojectVersion(): string {
  const source = readFileSync(path.join(ROOT, '..', 'pyproject.toml'), 'utf8');
  const match = /^version = "([^"]+)"$/m.exec(source);
  expect(match, 'no version line in pyproject.toml').not.toBeNull();
  return match![1]!;
}

describe('[version]', () => {
  it('should show the package.json version, prefixed with v', () => {
    expect(APP_VERSION).toBe(`v${packageVersion()}`);
    expect(APP_VERSION).toMatch(/^v\d+\.\d+\.\d+$/);
  });

  it('should agree with the backend pyproject.toml', () => {
    expect(packageVersion()).toBe(pyprojectVersion());
  });
});
