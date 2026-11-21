/**
 * The sidebar's Manual link points at the address the manual is actually published at.
 *
 * The published address is `manual/docusaurus.config.ts`'s `url` + `baseUrl`. The two files are
 * compared here so that moving the site (or the link) without the other is a red test rather
 * than a dead link in every installation. The config is read as text; each extraction must
 * match, so a reshaped config fails loudly instead of passing over nothing.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { MANUAL_URL, PUBLISHED_MANUAL_URL } from './manual';

const CONFIG = path.resolve(__dirname, '../../../manual/docusaurus.config.ts');

function constant(source: string, name: string): string {
  const match = new RegExp(`const ${name} = '([^']+)';`).exec(source);
  expect(match, `${name} not found in manual/docusaurus.config.ts`).not.toBeNull();
  return match![1]!;
}

describe('[manual link]', () => {
  it('should equal the manual site config url + baseUrl', () => {
    const source = readFileSync(CONFIG, 'utf8');
    // The config must actually USE the two constants, or comparing them proves nothing.
    expect(source).toMatch(/^\s*url: SITE_URL,$/m);
    expect(source).toMatch(/^\s*baseUrl: BASE_URL,$/m);
    expect(PUBLISHED_MANUAL_URL).toBe(constant(source, 'SITE_URL') + constant(source, 'BASE_URL'));
  });

  it('should default to the published manual when no build override is set', () => {
    expect(MANUAL_URL).toBe(PUBLISHED_MANUAL_URL);
    expect(MANUAL_URL).toMatch(/^https:\/\//);
  });
});
