/**
 * Every Help topic links a manual page and section that exist (ENH-004, 2026-10-20).
 *
 * Origin: miDataworks (Onegaishimas/miDataworks) `frontend/src/test/manualLinks.test.tsx`
 * @ 32cf406. Mode: adapt (`docs/REUSE.md`): miForge's manual is plain `manual/docs/**`, with no
 * generated pages, so a page maps straight to its Markdown file.
 *
 * The manual's own sources are read from disk, so a renamed page or heading is a red test here
 * rather than a dead link on the Help screen. The manual's build also refuses a broken anchor
 * (`onBrokenAnchors: 'throw'`), which covers links INSIDE the manual; this covers links INTO it.
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { HELP_TOPICS } from '@/config/help';
import { MANUAL_BASE_URL, PUBLISHED_MANUAL_URL, manualUrl } from '@/config/manual';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { HelpPanel } from '@/components/help/HelpPanel';
import { HelpLink } from '@/components/help/HelpLink';

const MANUAL = path.resolve(__dirname, '../../../manual');
const DOCS = path.join(MANUAL, 'docs');

/** github-slugger, as Docusaurus uses it, for the headings this manual writes. */
export function slug(heading: string): string {
  return heading
    .replace(/\s*\{#[^}]+\}\s*$/, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[*_`]/g, (mark) => (mark === '_' ? '_' : ''))
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s_-]/gu, '')
    .replace(/\s/g, '-');
}

/** The file a page path names, or `null`. */
function sourceOf(page: string): string | null {
  for (const candidate of [path.join(DOCS, `${page}.md`), path.join(DOCS, page, 'index.md')]) {
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

/** The anchors a Markdown file's headings produce. Fenced code is not a heading. */
function anchorsOf(file: string): Set<string> {
  const anchors = new Set<string>();
  let fenced = false;
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) {
      fenced = !fenced;
      continue;
    }
    if (fenced) continue;
    const match = /^#{1,6}\s+(.*)$/.exec(line);
    if (match === null) continue;
    const explicit = /\{#([^}]+)\}\s*$/.exec(match[1]!);
    anchors.add(explicit === null ? slug(match[1]!) : explicit[1]!);
  }
  return anchors;
}

describe('[manual links]', () => {
  it('should build every link from the published base', () => {
    expect(MANUAL_BASE_URL).toBe(PUBLISHED_MANUAL_URL);
    expect(MANUAL_BASE_URL.endsWith('/')).toBe(true);
    expect(manualUrl('guides/runs#the-screens')).toBe(
      'https://hitsainet.github.io/miForge/guides/runs#the-screens',
    );
    expect(manualUrl('/glossary', 'https://example.com/m')).toBe('https://example.com/m/glossary');
  });

  it('should slug headings as Docusaurus does', () => {
    expect(slug('The **Checkpoints** screen')).toBe('the-checkpoints-screen');
    expect(slug('Authoring a pack: **Onboard a domain**')).toBe(
      'authoring-a-pack-onboard-a-domain',
    );
    expect(slug('1. Make a model and data available')).toBe('1-make-a-model-and-data-available');
    expect(slug('Test, retention and the floor')).toBe('test-retention-and-the-floor');
  });

  it('should not read a comment in a code block as a heading', () => {
    // operators/settings.md has `# set an override` inside a bash block.
    const anchors = anchorsOf(path.join(DOCS, 'operators', 'settings.md'));
    expect(anchors.has('the-settings-screen')).toBe(true);
    expect(anchors.has('set-an-override')).toBe(false);
  });

  it('should name a manual page and section that exist, for every topic', () => {
    expect(HELP_TOPICS.length).toBeGreaterThanOrEqual(12);
    for (const topic of HELP_TOPICS) {
      const [page, anchor] = topic.manual.split('#') as [string, string | undefined];
      const file = sourceOf(page);
      expect(file, `${topic.id}: no manual page ${page}`).not.toBeNull();
      if (anchor !== undefined) {
        expect(anchorsOf(file!).has(anchor), `${topic.id}: no section #${anchor} in ${page}`).toBe(
          true,
        );
      }
    }
  });

  it('should link each summary to its manual section, once, in a new tab', () => {
    render(
      <NavigationProvider initialScreen="help">
        <HelpPanel />
      </NavigationProvider>,
    );
    for (const topic of HELP_TOPICS) {
      const card = screen.getByTestId(`help-topic-${topic.id}`);
      const links = within(card).getAllByRole('link');
      expect(links, topic.id).toHaveLength(1);
      expect(links[0]).toHaveAttribute('href', manualUrl(topic.manual));
      expect(links[0]).toHaveAttribute('target', '_blank');
      expect(links[0]?.getAttribute('rel')).toContain('noopener');
    }
    expect(screen.getByTestId('manual-home')).toHaveAttribute('href', MANUAL_BASE_URL);
  });

  it('should give each HelpLink exactly one manual link, and an unknown topic none', () => {
    for (const topic of HELP_TOPICS) {
      const { container, unmount } = render(
        <NavigationProvider initialScreen="runs">
          <HelpLink topic={topic.id} />
        </NavigationProvider>,
      );
      const links = container.querySelectorAll('a');
      expect(links, topic.id).toHaveLength(1);
      expect(links[0]).toHaveAttribute('href', manualUrl(topic.manual));
      unmount();
    }
    const { container } = render(
      <NavigationProvider initialScreen="runs">
        <HelpLink topic="no-such-topic" />
      </NavigationProvider>,
    );
    expect(container.querySelectorAll('a, button')).toHaveLength(0);
  });
});
