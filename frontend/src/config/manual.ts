/**
 * Where the published miForge manual lives - the sidebar's **Manual** link.
 *
 * The manual is a Docusaurus site under `manual/`, built and deployed to GitHub Pages from the
 * public mirror (`.github/workflows/deploy-manual.yml`). Its address is the site config's
 * `url` + `baseUrl` (`manual/docusaurus.config.ts`), and `manual.test.ts` compares the two, so
 * moving the site without moving this link - or the reverse - is a red test, not a dead link.
 *
 * A build can point the link elsewhere (an installation that hosts its own copy) with
 * `VITE_MANUAL_URL`; unset, it is the published site.
 */

/** The published manual's home page. */
export const PUBLISHED_MANUAL_URL = 'https://hitsainet.github.io/miForge/';

function fromEnv(): string | undefined {
  // Vite replaces `import.meta.env` at build time. The project declares no `vite/client`
  // ambient types, so the shape is stated here rather than widened for every module.
  const env = (import.meta as ImportMeta & { readonly env?: Record<string, unknown> }).env;
  const value: unknown = env?.['VITE_MANUAL_URL'];
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : undefined;
}

/** The address the sidebar links to. */
export const MANUAL_URL: string = fromEnv() ?? PUBLISHED_MANUAL_URL;

/**
 * The manual's base, always ending in `/`, so a page path can be appended (ENH-004, 2026-10-20).
 * Every deep link is built by {@link manualUrl} from this - never typed as a literal.
 */
export const MANUAL_BASE_URL: string = MANUAL_URL.endsWith('/') ? MANUAL_URL : `${MANUAL_URL}/`;

/**
 * The address of a manual page and, optionally, a section: `manualUrl('guides/runs#the-screens')`.
 * `test/manualLinks.test.tsx` checks every page and anchor the Help topics use against the
 * manual's own sources, and the manual's build refuses a broken anchor.
 */
export function manualUrl(page: string, base: string = MANUAL_BASE_URL): string {
  const root = base.endsWith('/') ? base : `${base}/`;
  return root + page.replace(/^\/+/, '');
}
