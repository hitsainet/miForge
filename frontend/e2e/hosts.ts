/**
 * The hosts the UI may contact besides its own origin: Google Fonts, and nothing else.
 *
 * Compared EXACTLY by hostname (CodeQL js/incomplete-url-substring-sanitization, alerts 3-4).
 * The earlier `hostname.endsWith('googleapis.com')` admitted `evilgoogleapis.com`, so the
 * no-foreign-request test could pass while the UI called a host nobody allowed.
 */
export const FONT_HOSTS: ReadonlySet<string> = new Set([
  'fonts.googleapis.com',
  'fonts.gstatic.com',
]);

/** True iff `url` parses and its hostname is exactly one of {@link FONT_HOSTS}. */
export function isAllowedFontHost(url: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  return parsed.protocol === 'https:' && FONT_HOSTS.has(parsed.hostname);
}
