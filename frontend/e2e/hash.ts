/**
 * Match a page's URL by its hash EXACTLY (CodeQL js/incomplete-sanitization, alert 27).
 *
 * The persistence spec used to build `new RegExp(`${hash.replace(/\//g, '\\/')}$`)`. That
 * escaped `/` and nothing else, so a hash carrying a regex metacharacter (`.`, `?`, `+`,
 * `(`, `\` ...) became a pattern: `#/runs.v1` also matched `#/runsXv1`, and the URL assertion
 * could pass on a screen nobody navigated to. No pattern is built here at all - the URL is
 * parsed and its `hash` compared as a string.
 */

/** A `toHaveURL` predicate: true iff `url`'s hash is exactly `hash` (e.g. `'#/runs'`). */
export const hashIs =
  (hash: string) =>
  (url: URL): boolean =>
    url.hash === hash;
