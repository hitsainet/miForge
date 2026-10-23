/**
 * The screen in the URL (F9 task 13.1; operator request 2026-10-17; closes FPRD **F-6**).
 *
 * A reload used to land on Runs whatever was open, because the screen lived in `useState` and
 * nowhere else. This module turns a screen and its parameters into a **hash** and back, from
 * the `path` every `ROUTES` row already declares - so `config/routes.ts` stays the only
 * registry and no router dependency is added (ADR-011 is untouched).
 *
 * ## Why the hash
 *
 * The hash is never sent to a server. It therefore works the same under nginx, `vite dev`,
 * `vite preview` and a bare `kubectl port-forward`, and a screen path can never collide with a
 * location nginx owns itself (`/api/`, `/socket.io/`, `/health`, `/assets/`).
 *
 * ## Shape
 *
 * `#<path>?<extra params>` - `:name` segments of the route's path are filled from the params,
 * and every other param goes in the query: `#/runs/run-a`, `#/packs/onboard?draftId=d1&step=3`,
 * `#/checkpoints?node=run-a%2Fstep-40`.
 *
 * **A literal segment beats a parameter**, so `#/runs/new` is the New run form and not a run
 * named `new`. Anything that does not match a row, or does not decode, is `null` - and the
 * caller falls back to its default screen rather than rendering a URL it cannot read.
 *
 * **An optional last segment** (`:name?`, ENH-004 D2, 2026-10-20): Help's path is
 * `/help/:topic?`, so `#/help` and `#/help/runs` are the same screen and a topic can be linked,
 * bookmarked and reloaded. Without the parameter the segment is left out; only the LAST segment
 * may be optional.
 */

/** The two fields of a route row this module needs. Typed loosely so tests can pass a table. */
export interface PathRoute<Id extends string = string> {
  readonly id: Id;
  readonly path: string;
}

export interface UrlScreen<Id extends string = string> {
  readonly screen: Id;
  readonly params: Readonly<Record<string, string>>;
}

function segments(path: string): string[] {
  return path.split('/').filter((part) => part.length > 0);
}

/**
 * The hash for a screen, or `null` when a `:param` its path needs is missing or empty.
 *
 * `null` rather than a half-built path: `#/runs/` would read back as nothing, and a URL the
 * app cannot read back is worse than leaving the URL where it was.
 */
export function toHash<Id extends string>(
  screen: Id,
  params: Readonly<Record<string, string>>,
  routes: readonly PathRoute<Id>[],
): string | null {
  const route = routes.find((entry) => entry.id === screen);
  if (route === undefined) return null;
  const used = new Set<string>();
  const parts: string[] = [];
  for (const part of segments(route.path)) {
    if (part.startsWith(':')) {
      const optional = part.endsWith('?');
      const name = optional ? part.slice(1, -1) : part.slice(1);
      const value = params[name];
      if (value === undefined || value.length === 0) {
        if (optional) continue;
        return null;
      }
      used.add(name);
      parts.push(encodeURIComponent(value));
    } else {
      parts.push(part);
    }
  }
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params).sort(([a], [b]) => a.localeCompare(b))) {
    if (used.has(key) || value.length === 0) continue;
    query.append(key, value);
  }
  const qs = query.toString();
  return `#/${parts.join('/')}${qs.length > 0 ? `?${qs}` : ''}`;
}

/** The screen a hash names, or `null` for an empty, unknown or undecodable one. */
export function fromHash<Id extends string>(
  hash: string,
  routes: readonly PathRoute<Id>[],
): UrlScreen<Id> | null {
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  if (raw.length === 0) return null;
  const at = raw.indexOf('?');
  const pathPart = at === -1 ? raw : raw.slice(0, at);
  const queryPart = at === -1 ? '' : raw.slice(at + 1);
  const wanted = segments(pathPart);
  if (wanted.length === 0) return null;

  let best: { route: PathRoute<Id>; captured: Record<string, string>; count: number } | null = null;
  for (const route of routes) {
    let pattern = segments(route.path);
    const last = pattern[pattern.length - 1];
    // `:name?` last: one segment shorter matches too, as the route without its parameter.
    if (last !== undefined && last.startsWith(':') && last.endsWith('?')) {
      if (wanted.length === pattern.length - 1) pattern = pattern.slice(0, -1);
      else pattern = [...pattern.slice(0, -1), last.slice(0, -1)];
    }
    if (pattern.length !== wanted.length) continue;
    const captured: Record<string, string> = {};
    let matched = true;
    let count = 0;
    for (let index = 0; index < pattern.length; index += 1) {
      const expected = pattern[index] as string;
      const actual = wanted[index] as string;
      if (expected.startsWith(':')) {
        let decoded: string;
        try {
          decoded = decodeURIComponent(actual);
        } catch {
          matched = false;
          break;
        }
        if (decoded.length === 0) {
          matched = false;
          break;
        }
        captured[expected.slice(1)] = decoded;
        count += 1;
      } else if (expected !== actual) {
        matched = false;
        break;
      }
    }
    // Fewest parameters wins: a literal segment is a more specific match than a capture.
    if (matched && (best === null || count < best.count)) best = { route, captured, count };
  }
  if (best === null) return null;

  const params: Record<string, string> = {};
  try {
    for (const [key, value] of new URLSearchParams(queryPart)) {
      if (value.length > 0) params[key] = value;
    }
  } catch {
    return null;
  }
  // The path's own captures win over a query key of the same name.
  return { screen: best.route.id, params: { ...params, ...best.captured } };
}
