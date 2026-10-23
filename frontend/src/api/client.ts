/**
 * The API client (foundation task 7.4; ADR-013, ADR-010, ADR-017).
 *
 * One module through which the browser reaches miForge, and **only** miForge. ADR-022
 * fences the core path off every sibling endpoint and a browser `fetch` is on that path,
 * so `API_BASE` is a relative path: there is no base-URL setting here to point at a
 * sibling host, and `clients/` on the backend remains the single outbound call site for
 * each optional sibling (ADR-010). F9's FTID 2.2 states the same rule - "no frontend
 * fetch to a sibling's host".
 *
 * **Success and failure are different shapes, deliberately.** A 2xx body is plain JSON
 * and is returned as-is; a non-2xx body is decoded by `decodeErrorEnvelope`. The client
 * never inspects a successful body for an `error` key - ADR-013 says the envelope wraps
 * failures only, and a client that looked for it on success would turn a resource with a
 * field called `error` into a spurious failure.
 */

import { ApiError, decodeErrorEnvelope, transportFailure, type ApiFailure } from './errors';

/** Relative on purpose. See the module docstring. */
export const API_BASE = '/api';

export interface RequestOptions {
  readonly signal?: AbortSignal;
  /** Query parameters. `undefined` values are omitted, never sent as the string "undefined". */
  readonly query?: Readonly<Record<string, string | number | boolean | undefined>>;
}

/**
 * A result the caller must branch on, for the many call sites where a refusal is an
 * expected answer rather than an exception - an unconfigured sibling, an oversized run,
 * a pack with a missing slot. ADR-013's own framing: refusing correctly is half this
 * system's behaviour, so a refusal is a first-class return.
 */
export type ApiResult<T> =
  { readonly ok: true; readonly value: T } | { readonly ok: false; readonly failure: ApiFailure };

type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

function buildUrl(path: string, query: RequestOptions['query']): string {
  const url = `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
  if (!query) return url;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined) continue;
    search.append(key, String(value));
  }
  const qs = search.toString();
  return qs ? `${url}?${qs}` : url;
}

/**
 * Issue one request and decode the outcome. The only place `fetch` is called.
 *
 * Never throws for a refusal - it returns `{ok: false}`. The only throw is a programming
 * error (a bad URL), which is not a refusal and must not be dressed as one.
 */
export async function request<T>(
  method: HttpMethod,
  path: string,
  body?: unknown,
  options: RequestOptions = {},
): Promise<ApiResult<T>> {
  const init: RequestInit = {
    method,
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    ...(options.signal ? { signal: options.signal } : {}),
  };

  let response: Response;
  try {
    response = await fetch(buildUrl(path, options.query), init);
  } catch (cause) {
    // No response at all. This is NOT a refusal: there is no code, and inventing one
    // ("request_failed") would put a word in the backend's mouth. The run view needs
    // this distinction to tell "miForge is down" from "miForge said no".
    return { ok: false, failure: transportFailure(causeText(cause)) };
  }

  const raw = await response.text();
  const parsed = parseJson(raw);

  if (!response.ok) {
    // `parsed.ok === false` hands the raw text to the decoder, which then reports
    // `malformed` and keeps the body - a proxy's HTML 502 arrives here.
    return {
      ok: false,
      failure: decodeErrorEnvelope(response.status, parsed.ok ? parsed.value : raw),
    };
  }

  // 204, or any empty success body, is `undefined` rather than a parse failure.
  if (raw.length === 0) {
    return { ok: true, value: undefined as T };
  }
  if (!parsed.ok) {
    // A 2xx whose body is not JSON is a broken contract, not a refusal; `malformed`
    // keeps the body so it is diagnosable.
    return {
      ok: false,
      failure: {
        kind: 'malformed',
        status: response.status,
        reason: 'a successful response carried a body that is not JSON',
        body: raw.slice(0, 2000),
      },
    };
  }
  return { ok: true, value: parsed.value as T };
}

export const apiGet = <T>(path: string, options?: RequestOptions): Promise<ApiResult<T>> =>
  request<T>('GET', path, undefined, options);

export const apiPost = <T>(
  path: string,
  body?: unknown,
  options?: RequestOptions,
): Promise<ApiResult<T>> => request<T>('POST', path, body, options);

/**
 * `PUT`, added for F7's step commit.
 *
 * `PUT /api/v1/pack-drafts/{id}/steps/{n}` is the one `PUT` on the whole live surface -
 * verified by enumerating `create_app().routes` - and it is a `PUT` because committing step
 * 4 twice must leave the draft in the same state rather than appending a second answer.
 * Added here rather than reached with `apiPost` so the method the server sees is the method
 * the route declares.
 */
export const apiPut = <T>(
  path: string,
  body?: unknown,
  options?: RequestOptions,
): Promise<ApiResult<T>> => request<T>('PUT', path, body, options);

export const apiPatch = <T>(
  path: string,
  body?: unknown,
  options?: RequestOptions,
): Promise<ApiResult<T>> => request<T>('PATCH', path, body, options);

export const apiDelete = <T>(path: string, options?: RequestOptions): Promise<ApiResult<T>> =>
  request<T>('DELETE', path, undefined, options);

/**
 * The throwing form, for a call site that genuinely cannot continue - a store's initial
 * load, where there is nothing to render either way.
 *
 * Prefer `apiGet` and a branch. A `try`/`catch` around several calls catches the first
 * failure and discards the rest, and F6's validator returns **every** failure together
 * precisely so a form can show them all at once (F9 FTID 3.5a).
 */
export async function requestOrThrow<T>(
  method: HttpMethod,
  path: string,
  body?: unknown,
  options?: RequestOptions,
): Promise<T> {
  const result = await request<T>(method, path, body, options);
  if (!result.ok) throw new ApiError(result.failure);
  return result.value;
}

function parseJson(raw: string): { ok: true; value: unknown } | { ok: false } {
  if (raw.length === 0) return { ok: false };
  try {
    return { ok: true, value: JSON.parse(raw) };
  } catch {
    return { ok: false };
  }
}

function causeText(cause: unknown): string {
  if (cause instanceof Error) return `${cause.name}: ${cause.message}`;
  return String(cause);
}
