/**
 * The error envelope, decoded into a typed failure (foundation task 7.4; ADR-013, ADR-010).
 *
 * Every failure leaves the backend as `{"error": {"code", "message", "details"}}`
 * (`backend/src/core/errors.py`); success bodies are plain JSON and are **not** wrapped.
 *
 * **Why this module exists at all.** A client that catches a failure and produces a
 * generic "request failed" throws away the `code` - and the code is the only thing in the
 * response the UI can act on. ADR-010 fixes the spelling for an absent optional service
 * as `<service>_not_configured` and requires that it render as *unavailable by name*; a
 * generic error cannot do that, so the screen either shows a blank panel or silently
 * falls back, and both of those are the fabricated value ADR-018 forbids wearing a
 * different hat. So the decode is a typed narrowing, not a string.
 *
 * **Four kinds, and the two that are easy to leave out.** `refusal` and
 * `service_not_configured` are the backend speaking. `malformed` is a failure whose body
 * is *not* an envelope - a proxy's HTML 502, a truncated response - and it **keeps the
 * raw body**, because a decoder that discards what it could not parse leaves the operator
 * with nothing to diagnose. `transport` is no response at all. Collapsing either of the
 * last two into `refusal` would invent a `code` that nobody sent, which is the same
 * defect class as the four claims in this project that cited sources not supporting them.
 */

/** The wire shape. Validated, never assumed. */
export interface ErrorEnvelope {
  readonly error: {
    readonly code: string;
    readonly message: string;
    readonly details: Readonly<Record<string, unknown>>;
  };
}

interface FailureBase {
  /** HTTP status, or 0 when no response arrived. */
  readonly status: number;
}

/** A named refusal from the backend. The normal way this system says no. */
export interface RefusalFailure extends FailureBase {
  readonly kind: 'refusal';
  readonly code: string;
  readonly message: string;
  readonly details: Readonly<Record<string, unknown>>;
}

/**
 * A refusal whose code matches `<service>_not_configured` (ADR-010).
 *
 * Narrowed out of `refusal` so that a screen cannot handle refusals in general and
 * *forget* this case: it is the one refusal that is **normal** rather than a fault - an
 * unset base URL means absent-and-normal - and it must be rendered as the service's name
 * plus the capabilities lost, never as a failure and never substituted.
 */
export interface NotConfiguredFailure extends FailureBase {
  readonly kind: 'service_not_configured';
  readonly code: string;
  readonly message: string;
  readonly details: Readonly<Record<string, unknown>>;
  /** The service, from the code's prefix. `midataworks_not_configured` -> `midataworks`. */
  readonly service: string;
  /** What cannot be done without it, from `details.capabilities_unavailable`. */
  readonly capabilitiesUnavailable: readonly string[];
}

/** A failure response whose body is not the envelope. Keeps the body verbatim. */
export interface MalformedFailure extends FailureBase {
  readonly kind: 'malformed';
  /** Why the body was rejected as an envelope - which key was missing or wrongly typed. */
  readonly reason: string;
  /** The body as received, truncated for display but never dropped. */
  readonly body: string;
}

/** No response: DNS, connection refused, abort, CORS. `status` is 0. */
export interface TransportFailure extends FailureBase {
  readonly kind: 'transport';
  readonly status: 0;
  readonly cause: string;
}

export type ApiFailure =
  RefusalFailure | NotConfiguredFailure | MalformedFailure | TransportFailure;

/** The longest body kept on a `malformed` failure. A 502 HTML page is not worth more. */
export const MALFORMED_BODY_LIMIT = 2000;

/** ADR-010's spelling. The suffix is the contract, so it is a constant and not a literal. */
export const NOT_CONFIGURED_SUFFIX = '_not_configured';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Validate a parsed body as an envelope, or say why it is not one.
 *
 * `details` is permitted to be absent and becomes `{}` - the backend's `refuse()` sends
 * `{}` when no details were given, but a hand-written error or an older build may omit
 * the key, and refusing the whole envelope over an empty bag would discard a perfectly
 * good `code`. `code` and `message` are **not** optional: a refusal with no code is
 * unactionable, which is exactly why `Refusal.__init__` raises on an empty one.
 */
export function parseErrorEnvelope(
  body: unknown,
): { ok: true; envelope: ErrorEnvelope } | { ok: false; reason: string } {
  if (!isRecord(body)) {
    return { ok: false, reason: 'body is not a JSON object' };
  }
  const error = body['error'];
  if (!isRecord(error)) {
    return { ok: false, reason: "no 'error' object" };
  }
  const code = error['code'];
  if (typeof code !== 'string' || code.length === 0) {
    return { ok: false, reason: "'error.code' is missing or not a non-empty string" };
  }
  const message = error['message'];
  if (typeof message !== 'string') {
    return { ok: false, reason: "'error.message' is missing or not a string" };
  }
  const rawDetails = error['details'];
  const details = isRecord(rawDetails) ? rawDetails : {};
  return { ok: true, envelope: { error: { code, message, details } } };
}

/**
 * Decode a failed response into exactly one `ApiFailure`.
 *
 * @param status HTTP status. Pass 0 only for a transport failure (`transportFailure`).
 * @param body   The parsed JSON body, or the raw text when parsing failed.
 */
export function decodeErrorEnvelope(status: number, body: unknown): ApiFailure {
  const parsed = parseErrorEnvelope(body);
  if (!parsed.ok) {
    return {
      kind: 'malformed',
      status,
      reason: parsed.reason,
      body: renderBody(body),
    };
  }
  const { code, message, details } = parsed.envelope.error;

  // ADR-010's narrowing. `endsWith` and not a regex because the service name is whatever
  // the backend's `not_configured()` was given, and a pattern here would be a second,
  // stricter contract than the one that side actually enforces.
  if (code.endsWith(NOT_CONFIGURED_SUFFIX)) {
    return {
      kind: 'service_not_configured',
      status,
      code,
      message,
      details,
      service: code.slice(0, -NOT_CONFIGURED_SUFFIX.length),
      capabilitiesUnavailable: readStringArray(details['capabilities_unavailable']),
    };
  }

  return { kind: 'refusal', status, code, message, details };
}

export function transportFailure(cause: string): TransportFailure {
  return { kind: 'transport', status: 0, cause };
}

function readStringArray(value: unknown): readonly string[] {
  // A missing or wrongly-typed list becomes empty rather than throwing: the caller's
  // reason for existing is the service name, which is in the code itself. But it is
  // never *invented* - an empty list renders as "capabilities not stated", not as a
  // guess at what was lost.
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

function renderBody(body: unknown): string {
  const text = typeof body === 'string' ? body : safeStringify(body);
  return text.length > MALFORMED_BODY_LIMIT
    ? `${text.slice(0, MALFORMED_BODY_LIMIT)}… (truncated)`
    : text;
}

function safeStringify(body: unknown): string {
  try {
    return JSON.stringify(body) ?? String(body);
  } catch {
    return String(body);
  }
}

/**
 * The thrown form, so a caller may use `try`/`catch` without losing the narrowing.
 *
 * `failure` is the payload; `message` is only for a stack trace. Code that branches on
 * `error.message` is code that will break when a message is reworded, which is why the
 * backend's messages say what to do next and the **code** is the stable part.
 */
export class ApiError extends Error {
  readonly failure: ApiFailure;

  constructor(failure: ApiFailure) {
    super(describeFailure(failure));
    this.name = 'ApiError';
    this.failure = failure;
  }
}

export function isApiError(value: unknown): value is ApiError {
  return value instanceof ApiError;
}

/**
 * One sentence per failure kind, for a log line or a last-resort banner.
 *
 * **Not** for a refusal the UI is expected to handle: ADR-013 wants the backend's own
 * `message`, which says what to do next, and F9's `RefusalBanner` renders an unrecognised
 * code as the code itself rather than as a generic sentence, "because a generic message
 * discards the only diagnosable thing in the response".
 */
export function describeFailure(failure: ApiFailure): string {
  switch (failure.kind) {
    case 'refusal':
      return `${failure.code}: ${failure.message}`;
    case 'service_not_configured':
      return `${failure.service} is not configured: ${failure.message}`;
    case 'malformed':
      return `HTTP ${failure.status} with a body that is not an error envelope (${failure.reason})`;
    case 'transport':
      return `no response from miForge (${failure.cause})`;
  }
}
