/**
 * The error envelope's decoding (foundation task 7.4; ADR-013, ADR-010, ADR-018).
 *
 * The load-bearing assertions are the two that say a refusal is **not** flattened: the
 * `code` survives, and `<service>_not_configured` narrows to its own kind. Everything
 * else in the UI's error handling depends on those two, because the code is the only
 * thing in the response a screen can switch on.
 */

import { describe, expect, it } from 'vitest';
import {
  ApiError,
  decodeErrorEnvelope,
  describeFailure,
  isApiError,
  MALFORMED_BODY_LIMIT,
  parseErrorEnvelope,
  transportFailure,
} from './errors';

const refusal = {
  error: {
    code: 'budget_exceeded',
    message: 'The estimate exceeds free VRAM. Reduce the batch size or choose a smaller model.',
    details: { estimate_bytes: 14_000_000_000, free_bytes: 11_000_000_000 },
  },
};

describe('[parseErrorEnvelope]', () => {
  it('should accept the backend envelope', () => {
    const parsed = parseErrorEnvelope(refusal);
    expect(parsed.ok).toBe(true);
  });

  it('should accept an envelope whose details key is absent and default it to empty', () => {
    // `refuse()` sends `{}`, but a hand-written error may omit the key, and refusing a
    // whole envelope over an empty bag would discard a perfectly good code.
    const parsed = parseErrorEnvelope({ error: { code: 'x', message: 'y' } });
    expect(parsed.ok && parsed.envelope.error.details).toEqual({});
  });

  it('should reject an envelope with no code, because a codeless refusal is unactionable', () => {
    // Mirrors `Refusal.__init__`, which raises on an empty code for the same reason.
    const parsed = parseErrorEnvelope({ error: { code: '', message: 'y' } });
    expect(parsed).toEqual({
      ok: false,
      reason: "'error.code' is missing or not a non-empty string",
    });
  });

  it('should reject a body that is not an envelope', () => {
    expect(parseErrorEnvelope('Bad Gateway').ok).toBe(false);
    expect(parseErrorEnvelope(null).ok).toBe(false);
    expect(parseErrorEnvelope([refusal]).ok).toBe(false);
    expect(parseErrorEnvelope({ detail: 'FastAPI default shape' }).ok).toBe(false);
  });
});

describe('[decodeErrorEnvelope]', () => {
  it('should keep the code, the message and the details of a refusal', () => {
    const failure = decodeErrorEnvelope(422, refusal);
    expect(failure).toEqual({
      kind: 'refusal',
      status: 422,
      code: 'budget_exceeded',
      message: refusal.error.message,
      details: refusal.error.details,
    });
  });

  it('should narrow <service>_not_configured to its own kind, naming the service', () => {
    // ADR-010. This is the assertion that stops the one refusal that is *normal* from
    // being handled as a fault - and the one a screen is most likely to forget.
    const failure = decodeErrorEnvelope(409, {
      error: {
        code: 'midataworks_not_configured',
        message: 'midataworks is not configured; set its base URL to enable contract parts.',
        details: {
          service: 'midataworks',
          capabilities_unavailable: ['contract parts', 'human-review selector'],
        },
      },
    });
    expect(failure.kind).toBe('service_not_configured');
    expect(failure).toMatchObject({
      service: 'midataworks',
      code: 'midataworks_not_configured',
      capabilitiesUnavailable: ['contract parts', 'human-review selector'],
    });
  });

  it('should not invent capabilities when the details do not state them', () => {
    const failure = decodeErrorEnvelope(409, {
      error: { code: 'millm_not_configured', message: 'no base URL', details: {} },
    });
    expect(failure).toMatchObject({
      kind: 'service_not_configured',
      service: 'millm',
      capabilitiesUnavailable: [],
    });
  });

  it('should report a non-envelope failure body as malformed and keep the body', () => {
    const html = '<html><body>502 Bad Gateway</body></html>';
    const failure = decodeErrorEnvelope(502, html);
    expect(failure).toMatchObject({ kind: 'malformed', status: 502, body: html });
  });

  it('should truncate but never drop an oversized malformed body', () => {
    const huge = 'x'.repeat(MALFORMED_BODY_LIMIT + 500);
    const failure = decodeErrorEnvelope(500, huge);
    expect(failure.kind).toBe('malformed');
    if (failure.kind !== 'malformed') throw new Error('unreachable');
    expect(failure.body.startsWith('x'.repeat(100))).toBe(true);
    expect(failure.body.endsWith('(truncated)')).toBe(true);
  });
});

describe('[transportFailure]', () => {
  it('should be its own kind with status 0, distinct from any refusal', () => {
    // "miForge is down" and "miForge said no" are different facts. Inventing a code for
    // the first would put a word in the backend's mouth.
    const failure = transportFailure('TypeError: Failed to fetch');
    expect(failure).toEqual({
      kind: 'transport',
      status: 0,
      cause: 'TypeError: Failed to fetch',
    });
  });
});

describe('[describeFailure]', () => {
  it('should lead with the code for a refusal', () => {
    expect(describeFailure(decodeErrorEnvelope(422, refusal))).toBe(
      `budget_exceeded: ${refusal.error.message}`,
    );
  });

  it('should name the service for an unconfigured integration', () => {
    const failure = decodeErrorEnvelope(409, {
      error: { code: 'midataworks_not_configured', message: 'set its base URL', details: {} },
    });
    expect(describeFailure(failure)).toContain('midataworks is not configured');
  });
});

describe('[ApiError]', () => {
  it('should carry the failure, not just a message', () => {
    const failure = decodeErrorEnvelope(422, refusal);
    const error = new ApiError(failure);
    expect(isApiError(error)).toBe(true);
    expect(error.failure).toBe(failure);
    // Code branches on `failure.code`, never on `message`: a reworded message must not
    // change behaviour, which is why the backend's messages say what to do next and the
    // code is the stable part.
    expect(error.failure.status).toBe(422);
  });
});
