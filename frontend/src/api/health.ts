/**
 * `GET /api/health` (foundation tasks 5.2 and 7.4; ADR-013, ADR-010).
 *
 * Foundation 5.2: *"each dependency separately (Postgres, Redis), plus **which optional
 * integrations are configured**, naming the unconfigured ones."*
 *
 * **The response is parsed defensively, not cast.** The endpoint does not exist yet
 * (foundation 5.2 is another agent's task) and this module is the frontend's half of the
 * same contract. A `response as HealthResponse` would make the first shape disagreement
 * a `TypeError` inside a component; `parseHealth` turns it into a `malformed` failure with
 * the body kept, which is the one outcome an operator can act on. This project's
 * recurring defect is a claim its cited source did not support - a cast is that claim in
 * TypeScript.
 */

import { apiGet, type ApiResult } from './client';
import type { ApiFailure } from './errors';

export type DependencyStatus = 'ok' | 'degraded' | 'unavailable';

export interface DependencyHealth {
  readonly name: string;
  readonly status: DependencyStatus;
  readonly detail: string | null;
}

/**
 * One optional sibling. `configured: false` is **normal** (ADR-010), not a fault, which
 * is why it is a boolean beside the name rather than a `status` with an error value.
 */
export interface IntegrationHealth {
  readonly name: string;
  readonly configured: boolean;
  readonly detail: string | null;
}

export interface HealthResponse {
  readonly status: DependencyStatus;
  readonly dependencies: readonly DependencyHealth[];
  readonly integrations: readonly IntegrationHealth[];
}

export const HEALTH_PATH = '/health';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asStatus(value: unknown): DependencyStatus {
  return value === 'ok' || value === 'degraded' || value === 'unavailable' ? value : 'unavailable';
}

function asDetail(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

/**
 * Parse a health body, or say what was wrong with it.
 *
 * An unrecognised `status` string becomes `unavailable` rather than `ok`: failing closed
 * is the rule where correctness is at stake (ADR-013), and a health check that reads an
 * unknown value as healthy is the one direction of error that matters.
 */
export function parseHealth(
  body: unknown,
): { ok: true; value: HealthResponse } | { ok: false; reason: string } {
  if (!isRecord(body)) return { ok: false, reason: 'health body is not a JSON object' };

  const dependencies = body['dependencies'];
  if (!Array.isArray(dependencies)) {
    return { ok: false, reason: "'dependencies' is missing or not an array" };
  }
  const integrations = body['integrations'];
  if (!Array.isArray(integrations)) {
    return { ok: false, reason: "'integrations' is missing or not an array" };
  }

  return {
    ok: true,
    value: {
      status: asStatus(body['status']),
      dependencies: dependencies.filter(isRecord).map((row) => ({
        name: String(row['name'] ?? 'unnamed'),
        status: asStatus(row['status']),
        detail: asDetail(row['detail']),
      })),
      integrations: integrations.filter(isRecord).map((row) => ({
        name: String(row['name'] ?? 'unnamed'),
        // Only an explicit `true` counts as configured. A missing key is "not
        // configured", which is the safe reading: it reports a capability as unavailable
        // rather than offering one that is not there.
        configured: row['configured'] === true,
        detail: asDetail(row['detail']),
      })),
    },
  };
}

export async function fetchHealth(): Promise<ApiResult<HealthResponse>> {
  const result = await apiGet<unknown>(HEALTH_PATH);
  if (!result.ok) return result;
  const parsed = parseHealth(result.value);
  if (parsed.ok) return { ok: true, value: parsed.value };
  const failure: ApiFailure = {
    kind: 'malformed',
    status: 200,
    reason: parsed.reason,
    body: JSON.stringify(result.value).slice(0, 2000),
  };
  return { ok: false, failure };
}

/** The unconfigured integrations, by name. Empty when every one is configured. */
export function unconfiguredIntegrations(health: HealthResponse): readonly string[] {
  return health.integrations.filter((row) => !row.configured).map((row) => row.name);
}
