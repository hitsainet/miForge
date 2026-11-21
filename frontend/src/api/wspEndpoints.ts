/**
 * WSP editor endpoints: list, register, change, remove, and the protocol test (2026-11-20).
 *
 * The Editor endpoints screen's calls, made as the operator (no agent header):
 *
 * - `GET /api/v1/wsp/endpoints` - the live rows, never the key or its ciphertext;
 * - `POST /api/v1/wsp/endpoints` - register one (`wsp/endpoints.py` `register_endpoint`);
 * - `PATCH /api/v1/wsp/endpoints/{id}` - `action_protocol`, `context_window_tokens`, `thinking`,
 *   `thinking_tokens`, `output_cap` and `paging` only (`PATCHABLE`); every other field is bound
 *   to the sealed key, so it is a new registration;
 * - `DELETE /api/v1/wsp/endpoints/{id}` - remove it (its id is never reused);
 * - `POST /api/v1/wsp/endpoints/{id}/protocol-test` - queue a short inspect run on the built-in
 *   bundle, answered `{run_id, status}`.
 *
 * **The key.** `RegisterBody.api_key` is sent once, in the registration's body, and is not part
 * of anything this module returns or stores. A listing row is read through `publicEndpoint`,
 * which copies the public fields by name: a field the backend might add (a ciphertext, a key)
 * never reaches the screen, whatever the screen renders.
 */
import { apiDelete, apiGet, apiPatch, apiPost, type ApiResult } from './client';
import { V1 } from './v1';

export const WSP_ENDPOINTS_PATH = `${V1}/wsp/endpoints`;
export const WSP_ENDPOINT_PATH = (endpointId: string): string =>
  `${WSP_ENDPOINTS_PATH}/${encodeURIComponent(endpointId)}`;

/** `wsp/models.py`'s closed enums. */
export const ENDPOINT_KINDS = ['openai_compatible', 'anthropic'] as const;
export const SERVINGS = ['generic', 'miserver'] as const;
export const OUTPUT_CAP_FIELDS = ['max_tokens', 'max_completion_tokens'] as const;
export const ACTION_PROTOCOLS = ['tool_calls', 'structured'] as const;
export const DEFAULT_ACTION_PROTOCOL = 'tool_calls';
/** `wsp.models.THINKING_VALUES` (2026-11-21): whether the engine asks the editor to think. */
export const THINKING_VALUES = ['off', 'on', 'default'] as const;
export const DEFAULT_THINKING = 'off';
/** `wsp.endpoints.DEFAULT_THINKING_TOKENS`: the reasoning allowance when none is set. */
export const DEFAULT_THINKING_TOKENS = 4096;
/** `wsp.models.OUTPUT_CAP_VALUES` (2026-11-21): whether each request names the output cap. */
export const OUTPUT_CAP_VALUES = ['on', 'off'] as const;
export const DEFAULT_OUTPUT_CAP = 'on';
/**
 * `wsp.models.PAGING_VALUES` (2026-11-21): whether `miwsp_engine` pages bulky tool results out
 * of the editor's history and keeps its experiment ledger.
 */
export const PAGING_VALUES = ['on', 'off'] as const;
export const DEFAULT_PAGING = 'on';
/** `wsp.endpoints.PRICE_FIELDS`, in US dollars per million tokens. */
export const PRICE_FIELDS = [
  'price_in_per_mtok',
  'price_out_per_mtok',
  'price_cache_write_per_mtok',
  'price_cache_read_per_mtok',
] as const;
export type PriceField = (typeof PRICE_FIELDS)[number];
/** `wsp.endpoints.PATCHABLE`: the only fields a change may name. */
export const PATCHABLE = [
  'action_protocol',
  'context_window_tokens',
  'thinking',
  'thinking_tokens',
  'output_cap',
  'paging',
] as const;
/** `wsp.endpoints.ENDPOINT_ID`. */
export const ENDPOINT_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,39}$/;
/** `wsp.endpoints.MAX_CONTEXT_WINDOW_TOKENS`. */
export const MAX_CONTEXT_WINDOW_TOKENS = 10_000_000;

/** One row of `GET /api/v1/wsp/endpoints` (`wsp.endpoints.public_document`). */
export interface WspEndpoint {
  readonly id: string;
  readonly kind: string;
  readonly base_url: string;
  readonly model: string;
  readonly locality: string;
  readonly serving: string;
  readonly output_cap_field: string;
  readonly price_in_per_mtok: string | null;
  readonly price_out_per_mtok: string | null;
  readonly price_cache_write_per_mtok: string | null;
  readonly price_cache_read_per_mtok: string | null;
  readonly has_key: boolean;
  readonly key_fingerprint: string | null;
  readonly created_at: string;
  readonly created_by: string;
  readonly action_protocol: string;
  readonly context_window_tokens: number | null;
  readonly thinking: string;
  readonly thinking_tokens: number | null;
  readonly output_cap: string;
  readonly paging: string;
}

export interface WspEndpointListing {
  readonly endpoints: readonly WspEndpoint[];
}

const text = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));
const maybeText = (value: unknown): string | null =>
  value === null || value === undefined ? null : text(value);

/**
 * The public fields of one row, copied by name. **The key-not-displayed guarantee's first half:**
 * nothing outside this list reaches a component, so a `key_ciphertext` or `api_key` the backend
 * ever served would stop here.
 */
export function publicEndpoint(raw: unknown): WspEndpoint {
  const row = (raw !== null && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  return {
    id: text(row['id']),
    kind: text(row['kind']),
    base_url: text(row['base_url']),
    model: text(row['model']),
    locality: text(row['locality']),
    serving: text(row['serving']),
    output_cap_field: text(row['output_cap_field']),
    price_in_per_mtok: maybeText(row['price_in_per_mtok']),
    price_out_per_mtok: maybeText(row['price_out_per_mtok']),
    price_cache_write_per_mtok: maybeText(row['price_cache_write_per_mtok']),
    price_cache_read_per_mtok: maybeText(row['price_cache_read_per_mtok']),
    has_key: row['has_key'] === true,
    key_fingerprint: maybeText(row['key_fingerprint']),
    created_at: text(row['created_at']),
    created_by: text(row['created_by']),
    action_protocol: text(row['action_protocol'] ?? DEFAULT_ACTION_PROTOCOL),
    context_window_tokens:
      typeof row['context_window_tokens'] === 'number' ? row['context_window_tokens'] : null,
    thinking: text(row['thinking'] ?? DEFAULT_THINKING),
    thinking_tokens: typeof row['thinking_tokens'] === 'number' ? row['thinking_tokens'] : null,
    output_cap: text(row['output_cap'] ?? DEFAULT_OUTPUT_CAP),
    paging: text(row['paging'] ?? DEFAULT_PAGING),
  };
}

/** The registration body. Only the fields given are sent; `api_key` only when typed. */
export interface RegisterBody {
  readonly id?: string;
  readonly kind: string;
  readonly base_url: string;
  readonly model: string;
  readonly serving?: string;
  readonly output_cap_field?: string;
  readonly action_protocol?: string;
  readonly context_window_tokens?: number;
  readonly thinking?: string;
  readonly thinking_tokens?: number;
  readonly output_cap?: string;
  readonly paging?: string;
  readonly price_in_per_mtok?: string;
  readonly price_out_per_mtok?: string;
  readonly price_cache_write_per_mtok?: string;
  readonly price_cache_read_per_mtok?: string;
  readonly api_key?: string;
}

export type PatchBody = Partial<{
  readonly action_protocol: string;
  readonly context_window_tokens: number | null;
  readonly thinking: string;
  readonly thinking_tokens: number | null;
  readonly output_cap: string;
  readonly paging: string;
}>;

export interface ProtocolTestRun {
  readonly run_id: string;
  readonly status: string;
  /** The thinking the test runs with (the endpoint's, frozen in its request), 2026-11-21. */
  readonly thinking?: string | null;
}

export async function listWspEndpoints(): Promise<ApiResult<WspEndpointListing>> {
  const result = await apiGet<{ endpoints?: unknown }>(WSP_ENDPOINTS_PATH);
  if (!result.ok) return result;
  const rows = Array.isArray(result.value?.endpoints) ? result.value.endpoints : [];
  return { ok: true, value: { endpoints: rows.map(publicEndpoint) } };
}

export async function registerWspEndpoint(body: RegisterBody): Promise<ApiResult<WspEndpoint>> {
  const result = await apiPost<unknown>(WSP_ENDPOINTS_PATH, body);
  return result.ok ? { ok: true, value: publicEndpoint(result.value) } : result;
}

export async function updateWspEndpoint(
  endpointId: string,
  body: PatchBody,
): Promise<ApiResult<WspEndpoint>> {
  const result = await apiPatch<unknown>(WSP_ENDPOINT_PATH(endpointId), body);
  return result.ok ? { ok: true, value: publicEndpoint(result.value) } : result;
}

export const removeWspEndpoint = (
  endpointId: string,
): Promise<ApiResult<{ readonly id: string; readonly removed_at: string }>> =>
  apiDelete(WSP_ENDPOINT_PATH(endpointId));

export const testWspEndpointProtocol = (endpointId: string): Promise<ApiResult<ProtocolTestRun>> =>
  apiPost<ProtocolTestRun>(`${WSP_ENDPOINT_PATH(endpointId)}/protocol-test`, {});

const PRIVATE_V4: readonly [number, number][] = [
  [0x0a000000, 8], // 10.0.0.0/8
  [0xac100000, 12], // 172.16.0.0/12
  [0xc0a80000, 16], // 192.168.0.0/16
  [0x7f000000, 8], // 127.0.0.0/8
  [0xa9fe0000, 16], // 169.254.0.0/16
];

function v4(host: string): number | null {
  const parts = host.split('.');
  if (parts.length !== 4) return null;
  let value = 0;
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part) || Number(part) > 255) return null;
    value = value * 256 + Number(part);
  }
  return value;
}

/**
 * What the backend will derive the locality as, for the form's hint only: **the backend decides**
 * (`wsp.endpoints.derive_locality`), and the list shows what it stored. Local is a private,
 * loopback or link-local address, `localhost`, or a cluster name (`*.svc`, `*.cluster.local`).
 * `null` when the URL has no host yet.
 */
export function expectedLocality(baseUrl: string): 'local' | 'external' | null {
  let host: string;
  try {
    host = new URL(baseUrl).hostname.toLowerCase().replace(/\.$/, '');
  } catch {
    return null;
  }
  if (host === '') return null;
  if (host === 'localhost' || host.endsWith('.svc') || host.endsWith('.cluster.local')) {
    return 'local';
  }
  const address = v4(host);
  if (address !== null) {
    const local = PRIVATE_V4.some(
      ([network, bits]) => address >>> (32 - bits) === network >>> (32 - bits),
    );
    return local ? 'local' : 'external';
  }
  if (host.startsWith('[') && host.endsWith(']')) {
    const v6 = host.slice(1, -1);
    if (v6 === '::1' || /^f[cd]/.test(v6) || /^fe[89ab]/.test(v6)) return 'local';
    return 'external';
  }
  return 'external';
}
