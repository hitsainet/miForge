/**
 * WSP editor endpoints: list, register, change, remove, and the protocol test (2026-11-20).
 *
 * The Editor endpoints screen's calls, made as the operator (no agent header):
 *
 * - `GET /api/v1/wsp/endpoints` - the live rows, never the key or its ciphertext;
 * - `POST /api/v1/wsp/endpoints` - register one (`wsp/endpoints.py` `register_endpoint`);
 * - `PATCH /api/v1/wsp/endpoints/{id}` - `action_protocol`, `context_window_tokens`, `thinking`,
 *   `thinking_tokens`, `output_cap`, `paging`, `output_ceiling_tokens` and (feature 020)
 *   `history_mode` only (`PATCHABLE`); every other field is bound to the sealed key, so it is a
 *   new registration;
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
/**
 * `wsp.models.EFFORTS` (feature 020, FR-020.4, F20-D6): the effort levels a hosted endpoint's
 * thinking may name. Never above `high` (W3-D9).
 */
export const EFFORTS = ['low', 'medium', 'high'] as const;
/** `wsp.models.THINKING_VALUES` (2026-11-21; the efforts since feature 020). */
export const THINKING_VALUES = ['off', 'on', 'default', ...EFFORTS] as const;
export const DEFAULT_THINKING = 'off';
/**
 * `wsp.models.HISTORY_MODES` (feature 020, FR-020.1): how `miwsp_engine` keeps a session's
 * conversation. `paging` rewrites older results into stubs; `append_only` never changes what it
 * has sent, so a hosted model's cache can serve it.
 */
export const HISTORY_MODES = ['paging', 'append_only'] as const;
export const DEFAULT_HISTORY_MODE = 'paging';
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
  'output_ceiling_tokens',
  'history_mode',
] as const;
/** `wsp.endpoints.ENDPOINT_ID`. */
export const ENDPOINT_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,39}$/;
/** `wsp.endpoints.MAX_CONTEXT_WINDOW_TOKENS`. */
export const MAX_CONTEXT_WINDOW_TOKENS = 10_000_000;
/** `wsp_worker.editor_limits` (feature 019): an empty ceiling's value, and the least one. */
export const DEFAULT_OUTPUT_CEILING_TOKENS = 32_000;
export const MIN_OUTPUT_CEILING_TOKENS = 256;
/** `wsp_worker.editor_limits.DEFAULT_WINDOW_TOKENS`: the window a run assumes when none is known. */
export const DEFAULT_WINDOW_TOKENS = 8192;

/** A window read: its value (null when nothing was reported) and when it was read. */
export interface WindowRead {
  readonly tokens: number | null;
  readonly read_at: string | null;
}

/** The latest protocol test (feature 019 P2, FR-019.16): its result, run, time and tested values. */
export interface ProtocolTestState {
  readonly result: 'passed' | 'failed' | null;
  readonly run_id: string | null;
  readonly at: string | null;
  readonly tested: {
    readonly base_url: string | null;
    readonly model: string | null;
    readonly action_protocol: string | null;
  };
  readonly current: boolean;
  readonly cleared: boolean;
  readonly required: boolean;
}

/** The newest run's window read (feature 019, F19-D14): its value, its source, the run, when. */
export interface LatestRunWindow {
  readonly tokens: number | null;
  readonly source: string;
  readonly run_id: string;
  readonly at: string | null;
}

/**
 * Why a run on this endpoint would be refused today (feature 020, FR-020.5, F20-D12; TD-2): the
 * backend's `endpoints.run_refusal`, the same check filing and push make. `null` when runs start.
 */
export interface RunsRefused {
  readonly code: string;
  readonly message: string;
}

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
  /** Feature 019 (FR-019.14): the endpoint's ceiling (null: 32,000) and the one in effect. */
  readonly output_ceiling_tokens: number | null;
  readonly output_ceiling_effective: number;
  /** Feature 019 (FR-019.3): the read at registration, the latest run's, what a run uses now. */
  readonly registration_window: WindowRead;
  readonly latest_run_window: LatestRunWindow | null;
  /**
   * The next run's expected window (operator, 2026-11-25, F-P4-1): stated, else the latest run's
   * reported read (its run and time), else 8,192 assumed; never the registration read (F19-D11).
   * A miServer endpoint that states no window reads `waits` instead of assumed (2026-11-27): its
   * runs wait for miServer to report a window. The backend sends tokens 0 there; it reads null.
   */
  readonly window_now: WindowNow;
  /**
   * What a run uses when its own push read fails: stated, else 8,192 assumed; `waits` (the
   * backend's tokens 0, read as null) on a miServer endpoint that states no window, whose run
   * waits and then fails `editor_not_resident` (2026-11-27).
   */
  readonly window_if_read_fails: {
    readonly tokens: number | null;
    readonly source: 'stated' | 'assumed' | 'waits';
    /**
     * Operator decision D1 (2026-11-27): what a run does when the server gives no answer at
     * all; null from an older backend. For a generic endpoint the two fields above are then the
     * case of a server that answers without a window.
     */
    readonly no_answer: 'stated' | 'waits' | 'not_read' | null;
  };
  readonly window_warnings: readonly string[];
  /** Feature 019 P2 (FR-019.16): the latest protocol test, or "not tested" (`result` null). */
  readonly protocol_test: ProtocolTestState;
  /** Feature 020 (FR-020.1): `paging` or `append_only`; an older backend's row reads `paging`. */
  readonly history_mode: string;
  /** Feature 020 (FR-020.5, F20-D12): why runs are refused now, or null. */
  readonly runs_refused: RunsRefused | null;
}

export interface WindowNow {
  readonly tokens: number | null;
  readonly source: 'stated' | 'reported' | 'assumed' | 'waits';
  readonly run_id: string | null;
  readonly at: string | null;
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
    ...windowFields(row),
    protocol_test: protocolTestOf(row['protocol_test']),
    history_mode: text(row['history_mode'] ?? DEFAULT_HISTORY_MODE),
    runs_refused: runsRefusedOf(row['runs_refused']),
  };
}

/** `runs_refused`, copied by name: `{code, message}` or null (an older backend serves none). */
function runsRefusedOf(raw: unknown): RunsRefused | null {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const value = raw as Record<string, unknown>;
  if (typeof value['code'] !== 'string' || value['code'] === '') return null;
  return { code: value['code'], message: text(value['message']) };
}

/**
 * The thinking values an endpoint takes, as the backend's `wsp.endpoints.thinking_accepted`
 * (feature 020, FR-020.4) states them, for the forms' choices only: **the backend decides**, and
 * refuses anything else by name (`wsp_endpoint_thinking_unsupported`).
 *
 * - hosted Anthropic: `default` and the efforts;
 * - hosted OpenAI-compatible: `off`, `default` and the efforts (`reasoning_effort`);
 * - local: `off` and `default`, and `on` for miServer's OpenAI kind. With `serving` not given
 *   (the add form, where it may not be chosen yet), `on` is offered and the backend decides.
 *
 * With no locality yet (no host typed), every value is offered.
 */
export function thinkingChoices(
  kind: string,
  locality: string | null,
  serving?: string,
): readonly string[] {
  if (locality === null) return THINKING_VALUES;
  if (locality === 'external') {
    return kind === 'anthropic' ? ['default', ...EFFORTS] : ['off', 'default', ...EFFORTS];
  }
  const on = serving === undefined || (kind === 'openai_compatible' && serving === 'miserver');
  return on ? ['off', 'on', 'default'] : ['off', 'default'];
}

const count = (value: unknown): number | null =>
  typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : null;

const record = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;

/** Feature 019's endpoint fields, copied by name (a row from an older backend reads as unknown). */
function windowFields(
  row: Record<string, unknown>,
): Pick<
  WspEndpoint,
  | 'output_ceiling_tokens'
  | 'output_ceiling_effective'
  | 'registration_window'
  | 'latest_run_window'
  | 'window_now'
  | 'window_if_read_fails'
  | 'window_warnings'
> {
  const ceiling = count(row['output_ceiling_tokens']);
  const registration = record(row['registration_window']);
  const latest = record(row['latest_run_window']);
  const now = record(row['window_now']);
  const fails = record(row['window_if_read_fails']);
  const stated = count(row['context_window_tokens']);
  const fallback =
    stated !== null
      ? ({ tokens: stated, source: 'stated' } as const)
      : ({ tokens: DEFAULT_WINDOW_TOKENS, source: 'assumed' } as const);
  return {
    output_ceiling_tokens: ceiling,
    output_ceiling_effective:
      count(row['output_ceiling_effective']) ?? ceiling ?? DEFAULT_OUTPUT_CEILING_TOKENS,
    registration_window: {
      tokens: count(registration?.['tokens']),
      read_at: maybeText(registration?.['read_at']),
    },
    latest_run_window:
      latest === null
        ? null
        : {
            tokens: count(latest['tokens']),
            source: text(latest['source']),
            run_id: text(latest['run_id']),
            at: maybeText(latest['at']),
          },
    window_now:
      now !== null && now['source'] === 'waits'
        ? { tokens: null, source: 'waits', run_id: null, at: null }
        : now !== null && count(now['tokens']) !== null
          ? {
              tokens: count(now['tokens']) as number,
              source:
                now['source'] === 'stated' || now['source'] === 'reported'
                  ? now['source']
                  : 'assumed',
              run_id: maybeText(now['run_id']),
              at: maybeText(now['at']),
            }
          : { ...fallback, run_id: null, at: null },
    window_if_read_fails: {
      ...(fails !== null && fails['source'] === 'waits'
        ? { tokens: null, source: 'waits' as const }
        : fails !== null && count(fails['tokens']) !== null
          ? {
              tokens: count(fails['tokens']) as number,
              source: fails['source'] === 'stated' ? ('stated' as const) : ('assumed' as const),
            }
          : fallback),
      no_answer:
        fails !== null &&
        (fails['no_answer'] === 'stated' ||
          fails['no_answer'] === 'waits' ||
          fails['no_answer'] === 'not_read')
          ? fails['no_answer']
          : null,
    },
    window_warnings: Array.isArray(row['window_warnings'])
      ? row['window_warnings'].filter((w): w is string => typeof w === 'string')
      : [],
  };
}

/** The protocol test, copied by name; an older backend's row reads "not tested". */
function protocolTestOf(raw: unknown): ProtocolTestState {
  const value = record(raw) ?? {};
  const tested = record(value['tested']) ?? {};
  const result = value['result'];
  return {
    result: result === 'passed' || result === 'failed' ? result : null,
    run_id: maybeText(value['run_id']),
    at: maybeText(value['at']),
    tested: {
      base_url: maybeText(tested['base_url']),
      model: maybeText(tested['model']),
      action_protocol: maybeText(tested['action_protocol']),
    },
    current: value['current'] === true,
    cleared: value['cleared'] === true,
    required: value['required'] === true,
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
  readonly output_ceiling_tokens?: number;
  readonly history_mode?: string;
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
  readonly output_ceiling_tokens: number | null;
  readonly history_mode: string;
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
