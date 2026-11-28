/**
 * Editor endpoints: the model servers a weight-space programming (WSP) session talks to as its
 * editor (2026-11-20).
 *
 * Until this screen, the operator registered an endpoint by hand with `curl`: every write is the
 * operator's, the backend's refusal to an agent sent it to "miForge's own screens", and none
 * existed. This is that screen. It calls the API **as the operator** (no agent header):
 *
 * - **the list** (`GET /api/v1/wsp/endpoints`), re-read every 30 s, after each change and on
 *   "Re-read the list", so an endpoint an agent proposed appears once you approve it;
 * - **Add** (`POST`): every field the registration accepts. The locality is never sent: the
 *   backend derives it from the host, and the form only says what it expects. The API key is a
 *   password field held in plain component state, sent once in the registration's body and
 *   cleared as soon as it is sent, whatever the answer; it is not part of the saved draft, and
 *   no answer carries it back. Every other entry survives a reload (`usePersistentState`);
 * - **Change** (`PATCH`): the action protocol, the context window, thinking and its reasoning
 *   allowance (2026-11-21), and the output ceiling (feature 019), the only fields a change may
 *   name; anything else is bound to the sealed key, so it is a new registration;
 * - **The window** (feature 019, FR-019.16): what the endpoint states, what the server reported
 *   at registration and at the latest run's push, what a run uses now, and the mismatch warning,
 *   as the backend serves them (`windowFacts`); never "the window the server reports" unread;
 * - **Hosted models** (feature 020, FR-020.22): the history mode (`paging` or `append_only`, with
 *   the kind's default and the refused choice explained), the window and, for an external
 *   Anthropic host, both cache prices marked required, the effort levels in place of on/off for
 *   an external host, and the detail's history mode, effort and `runs_refused` callout ("window
 *   required: set it to run", or "cache prices required: register this endpoint again"), as the
 *   backend serves them. The backend decides every one; the form only says what it expects;
 * - **Remove** (`DELETE`), after a confirmation. An id is never reused;
 * - **Run protocol test** (`POST .../protocol-test`): queues a short inspect run and links it.
 *
 * Every refusal is the backend's, rendered by `FailureNotice` with its code, message and details.
 * Every value is a text node. A row shows `has_key` and the key's fingerprint, never a key: rows
 * are read through `publicEndpoint`, which copies the public fields by name.
 */
import { useCallback, useState, type FormEvent, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Callout } from '@/components/common/Callout';
import { MonoId } from '@/components/common/MonoId';
import { Pill } from '@/components/common/Pill';
import { Field, Input, Select } from '@/components/common/Field';
import { FailureNotice } from '@/components/common/FailureNotice';
import { DraftRestoredNote } from '@/components/common/DraftRestoredNote';
import { formatEt } from '@/components/common/time';
import { useNavigation } from '@/contexts/NavigationContext';
import { usePolledRead } from '@/hooks/usePolledRead';
import { usePersistentState } from '@/hooks/usePersistentState';
import { DRAFT_KEYS } from '@/stores/draftStorage';
import type { ApiFailure } from '@/api/errors';
import type { ApiResult } from '@/api/client';
import {
  ACTION_PROTOCOLS,
  DEFAULT_ACTION_PROTOCOL,
  DEFAULT_OUTPUT_CAP,
  DEFAULT_OUTPUT_CEILING_TOKENS,
  DEFAULT_WINDOW_TOKENS,
  MIN_OUTPUT_CEILING_TOKENS,
  DEFAULT_PAGING,
  DEFAULT_THINKING,
  DEFAULT_THINKING_TOKENS,
  EFFORTS,
  ENDPOINT_KINDS,
  HISTORY_MODES,
  OUTPUT_CAP_FIELDS,
  OUTPUT_CAP_VALUES,
  PAGING_VALUES,
  PRICE_FIELDS,
  SERVINGS,
  expectedLocality,
  thinkingChoices,
  listWspEndpoints,
  registerWspEndpoint,
  removeWspEndpoint,
  testWspEndpointProtocol,
  updateWspEndpoint,
  type PatchBody,
  type PriceField,
  type ProtocolTestRun,
  type RegisterBody,
  type WspEndpoint,
  type WspEndpointListing,
} from '@/api/wspEndpoints';

export const WSP_ENDPOINTS_TITLE = 'Editor endpoints';
export const WSP_ENDPOINTS_SUBTITLE =
  'The model servers a weight-space programming session uses as its editor';
/** The list re-reads itself this often: an agent's proposal appears once you approve it. */
export const WSP_ENDPOINTS_POLL_MS = 30_000;

/** Everything the Add form holds, as typed. The API key is NOT here: it is never saved. */
export interface EndpointDraft {
  readonly id: string;
  readonly kind: string;
  readonly serving: string;
  readonly base_url: string;
  readonly model: string;
  readonly output_cap_field: string;
  readonly action_protocol: string;
  /** The window as typed. Not `context_window_tokens`: a draft key with the word "tokens" is
   * dropped by the secret rule (`stores/draftStorage.ts`), so it would not survive a reload. */
  readonly context_window: string;
  /** 2026-11-21: off | on | default, or '' for the default (off). */
  readonly thinking: string;
  /** The reasoning allowance as typed; named without "tokens" for the same draft rule. */
  readonly thinking_allowance: string;
  /** 2026-11-21: on | off, or '' for the default (on). */
  readonly output_cap: string;
  /** 2026-11-21: on | off, or '' for the default (on): the engine's paging and ledger. */
  readonly paging: string;
  /** Feature 019: the output ceiling as typed; named without "tokens" for the draft rule. */
  readonly output_ceiling: string;
  /** Feature 020: paging | append_only, or '' for the host's default. */
  readonly history_mode: string;
  readonly price_in_per_mtok: string;
  readonly price_out_per_mtok: string;
  readonly price_cache_write_per_mtok: string;
  readonly price_cache_read_per_mtok: string;
}

export const EMPTY_ENDPOINT_DRAFT: EndpointDraft = {
  id: '',
  kind: '',
  serving: '',
  base_url: '',
  model: '',
  output_cap_field: '',
  action_protocol: '',
  context_window: '',
  thinking: '',
  thinking_allowance: '',
  output_cap: '',
  paging: '',
  output_ceiling: '',
  history_mode: '',
  price_in_per_mtok: '',
  price_out_per_mtok: '',
  price_cache_write_per_mtok: '',
  price_cache_read_per_mtok: '',
};

function decodeDraft(data: unknown): EndpointDraft | null {
  if (data === null || typeof data !== 'object') return null;
  const row = data as Record<string, unknown>;
  const out: Record<string, string> = {};
  for (const key of Object.keys(EMPTY_ENDPOINT_DRAFT)) {
    const value = row[key];
    out[key] = typeof value === 'string' ? value : '';
  }
  return out as unknown as EndpointDraft;
}

const PRICE_LABEL: Readonly<Record<PriceField, string>> = {
  price_in_per_mtok: 'Input price',
  price_out_per_mtok: 'Output price',
  price_cache_write_per_mtok: 'Cache write price',
  price_cache_read_per_mtok: 'Cache read price',
};

/**
 * A typed number as the backend should see it: a finite number as a number, anything else as
 * the text itself, so the backend refuses it by name rather than the form turning it into nothing.
 */
function windowValue(typed: string): number | string {
  const value = Number(typed);
  return Number.isFinite(value) ? value : typed;
}

/** The registration body: the fields given, trimmed; the key only when typed. */
export function registrationBody(draft: EndpointDraft, apiKey: string): RegisterBody {
  const body: Record<string, unknown> = {
    kind: draft.kind,
    base_url: draft.base_url.trim(),
    model: draft.model.trim(),
  };
  if (draft.id.trim() !== '') body['id'] = draft.id.trim();
  if (draft.serving !== '') body['serving'] = draft.serving;
  if (draft.output_cap_field !== '') body['output_cap_field'] = draft.output_cap_field;
  if (draft.action_protocol !== '') body['action_protocol'] = draft.action_protocol;
  if (draft.context_window.trim() !== '') {
    body['context_window_tokens'] = windowValue(draft.context_window.trim());
  }
  if (draft.thinking !== '') body['thinking'] = draft.thinking;
  if (draft.output_cap !== '') body['output_cap'] = draft.output_cap;
  if (draft.paging !== '') body['paging'] = draft.paging;
  if (draft.history_mode !== '') body['history_mode'] = draft.history_mode;
  if (draft.thinking_allowance.trim() !== '') {
    body['thinking_tokens'] = windowValue(draft.thinking_allowance.trim());
  }
  if (draft.output_ceiling.trim() !== '') {
    body['output_ceiling_tokens'] = windowValue(draft.output_ceiling.trim());
  }
  for (const name of PRICE_FIELDS) {
    if (draft[name].trim() !== '') body[name] = draft[name].trim();
  }
  if (apiKey !== '') body['api_key'] = apiKey;
  return body as unknown as RegisterBody;
}

const isEffort = (value: string): boolean => (EFFORTS as readonly string[]).includes(value);

/**
 * What an endpoint's thinking setting does, in words (2026-11-21; the efforts since feature 020,
 * FR-020.22). With the endpoint's kind and locality, a hosted endpoint's words say what the engine
 * sends there (FR-020.15): a hosted Anthropic endpoint always thinks (adaptive), at the effort
 * named or the model's own; a hosted OpenAI-compatible one sends `reasoning_effort` for an effort.
 */
export function thinkingText(
  thinking: string,
  endpoint?: { readonly kind: string; readonly locality: string },
): string {
  const hostedAnthropic = endpoint?.locality === 'external' && endpoint.kind === 'anthropic';
  if (isEffort(thinking)) {
    if (hostedAnthropic) return `${thinking}: adaptive thinking at effort ${thinking}`;
    if (endpoint?.locality === 'external') {
      return `${thinking}: reasoning_effort ${thinking} is sent with each request`;
    }
    return `${thinking}: effort ${thinking} (an external endpoint's setting)`;
  }
  if (thinking === 'on') return 'on: the editor reasons before each action';
  if (thinking === 'default') {
    return hostedAnthropic
      ? "default: adaptive thinking at the model's own effort (no effort is sent)"
      : "default: nothing is sent; the server's template decides";
  }
  return 'off: the engine asks the editor not to think';
}

/** The history mode in words (feature 020, FR-020.1). */
export function historyModeText(mode: string): string {
  if (mode === 'append_only') {
    return 'append_only: nothing already sent changes, so a hosted model can serve the history from its cache; a long result is cut once and recalled by offset';
  }
  return 'paging: bulky tool results are replaced by page stubs after a few turns, so older requests change';
}

/**
 * The history mode's hint for what the host and kind look like (FR-020.22): the default for that
 * host, and the choice the backend refuses (`paging` on an external Anthropic endpoint).
 */
export function historyModeHint(locality: 'local' | 'external' | null, kind: string): string {
  const both =
    'paging rewrites older tool results into page stubs; append_only never changes what was sent, so a hosted model can cache it.';
  if (locality === 'external' && kind === 'anthropic') {
    return `${both} Empty: append_only. An external Anthropic endpoint cannot be paging: its thinking binds the exact earlier conversation, so paging is refused.`;
  }
  if (locality === 'external')
    return `${both} Empty: append_only (the default for an external host).`;
  if (locality === 'local') return `${both} Empty: paging (the default for a local host).`;
  return `${both} Empty: append_only for an external host, paging for a local one.`;
}

/**
 * Whether the form marks a price required for what the host and kind look like (FR-020.3): the
 * input and output prices for an external host, and both cache prices for an external Anthropic
 * one. The backend decides, and its refusal names the field.
 */
export function requiredPrice(
  name: PriceField,
  locality: 'local' | 'external' | null,
  kind: string,
): boolean {
  if (locality !== 'external') return false;
  if (name === 'price_in_per_mtok' || name === 'price_out_per_mtok') return true;
  return kind === 'anthropic';
}

/** The two `runs_refused` states (FR-020.22, F20-D12), in words, for the detail's callout. */
export const RUNS_REFUSED_TITLE: Readonly<Record<string, string>> = {
  endpoint_window_required: 'Window required: set it to run',
  wsp_endpoint_cache_prices_required: 'Cache prices required: register this endpoint again',
};

function price(value: string | null): string {
  return value === null ? 'not set' : `$${value} per million tokens`;
}

const tokens = (value: number): string => `${value.toLocaleString('en-US')} tokens`;

/** The hint beside both forms' ceiling field (FR-019.16). */
export const OUTPUT_CEILING_HINT = `Empty: ${DEFAULT_OUTPUT_CEILING_TOKENS.toLocaleString('en-US')}. At least ${MIN_OUTPUT_CEILING_TOKENS}, and at most the context window when it is known.`;
/** The hint beside both forms' window field: what an empty window really does (FR-019.16). */
export const CONTEXT_WINDOW_HINT = `Empty: each run reads the server's window when it starts (miServer, vLLM, SGLang or llama.cpp), else assumes ${DEFAULT_WINDOW_TOKENS.toLocaleString('en-US')}.`;
/**
 * The window's hint for an external host (feature 020, FR-020.2): required, never read from the
 * server, and also where the handoff falls.
 */
export const CONTEXT_WINDOW_REQUIRED_HINT =
  "Required for an external endpoint, and it cannot be cleared: a hosted server is never read for one. It is also where the session hands off, so a window below the model's own bounds the size of every request.";
/** The cache prices' hint (feature 020, FR-020.3, F20-D5). */
export const CACHE_PRICE_HINT =
  'Required for an external Anthropic endpoint (the write price is the 5-minute price); optional otherwise. Sealed with the key: it changes only by registering again.';

/**
 * The window as the endpoint list serves it (feature 019, FR-019.3, F19-D14), in words: the stated
 * value, the registration read with its time, the latest run's read with its run and time, and
 * what a run uses now. Nothing here claims a server reports a window it was never read for.
 */
export function windowFacts(endpoint: WspEndpoint): readonly [string, string][] {
  const stated = endpoint.context_window_tokens;
  const registration = endpoint.registration_window;
  const latest = endpoint.latest_run_window;
  const rows: [string, string][] = [
    ['Context window', stated === null ? 'not stated' : `${tokens(stated)} (stated)`],
    [
      'Read at registration',
      registration.tokens === null
        ? `none reported${registration.read_at === null ? '' : `, ${formatEt(registration.read_at)}`}`
        : `${tokens(registration.tokens)}${registration.read_at === null ? '' : `, ${formatEt(registration.read_at)}`}`,
    ],
    [
      'Latest run',
      latest === null
        ? 'no run has recorded a window yet'
        : `${latest.tokens === null ? 'none reported' : tokens(latest.tokens)}, run ${latest.run_id}${latest.at === null ? '' : `, ${formatEt(latest.at)}`}`,
    ],
    // Operator, 2026-11-25 (F-P4-1): the next run's expected window, then the failed-read case.
    ['Next run expects', nextRunText(endpoint)],
    ...ifReadFailsRows(endpoint),
  ];
  return rows;
}

/**
 * `window_now` in words: stated; the latest run's reported read with its run and time (each run
 * still reads the server when it starts); or, with nothing stated or reported, the assumption.
 * The registration read never predicts a run (F19-D11).
 */
export function nextRunText(endpoint: WspEndpoint): string {
  const now = endpoint.window_now;
  if (now.source === 'waits' || now.tokens === null)
    return `none reported yet: each run waits in the queue until miServer reports a window (a model is loaded); none is assumed`;
  if (now.source === 'stated') return `${tokens(now.tokens)} (stated)`;
  if (now.source === 'reported')
    return `${tokens(now.tokens)}, as the server reported for run ${now.run_id ?? 'unknown'}${now.at === null ? '' : `, ${formatEt(now.at)}`}; each run reads it again when it starts`;
  return `none reported: runs assume ${tokens(now.tokens)} unless the server reports a window when the run starts`;
}

/**
 * The detail's failed-read rows (operator decision D1, 2026-11-27): one row, unless the endpoint is
 * a generic one that states no window, whose two cases differ: a server that answers without a
 * window (8,192 assumed, F19-D10) and one that does not answer at all (the run waits, then fails
 * `editor_not_resident`).
 */
export function ifReadFailsRows(endpoint: WspEndpoint): [string, string][] {
  const fails = endpoint.window_if_read_fails;
  // D1 (operator, 2026-11-27): a generic endpoint that states no window has two cases.
  if (fails.no_answer === 'waits' && fails.source !== 'waits') {
    return [
      ['If the server answers without a window', ifReadFailsText(endpoint)],
      [
        'If the server does not answer',
        'the run waits in the queue until the server answers, and fails editor_not_resident if it has not answered within the wait (15 minutes by default); no window is assumed',
      ],
    ];
  }
  return [['If the run cannot read the server', ifReadFailsText(endpoint)]];
}

/**
 * `window_if_read_fails` in words (2026-11-27): the stated window, the 8,192 assumption, or, on a
 * miServer endpoint that states none, the wait that ends in `editor_not_resident`.
 */
export function ifReadFailsText(endpoint: WspEndpoint): string {
  const fails = endpoint.window_if_read_fails;
  if (fails.source === 'waits' || fails.tokens === null)
    return 'the run waits in the queue for miServer to report a window, then fails editor_not_resident; no window is assumed';
  return `${tokens(fails.tokens)} (${fails.source === 'stated' ? 'stated' : 'assumed'})`;
}

/**
 * The latest protocol test in words (feature 019 P2, FR-019.16): passed, failed, not tested, or
 * cleared because the protocol changed since; on miServer a pass is not required (F19-D8).
 */
export function protocolTestText(endpoint: WspEndpoint): string {
  const state = endpoint.protocol_test;
  const when = state.at === null ? '' : `, ${formatEt(state.at)}`;
  const run = state.run_id === null ? '' : `, run ${state.run_id}`;
  const tested = state.tested;
  const what =
    tested.action_protocol === null
      ? ''
      : ` (tested ${tested.action_protocol} on ${tested.model ?? 'the model'} at ${tested.base_url ?? 'the base URL'})`;
  let text: string;
  if (state.result === null) text = 'not tested';
  else if (state.cleared) text = `cleared: the protocol changed since the test${run}${when}`;
  else text = `${state.result}${run}${when}${what}`;
  if (endpoint.serving === 'miserver') return `${text}; not required on a miServer endpoint`;
  if (state.required && !state.current) {
    return `${text}; required before any run (structured on a generic endpoint): run the protocol test`;
  }
  return text;
}

/** The ceiling in words: the endpoint's, or the default when it sets none. */
export function ceilingText(endpoint: WspEndpoint): string {
  return endpoint.output_ceiling_tokens === null
    ? `${tokens(endpoint.output_ceiling_effective)} (the default)`
    : tokens(endpoint.output_ceiling_tokens);
}

/** The reads below a stated window, for the mismatch warning. */
function mismatchReads(endpoint: WspEndpoint): string {
  const stated = endpoint.context_window_tokens ?? Number.POSITIVE_INFINITY;
  const reads: string[] = [];
  const registration = endpoint.registration_window.tokens;
  if (registration !== null && registration < stated) {
    reads.push(`${tokens(registration)} at registration`);
  }
  const latest = endpoint.latest_run_window?.tokens ?? null;
  if (latest !== null && latest < stated) reads.push(`${tokens(latest)} at the latest run`);
  return reads.join(', ');
}

/** One endpoint's facts, in reading order. The key is shown as present or absent, never as text. */
export function endpointFacts(endpoint: WspEndpoint): readonly [string, string][] {
  return [
    ['Model', endpoint.model],
    ['Base URL', endpoint.base_url],
    ['Kind', endpoint.kind],
    ['Serving', endpoint.serving],
    ['Locality', endpoint.locality],
    ['Action protocol', endpoint.action_protocol],
    ...windowFacts(endpoint),
    ['Output ceiling', ceilingText(endpoint)],
    ['Protocol test', protocolTestText(endpoint)],
    ['History mode', historyModeText(endpoint.history_mode)],
    ['Thinking', thinkingText(endpoint.thinking, endpoint)],
    [
      'Reasoning allowance',
      `${(endpoint.thinking_tokens ?? DEFAULT_THINKING_TOKENS).toLocaleString('en-US')} tokens per turn${
        endpoint.thinking_tokens === null ? " (the engine's default)" : ''
      }${endpoint.thinking === 'on' ? '' : ', used when thinking is on'}`,
    ],
    [
      'Output cap',
      endpoint.output_cap === 'off'
        ? 'off: requests name no per-turn cap; the gateway sizes one to the window'
        : 'on: each request names the per-turn cap',
    ],
    ['Output cap field', endpoint.output_cap_field],
    [
      'Paging',
      endpoint.history_mode === 'append_only'
        ? `not applicable: the history mode is append_only (stored: ${endpoint.paging})`
        : endpoint.paging === 'off'
          ? 'off: every tool result stays in the conversation, and no ledger is kept'
          : 'on: bulky tool results are paged out after a few turns (the editor can recall them), and the engine keeps an experiment ledger',
    ],
    ...PRICE_FIELDS.map((name): [string, string] => [PRICE_LABEL[name], price(endpoint[name])]),
    [
      'API key',
      endpoint.has_key
        ? `sealed, fingerprint ${endpoint.key_fingerprint ?? 'not recorded'}`
        : 'none',
    ],
    ['Registered', `${formatEt(endpoint.created_at)} by ${endpoint.created_by}`],
  ];
}

// --------------------------------------------------------------------------- one endpoint

interface Busy {
  readonly failure: ApiFailure | null;
  readonly what: string;
}

/** The choices, with the stored value kept first if the rule would not offer it. */
function withCurrent(choices: readonly string[], current: string): readonly string[] {
  return choices.includes(current) ? choices : [current, ...choices];
}

function ChangeForm({
  endpoint,
  update,
  onDone,
  onCancel,
}: {
  readonly endpoint: WspEndpoint;
  readonly update: typeof updateWspEndpoint;
  readonly onDone: () => void;
  readonly onCancel: () => void;
}): ReactNode {
  const [protocol, setProtocol] = useState(endpoint.action_protocol);
  const [windowText, setWindowText] = useState(
    endpoint.context_window_tokens === null ? '' : String(endpoint.context_window_tokens),
  );
  const [thinking, setThinking] = useState(endpoint.thinking);
  const [outputCap, setOutputCap] = useState(endpoint.output_cap);
  const [paging, setPaging] = useState(endpoint.paging);
  const [historyMode, setHistoryMode] = useState(endpoint.history_mode);
  const [allowanceText, setAllowanceText] = useState(
    endpoint.thinking_tokens === null ? '' : String(endpoint.thinking_tokens),
  );
  const [ceilingTyped, setCeilingTyped] = useState(
    endpoint.output_ceiling_tokens === null ? '' : String(endpoint.output_ceiling_tokens),
  );
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [nothing, setNothing] = useState(false);
  const id = endpoint.id;

  const save = async (): Promise<void> => {
    const body: Record<string, unknown> = {};
    if (protocol !== endpoint.action_protocol) body['action_protocol'] = protocol;
    const typed = windowText.trim();
    const current = endpoint.context_window_tokens;
    const next = typed === '' ? null : windowValue(typed);
    if (next !== current) body['context_window_tokens'] = next;
    if (thinking !== endpoint.thinking) body['thinking'] = thinking;
    if (outputCap !== endpoint.output_cap) body['output_cap'] = outputCap;
    if (paging !== endpoint.paging) body['paging'] = paging;
    if (historyMode !== endpoint.history_mode) body['history_mode'] = historyMode;
    const allowance = allowanceText.trim() === '' ? null : windowValue(allowanceText.trim());
    if (allowance !== endpoint.thinking_tokens) body['thinking_tokens'] = allowance;
    const ceiling = ceilingTyped.trim() === '' ? null : windowValue(ceilingTyped.trim());
    if (ceiling !== endpoint.output_ceiling_tokens) body['output_ceiling_tokens'] = ceiling;
    if (Object.keys(body).length === 0) {
      setNothing(true);
      return;
    }
    setNothing(false);
    setBusy(true);
    const result = await update(id, body as PatchBody);
    setBusy(false);
    if (result.ok) onDone();
    else setFailure(result.failure);
  };

  return (
    <div
      className="mt-3 rounded-control border border-line p-3"
      data-testid={`endpoint-change-${id}`}
    >
      <p className="mb-3 text-hint text-dim">
        Only the action protocol, the context window, the output ceiling, the history mode,
        thinking, the reasoning allowance, the output cap and paging can change. A queued or
        approved run on this endpoint then fails at its start, because the endpoint it was approved
        for changed. Anything else is bound to the sealed key: remove the endpoint and register it
        again.
      </p>
      <Field
        label="Action protocol"
        hint="structured: any OpenAI-compatible endpoint; a generic one runs only its protocol test until it passes, and a change clears the pass."
      >
        {({ id: controlId, describedBy }) => (
          <Select
            id={controlId}
            aria-describedby={describedBy}
            closedEnum="the backend's ACTION_PROTOCOLS"
            placeholder="Choose a protocol"
            options={ACTION_PROTOCOLS.map((value) => ({ value, label: value }))}
            value={protocol}
            onChange={(event) => setProtocol(event.target.value)}
            data-testid={`endpoint-change-protocol-${id}`}
          />
        )}
      </Field>
      <Field
        label={
          endpoint.locality === 'external'
            ? 'Context window (tokens, required)'
            : 'Context window (tokens)'
        }
        hint={endpoint.locality === 'external' ? CONTEXT_WINDOW_REQUIRED_HINT : CONTEXT_WINDOW_HINT}
      >
        {({ id: controlId, describedBy }) => (
          <Input
            id={controlId}
            aria-describedby={describedBy}
            entry="number"
            inputMode="numeric"
            value={windowText}
            onChange={(event) => setWindowText(event.target.value)}
            data-testid={`endpoint-change-window-${id}`}
          />
        )}
      </Field>
      <Field
        label="History mode"
        hint={historyModeHint(
          endpoint.locality === 'external' ? 'external' : 'local',
          endpoint.kind,
        )}
      >
        {({ id: controlId, describedBy }) => (
          <Select
            id={controlId}
            aria-describedby={describedBy}
            closedEnum="the backend's HISTORY_MODES"
            placeholder="Choose the history mode"
            options={HISTORY_MODES.map((value) => ({ value, label: value }))}
            value={historyMode}
            onChange={(event) => setHistoryMode(event.target.value)}
            data-testid={`endpoint-change-history-mode-${id}`}
          />
        )}
      </Field>
      <Field label="Output ceiling (tokens)" hint={OUTPUT_CEILING_HINT}>
        {({ id: controlId, describedBy }) => (
          <Input
            id={controlId}
            aria-describedby={describedBy}
            entry="number"
            inputMode="numeric"
            value={ceilingTyped}
            onChange={(event) => setCeilingTyped(event.target.value)}
            data-testid={`endpoint-change-ceiling-${id}`}
          />
        )}
      </Field>
      <Field
        label={endpoint.locality === 'external' ? 'Thinking (effort)' : 'Thinking'}
        hint={
          endpoint.locality === 'external'
            ? 'An external endpoint takes an effort level (low, medium or high) or default, the model’s own; an external OpenAI-compatible one may also be off. A run can override it.'
            : 'on: an OpenAI-compatible miServer endpoint only. default: nothing is sent. A run can override it.'
        }
      >
        {({ id: controlId, describedBy }) => (
          <Select
            id={controlId}
            aria-describedby={describedBy}
            closedEnum="the backend's THINKING_VALUES for this endpoint"
            placeholder="Choose thinking"
            options={withCurrent(
              thinkingChoices(endpoint.kind, endpoint.locality, endpoint.serving),
              endpoint.thinking,
            ).map((value) => ({ value, label: value }))}
            value={thinking}
            onChange={(event) => setThinking(event.target.value)}
            data-testid={`endpoint-change-thinking-${id}`}
          />
        )}
      </Field>
      <Field
        label="Reasoning allowance (tokens per turn)"
        hint={`Used with thinking on, beside the action's own cap; below the context window. Empty: ${DEFAULT_THINKING_TOKENS.toLocaleString('en-US')}.`}
      >
        {({ id: controlId, describedBy }) => (
          <Input
            id={controlId}
            aria-describedby={describedBy}
            entry="number"
            inputMode="numeric"
            value={allowanceText}
            onChange={(event) => setAllowanceText(event.target.value)}
            data-testid={`endpoint-change-allowance-${id}`}
          />
        )}
      </Field>
      <Field
        label="Output cap"
        hint="off: requests name no per-turn cap, and the gateway sizes one to the context window. A run can override it."
      >
        {({ id: controlId, describedBy }) => (
          <Select
            id={controlId}
            aria-describedby={describedBy}
            closedEnum="the backend's OUTPUT_CAP_VALUES"
            placeholder="Choose the output cap"
            options={OUTPUT_CAP_VALUES.map((value) => ({ value, label: value }))}
            value={outputCap}
            onChange={(event) => setOutputCap(event.target.value)}
            data-testid={`endpoint-change-output-cap-${id}`}
          />
        )}
      </Field>
      <Field
        label="Paging"
        hint="miwsp_engine only. on: bulky tool results leave the conversation after a few turns (the editor can recall them) and an experiment ledger rides in every request. A run can override it."
      >
        {({ id: controlId, describedBy }) => (
          <Select
            id={controlId}
            aria-describedby={describedBy}
            closedEnum="the backend's PAGING_VALUES"
            placeholder="Choose paging"
            options={PAGING_VALUES.map((value) => ({ value, label: value }))}
            value={paging}
            onChange={(event) => setPaging(event.target.value)}
            data-testid={`endpoint-change-paging-${id}`}
          />
        )}
      </Field>
      {nothing ? (
        <p className="mb-2 text-hint text-dim" data-testid={`endpoint-change-nothing-${id}`}>
          Nothing changed.
        </p>
      ) : null}
      {failure === null ? null : (
        <FailureNotice
          failure={failure}
          surface={`the change to ${id}`}
          className="mb-3"
          data-testid={`endpoint-change-failure-${id}`}
        />
      )}
      <div className="flex flex-wrap gap-2">
        <Button
          variant="primary"
          disabled={busy}
          onClick={() => void save()}
          data-testid={`endpoint-change-save-${id}`}
        >
          Save change
        </Button>
        <Button disabled={busy} onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

export function EndpointItem({
  endpoint,
  update,
  remove,
  protocolTest,
  onChanged,
}: {
  readonly endpoint: WspEndpoint;
  readonly update: typeof updateWspEndpoint;
  readonly remove: typeof removeWspEndpoint;
  readonly protocolTest: typeof testWspEndpointProtocol;
  readonly onChanged: () => void;
}): ReactNode {
  const navigation = useNavigation();
  const [mode, setMode] = useState<'view' | 'change' | 'confirm-remove'>('view');
  const [busy, setBusy] = useState<Busy | null>(null);
  const [queued, setQueued] = useState<ProtocolTestRun | null>(null);
  const id = endpoint.id;

  const doRemove = async (): Promise<void> => {
    setBusy({ failure: null, what: 'remove' });
    const result = await remove(id);
    if (result.ok) {
      setBusy(null);
      setMode('view');
      onChanged();
    } else setBusy({ failure: result.failure, what: 'remove' });
  };

  const doTest = async (): Promise<void> => {
    setBusy({ failure: null, what: 'test' });
    setQueued(null);
    const result = await protocolTest(id);
    if (result.ok) {
      setBusy(null);
      setQueued(result.value);
    } else setBusy({ failure: result.failure, what: 'test' });
  };

  const working = busy !== null && busy.failure === null;
  return (
    <Card as="li" data-testid={`endpoint-row-${id}`}>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <MonoId value={id} />
        <Pill tone={endpoint.locality === 'external' ? 'amber' : 'neutral'}>
          {endpoint.locality}
        </Pill>
        <Pill tone={endpoint.has_key ? 'green' : 'neutral'} data-testid={`endpoint-key-${id}`}>
          {endpoint.has_key ? 'key sealed' : 'no key'}
        </Pill>
      </div>
      <dl
        className="mt-2 grid grid-cols-[10rem_1fr] gap-x-4 gap-y-1 text-body"
        data-testid={`endpoint-facts-${id}`}
      >
        {endpointFacts(endpoint).map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-dim">{label}</dt>
            <dd className="min-w-0 wrap-break-word">{value}</dd>
          </div>
        ))}
      </dl>
      {endpoint.runs_refused === null ? null : (
        <Callout
          tone="warning"
          title={
            RUNS_REFUSED_TITLE[endpoint.runs_refused.code] ?? 'Runs on this endpoint are refused'
          }
          className="mt-2"
          data-testid={`endpoint-runs-refused-${id}`}
        >
          <MonoId value={endpoint.runs_refused.code} /> {endpoint.runs_refused.message}
          {endpoint.runs_refused.code === 'endpoint_window_required'
            ? ' Set the context window under Change.'
            : endpoint.runs_refused.code === 'wsp_endpoint_cache_prices_required'
              ? ' Remove it and register it again with both cache prices and the key.'
              : ''}
        </Callout>
      )}
      {endpoint.window_warnings.includes('window_stated_above_reported') ? (
        <Callout
          tone="warning"
          title="The stated window is larger than the server reported"
          className="mt-2"
          data-testid={`endpoint-window-mismatch-${id}`}
        >
          This endpoint states{' '}
          {endpoint.context_window_tokens === null
            ? 'a window'
            : tokens(endpoint.context_window_tokens)}
          , and the server reported less ({mismatchReads(endpoint)}). Runs use the stated window, so
          a long session can overflow the server. Lower the context window under Change, or check
          the server's configuration.
        </Callout>
      ) : null}
      {endpoint.protocol_test.run_id === null ? null : (
        <button
          type="button"
          className="mt-1 mr-3 text-hint underline"
          onClick={() => navigation.go('wsp-run', { runId: endpoint.protocol_test.run_id ?? '' })}
          data-testid={`endpoint-protocol-test-run-${id}`}
        >
          Open the latest protocol test
        </button>
      )}
      {endpoint.latest_run_window === null ? null : (
        <button
          type="button"
          className="mt-1 text-hint underline"
          onClick={() =>
            navigation.go('wsp-run', { runId: endpoint.latest_run_window?.run_id ?? '' })
          }
          data-testid={`endpoint-latest-run-${id}`}
        >
          Open the latest run that read the window
        </button>
      )}
      <details className="mt-2 text-hint" data-testid={`endpoint-record-${id}`}>
        <summary className="cursor-pointer text-dim">The stored record</summary>
        <pre className="mt-1 font-mono text-ident wrap-break-word whitespace-pre-wrap">
          {JSON.stringify(endpoint, null, 2)}
        </pre>
      </details>
      {mode === 'change' ? (
        <ChangeForm
          endpoint={endpoint}
          update={update}
          onDone={() => {
            setMode('view');
            onChanged();
          }}
          onCancel={() => setMode('view')}
        />
      ) : null}
      {mode === 'confirm-remove' ? (
        <Callout
          tone="warning"
          title={`Remove ${id}?`}
          className="mt-3"
          data-testid={`endpoint-confirm-remove-${id}`}
        >
          A run that names this endpoint can no longer start, and the id is never used again.
          <div className="mt-2 flex flex-wrap gap-2">
            <Button
              variant="primary"
              disabled={working}
              onClick={() => void doRemove()}
              data-testid={`endpoint-remove-confirm-${id}`}
            >
              Remove {id}
            </Button>
            <Button
              disabled={working}
              onClick={() => {
                setMode('view');
                setBusy(null);
              }}
              data-testid={`endpoint-remove-cancel-${id}`}
            >
              Keep it
            </Button>
          </div>
        </Callout>
      ) : null}
      {busy?.failure === null || busy === null ? null : (
        <FailureNotice
          failure={busy.failure}
          surface={busy.what === 'remove' ? `removing ${id}` : `the protocol test of ${id}`}
          className="mt-3"
          data-testid={`endpoint-failure-${id}`}
        />
      )}
      {queued === null ? null : (
        <Callout tone="info" className="mt-3" data-testid={`endpoint-test-queued-${id}`}>
          Protocol test queued as <MonoId value={queued.run_id} /> ({queued.status}). Its row in WSP
          runs shows how many turns the editor broke the one-action rule; 0 is clean.
          {queued.thinking === 'on'
            ? ' Thinking is on for it: the row also says whether the server sent the reasoning back.'
            : ''}{' '}
          <button
            type="button"
            className="underline"
            onClick={() => navigation.go('wsp-run', { runId: queued.run_id })}
            data-testid={`endpoint-test-open-${id}`}
          >
            Open the run
          </button>
        </Callout>
      )}
      {mode !== 'view' ? null : (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button onClick={() => setMode('change')} data-testid={`endpoint-change-open-${id}`}>
            Change protocol, window, history mode, ceiling or thinking
          </Button>
          <Button
            disabled={working}
            onClick={() => void doTest()}
            data-testid={`endpoint-test-${id}`}
          >
            Run protocol test
          </Button>
          <Button
            onClick={() => {
              setBusy(null);
              setMode('confirm-remove');
            }}
            data-testid={`endpoint-remove-${id}`}
          >
            Remove
          </Button>
        </div>
      )}
    </Card>
  );
}

// ------------------------------------------------------------------------------ the form

function AddForm({
  register,
  onRegistered,
}: {
  readonly register: typeof registerWspEndpoint;
  readonly onRegistered: (endpoint: WspEndpoint) => void;
}): ReactNode {
  const persisted = usePersistentState<EndpointDraft>(
    DRAFT_KEYS.wspEndpoint,
    EMPTY_ENDPOINT_DRAFT,
    { version: 1, decode: decodeDraft },
  );
  const draft = persisted.value;
  // The key lives here and nowhere else: not in the draft, not in a store, not in a URL.
  const [apiKey, setApiKey] = useState('');
  const [keyCleared, setKeyCleared] = useState(false);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const set = (name: keyof EndpointDraft) => (value: string) =>
    persisted.set((current) => ({ ...current, [name]: value }));
  const locality = expectedLocality(draft.base_url.trim());

  const submit = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    const sent = apiKey;
    // Sent once: the field is emptied before the answer arrives, whatever it is.
    setApiKey('');
    setKeyCleared(sent !== '');
    setBusy(true);
    setFailure(null);
    const result = await register(registrationBody(draft, sent));
    setBusy(false);
    if (result.ok) {
      persisted.clear();
      setKeyCleared(false);
      onRegistered(result.value);
    } else setFailure(result.failure);
  };

  const text = (
    name: keyof EndpointDraft,
    label: string,
    hint: string,
    extra: Record<string, unknown>,
  ): ReactNode => (
    <Field label={label} hint={hint}>
      {({ id, describedBy }) => (
        <Input
          id={id}
          aria-describedby={describedBy}
          value={draft[name]}
          onChange={(event) => set(name)(event.target.value)}
          data-testid={`endpoint-add-${name.replaceAll('_', '-')}`}
          {...extra}
        />
      )}
    </Field>
  );

  const select = (
    name: keyof EndpointDraft,
    label: string,
    hint: string,
    values: readonly string[],
    placeholder: string,
  ): ReactNode => (
    <Field label={label} hint={hint}>
      {({ id, describedBy }) => (
        <Select
          id={id}
          aria-describedby={describedBy}
          closedEnum={`the backend's fixed ${label.toLowerCase()} values`}
          placeholder={placeholder}
          options={values.map((value) => ({ value, label: value }))}
          value={draft[name]}
          onChange={(event) => set(name)(event.target.value)}
          data-testid={`endpoint-add-${name.replaceAll('_', '-')}`}
        />
      )}
    </Field>
  );

  return (
    <Card as="section" className="mb-6" data-testid="endpoint-add">
      <h2 className="mb-3 font-semibold">Add an editor endpoint</h2>
      <DraftRestoredNote
        restoredAt={persisted.restoredAt}
        onDiscard={persisted.clear}
        className="mb-3"
      />
      <form onSubmit={(event) => void submit(event)} noValidate>
        <div className="grid gap-x-6 md:grid-cols-2">
          {text(
            'id',
            'Id',
            'a-z, 0-9, _ and -, at most 40; empty: miForge names it. Never reused.',
            {
              entry: 'new-name',
            },
          )}
          {text(
            'model',
            'Model',
            'The model name the server serves, exactly as it expects it in a request.',
            {
              entry: 'id-fallback',
              entryReason: 'the model is named by the editor server, which miForge cannot list',
            },
          )}
          {text(
            'base_url',
            'Base URL',
            'http or https, no credentials, the path empty or exactly /v1. An external host needs https and both allowlists.',
            { entry: 'address' },
          )}
          {select(
            'kind',
            'Kind',
            'The wire protocol the server speaks.',
            ENDPOINT_KINDS,
            'Choose a kind',
          )}
          {select(
            'serving',
            'Serving',
            'miserver for a miServer host (required there); empty: generic.',
            SERVINGS,
            'generic (the default)',
          )}
          {select(
            'action_protocol',
            'Action protocol',
            'How the engine asks for one action per turn; structured needs an OpenAI-compatible endpoint, and a generic one runs only its protocol test until it passes.',
            ACTION_PROTOCOLS,
            `${DEFAULT_ACTION_PROTOCOL} (the default)`,
          )}
          {select(
            'output_cap_field',
            'Output cap field',
            'The request field that caps output; anthropic takes max_tokens only.',
            OUTPUT_CAP_FIELDS,
            "The kind's default",
          )}
          {text(
            'context_window',
            locality === 'external'
              ? 'Context window (tokens, required)'
              : 'Context window (tokens)',
            locality === 'external' ? CONTEXT_WINDOW_REQUIRED_HINT : CONTEXT_WINDOW_HINT,
            {
              entry: 'number',
              inputMode: 'numeric',
            },
          )}
          {select(
            'history_mode',
            'History mode',
            historyModeHint(locality, draft.kind),
            HISTORY_MODES,
            locality === 'external'
              ? 'append_only (the default for an external host)'
              : locality === 'local'
                ? 'paging (the default for a local host)'
                : "the host's default",
          )}
          {text('output_ceiling', 'Output ceiling (tokens)', OUTPUT_CEILING_HINT, {
            entry: 'number',
            inputMode: 'numeric',
          })}
          {select(
            'thinking',
            locality === 'external' ? 'Thinking (effort)' : 'Thinking',
            locality === 'external'
              ? draft.kind === 'anthropic'
                ? 'An external Anthropic endpoint always thinks (adaptive): default is the model’s own effort, or name low, medium or high. A run can override it.'
                : 'An external OpenAI-compatible endpoint takes an effort level (sent as reasoning_effort), default (nothing sent) or off. A run can override it.'
              : 'Whether the engine asks the editor to think; on needs an OpenAI-compatible miServer endpoint, default sends nothing. A run can override it.',
            thinkingChoices(draft.kind, locality),
            locality === 'external' && draft.kind === 'anthropic'
              ? 'default (the default)'
              : `${DEFAULT_THINKING} (the default)`,
          )}
          {select(
            'output_cap',
            'Output cap',
            'Whether each request names the per-turn output cap; off lets a reply use the room the context window has left. A run can override it.',
            OUTPUT_CAP_VALUES,
            `${DEFAULT_OUTPUT_CAP} (the default)`,
          )}
          {select(
            'paging',
            'Paging',
            'miwsp_engine only: whether bulky tool results leave the conversation after a few turns (the editor can recall them) and an experiment ledger rides in every request. A run can override it.',
            PAGING_VALUES,
            `${DEFAULT_PAGING} (the default)`,
          )}
          {text(
            'thinking_allowance',
            'Reasoning allowance (tokens per turn)',
            `Used with thinking on, beside the action's own cap; below the context window. Empty: ${DEFAULT_THINKING_TOKENS.toLocaleString('en-US')}.`,
            { entry: 'number', inputMode: 'numeric' },
          )}
          {PRICE_FIELDS.map((name) => (
            <div key={name}>
              {text(
                name,
                `${PRICE_LABEL[name]} (USD per million tokens${requiredPrice(name, locality, draft.kind) ? ', required' : ''})`,
                name === 'price_in_per_mtok' || name === 'price_out_per_mtok'
                  ? 'Required for an external endpoint; a local one needs none.'
                  : CACHE_PRICE_HINT,
                { entry: 'number', inputMode: 'decimal' },
              )}
            </div>
          ))}
          <Field
            label="API key"
            hint="Sent once with this registration and sealed for the gateway; it is never shown again or kept by this page. Required for an external endpoint."
          >
            {({ id, describedBy }) => (
              <Input
                id={id}
                aria-describedby={describedBy}
                type="password"
                entry="secret"
                autoComplete="new-password"
                spellCheck={false}
                value={apiKey}
                onChange={(event) => {
                  setApiKey(event.target.value);
                  setKeyCleared(false);
                }}
                data-testid="endpoint-add-api-key"
              />
            )}
          </Field>
        </div>
        <p className="mb-3 text-hint text-dim" data-testid="endpoint-add-locality">
          {locality === null
            ? 'The locality is derived from the host when you register: local for a private, loopback or cluster host, external otherwise.'
            : locality === 'local'
              ? 'This host looks local: no key and no prices are needed. miForge decides when you register.'
              : draft.kind === 'anthropic'
                ? 'This host looks external: data leaves this network, so it needs https, an API key, its context window, input and output prices and both cache prices. miForge decides when you register.'
                : 'This host looks external: data leaves this network, so it needs https, an API key, its context window and input and output prices. miForge decides when you register.'}
        </p>
        {keyCleared ? (
          <p className="mb-3 text-hint text-dim" data-testid="endpoint-add-key-cleared">
            The API key was sent and is not kept here: type it again to register again.
          </p>
        ) : null}
        {failure === null ? null : (
          <FailureNotice
            failure={failure}
            surface="the registration"
            className="mb-3"
            data-testid="endpoint-add-failure"
          />
        )}
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" type="submit" disabled={busy} data-testid="endpoint-add-submit">
            Register endpoint
          </Button>
          <Button type="button" disabled={busy} onClick={persisted.clear}>
            Clear the form
          </Button>
        </div>
      </form>
    </Card>
  );
}

// ------------------------------------------------------------------------------ the page

export interface WspEndpointsPageProps {
  /** Injected in tests. The real ones are the routes in `api/wspEndpoints.ts`. */
  readonly list?: typeof listWspEndpoints;
  readonly register?: typeof registerWspEndpoint;
  readonly update?: typeof updateWspEndpoint;
  readonly remove?: typeof removeWspEndpoint;
  readonly protocolTest?: typeof testWspEndpointProtocol;
}

export function WspEndpointsPage({
  list = listWspEndpoints,
  register = registerWspEndpoint,
  update = updateWspEndpoint,
  remove = removeWspEndpoint,
  protocolTest = testWspEndpointProtocol,
}: WspEndpointsPageProps = {}): ReactNode {
  const [asked, setAsked] = useState(0);
  const [registered, setRegistered] = useState<WspEndpoint | null>(null);
  const load = useCallback(async (): Promise<ApiResult<WspEndpointListing>> => {
    void asked;
    return list();
  }, [list, asked]);
  const { value, failure } = usePolledRead(load, WSP_ENDPOINTS_POLL_MS);
  const reread = (): void => setAsked((n) => n + 1);
  const rows = value?.endpoints ?? [];

  return (
    <>
      <PageHead
        title={WSP_ENDPOINTS_TITLE}
        subtitle={WSP_ENDPOINTS_SUBTITLE}
        actions={
          <Button onClick={reread} data-testid="endpoints-reread">
            Re-read the list
          </Button>
        }
      />
      <p className="mb-4 text-hint text-dim">
        A WSP session names one of these as its editor. You register, change and remove them here.
        An agent can propose a local endpoint with no key and no prices; it waits in Approvals, and
        appears here once you approve it.
      </p>
      {registered === null ? null : (
        <Callout tone="info" className="mb-4" data-testid="endpoint-registered">
          Registered <MonoId value={registered.id} /> ({registered.locality},{' '}
          {registered.has_key ? 'key sealed' : 'no key'}).
        </Callout>
      )}
      <AddForm
        register={register}
        onRegistered={(endpoint) => {
          setRegistered(endpoint);
          reread();
        }}
      />
      {failure === null ? null : (
        <FailureNotice
          failure={failure}
          surface="the editor endpoint list"
          className="mb-4"
          data-testid="endpoints-failure"
        />
      )}
      {value === null ? (
        failure === null ? (
          <Card data-testid="endpoints-loading">Reading the editor endpoints…</Card>
        ) : null
      ) : rows.length === 0 ? (
        <Card data-testid="endpoints-empty">
          No editor endpoint is registered. Add one above, or approve one an agent proposed.
        </Card>
      ) : (
        <>
          <div className="mb-2 text-hint text-dim" data-testid="endpoints-count">
            {rows.length} {rows.length === 1 ? 'endpoint' : 'endpoints'}
          </div>
          <ul className="stack-y-2" data-testid="endpoints-rows">
            {rows.map((endpoint) => (
              <EndpointItem
                key={endpoint.id}
                endpoint={endpoint}
                update={update}
                remove={remove}
                protocolTest={protocolTest}
                onChanged={reread}
              />
            ))}
          </ul>
        </>
      )}
    </>
  );
}
