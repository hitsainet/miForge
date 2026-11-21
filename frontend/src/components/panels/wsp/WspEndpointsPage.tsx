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
 *   allowance (2026-11-21), the only fields a change may name; anything else is bound to the
 *   sealed key, so it is a new registration;
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
  DEFAULT_PAGING,
  DEFAULT_THINKING,
  DEFAULT_THINKING_TOKENS,
  ENDPOINT_KINDS,
  OUTPUT_CAP_FIELDS,
  OUTPUT_CAP_VALUES,
  PAGING_VALUES,
  PRICE_FIELDS,
  SERVINGS,
  THINKING_VALUES,
  expectedLocality,
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
  if (draft.thinking_allowance.trim() !== '') {
    body['thinking_tokens'] = windowValue(draft.thinking_allowance.trim());
  }
  for (const name of PRICE_FIELDS) {
    if (draft[name].trim() !== '') body[name] = draft[name].trim();
  }
  if (apiKey !== '') body['api_key'] = apiKey;
  return body as unknown as RegisterBody;
}

/** What an endpoint's thinking setting does, in words (2026-11-21). */
export function thinkingText(thinking: string): string {
  if (thinking === 'on') return 'on: the editor reasons before each action';
  if (thinking === 'default') return "default: nothing is sent; the server's template decides";
  return 'off: the engine asks the editor not to think';
}

function price(value: string | null): string {
  return value === null ? 'not set' : `$${value} per million tokens`;
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
    [
      'Context window',
      endpoint.context_window_tokens === null
        ? 'the window the server reports'
        : `${endpoint.context_window_tokens.toLocaleString('en-US')} tokens`,
    ],
    ['Thinking', thinkingText(endpoint.thinking)],
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
      endpoint.paging === 'off'
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
  const [allowanceText, setAllowanceText] = useState(
    endpoint.thinking_tokens === null ? '' : String(endpoint.thinking_tokens),
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
    const allowance = allowanceText.trim() === '' ? null : windowValue(allowanceText.trim());
    if (allowance !== endpoint.thinking_tokens) body['thinking_tokens'] = allowance;
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
        Only the action protocol, the context window, thinking and the reasoning allowance can
        change. A queued or approved run on this endpoint then fails at its start, because the
        endpoint it was approved for changed. Anything else is bound to the sealed key: remove the
        endpoint and register it again.
      </p>
      <Field
        label="Action protocol"
        hint="structured: an OpenAI-compatible miServer endpoint only."
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
      <Field label="Context window (tokens)" hint="Empty: the window the server reports.">
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
        label="Thinking"
        hint="on: an OpenAI-compatible miServer endpoint only. default: nothing is sent. A run can override it."
      >
        {({ id: controlId, describedBy }) => (
          <Select
            id={controlId}
            aria-describedby={describedBy}
            closedEnum="the backend's THINKING_VALUES"
            placeholder="Choose thinking"
            options={THINKING_VALUES.map((value) => ({ value, label: value }))}
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
            Change protocol, window or thinking
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
            'How the engine asks for one action per turn; structured needs an OpenAI-compatible miServer endpoint.',
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
            'Context window (tokens)',
            'Empty: the window the server reports, else the engine’s default.',
            { entry: 'number', inputMode: 'numeric' },
          )}
          {select(
            'thinking',
            'Thinking',
            'Whether the engine asks the editor to think; on needs an OpenAI-compatible miServer endpoint, default sends nothing. A run can override it.',
            THINKING_VALUES,
            `${DEFAULT_THINKING} (the default)`,
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
                `${PRICE_LABEL[name]} (USD per million tokens)`,
                name === 'price_in_per_mtok' || name === 'price_out_per_mtok'
                  ? 'Required for an external endpoint; a local one needs none.'
                  : 'Optional.',
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
              : 'This host looks external: data leaves this network, so it needs https, an API key and input and output prices. miForge decides when you register.'}
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
