/**
 * A backend at the HTTP boundary, for the tests that render **every `ROUTES` screen**: the
 * persistence wiring test and the field-declaration test (tasks 13.6, 14.7).
 *
 * The answers are identical on every call, so a value that comes back after a remount can
 * only have come from what the browser kept. Unknown routes get a real 404 envelope.
 */
import healthFixture from './fixtures/health.json' with { type: 'json' };

export const DRAFT_ID = 'draft-1';

export interface Call {
  readonly method: string;
  readonly path: string;
}

export function draftRecord(): Readonly<Record<string, unknown>> {
  return {
    draft_id: DRAFT_ID,
    name: null,
    version: '1',
    domain: null,
    current_step: 1,
    step_count: 7,
    step_states: { '1': 'current' },
    slots: { retention_set: null },
    created_at: '2026-10-17T00:00:00Z',
    updated_at: '2026-10-17T00:00:00Z',
    last_dry_run: null,
  };
}

const CHECKPOINT = {
  name: 'run-a/step-40',
  run_id: 'run-a',
  step: 40,
  domain: 'notary',
  domain_state: 'given',
  domain_display: 'notary',
  signal_source: 'corpus',
  parent: null,
  base_model: 'base',
  merged_adapter: null,
  weights_sha256: 'a'.repeat(64),
  manifest_sha256: 'b'.repeat(64),
  selector_hash: null,
  grader_hash: null,
  test: null,
  retention: null,
  path: '/data/run-a/step-40',
  created_at: '2026-10-17T00:00:00Z',
};

const RUN_A = {
  run_id: 'run-a',
  status: 'stopped',
  run_type: 'change',
  last_step: 40,
  cancel_requested_at: null,
  failure_code: null,
  heartbeat: {
    heartbeat_at: '2026-10-17T00:00:00+00:00',
    age_seconds: 12,
    longest_gap_seconds: 180,
    lapse_threshold_seconds: 900,
    lapsed: false,
  },
};

export const WSP_RUN_ID = 'wsp_0123456789ab';

/** One row of `GET /api/v1/wsp/endpoints` (`wsp.endpoints.public_document`): local, no key. */
export const WSP_ENDPOINT = {
  id: 'miserver-gemma4-26b',
  kind: 'openai_compatible',
  base_url: 'http://miserver-backend.miserver.svc:8000/v1',
  model: 'gemma-4-26b',
  locality: 'local',
  serving: 'miserver',
  output_cap_field: 'max_completion_tokens',
  price_in_per_mtok: null,
  price_out_per_mtok: null,
  price_cache_write_per_mtok: null,
  price_cache_read_per_mtok: null,
  has_key: false,
  key_fingerprint: null,
  created_at: '2026-11-20T15:00:00+00:00',
  created_by: 'operator',
  action_protocol: 'structured',
  context_window_tokens: 32768,
  thinking: 'off',
  thinking_tokens: null,
  output_cap: 'on',
  paging: 'on',
  // Feature 019 (FR-019.3, FR-019.14): the ceiling and the window reads, as the backend serves them.
  output_ceiling_tokens: 4096,
  output_ceiling_effective: 4096,
  registration_window: { tokens: 32768, read_at: '2026-11-20T15:00:01+00:00' },
  latest_run_window: {
    tokens: 16384,
    source: 'reported',
    run_id: WSP_RUN_ID,
    at: '2026-11-20T15:00:05+00:00',
  },
  window_now: { tokens: 32768, source: 'stated', run_id: null, at: null },
  window_if_read_fails: { tokens: 32768, source: 'stated', no_answer: 'stated' },
  window_warnings: ['window_stated_above_reported'],
  // Feature 019 P2 (FR-019.16): the latest protocol test. miServer needs no pass.
  protocol_test: {
    result: 'passed',
    run_id: 'wsp_0000000000aa',
    at: '2026-11-20T14:00:00+00:00',
    tested: {
      base_url: 'http://miserver-backend.miserver.svc:8000/v1',
      model: 'gemma-4-26b',
      action_protocol: 'structured',
    },
    current: true,
    cleared: false,
    required: false,
  },
  // Feature 020 (FR-020.1, FR-020.5): a local endpoint keeps paging, and runs start on it.
  history_mode: 'paging',
  runs_refused: null,
} as const;

/** `GET /api/v1/wsp/pool` (feature 019 P3, `wsp/pool.py`): one idle worker, nothing queued. */
export const WSP_POOL = {
  pool_size: 1,
  maximum: 1,
  effective: 1,
  busy: 0,
  idle: 1,
  not_answering: 0,
  workers: [
    {
      index: 0,
      state: 'idle',
      worker_state: 'idle',
      run_id: null,
      boot_id: 'b'.repeat(32),
      read_at: '2026-11-25T06:00:00+00:00',
    },
  ],
  queued_by_endpoint: [],
  queued_without_endpoint: 0,
  shared_base_urls: [],
};

/** One row of `GET /api/v1/wsp/runs`, in the backend's shape (`list_runs`). */
export const WSP_RUN_ROW = {
  run_id: WSP_RUN_ID,
  status: 'completed',
  kind: 'repair',
  mode: 'session',
  caller: 'agent:forge-mcp',
  caller_ref: 'probe-17',
  endpoint_id: 'ep-local',
  stop_reason: 'gateway_refused',
  outcome: 'no_change',
  created_at: '2026-11-20T15:00:00+00:00',
  provider: 'miwsp_engine',
  provider_version: '0.1.0',
  failure: {
    code: 'editor_context_exceeded',
    message: 'The editor session ended; it is not retried. (code: editor_context_exceeded)',
  },
  usage: { requests: 1, input_tokens: 7000, output_tokens: 32000, cost_usd: 0 },
  started_at: '2026-11-20T15:00:05+00:00',
  finished_at: '2026-11-20T15:00:09+00:00',
  action_protocol: null,
  handoffs: null,
  refused_turns: null,
  kernel_restarts: null,
  thinking: null,
  thinking_tokens: null,
  output_cap: 'on',
  reasoning_turns: null,
  reasoning_returned: null,
  repeated_turns: null,
  paging: null,
  pages_out: null,
  recalls: null,
  ledger_rows: null,
  peak_prompt_tokens: null,
  forward_passes_used: 0,
  window_tokens: 32768,
  window_source: 'reported',
  window_reported_tokens: 32768,
  output_ceiling_tokens: 32000,
  output_ceiling_lowered: false,
  window_warnings: [],
  editor_wait_since: null,
  // Feature 020 (FR-020.21): cost and cache totals from the calls; a local call reports no cache.
  metered: { cost_usd: 0, cache_read_tokens: null, cache_write_tokens: null },
} as const;

/** The run's final progress (`GET /api/v1/wsp/runs/{id}/progress`, live progress 2026-11-20). */
export const WSP_RUN_PROGRESS = {
  run_id: WSP_RUN_ID,
  status: 'completed',
  live: false,
  now: '2026-11-20T15:05:00Z',
  created_at: '2026-11-20T15:00:00Z',
  started_at: '2026-11-20T15:00:05Z',
  finished_at: '2026-11-20T15:00:09Z',
  clock_from: 'started_at',
  elapsed_s: 4,
  // Feature 019 (FR-019.17): the window and ceiling the push used.
  window_tokens: 32768,
  window_source: 'reported',
  window_reported_tokens: 32768,
  output_ceiling_tokens: 16384,
  output_ceiling_lowered: true,
  window_warnings: [],
  // Debt D3: the run never waited for its editor model.
  editor_wait_since: null,
  worker: {
    state: 'finished',
    elapsed_s: 3,
    last_heartbeat_at: '2026-11-20T15:00:08Z',
    heartbeat_age_s: null,
  },
  metered: {
    calls: 1,
    refused: 0,
    input_tokens: 7000,
    output_tokens: 32000,
    tokens: 39000,
    cost_usd: 0,
    last_call_at: '2026-11-20T15:00:08Z',
    // Feature 020 (FR-020.21, DEP-4): the run's cache totals; null: no call recorded one.
    cache_read_tokens: null,
    cache_write_tokens: null,
  },
  in_flight: null,
  budgets: [
    { name: 'tokens', unit: 'tokens', spent: 39000, cap: 100000, fraction: 0.39, estimate: false },
    { name: 'cost_usd', unit: 'USD', spent: 0, cap: null, fraction: null, estimate: false },
    { name: 'time_s', unit: 's', spent: 4, cap: 3600, fraction: 0.0011, estimate: false },
    { name: 'steps', unit: 'editor calls', spent: 1, cap: 50, fraction: 0.02, estimate: true },
  ],
  queue: null,
} as const;

/** Its one editor call (`GET /api/v1/wsp/runs/{id}/calls`): counts and codes only. */
export const WSP_RUN_CALLS = {
  run_id: WSP_RUN_ID,
  status: 'completed',
  live: false,
  calls: [
    {
      seq: 0,
      at: '2026-11-20T15:00:08Z',
      outcome: 'session_ended',
      http_status: 400,
      latency_ms: 2300,
      finish_reason: null,
      refusal_code: 'editor_context_exceeded',
      input_tokens: 7000,
      output_tokens: 32000,
      cost_usd: 0,
      // Feature 020 (FR-020.20): null, "not recorded".
      cache_read_tokens: null,
      cache_write_tokens: null,
    },
  ],
  more: false,
} as const;

/** The same run as `GET /api/v1/wsp/runs/{id}` serves it (`miforge-wsp-run-v1`). */
export const WSP_RUN_DOCUMENT = {
  kind: 'miforge.wsp-run/v1',
  run_id: WSP_RUN_ID,
  status: 'completed',
  created_at: '2026-11-20T15:00:00Z',
  request: { kind: 'repair', mode: 'session', task: 'Repair the probe.' },
  request_sha256: 'a'.repeat(64),
  caller_ref: 'probe-17',
  approval: { approval_id: 'appr-1', action: 'wsp_session' },
  boot_id: null,
  sensitivity: { value: 'not_sensitive', effective: 'not_sensitive', source: 'none' },
  usage: {
    metered_by: 'wsp_gateway',
    requests: 1,
    input_tokens: 7000,
    output_tokens: 32000,
    cost_usd: 0,
    self_reported: null,
  },
  forward_passes_used: 0,
  stop_reason: 'gateway_refused',
  outcome: 'no_change',
  failure: WSP_RUN_ROW.failure,
  content_sha256: null,
  transcript_sha256: 'c'.repeat(64),
  result_sha256: 'd'.repeat(64),
  patch_sha256: null,
  program_sha256: null,
  program_reruns: [],
  replay: null,
  candidate_id: null,
  verdict: null,
  run_sha256: null,
  lineage: {
    bundle_sha256: 'b'.repeat(64),
    chain: [],
    parent: { kind: 'bundle', ref: 'b'.repeat(64) },
    provider: 'miwsp_engine',
    provider_version: '0.1.0',
  },
} as const;

/** `GET /api/v1/settings`, in the backend's shape. */
const NOW = {
  when: 'now',
  text: 'Immediately in the API; each worker applies it before its next task.',
} as const;
const SWEEP = { when: 'next_sweep', text: 'At the next checkpoint pruning sweep.' } as const;

/** One recorded change, as `GET /settings/history` serves it (ENH-003 D12). */
export const SETTING_CHANGE = {
  id: 7,
  key: 'checkpoint_lifecycle_keep_last_n',
  action: 'set',
  previous: 12,
  previous_source: 'env',
  value: 10,
  source: 'override',
  actor: 'agent:forge-mcp',
  approval_id: 'apr_0123',
  reason: 'the volume is filling',
  changed_at: '2026-10-20T14:05:00+00:00',
} as const;

export const SETTINGS_HISTORY = { changes: [SETTING_CHANGE], count: 1 } as const;

export const SETTINGS = {
  settings: {
    checkpoint_lifecycle_enabled: {
      value: false,
      source: 'env',
      runtime_writable: true,
      label: 'Prune old checkpoints',
      description: 'Whether the pruning sweep removes old checkpoints.',
      group: 'checkpoint_pruning',
      effect: SWEEP,
      env_value: false,
      updated_at: null,
      updated_by: null,
      last_change: null,
    },
    checkpoint_lifecycle_keep_last_n: {
      value: 10,
      source: 'override',
      runtime_writable: true,
      label: 'Keep the last N saves',
      description: "How many of each run's most recent saves pruning always keeps.",
      group: 'checkpoint_pruning',
      effect: SWEEP,
      env_value: 12,
      updated_at: '2026-10-20T14:05:00+00:00',
      updated_by: 'agent:forge-mcp',
      last_change: SETTING_CHANGE,
    },
    millm_base_url: {
      value: null,
      source: 'env',
      runtime_writable: true,
      label: 'miServer address',
      description: "Where miServer, the suite's inference server, answers.",
      group: 'connections',
      effect: NOW,
      env_value: null,
      updated_at: null,
      updated_by: null,
      last_change: null,
    },
    log_level: {
      value: 'INFO',
      source: 'env',
      runtime_writable: true,
      label: 'Log level',
      description: 'How much miForge writes to its logs: DEBUG, INFO, WARNING or ERROR.',
      group: 'general',
      effect: NOW,
      env_value: 'INFO',
      updated_at: null,
      updated_by: null,
      last_change: null,
    },
    database_url: {
      value: 'postgresql://miforge:<redacted>@db/miforge',
      source: 'env',
      runtime_writable: false,
      label: 'Database address',
      description: "Where miForge's PostgreSQL database is. The password is never shown.",
      group: 'storage',
      effect: {
        when: 'restart',
        text: 'Read at startup. Change it in the environment and restart the miforge-backend pod; a restart stops running jobs.',
        pod: 'miforge-backend',
      },
      env_value: 'postgresql://miforge:<redacted>@db/miforge',
      updated_at: null,
      updated_by: null,
      last_change: null,
    },
  },
  count: 5,
  runtime_keys: [
    'checkpoint_lifecycle_enabled',
    'checkpoint_lifecycle_keep_last_n',
    'log_level',
    'millm_base_url',
  ],
  overrides: ['checkpoint_lifecycle_keep_last_n'],
  // `core/settings_catalog.py::GROUPS`, in the order the backend serves them.
  groups: [
    { id: 'general', label: 'General' },
    { id: 'storage', label: 'Storage' },
    { id: 'jobs', label: 'Jobs, GPUs and approvals' },
    { id: 'checkpoint_pruning', label: 'Checkpoint pruning' },
    { id: 'connections', label: 'Suite connections' },
    { id: 'hot_update', label: 'Hot-update runs' },
  ],
} as const;

/** The same answers on every mount. Unknown routes get a real 404 envelope. */
export function answer(method: string, path: string): { status: number; body: unknown } {
  // The backend's own answer, generated by backend/scripts/write_health_fixture.py and diffed
  // by backend/tests/contract/test_health_fixture.py (it was a hand-written array shape).
  if (path === '/api/health') return { status: 200, body: healthFixture };
  if (path === '/api/v1/run-types')
    return {
      status: 200,
      body: {
        run_types: [
          { name: 'change', stored_as: 'change', changes: 'everything' },
          { name: 'adapter', stored_as: 'adapter', changes: 'an adapter' },
        ],
        count: 2,
      },
    };
  if (path === '/api/v1/packs') return { status: 200, body: { packs: [] } };
  // The Editor endpoints screen (2026-11-20): one local, keyless endpoint.
  if (path === '/api/v1/wsp/endpoints' && method === 'GET')
    return { status: 200, body: { endpoints: [WSP_ENDPOINT] } };
  // The WSP runs screens (2026-11-20): one finished run, its document and its transcript.
  if (path === '/api/v1/wsp/pool' && method === 'GET') return { status: 200, body: WSP_POOL };
  if (path === '/api/v1/wsp/runs' && method === 'GET')
    return { status: 200, body: { runs: [WSP_RUN_ROW] } };
  if (path === `/api/v1/wsp/runs/${WSP_RUN_ID}` && method === 'GET')
    return { status: 200, body: WSP_RUN_DOCUMENT };
  if (path === `/api/v1/wsp/runs/${WSP_RUN_ID}/progress` && method === 'GET')
    return { status: 200, body: WSP_RUN_PROGRESS };
  if (path === `/api/v1/wsp/runs/${WSP_RUN_ID}/calls` && method === 'GET')
    return { status: 200, body: WSP_RUN_CALLS };
  if (path === `/api/v1/wsp/runs/${WSP_RUN_ID}/transcript` && method === 'GET')
    return {
      status: 200,
      body: {
        run_id: WSP_RUN_ID,
        transcript_sha256: 'c'.repeat(64),
        entries: [{ role: 'user', content: 'Repair the probe.' }],
        partial: false,
      },
    };
  // F12: the approval queue, read by the shell's badge on every screen.
  if (path === '/api/v1/approvals' && method === 'GET')
    return { status: 200, body: { approvals: [], count: 0, pending: 0, limit: 200 } };
  // F12: the Settings screen, one row per editor kind (boolean, number, address, log level)
  // plus a fixed one, so the field-declaration walk sees every control the screen can draw.
  if (path === '/api/v1/settings' && method === 'GET') return { status: 200, body: SETTINGS };
  if (path === '/api/v1/settings/history' && method === 'GET')
    return { status: 200, body: SETTINGS_HISTORY };
  // Task 14.0's listings, so the pickers they feed have something to choose - and the derived
  // test below then requires each CHOICE back after the reload, not only each typed value.
  if (path === '/api/v1/runs' && method === 'GET')
    return {
      status: 200,
      body: {
        runs: [],
        total: 0,
        limit: 25,
        offset: 0,
        statuses: ['completed', 'failed', 'running'],
        run_types: ['adapter', 'change'],
      },
    };
  if (path === '/api/v1/models/local')
    return {
      status: 200,
      body: {
        models: [
          {
            path: '/data/checkpoints/bases/tiny',
            root: '/data/checkpoints',
            relative_path: 'bases/tiny',
            kind: 'directory',
            architecture: 'llama',
            has_weights: true,
            has_tokenizer: true,
            config_path: '/data/checkpoints/bases/tiny/config.json',
            modified_at: '2026-10-17T00:00:00+00:00',
          },
        ],
        total: 1,
        limit: 500,
        offset: 0,
        roots: ['/data/checkpoints'],
        unreadable_roots: [],
        max_depth: 4,
      },
    };
  // The create run's tokenizer picker (2026-10-20): one tokenizer-only folder.
  if (path === '/api/v1/tokenizers/local')
    return {
      status: 200,
      body: {
        tokenizers: [
          {
            path: '/data/checkpoints/tokenizers/tiny-tok',
            root: '/data/checkpoints',
            relative_path: 'tokenizers/tiny-tok',
            kind: 'tokenizer',
            tokenizer_files: ['tokenizer.json'],
            modified_at: '2026-10-17T00:00:00+00:00',
          },
        ],
        total: 1,
        limit: 500,
        offset: 0,
        roots: ['/data/checkpoints'],
        unreadable_roots: [],
        max_depth: 4,
      },
    };
  if (path === '/api/v1/hub/revision')
    return {
      status: 200,
      body: {
        repo: 'org/tok',
        revision: 'main',
        commit: '0123456789abcdef0123456789abcdef01234567',
        is_commit: false,
        tokenizer_files: ['tokenizer.json'],
        has_tokenizer: true,
        has_config: false,
        has_weights: false,
      },
    };
  if (path === '/api/v1/parts')
    return {
      status: 200,
      body: {
        items: [
          {
            id: 'part-1',
            part_kind: 'prompt_set',
            source: 'midataworks_contract',
            contract_reference: { type: 'prompt-set', name: 'support', version: '3' },
            rendered: 'prompt-set/support@3',
            dataset: 'support',
            version_number: 3,
            row_count: 10,
            caveats: [],
            licence_classes: [],
            imported_at: '2026-10-17T00:00:00Z',
          },
        ],
        total: 1,
      },
    };
  if (path === '/api/v1/gpus')
    return {
      status: 200,
      body: {
        devices: [
          {
            uuid: 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57',
            name: 'RTX 3090',
            free_memory_bytes: 1,
          },
        ],
        available: true,
        unavailable: null,
        auto: true,
      },
    };
  if (path === '/api/v1/stop-conditions')
    return {
      status: 200,
      body: {
        conditions: [
          {
            condition: 'step_count',
            configured: true,
            required_parameters: ['total'],
            precedence_rank: 4,
            summary: 'steps',
          },
        ],
        count: 1,
        closed: true,
        precedence: ['step_count'],
        minimum_enabled: 1,
      },
    };
  if (path === '/api/v1/selectors') return { status: 200, body: [] };
  if (path === '/api/v1/host/resources')
    return {
      status: 200,
      body: {
        observed_at: '2026-10-17T00:00:00+00:00',
        cpu: { percent: 12, logical_cores: 32 },
        memory: { used_bytes: 8 * 1024 ** 3, total_bytes: 64 * 1024 ** 3 },
        gpus: [
          {
            uuid: 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57',
            name: 'NVIDIA GeForce RTX 3090',
            utilization_percent: 0,
            memory_used_bytes: 0,
            memory_total_bytes: 24 * 1024 ** 3,
            temperature_c: 40,
          },
        ],
        unavailable: [],
      },
    };
  if (path === '/api/v1/attachment') return { status: 200, body: { configured: false } };
  if (path === '/api/v1/models/sources')
    return {
      status: 200,
      body: {
        modes: [
          {
            mode: 'load',
            available: true,
            default: true,
            reason: null,
            message: null,
            capabilities_unavailable: [],
            sources: ['local_dir'],
          },
          {
            mode: 'create',
            available: true,
            default: false,
            reason: null,
            message: null,
            capabilities_unavailable: [],
            sources: ['config'],
          },
        ],
        default: 'load',
      },
    };
  if (path === '/api/v1/signal-sources')
    return {
      status: 200,
      body: {
        sources: [{ name: 'corpus', available: true, reason: null, requires: [] }],
        count: 1,
        available: 1,
      },
    };
  if (path === '/api/v1/pack-drafts' && method === 'POST')
    return { status: 201, body: draftRecord() };
  if (path === `/api/v1/pack-drafts/${DRAFT_ID}` && method === 'GET')
    return { status: 200, body: draftRecord() };
  if (path === '/api/v1/tree')
    return {
      status: 200,
      body: {
        nodes: [
          {
            name: CHECKPOINT.name,
            kind: 'checkpoint',
            run_id: CHECKPOINT.run_id,
            step: CHECKPOINT.step,
            domain: CHECKPOINT.domain,
            domain_state: 'recorded',
            domain_display: CHECKPOINT.domain_display,
            parent: { target: null, state: 'root' },
            base: null,
            base_manifest_sha256: null,
            merged: null,
            merged_step: null,
            merged_verification: null,
            test: null,
            manifest_sha256: CHECKPOINT.manifest_sha256,
            path: CHECKPOINT.path,
            run_indexed: true,
            index_stale: false,
          },
        ],
        edges: [],
        page: { limit: 2000, offset: 0, total: 1 },
        index_stale: [],
      },
    };
  if (path === '/api/v1/runs/run-a/checkpoints/40')
    return {
      status: 200,
      body: { run_id: 'run-a', step: 40, manifest: { kind: 'checkpoint', step: 40 } },
    };
  if (path === '/api/v1/runs/run-a' && method === 'GET') return { status: 200, body: RUN_A };
  if (path === '/api/v1/runs/run-a/scores') return { status: 200, body: { scores: [] } };
  if (path === '/api/v1/runs/run-gone')
    return {
      status: 404,
      body: { error: { code: 'run_not_found', message: 'no such run', details: {} } },
    };
  if (path === '/api/v1/runs/run-down')
    return {
      status: 503,
      body: { error: { code: 'database_unavailable', message: 'x', details: {} } },
    };
  return {
    status: 404,
    body: { error: { code: 'not_stubbed', message: `${method} ${path}`, details: {} } },
  };
}
