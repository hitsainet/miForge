/**
 * A backend at the HTTP boundary, for the tests that render **every `ROUTES` screen**: the
 * persistence wiring test and the field-declaration test (tasks 13.6, 14.7).
 *
 * The answers are identical on every call, so a value that comes back after a remount can
 * only have come from what the browser kept. Unknown routes get a real 404 envelope.
 */
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
      label: 'miLLM address',
      description: "Where miLLM, the suite's inference server, answers.",
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
} as const;

/** The same answers on every mount. Unknown routes get a real 404 envelope. */
export function answer(method: string, path: string): { status: number; body: unknown } {
  if (path === '/api/health') return { status: 200, body: { status: 'ok', dependencies: [] } };
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
