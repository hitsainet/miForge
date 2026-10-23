/**
 * Each New run field that names something which exists is a picker over its route (task 14.0;
 * AC-009.45, .47). Per field: what it lists, what it filters out, that a choice submits the
 * **identifier** (a path, a UUID, a condition name - never the label), and its empty and
 * failed states. Loaders are injected, at the same seam the form uses.
 */
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import {
  GpuRequestField,
  ModelSourceField,
  SignalSourceField,
  StopConditionsField,
} from './DefinitionFields';
import { PackField } from './PackField';
import type { StopConditionEntry } from './draft';
import type { ApiResult } from '@/api/client';
import type { LocalModelListing } from '@/api/modelsLocal';
import type { GpuListing } from '@/api/gpus';
import type { HubRevisionCheck, LocalTokenizerListing } from '@/api/tokenizers';
import type { StopConditionRegistry } from '@/api/stopConditions';
import type { ModelSourceListing } from '@/api/modelSources';
import type { SignalSourceListing } from '@/api/signalSources';
import type { PackListing } from '@/types/packs';
import {
  choose,
  openPicker,
  pickedValue,
  pickerInput,
  pickerValues,
} from '@/components/common/pickerTestKit';

const ok =
  <T,>(value: T) =>
  () =>
    Promise.resolve<ApiResult<T>>({ ok: true, value });
const refused = () =>
  Promise.resolve<ApiResult<never>>({
    ok: false,
    failure: { kind: 'transport', status: 0, cause: 'connection refused' },
  });

const MODES: ModelSourceListing = {
  modes: [
    {
      mode: 'load',
      available: true,
      default: true,
      reason: null,
      message: null,
      capabilities_unavailable: [],
      sources: ['local_dir', 'hub'],
    },
    {
      mode: 'attach',
      available: false,
      default: false,
      reason: 'millm_not_configured',
      message: 'MILLM_BASE_URL is unset',
      capabilities_unavailable: ['hot update'],
      sources: [],
    },
  ],
  default: 'load',
};

const MODELS: LocalModelListing = {
  models: [
    {
      path: '/data/checkpoints/run-a/step-40',
      root: '/data/checkpoints',
      relative_path: 'run-a/step-40',
      kind: 'checkpoint',
      architecture: 'LlamaForCausalLM',
      has_weights: true,
      has_tokenizer: true,
      config_path: '/data/checkpoints/run-a/step-40/config.json',
      modified_at: '2026-10-17T12:00:00+00:00',
    },
    {
      path: '/data/checkpoints/arch/tiny',
      root: '/data/checkpoints',
      relative_path: 'arch/tiny',
      kind: 'directory',
      architecture: 'llama',
      has_weights: false,
      has_tokenizer: false,
      config_path: '/data/checkpoints/arch/tiny/config.json',
      modified_at: '2026-10-16T12:00:00+00:00',
    },
  ],
  total: 2,
  limit: 500,
  offset: 0,
  roots: ['/data/checkpoints', '/data/parts'],
  unreadable_roots: [
    { root: '/data/parts', code: 'model_root_missing', reason: 'not a directory on this host' },
  ],
  max_depth: 4,
};

function ModelSourceHarness({
  onChange,
  loadModels = ok(MODELS),
}: {
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
  readonly loadModels?: () => Promise<ApiResult<LocalModelListing>>;
}) {
  const [state, setState] = useState<Record<string, string>>({});
  return (
    <ModelSourceField
      mode={state['mode'] ?? ''}
      path={state['path'] ?? ''}
      repo=""
      revision=""
      configPath={state['configPath'] ?? ''}
      onChange={(patch) => {
        onChange(patch);
        setState((current) => ({ ...current, ...patch }));
      }}
      load={ok(MODES)}
      loadModels={loadModels}
    />
  );
}

describe('[ModelSourceField] the mode, the path and the config are pickers', () => {
  it('should list every mode and refuse to let an unavailable one be chosen', async () => {
    const onChange = vi.fn();
    render(<ModelSourceHarness onChange={onChange} />);
    expect(await pickerValues('mode-select')).toEqual(['load', 'attach']);
    const listbox = await openPicker('mode-select');
    expect(listbox).toHaveTextContent('MILLM_BASE_URL is unset');
    await choose('mode-select', 'attach');
    expect(onChange).not.toHaveBeenCalled();
    await choose('mode-select', 'load');
    expect(onChange).toHaveBeenCalledWith({ mode: 'load' });
  });

  it('should offer only directories with weights for a load, and submit the PATH', async () => {
    const onChange = vi.fn();
    render(<ModelSourceHarness onChange={onChange} />);
    expect(await pickerValues('path-input')).toEqual(['/data/checkpoints/run-a/step-40']);
    await choose('path-input', '/data/checkpoints/run-a/step-40');
    expect(onChange).toHaveBeenLastCalledWith({ path: '/data/checkpoints/run-a/step-40' });
    // The label is the path under its root; the submitted value is the absolute path.
    expect(pickerInput('path-input').value).toBe('/data/checkpoints/run-a/step-40');
    expect(pickedValue('path-input')).toBe('/data/checkpoints/run-a/step-40');
  });

  it('should offer every config.json for a create, weights or not, and submit its path', async () => {
    const onChange = vi.fn();
    render(<ModelSourceHarness onChange={onChange} />);
    expect(await pickerValues('config-path-input')).toEqual([
      '/data/checkpoints/run-a/step-40/config.json',
      '/data/checkpoints/arch/tiny/config.json',
    ]);
    await choose('config-path-input', '/data/checkpoints/arch/tiny/config.json');
    expect(onChange).toHaveBeenLastCalledWith({
      configPath: '/data/checkpoints/arch/tiny/config.json',
    });
  });

  it('should keep a typed path, and say why the list cannot be complete', async () => {
    const onChange = vi.fn();
    render(<ModelSourceHarness onChange={onChange} />);
    await openPicker('path-input');
    await act(async () => {
      fireEvent.change(pickerInput('path-input'), { target: { value: '/data/parts/deep/x' } });
    });
    expect(onChange).toHaveBeenLastCalledWith({ path: '/data/parts/deep/x' });
    expect(pickerInput('path-input').getAttribute('data-free-text-reason')).toMatch(
      /looks 4 levels/,
    );
  });

  it('should name an unreadable root, so an empty list is not read as no models', async () => {
    render(<ModelSourceHarness onChange={() => {}} />);
    expect(await screen.findByTestId('unreadable-roots')).toHaveTextContent(
      '/data/parts: not a directory on this host',
    );
  });

  it('should name the directory listing as unavailable when it cannot be read', async () => {
    render(<ModelSourceHarness onChange={() => {}} loadModels={refused} />);
    expect(await screen.findAllByText(/model directory list is unavailable/)).not.toHaveLength(0);
  });

  it('should declare Repository and Revision as typed fallbacks with their reason', () => {
    render(<ModelSourceHarness onChange={() => {}} />);
    for (const id of ['repo-input', 'revision-input']) {
      expect(screen.getByTestId(id).getAttribute('data-entry')).toBe('id-fallback');
      expect(screen.getByTestId(id).getAttribute('data-entry-reason')).toMatch(/Hub/);
    }
  });
});

describe('[SignalSourceField]', () => {
  const SOURCES: SignalSourceListing = {
    sources: [
      {
        name: 'corpus',
        part_kind: 'corpus',
        objective: 'sft',
        requires: {
          part_kind: 'corpus',
          selector: false,
          grader: false,
          generation_provider: false,
        },
        available: true,
        unavailable_reason: null,
        summary: 'next-token loss on a corpus',
      },
      {
        name: 'reward',
        part_kind: null,
        objective: 'reward',
        requires: { part_kind: null, selector: true, grader: true, generation_provider: true },
        available: false,
        // The wire sends a CODE, not a sentence. This fixture once carried a sentence, which
        // hid that the screen printed the bare code (copy review, 2026-10-20).
        unavailable_reason: 'inference_provider_unavailable',
        summary: 'reward from a grader',
      },
    ],
    count: 2,
    available: 1,
  };

  it('should list every source, the unavailable one disabled with its reason', async () => {
    const onChange = vi.fn();
    render(<SignalSourceField value="" onChange={onChange} load={ok(SOURCES)} />);
    expect(await pickerValues('signal-source-select')).toEqual(['corpus', 'reward']);
    await choose('signal-source-select', 'reward');
    expect(onChange).not.toHaveBeenCalled();
    await choose('signal-source-select', 'corpus');
    expect(onChange).toHaveBeenCalledWith('corpus');
  });

  it('should say why a source is unavailable in words, and spell out the objective', async () => {
    render(<SignalSourceField value="corpus" onChange={() => {}} load={ok(SOURCES)} />);
    const notes = await screen.findByTestId('signal-source-notes');
    expect(notes).toHaveTextContent(
      'unavailable: no generation provider is available (inference_provider_unavailable)',
    );
    const requires = screen.getByTestId('signal-source-requires');
    expect(requires).toHaveTextContent('a corpus part in the pack');
    expect(requires).toHaveTextContent('supervised fine-tuning (SFT)');
  });
});

describe('[GpuRequestField] a card by UUID, never by index', () => {
  const GPUS: GpuListing = {
    devices: [
      {
        uuid: 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57',
        name: 'NVIDIA GeForce RTX 3090',
        free_memory_bytes: 20 * 1024 ** 3,
      },
      {
        uuid: 'GPU-f47ba814-49a2-603f-3595-275284140251',
        name: 'NVIDIA GeForce RTX 3080 Ti',
        free_memory_bytes: null,
      },
    ],
    available: true,
    unavailable: null,
    auto: true,
  };

  it('should offer auto and each card, and submit the UUID - not the name', async () => {
    const onChange = vi.fn();
    render(<GpuRequestField value="" onChange={onChange} load={ok(GPUS)} />);
    expect(await pickerValues('gpu-request')).toEqual([
      'auto',
      'GPU-247aa582-0d1b-e161-8156-983ed1fefc57',
      'GPU-f47ba814-49a2-603f-3595-275284140251',
    ]);
    const listbox = await openPicker('gpu-request');
    expect(listbox).toHaveTextContent('20.0 GiB free now');
    expect(listbox).toHaveTextContent('free memory not read');
    expect(listbox.textContent ?? '').not.toMatch(/\bindex\b|GPU 0|GPU 1/);
    await choose('gpu-request', 'GPU-f47ba814-49a2-603f-3595-275284140251');
    expect(onChange).toHaveBeenLastCalledWith('GPU-f47ba814-49a2-603f-3595-275284140251');
  });

  it('should still offer auto, and name why, when no card is readable here', async () => {
    const onChange = vi.fn();
    render(
      <GpuRequestField
        value=""
        onChange={onChange}
        load={ok<GpuListing>({
          devices: [],
          available: false,
          unavailable: { code: 'nvidia_smi_unavailable', message: 'nvidia-smi is not on PATH' },
          auto: true,
        })}
      />,
    );
    expect(await pickerValues('gpu-request')).toEqual(['auto']);
    expect(screen.getByTestId('gpu-unavailable')).toHaveTextContent('nvidia_smi_unavailable');
  });
});

const REGISTRY: StopConditionRegistry = {
  conditions: [
    {
      condition: 'operator_stop',
      configured: false,
      required_parameters: [],
      precedence_rank: 0,
      summary: 'the operator asked',
    },
    {
      condition: 'step_count',
      configured: true,
      required_parameters: ['total'],
      precedence_rank: 4,
      summary: 'a number of steps',
    },
    {
      condition: 'loss_plateau',
      configured: true,
      required_parameters: ['metric', 'window', 'min_relative_improvement', 'min_steps'],
      precedence_rank: 3,
      summary: 'the loss stopped improving',
      metrics: ['raw', 'window_mean'],
    },
  ],
  count: 3,
  closed: true,
  precedence: ['operator_stop', 'loss_plateau', 'step_count'],
  minimum_enabled: 1,
};

function StopHarness({
  registry = REGISTRY,
  onEntries,
}: {
  readonly registry?: StopConditionRegistry;
  readonly onEntries: (entries: readonly StopConditionEntry[]) => void;
}) {
  const [entries, setEntries] = useState<readonly StopConditionEntry[]>([
    { condition: '', params: {} },
  ]);
  return (
    <StopConditionsField
      entries={entries}
      onChange={(next) => {
        onEntries(next);
        setEntries(next);
      }}
      load={ok(registry)}
    />
  );
}

describe('[StopConditionsField] the live registry, and its parameter names (AC-009.47)', () => {
  it('should offer exactly the registrys conditions, not a hard-coded list', async () => {
    const extended: StopConditionRegistry = {
      ...REGISTRY,
      conditions: [
        ...REGISTRY.conditions,
        {
          condition: 'brand_new',
          configured: false,
          required_parameters: [],
          precedence_rank: 9,
          summary: 'a sixth',
        },
      ],
    };
    render(<StopHarness registry={extended} onEntries={() => {}} />);
    expect(await pickerValues('stop-condition-0')).toEqual([
      'operator_stop',
      'step_count',
      'loss_plateau',
      'brand_new',
    ]);
  });

  it('should ask step_count for its total, under the name the registry gives', async () => {
    const onEntries = vi.fn();
    render(<StopHarness onEntries={onEntries} />);
    await choose('stop-condition-0', 'step_count');
    const total = await screen.findByTestId('stop-param-0-total');
    expect(total.getAttribute('data-entry')).toBe('number');
    fireEvent.change(total, { target: { value: '40' } });
    expect(onEntries).toHaveBeenLastCalledWith([
      { condition: 'step_count', params: { total: '40' } },
    ]);
  });

  it('should ask loss_plateau for all four, its metric from the registrys own list', async () => {
    render(<StopHarness onEntries={() => {}} />);
    await choose('stop-condition-0', 'loss_plateau');
    for (const name of ['window', 'min_relative_improvement', 'min_steps']) {
      expect(screen.getByTestId(`stop-param-0-${name}`)).toBeInTheDocument();
    }
    expect(await pickerValues('stop-param-0-metric')).toEqual(['raw', 'window_mean']);
  });

  it('should say a condition takes no parameters rather than offer a box for one', async () => {
    render(<StopHarness onEntries={() => {}} />);
    await choose('stop-condition-0', 'operator_stop');
    expect(screen.getByTestId('stop-conditions')).toHaveTextContent('takes no parameters');
    expect(screen.queryByTestId(/stop-param-0-/)).toBeNull();
  });
});

describe('[PackField] chosen by name and version, submitted as pack_id and version', () => {
  const PACKS: PackListing = {
    packs: [
      {
        pack_id: 'pk-7f3a',
        version: '3',
        name: 'notary',
        domain: 'notarial acts',
        parts_filled: 7,
        parts_total: 7,
        prompt_count: 120,
        runs_using: [],
        validity: {
          state: 'valid',
          missing_slots: [],
          unresolved: [],
          service_unavailable: [],
          retention_floor_missing: false,
        },
        derived_from: null,
        is_example: false,
        definition_sha256: 'a'.repeat(64),
        created_at: '2026-10-17T00:00:00Z',
      },
      {
        pack_id: 'pk-91bb',
        version: '1',
        name: 'tax',
        domain: null,
        parts_filled: 4,
        parts_total: 7,
        prompt_count: null,
        runs_using: [],
        validity: {
          state: 'incomplete',
          missing_slots: ['test_set'],
          unresolved: [],
          service_unavailable: [],
          retention_floor_missing: false,
        },
        derived_from: null,
        is_example: false,
        definition_sha256: 'b'.repeat(64),
        created_at: '2026-10-17T00:00:00Z',
      },
    ],
  };

  it('should submit the pack id and version, never the label (M41)', async () => {
    const onChange = vi.fn();
    render(<PackField packId="" version="" onChange={onChange} load={ok(PACKS)} />);
    await choose('pack-select', 'pk-7f3a@3');
    expect(onChange).toHaveBeenCalledWith('pk-7f3a', '3');
    expect(onChange).not.toHaveBeenCalledWith('notary', expect.anything());
  });

  it('should list an incomplete pack, with what it is missing, rather than hide it', async () => {
    render(<PackField packId="" version="" onChange={() => {}} load={ok(PACKS)} />);
    expect(await pickerValues('pack-select')).toEqual(['pk-7f3a@3', 'pk-91bb@1']);
    expect(await openPicker('pack-select')).toHaveTextContent('incomplete: missing test_set');
  });

  it('should say how to get a pack when none is registered', async () => {
    render(<PackField packId="" version="" onChange={() => {}} load={ok({ packs: [] })} />);
    expect(await screen.findByTestId('pack-select-status')).toHaveTextContent(
      'Author one on Onboard a domain',
    );
  });
});

/* ------------------------------------------------------------------------------------ *
 * A create run's tokenizer (operator request 2026-10-20)
 * ------------------------------------------------------------------------------------ */

const TOKENIZERS: LocalTokenizerListing = {
  tokenizers: [
    {
      path: '/data/checkpoints/tokenizers/tiny-tok',
      root: '/data/checkpoints',
      relative_path: 'tokenizers/tiny-tok',
      kind: 'tokenizer',
      tokenizer_files: ['tokenizer.json', 'tokenizer_config.json'],
      modified_at: '2026-10-17T12:00:00+00:00',
    },
    {
      path: '/data/checkpoints/run-a/step-40',
      root: '/data/checkpoints',
      relative_path: 'run-a/step-40',
      kind: 'checkpoint',
      tokenizer_files: ['tokenizer.json'],
      modified_at: '2026-10-16T12:00:00+00:00',
    },
  ],
  total: 2,
  limit: 500,
  offset: 0,
  roots: ['/data/checkpoints'],
  unreadable_roots: [],
  max_depth: 4,
};

const COMMIT = '0123456789abcdef0123456789abcdef01234567';
const CHECKED: HubRevisionCheck = {
  repo: 'org/tok',
  revision: 'main',
  commit: COMMIT,
  is_commit: false,
  tokenizer_files: ['tokenizer.json'],
  has_tokenizer: true,
  has_config: false,
  has_weights: false,
};

function TokenizerHarness({
  initialMode,
  onChange,
  loadTokenizers = ok(TOKENIZERS),
  checkRevision = vi.fn(() =>
    Promise.resolve<ApiResult<HubRevisionCheck>>({ ok: true, value: CHECKED }),
  ),
}: {
  readonly initialMode: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
  readonly loadTokenizers?: () => Promise<ApiResult<LocalTokenizerListing>>;
  readonly checkRevision?: (repo: string, revision: string) => Promise<ApiResult<HubRevisionCheck>>;
}) {
  const [state, setState] = useState<Record<string, string>>({ mode: initialMode });
  return (
    <ModelSourceField
      mode={state['mode'] ?? ''}
      path=""
      repo={state['repo'] ?? ''}
      revision={state['revision'] ?? ''}
      configPath=""
      tokenizerPath={state['tokenizerPath'] ?? ''}
      tokenizerRepo={state['tokenizerRepo'] ?? ''}
      tokenizerRevision={state['tokenizerRevision'] ?? ''}
      onChange={(patch) => {
        onChange(patch);
        setState((current) => ({ ...current, ...patch }));
      }}
      load={ok(MODES)}
      loadModels={ok(MODELS)}
      loadTokenizers={loadTokenizers}
      checkRevision={checkRevision}
    />
  );
}

describe('[ModelSourceField] the tokenizer is asked for on a create, and only there', () => {
  it('should show no tokenizer field for a load or before a mode is chosen', async () => {
    const loadTokenizers = vi.fn(ok(TOKENIZERS));
    for (const mode of ['', 'load', 'attach']) {
      const { unmount } = render(
        <TokenizerHarness initialMode={mode} onChange={vi.fn()} loadTokenizers={loadTokenizers} />,
      );
      await act(async () => {});
      expect(screen.queryByTestId('tokenizer-fields'), mode).toBeNull();
      expect(screen.queryByTestId('tokenizer-path-input'), mode).toBeNull();
      unmount();
    }
    // Not even read: a load has no tokenizer to choose.
    expect(loadTokenizers).not.toHaveBeenCalled();
  });

  it('should list the tokenizer folders for a create and submit the chosen PATH', async () => {
    const onChange = vi.fn();
    const loadTokenizers = vi.fn(ok(TOKENIZERS));
    render(
      <TokenizerHarness initialMode="create" onChange={onChange} loadTokenizers={loadTokenizers} />,
    );
    expect(await pickerValues('tokenizer-path-input')).toEqual([
      '/data/checkpoints/tokenizers/tiny-tok',
      '/data/checkpoints/run-a/step-40',
    ]);
    expect(loadTokenizers).toHaveBeenCalledTimes(1);
    const listbox = await openPicker('tokenizer-path-input');
    expect(listbox).toHaveTextContent('tokenizers/tiny-tok');
    expect(listbox).toHaveTextContent('tokenizer folder');
    expect(listbox).toHaveTextContent('checkpoint');
    await choose('tokenizer-path-input', '/data/checkpoints/tokenizers/tiny-tok');
    expect(onChange).toHaveBeenLastCalledWith({
      tokenizerPath: '/data/checkpoints/tokenizers/tiny-tok',
    });
    expect(pickedValue('tokenizer-path-input')).toBe('/data/checkpoints/tokenizers/tiny-tok');
    // The identifier is what is held - the folder's full path, not its label.
    expect(pickerInput('tokenizer-path-input').getAttribute('data-value')).toBe(
      '/data/checkpoints/tokenizers/tiny-tok',
    );
  });

  it('should say so when the tokenizer list could not be read, rather than show it empty', async () => {
    render(<TokenizerHarness initialMode="create" onChange={vi.fn()} loadTokenizers={refused} />);
    expect(await screen.findAllByText(/tokenizer folder list is unavailable/)).not.toHaveLength(0);
  });

  it('should resolve a Hub revision to its commit on Check, and pin it on request', async () => {
    const onChange = vi.fn();
    const checkRevision = vi.fn(() =>
      Promise.resolve<ApiResult<HubRevisionCheck>>({ ok: true, value: CHECKED }),
    );
    render(
      <TokenizerHarness initialMode="create" onChange={onChange} checkRevision={checkRevision} />,
    );
    const button = screen.getByTestId('tokenizer-hub-check-button');
    expect(button).toBeDisabled();
    fireEvent.change(screen.getByTestId('tokenizer-repo-input'), {
      target: { value: ' org/tok ' },
    });
    fireEvent.change(screen.getByTestId('tokenizer-revision-input'), {
      target: { value: 'main' },
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId('tokenizer-hub-check-button'));
    });
    expect(checkRevision).toHaveBeenCalledTimes(1);
    expect(checkRevision).toHaveBeenCalledWith('org/tok', 'main');
    const answer = screen.getByTestId('tokenizer-hub-check-answer');
    expect(answer).toHaveTextContent(`points at commit ${COMMIT} now`);
    expect(answer).toHaveTextContent('Tokenizer files at its root: tokenizer.json.');
    await act(async () => {
      fireEvent.click(screen.getByTestId('tokenizer-hub-check-pin'));
    });
    expect(onChange).toHaveBeenLastCalledWith({ tokenizerRevision: COMMIT });
    expect(screen.getByTestId('tokenizer-revision-input')).toHaveValue(COMMIT);
    // The answer was for "main"; the revision changed, so it is dropped rather than shown
    // beside a value it was not for.
    expect(screen.queryByTestId('tokenizer-hub-check-answer')).toBeNull();
  });

  it('should warn when the commit has no tokenizer at its root', async () => {
    const checkRevision = vi.fn(() =>
      Promise.resolve<ApiResult<HubRevisionCheck>>({
        ok: true,
        value: { ...CHECKED, tokenizer_files: [], has_tokenizer: false },
      }),
    );
    render(
      <TokenizerHarness initialMode="create" onChange={vi.fn()} checkRevision={checkRevision} />,
    );
    fireEvent.change(screen.getByTestId('tokenizer-repo-input'), { target: { value: 'org/tok' } });
    await act(async () => {
      fireEvent.click(screen.getByTestId('tokenizer-hub-check-button'));
    });
    expect(screen.getByTestId('tokenizer-hub-check-missing')).toHaveTextContent(
      'no tokenizer files at its root',
    );
  });

  it('should render a refused check as a failure, not an answer', async () => {
    const checkRevision = vi.fn(() =>
      Promise.resolve<ApiResult<HubRevisionCheck>>({
        ok: false,
        failure: {
          kind: 'refusal',
          status: 400,
          code: 'model_revision_not_found',
          message: "The repo 'org/tok' exists but has no revision 'nope'.",
          details: {},
        },
      }),
    );
    render(
      <TokenizerHarness initialMode="create" onChange={vi.fn()} checkRevision={checkRevision} />,
    );
    fireEvent.change(screen.getByTestId('tokenizer-repo-input'), { target: { value: 'org/tok' } });
    await act(async () => {
      fireEvent.click(screen.getByTestId('tokenizer-hub-check-button'));
    });
    expect(screen.queryByTestId('tokenizer-hub-check-answer')).toBeNull();
    expect(screen.getByTestId('tokenizer-hub-check')).toHaveTextContent('has no revision');
  });

  it('should offer the same Check for a model loaded from the Hub', async () => {
    const checkRevision = vi.fn(() =>
      Promise.resolve<ApiResult<HubRevisionCheck>>({
        ok: true,
        value: { ...CHECKED, has_config: true, has_weights: true },
      }),
    );
    render(
      <TokenizerHarness initialMode="load" onChange={vi.fn()} checkRevision={checkRevision} />,
    );
    fireEvent.change(screen.getByTestId('repo-input'), { target: { value: 'org/model' } });
    await act(async () => {
      fireEvent.click(screen.getByTestId('model-hub-check-button'));
    });
    expect(checkRevision).toHaveBeenCalledWith('org/model', '');
    expect(screen.getByTestId('model-hub-check-answer')).toHaveTextContent(
      'It has a config.json and safetensors weights at its root.',
    );
  });
});
