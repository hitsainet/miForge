/**
 * The remaining fields of R-02.53's definition (FR-009.33, .39).
 *
 * **Why several components in one file.** F9's FTASKS lists these as separate modules
 * (`ModelSourceField.tsx`, `OptimizerFields.tsx`, …). They are grouped here because each is a
 * handful of labelled inputs with no logic, every one shares the same two rules, and the two
 * components that *do* carry a rule a mutation targets - `RunTypeField` (M22) and `PackField`
 * (F7's OQ-9) - are in their own files with their own tests, as is `BudgetPreview` (M20) and
 * `RefusalBanner` (M21). A file per text input would spread one rule across seven modules,
 * which is how a rule comes to be followed in six of them. Recorded as a deviation.
 *
 * **The two rules, which are the same rule from both sides:**
 *
 * 1. **No field carries an initial value.** `ModelSourceField` does not preselect the mode
 *    the registry marks `default: true`; `OptimizerFields` does not preselect `8bit`; no
 *    cadence starts at a number. A non-nullable field with an initial value *is* a default,
 *    and ADR-018 says the run log would record it as the operator's choice. **M19.**
 * 2. **No field validates.** No required marker that blocks, no range check, no "at least one
 *    stop condition". The refusals are the backend's copy.
 *
 * An unavailable option is **shown, named and disabled with its reason** - never hidden and
 * never swapped. *An option missing from a response is indistinguishable from one that is
 * broken* (ADR-013), and the same is true of one missing from a form.
 *
 * **Every field that names something that already exists is a `Picker`** over the route that
 * lists it (task 14.0): the acquisition mode, the model directory, the architecture config,
 * the signal source, each stop condition and its `metric`, and the GPU. A text input remains
 * only for a number, a new name, or an identifier no list here can hold - and each such input
 * declares which it is (`Input`'s `entry`), which the ROUTES-derived test reads.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { Field, Input, Select } from '@/components/common/Field';
import { Picker, type PickerListing, type PickerOption } from '@/components/common/Picker';
import { Button } from '@/components/common/Button';
import { Card } from '@/components/common/Card';
import { MonoId } from '@/components/common/MonoId';
import { FailureNotice } from '@/components/common/FailureNotice';
import type { ApiFailure } from '@/api/errors';
import { useListing } from '@/hooks/useListing';
import { fetchModelSources, type ModelSourceListing } from '@/api/modelSources';
import { fetchSignalSources, type SignalSourceListing } from '@/api/signalSources';
import { fetchLocalModels, type LocalModelListing, type LocalModelRow } from '@/api/modelsLocal';
import { fetchGpus, type GpuListing } from '@/api/gpus';
import {
  checkHubRevision,
  fetchLocalTokenizers,
  type HubRevisionCheck,
  type LocalTokenizerListing,
  type LocalTokenizerRow,
} from '@/api/tokenizers';
import {
  fetchStopConditionRegistry,
  type StopConditionEntry as RegistryEntry,
  type StopConditionRegistry,
} from '@/api/stopConditions';
import type { StopConditionEntry } from './draft';
import { unavailableSentence } from './unavailableReasons';

/** A training objective's code, spelled out once with the code in brackets. */
const OBJECTIVES: Readonly<Record<string, string>> = {
  sft: 'supervised fine-tuning (SFT)',
  dpo: 'direct preference optimization (DPO)',
  grpo: 'group relative policy optimization (GRPO)',
};

/** `preference_pairs` -> "a preference pairs part in the pack". The wire may omit it. */
function partKindPhrase(kind: string | null | undefined): string {
  return kind === null || kind === undefined || kind.length === 0
    ? 'no pack part'
    : `a ${kind.replace(/_/g, ' ')} part in the pack`;
}

export function objectiveName(objective: string | null | undefined): string {
  if (objective === null || objective === undefined || objective.length === 0)
    return 'not recorded';
  return OBJECTIVES[objective] ?? objective;
}

/* ------------------------------------------------------------------------------------ *
 * Model source - R-02.46's three ways
 * ------------------------------------------------------------------------------------ */

/** Every acquisition mode; an unavailable one is listed, disabled, with its reason. */
export function modeOptions(listing: ModelSourceListing): readonly PickerOption[] {
  return listing.modes.map((row) => ({
    value: row.mode,
    label: row.mode,
    detail: `sources: ${row.sources.join(', ') || 'none'}`,
    ...(row.available
      ? {}
      : {
          disabledReason:
            (row.message ?? row.reason ?? 'the response gave no reason') +
            (row.capabilities_unavailable.length === 0
              ? ''
              : ` Without it, miForge cannot: ${row.capabilities_unavailable.join('; ')}.`),
        }),
  }));
}

function modelDetail(row: LocalModelRow): string {
  return [
    row.kind === 'checkpoint' ? 'checkpoint' : 'model directory',
    row.architecture ?? 'architecture not read',
    row.has_tokenizer ? null : 'no tokenizer in this directory, so it cannot be loaded',
    `under ${row.root}`,
  ]
    .filter((part): part is string => part !== null)
    .join(' · ');
}

/** The directories a `load` can read: those with safetensors shards, by their path. */
export function loadableOptions(listing: LocalModelListing): readonly PickerOption[] {
  return listing.models
    .filter((row) => row.has_weights)
    .map((row) => ({
      value: row.path,
      label: row.relative_path,
      detail: modelDetail(row),
      keywords: row.path,
    }));
}

/** Every listed directory's `config.json`, for a `create`. Weights are not needed. */
export function configOptions(listing: LocalModelListing): readonly PickerOption[] {
  return listing.models.map((row) => ({
    value: row.config_path,
    label: `${row.relative_path}/config.json`,
    detail: `${row.architecture ?? 'architecture not read'} · under ${row.root}`,
    keywords: row.config_path,
  }));
}

/** One read serves both pickers; the page bound is the route's maximum. */
const loadAllModels = (): ReturnType<typeof fetchLocalModels> => fetchLocalModels({ limit: 500 });

const TOKENIZER_KINDS: Readonly<Record<string, string>> = {
  tokenizer: 'tokenizer folder',
  model: 'model folder',
  checkpoint: 'checkpoint',
};

function tokenizerDetail(row: LocalTokenizerRow): string {
  return [
    TOKENIZER_KINDS[row.kind] ?? row.kind,
    row.tokenizer_files.join(', '),
    `under ${row.root}`,
  ].join(' · ');
}

/** Every folder under the model roots with tokenizer files at its root, by its path. */
export function tokenizerOptions(listing: LocalTokenizerListing): readonly PickerOption[] {
  return listing.tokenizers.map((row) => ({
    value: row.path,
    label: row.relative_path,
    detail: tokenizerDetail(row),
    keywords: row.path,
  }));
}

const loadAllTokenizers = (): ReturnType<typeof fetchLocalTokenizers> =>
  fetchLocalTokenizers({ limit: 500 });

export interface ModelSourceFieldProps {
  readonly mode: string;
  readonly path: string;
  readonly repo: string;
  readonly revision: string;
  readonly configPath: string;
  /** A create run's tokenizer: a folder on this host, or a Hub repository and revision. */
  readonly tokenizerPath?: string;
  readonly tokenizerRepo?: string;
  readonly tokenizerRevision?: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
  readonly load?: typeof fetchModelSources;
  /** Injected in tests. The real one reads `GET /api/v1/models/local`. */
  readonly loadModels?: () => ReturnType<typeof fetchLocalModels>;
  /** Injected in tests. The real one reads `GET /api/v1/tokenizers/local`. */
  readonly loadTokenizers?: () => ReturnType<typeof fetchLocalTokenizers>;
  /** Injected in tests. The real one reads `GET /api/v1/hub/revision`. */
  readonly checkRevision?: typeof checkHubRevision;
}

export function ModelSourceField({
  mode,
  path,
  repo,
  revision,
  configPath,
  tokenizerPath = '',
  tokenizerRepo = '',
  tokenizerRevision = '',
  onChange,
  load = fetchModelSources,
  loadModels = loadAllModels,
  loadTokenizers = loadAllTokenizers,
  checkRevision = checkHubRevision,
}: ModelSourceFieldProps): ReactNode {
  const modes = useListing(load, modeOptions);
  // One read, two views of it: the loadable directories, and every directory's config.
  const models = useListing(loadModels, loadableOptions);
  const loadable = models.listing;
  const configs: PickerListing =
    models.value === null
      ? models.listing
      : { kind: 'ready', options: configOptions(models.value) };
  const depth = models.value?.max_depth;
  const walkReason =
    `The list looks ${depth ?? 'a limited number of'} levels below each model root, so a ` +
    'deeper directory, or one added after this page loaded, is not on it. A typed path must ' +
    'still be under a model root.';

  return (
    <>
      <Field
        label="Model source"
        hint="Load a model from a directory on this host or from a Hub repository, or create a new model from an architecture config. A mode that cannot run here is listed with the reason."
      >
        {({ id, describedBy }) => (
          <>
            <Picker
              id={id}
              describedBy={describedBy}
              value={mode}
              onChange={(next) => onChange({ mode: next })}
              listing={modes.listing}
              placeholder="Choose how to get the model"
              empty="No model source modes are available."
              surface="the model source list"
              data-testid="mode-select"
            />
            {modes.value === null ? null : (
              <ul className="mt-2 stack-y-0.5 text-hint text-dim" data-testid="mode-reasons">
                {modes.value.modes
                  .filter((row) => !row.available)
                  .map((row) => (
                    <li key={row.mode} className="text-amber">
                      <span className="font-mono text-ident">{row.mode}</span> is unavailable:{' '}
                      {row.message ?? row.reason ?? 'the response gave no reason'}
                      {row.capabilities_unavailable.length === 0
                        ? null
                        : ` Without it, miForge cannot: ${row.capabilities_unavailable.join('; ')}.`}
                    </li>
                  ))}
              </ul>
            )}
          </>
        )}
      </Field>

      <Field
        label="Path"
        hint="For load mode: a model directory or checkpoint on this host. Only paths under a model root can be loaded."
      >
        {({ id, describedBy }) => (
          <>
            <Picker
              id={id}
              describedBy={describedBy}
              value={path}
              onChange={(next) => onChange({ path: next })}
              listing={loadable}
              placeholder="Choose a model directory"
              empty="No model directory with weights is under the model roots."
              surface="the model directory list"
              freeText={{ reason: walkReason }}
              data-testid="path-input"
            />
            <UnreadableRoots listing={models.value} />
          </>
        )}
      </Field>

      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-2">
        <Field
          label="Repository"
          hint="For load mode from the Hub: the repository ID, such as org/model."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={repo}
              onChange={(event) => onChange({ repo: event.target.value })}
              entry="id-fallback"
              entryReason="miForge does not search the Hub."
              data-testid="repo-input"
            />
          )}
        </Field>
        <Field
          label="Revision"
          hint="A commit hash pins the exact files. A branch name such as main can change between runs."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={revision}
              onChange={(event) => onChange({ revision: event.target.value })}
              entry="id-fallback"
              entryReason="miForge does not search the Hub, so revisions are not listed."
              data-testid="revision-input"
            />
          )}
        </Field>
      </div>
      <HubRevisionCheckButton
        repo={repo}
        revision={revision}
        want="model"
        onPin={(commit) => onChange({ revision: commit })}
        check={checkRevision}
        testId="model-hub-check"
      />

      <Field
        label="Architecture config path"
        hint="For create mode: the config.json a new model is built from, with randomly initialized weights."
      >
        {({ id, describedBy }) => (
          <Picker
            id={id}
            describedBy={describedBy}
            value={configPath}
            onChange={(next) => onChange({ configPath: next })}
            listing={configs}
            placeholder="Choose an architecture config"
            empty="No config.json is under the model roots."
            surface="the model directory list"
            freeText={{
              reason: `Only a config.json inside a model directory is listed. ${walkReason}`,
            }}
            data-testid="config-path-input"
          />
        )}
      </Field>

      {mode === 'create' ? (
        <TokenizerFields
          path={tokenizerPath}
          repo={tokenizerRepo}
          revision={tokenizerRevision}
          onChange={onChange}
          loadTokenizers={loadTokenizers}
          checkRevision={checkRevision}
          walkReason={walkReason}
        />
      ) : null}
    </>
  );
}

interface TokenizerFieldsProps {
  readonly path: string;
  readonly repo: string;
  readonly revision: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
  readonly loadTokenizers: () => ReturnType<typeof fetchLocalTokenizers>;
  readonly checkRevision: typeof checkHubRevision;
  readonly walkReason: string;
}

/**
 * A create run's tokenizer, **shown for create mode only**. A loaded model brings its own
 * tokenizer from its folder; a new model has none, and miForge does not train one, so the run
 * is given one: a folder on this host, or a Hugging Face repository. Choose one of the two -
 * naming both is refused by the server, not resolved here.
 */
function TokenizerFields({
  path,
  repo,
  revision,
  onChange,
  loadTokenizers,
  checkRevision,
  walkReason,
}: TokenizerFieldsProps): ReactNode {
  const tokenizers = useListing(loadTokenizers, tokenizerOptions);
  return (
    <div data-testid="tokenizer-fields">
      <Field
        label="Tokenizer"
        hint="A new model needs a tokenizer, and miForge does not make one. Choose a tokenizer, model or checkpoint folder on this host whose tokenizer the new model will use, or name a Hugging Face repository below. Its vocabulary must fit the vocabulary size in the architecture config."
      >
        {({ id, describedBy }) => (
          <>
            <Picker
              id={id}
              describedBy={describedBy}
              value={path}
              onChange={(next) => onChange({ tokenizerPath: next })}
              listing={tokenizers.listing}
              placeholder="Choose a tokenizer folder"
              empty="No folder with tokenizer files is under the model roots."
              surface="the tokenizer folder list"
              freeText={{ reason: walkReason }}
              data-testid="tokenizer-path-input"
            />
            {tokenizers.value === null || tokenizers.value.unreadable_roots.length === 0 ? null : (
              <ul className="mt-1 stack-y-0.5 text-hint text-amber">
                {tokenizers.value.unreadable_roots.map((row) => (
                  <li key={row.root}>
                    <span className="font-mono text-ident">{row.root}</span>: {row.reason}
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </Field>

      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-2">
        <Field
          label="Tokenizer repository"
          hint="Or a Hugging Face repository with tokenizer files at its root, such as org/model. Leave the folder above empty if you use this."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={repo}
              onChange={(event) => onChange({ tokenizerRepo: event.target.value })}
              entry="id-fallback"
              entryReason="miForge does not search the Hub."
              data-testid="tokenizer-repo-input"
            />
          )}
        </Field>
        <Field
          label="Tokenizer revision"
          hint="A branch, tag or commit. Empty means main. Check turns a branch into the commit it points at now."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={revision}
              onChange={(event) => onChange({ tokenizerRevision: event.target.value })}
              entry="id-fallback"
              entryReason="miForge does not search the Hub, so revisions are not listed."
              data-testid="tokenizer-revision-input"
            />
          )}
        </Field>
      </div>
      <HubRevisionCheckButton
        repo={repo}
        revision={revision}
        want="tokenizer"
        onPin={(commit) => onChange({ tokenizerRevision: commit })}
        check={checkRevision}
        testId="tokenizer-hub-check"
      />
    </div>
  );
}

type CheckState =
  | { readonly kind: 'idle' }
  | { readonly kind: 'checking' }
  | { readonly kind: 'done'; readonly value: HubRevisionCheck }
  | { readonly kind: 'failed'; readonly failure: ApiFailure };

export interface HubRevisionCheckButtonProps {
  readonly repo: string;
  readonly revision: string;
  /** What the repository is for, which decides what its answer is read for. */
  readonly want: 'model' | 'tokenizer';
  /** Called with the commit when the operator chooses to use it as the revision. */
  readonly onPin: (commit: string) => void;
  readonly check: typeof checkHubRevision;
  readonly testId: string;
}

/**
 * **Check**: resolve the revision to the commit it points at now, and say what that commit
 * holds at its root. "Use this commit" puts the commit in the revision field, so the run is
 * defined against exact files rather than a branch that can move. The answer is dropped when
 * the repository or revision changes, so it is never read against a value it was not for.
 */
export function HubRevisionCheckButton({
  repo,
  revision,
  want,
  onPin,
  check,
  testId,
}: HubRevisionCheckButtonProps): ReactNode {
  const [state, setState] = useState<CheckState>({ kind: 'idle' });
  useEffect(() => {
    setState({ kind: 'idle' });
  }, [repo, revision]);

  const trimmedRepo = repo.trim();
  const onCheck = async (): Promise<void> => {
    setState({ kind: 'checking' });
    const result = await check(trimmedRepo, revision.trim());
    setState(
      result.ok
        ? { kind: 'done', value: result.value }
        : { kind: 'failed', failure: result.failure },
    );
  };

  return (
    <div className="mb-3" data-testid={testId}>
      <Button
        onClick={() => void onCheck()}
        disabled={trimmedRepo.length === 0 || state.kind === 'checking'}
        data-testid={`${testId}-button`}
      >
        {state.kind === 'checking' ? 'Checking…' : 'Check'}
      </Button>
      {state.kind === 'failed' ? (
        <FailureNotice failure={state.failure} surface="the Hub check" className="mt-2" />
      ) : null}
      {state.kind === 'done' ? (
        <HubCheckAnswer value={state.value} want={want} onPin={onPin} testId={testId} />
      ) : null}
    </div>
  );
}

function HubCheckAnswer({
  value,
  want,
  onPin,
  testId,
}: {
  readonly value: HubRevisionCheck;
  readonly want: 'model' | 'tokenizer';
  readonly onPin: (commit: string) => void;
  readonly testId: string;
}): ReactNode {
  const missing =
    want === 'tokenizer'
      ? value.has_tokenizer
        ? null
        : 'This commit has no tokenizer files at its root, so a create run using it would be refused.'
      : [
          value.has_config ? null : 'no config.json',
          value.has_weights ? null : 'no safetensors weights',
        ]
          .filter((part): part is string => part !== null)
          .join(' and ') || null;
  return (
    <div className="mt-2 text-hint" data-testid={`${testId}-answer`}>
      <div>
        <MonoId value={value.repo} /> at <MonoId value={value.revision} />{' '}
        {value.is_commit ? 'is the commit' : 'points at commit'} <MonoId value={value.commit} />
        {value.is_commit ? '.' : ' now.'}
      </div>
      {want === 'tokenizer' && value.has_tokenizer ? (
        <div className="text-dim">
          Tokenizer files at its root: {value.tokenizer_files.join(', ')}.
        </div>
      ) : null}
      {want === 'model' && missing === null ? (
        <div className="text-dim">It has a config.json and safetensors weights at its root.</div>
      ) : null}
      {missing === null ? null : (
        <div className="text-amber" data-testid={`${testId}-missing`}>
          {want === 'tokenizer'
            ? missing
            : `This commit has ${missing} at its root, so it cannot be loaded.`}
        </div>
      )}
      {value.is_commit ? null : (
        <div className="mt-1">
          <Button onClick={() => onPin(value.commit)} data-testid={`${testId}-pin`}>
            Use this commit
          </Button>{' '}
          <span className="text-dim">
            A branch or tag can move. Using the commit keeps the run on these exact files.
          </span>
        </div>
      )}
    </div>
  );
}

/** A root that could not be read is named, so an empty list is not read as "no models". */
function UnreadableRoots({ listing }: { readonly listing: LocalModelListing | null }): ReactNode {
  if (listing === null || listing.unreadable_roots.length === 0) return null;
  return (
    <ul className="mt-1 stack-y-0.5 text-hint text-amber" data-testid="unreadable-roots">
      {listing.unreadable_roots.map((row) => (
        <li key={row.root}>
          <span className="font-mono text-ident">{row.root}</span>: {row.reason}
        </li>
      ))}
    </ul>
  );
}

/* ------------------------------------------------------------------------------------ *
 * Signal source - R-02.4-.7
 * ------------------------------------------------------------------------------------ */

export function signalSourceOptions(listing: SignalSourceListing): readonly PickerOption[] {
  return listing.sources.map((row) => ({
    value: row.name,
    label: row.name,
    detail: row.summary,
    // Named by its service, never hidden and never substituted (ADR-010).
    ...(row.available ? {} : { disabledReason: unavailableSentence(row.unavailable_reason) }),
  }));
}

export interface SignalSourceFieldProps {
  readonly value: string;
  readonly onChange: (value: string) => void;
  readonly load?: typeof fetchSignalSources;
}

export function SignalSourceField({
  value,
  onChange,
  load = fetchSignalSources,
}: SignalSourceFieldProps): ReactNode {
  const { value: listing, listing: pickerListing } = useListing(load, signalSourceOptions);
  const rows = listing?.sources ?? null;
  const selected = (rows ?? []).find((row) => row.name === value) ?? null;

  return (
    <Field
      label="Signal source"
      hint="Where the run's training examples come from. Availability is checked each time this list loads, so a newly configured provider shows up without a restart."
    >
      {({ id, describedBy }) => (
        <>
          <Picker
            id={id}
            describedBy={describedBy}
            value={value}
            onChange={onChange}
            listing={pickerListing}
            placeholder="Choose a signal source"
            empty="No signal sources are available."
            surface="the signal-source list"
            data-testid="signal-source-select"
          />
          {rows === null ? null : (
            <ul className="mt-2 stack-y-0.5 text-hint text-dim" data-testid="signal-source-notes">
              {rows.map((row) => (
                <li key={row.name} className={row.available ? '' : 'text-amber'}>
                  <span className="font-mono text-ident">{row.name}</span>: {row.summary}
                  {row.available
                    ? null
                    : ` — unavailable: ${unavailableSentence(row.unavailable_reason)}`}
                </li>
              ))}
            </ul>
          )}
          {selected === null ? null : (
            <div className="mt-2 text-hint text-dim" data-testid="signal-source-requires">
              This source needs: {partKindPhrase(selected.requires.part_kind)}
              {selected.requires.selector ? ', a selector configuration' : ''}
              {selected.requires.grader ? ', a grader configuration' : ''}
              {selected.requires.generation_provider ? ', a generation provider' : ''}. Training
              objective: {objectiveName(selected.objective)}.
            </div>
          )}
        </>
      )}
    </Field>
  );
}

/* ------------------------------------------------------------------------------------ *
 * GPU request - a card by UUID, never by index
 * ------------------------------------------------------------------------------------ */

/** `auto` first, then each card by UUID. **No index**: the response carries none. */
export function gpuOptions(listing: GpuListing): readonly PickerOption[] {
  return [
    {
      value: 'auto',
      label: 'auto',
      detail: 'miForge picks the card with the most free memory when the run starts',
    },
    ...listing.devices.map((device) => ({
      value: device.uuid,
      label: device.name,
      detail: `${device.uuid} · ${
        device.free_memory_bytes === null
          ? 'free memory not read'
          : `${(device.free_memory_bytes / 1024 ** 3).toFixed(1)} GiB free now`
      }`,
    })),
  ];
}

export interface GpuRequestFieldProps {
  readonly value: string;
  readonly onChange: (value: string) => void;
  readonly load?: typeof fetchGpus;
}

export function GpuRequestField({
  value,
  onChange,
  load = fetchGpus,
}: GpuRequestFieldProps): ReactNode {
  const { value: listing, listing: pickerListing } = useListing(load, gpuOptions);
  return (
    <Field
      label="GPU request"
      hint="Optional. Left empty, miForge picks a card when the run starts. A card is named by its universally unique identifier (UUID), never by index."
    >
      {({ id, describedBy }) => (
        <>
          <Picker
            id={id}
            describedBy={describedBy}
            value={value}
            onChange={onChange}
            listing={pickerListing}
            placeholder="Choose a card, or auto"
            empty="No GPU is visible to miForge."
            surface="the GPU list"
            freeText={{
              reason:
                'A card in use by a training worker may not be listed. You can type its GPU-… UUID, as shown by nvidia-smi -L on the host.',
            }}
            data-testid="gpu-request"
          />
          {listing === null || listing.unavailable === null ? null : (
            <div className="mt-1 text-hint text-amber" data-testid="gpu-unavailable">
              No GPU could be read ({listing.unavailable.code}): {listing.unavailable.message}
            </div>
          )}
        </>
      )}
    </Field>
  );
}

/* ------------------------------------------------------------------------------------ *
 * Optimizer and batch configuration - R-02.13
 * ------------------------------------------------------------------------------------ */

export interface OptimizerFieldsProps {
  readonly learningRate: string;
  readonly warmupSteps: string;
  readonly batchSize: string;
  readonly gradientAccumulation: string;
  readonly optimizerStatePrecision: string;
  readonly seed: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
}

export function OptimizerFields(props: OptimizerFieldsProps): ReactNode {
  const { onChange } = props;
  const rows: readonly {
    readonly key: keyof OptimizerFieldsProps;
    readonly label: string;
    readonly hint: string;
  }[] = [
    {
      key: 'learningRate',
      label: 'Learning rate',
      hint: 'Required. There is no default, so the run log records only a rate you chose.',
    },
    { key: 'warmupSteps', label: 'Warmup steps', hint: 'In steps.' },
    { key: 'batchSize', label: 'Batch size', hint: 'Examples per step.' },
    {
      key: 'gradientAccumulation',
      label: 'Gradient accumulation',
      hint: 'Micro-batches per step. Left empty, the server uses its default.',
    },
    {
      key: 'seed',
      label: 'Seed',
      hint: 'Recorded with the run, so a create run states the seed its starting weights came from.',
    },
  ];

  return (
    <>
      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-2">
        {rows.map((row) => (
          <Field key={row.key} label={row.label} hint={row.hint}>
            {({ id, describedBy }) => (
              <Input
                id={id}
                aria-describedby={describedBy}
                inputMode="decimal"
                entry="number"
                value={props[row.key] as string}
                onChange={(event) => onChange({ [row.key]: event.target.value })}
                data-testid={`optimizer-${row.key}`}
              />
            )}
          </Field>
        ))}
      </div>
      <Field
        label="Optimizer state precision"
        hint="8-bit optimizer states use less GPU memory than 32-bit. Left empty, the server uses its default."
      >
        {({ id, describedBy }) => (
          <Select
            id={id}
            aria-describedby={describedBy}
            value={props.optimizerStatePrecision}
            onChange={(event) => onChange({ optimizerStatePrecision: event.target.value })}
            placeholder="Choose a precision"
            closedEnum="the two optimizer-state precisions the run configuration accepts"
            options={[
              { value: '8bit', label: '8-bit' },
              { value: '32bit', label: '32-bit' },
            ]}
            data-testid="optimizer-precision"
          />
        )}
      </Field>
    </>
  );
}

/* ------------------------------------------------------------------------------------ *
 * Cadences, the floor, and the save cadence
 * ------------------------------------------------------------------------------------ */

export interface CadenceFieldsProps {
  readonly testEvery: string;
  readonly retentionEvery: string;
  readonly saveEvery: string;
  readonly retentionFloor: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
  /** The pack's floor, when a pack is chosen. Shown and **said to be the pack's**. */
  readonly packFloor?: number | null;
}

export function CadenceFields({
  testEvery,
  retentionEvery,
  saveEvery,
  retentionFloor,
  onChange,
  packFloor,
}: CadenceFieldsProps): ReactNode {
  return (
    <>
      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-3">
        <Field
          label="Test every"
          hint="In steps. A larger value means fewer points on the test curve."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="numeric"
              value={testEvery}
              onChange={(event) => onChange({ testEvery: event.target.value })}
              entry="number"
              data-testid="cadence-test"
            />
          )}
        </Field>
        <Field
          label="Retention every"
          hint="In steps. Checked every N steps, so a drop below the floor can go unseen for up to N−1 steps."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="numeric"
              value={retentionEvery}
              onChange={(event) => onChange({ retentionEvery: event.target.value })}
              entry="number"
              data-testid="cadence-retention"
            />
          )}
        </Field>
        <Field
          label="Save every"
          hint="In steps. A crash loses at most the steps since the last save."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="numeric"
              value={saveEvery}
              onChange={(event) => onChange({ saveEvery: event.target.value })}
              entry="number"
              data-testid="cadence-save"
            />
          )}
        </Field>
      </div>

      <Field
        label="Retention floor"
        hint={
          packFloor === undefined || packFloor === null
            ? 'Optional. Left empty, the run uses the pack’s floor. A value here overrides it for this run only, and the run’s scores record which floor was used.'
            : `Optional. Left empty, the run uses the pack’s floor, ${packFloor.toFixed(3)}. A value here overrides it for this run only, and the run’s scores record which floor was used.`
        }
      >
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            inputMode="decimal"
            value={retentionFloor}
            /* Deliberately NOT prefilled with the pack's floor. Showing the pack's value is
               information; putting it in the input is this form choosing an override the
               operator did not ask for. The override is wired end to end (definition ->
               launch -> scoring's `resolve_floor`, which records the layer in force). */
            onChange={(event) => onChange({ retentionFloor: event.target.value })}
            entry="number"
            data-testid="retention-floor"
          />
        )}
      </Field>
    </>
  );
}

/* ------------------------------------------------------------------------------------ *
 * Stop conditions - at least one, and the form does not enforce that
 * ------------------------------------------------------------------------------------ */

export function stopConditionOptions(registry: StopConditionRegistry): readonly PickerOption[] {
  return registry.conditions.map((entry) => ({
    value: entry.condition,
    label: entry.condition,
    detail:
      `${entry.summary} · ` +
      (entry.required_parameters.length === 0
        ? 'takes no parameters'
        : `takes ${entry.required_parameters.join(', ')}`),
  }));
}

/** A refusal's `details.available`, for when the registry itself could not be read. */
function fromAvailable(names: readonly string[]): StopConditionRegistry {
  return {
    conditions: names.map((condition, rank) => ({
      condition,
      configured: false,
      required_parameters: [],
      precedence_rank: rank,
      summary: 'the list of stop conditions could not be loaded',
    })),
    count: names.length,
    closed: true,
    precedence: names,
    minimum_enabled: 1,
  };
}

export interface StopConditionsFieldProps {
  readonly entries: readonly StopConditionEntry[];
  readonly onChange: (entries: readonly StopConditionEntry[]) => void;
  /** `details.available` from a `run_has_no_stop_condition` refusal, when one has arrived. */
  readonly available?: readonly string[];
  /** Injected in tests. The real one reads `GET /api/v1/stop-conditions`. */
  readonly load?: typeof fetchStopConditionRegistry;
}

export function StopConditionsField({
  entries,
  onChange,
  available,
  load = fetchStopConditionRegistry,
}: StopConditionsFieldProps): ReactNode {
  const registry = useListing(load, stopConditionOptions);
  // The live registry when it answered; the refusal's own list when only that is in hand.
  // **No hard-coded list**: until task 14.0 this field offered `DOCUMENTED_STOP_CONDITIONS`
  // while the route F6 added for it had no reader.
  const known: StopConditionRegistry | null =
    registry.value ?? (available === undefined ? null : fromAvailable(available));
  const listing: PickerListing =
    registry.value === null && known !== null
      ? { kind: 'ready', options: stopConditionOptions(known) }
      : registry.listing;
  const specFor = (condition: string): RegistryEntry | undefined =>
    known?.conditions.find((row) => row.condition === condition);

  const replace = (index: number, next: StopConditionEntry): void =>
    onChange(entries.map((row, at) => (at === index ? next : row)));

  return (
    <Card className="mb-4" data-testid="stop-conditions">
      <div className="mb-1 text-heading">Stop conditions</div>
      <div className="mb-2 text-hint text-dim">
        A run needs <strong>at least one</strong>. Every parameter a condition takes is required and
        has no default.
      </div>

      {entries.length === 0 ? (
        <div className="text-hint text-dim" data-testid="no-stop-conditions">
          No stop condition added yet.
        </div>
      ) : (
        <ul className="stack-y-3">
          {entries.map((entry, index) => {
            const spec = specFor(entry.condition);
            return (
              <li key={index} className="rounded-control border border-line p-3">
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_auto]">
                  <Picker
                    value={entry.condition}
                    onChange={(condition) => replace(index, { condition, params: {} })}
                    listing={listing}
                    placeholder="Choose a condition"
                    empty="No stop conditions are available."
                    surface="the stop-condition list"
                    data-testid={`stop-condition-${index}`}
                  />
                  <Button onClick={() => onChange(entries.filter((_, at) => at !== index))}>
                    Remove this condition
                  </Button>
                </div>
                {spec === undefined ? null : spec.required_parameters.length === 0 ? (
                  <div className="mt-2 text-hint text-dim">This condition takes no parameters.</div>
                ) : (
                  <div className="mt-2 grid grid-cols-1 gap-x-4 sm:grid-cols-2">
                    {spec.required_parameters.map((name) => (
                      <StopParameter
                        key={name}
                        name={name}
                        index={index}
                        value={entry.params[name] ?? ''}
                        choices={name === 'metric' ? spec.metrics : undefined}
                        onChange={(raw) =>
                          replace(index, { ...entry, params: { ...entry.params, [name]: raw } })
                        }
                      />
                    ))}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}

      <div className="mt-3">
        <Button onClick={() => onChange([...entries, { condition: '', params: {} }])}>
          Add a stop condition
        </Button>
      </div>
    </Card>
  );
}

/** One registry-named parameter: a picker when the registry lists its values, else a number. */
function StopParameter({
  name,
  index,
  value,
  choices,
  onChange,
}: {
  readonly name: string;
  readonly index: number;
  readonly value: string;
  readonly choices: readonly string[] | undefined;
  readonly onChange: (value: string) => void;
}): ReactNode {
  const testId = `stop-param-${index}-${name}`;
  return (
    <Field label={name} hint="Required by this condition. No default.">
      {({ id, describedBy }) =>
        choices === undefined ? (
          <Input
            id={id}
            aria-describedby={describedBy}
            inputMode="decimal"
            entry="number"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            data-testid={testId}
          />
        ) : (
          <Picker
            id={id}
            describedBy={describedBy}
            value={value}
            onChange={onChange}
            listing={{
              kind: 'ready',
              options: choices.map((choice) => ({ value: choice, label: choice })),
            }}
            placeholder={`Choose a ${name}`}
            empty="No values are available."
            surface={`the ${name} list`}
            data-testid={testId}
          />
        )
      }
    </Field>
  );
}

/* ------------------------------------------------------------------------------------ *
 * The adapter block - only for an adapter run
 * ------------------------------------------------------------------------------------ */

export interface AdapterBlockFieldsProps {
  readonly rank: string;
  readonly alpha: string;
  readonly targetModules: string;
  readonly dropout: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
}

export function AdapterBlockFields({
  rank,
  alpha,
  targetModules,
  dropout,
  onChange,
}: AdapterBlockFieldsProps): ReactNode {
  return (
    <Card className="mb-4" data-testid="adapter-block">
      <div className="mb-1 text-heading">Adapter</div>
      <div className="mb-2 text-hint text-dim">
        Low-rank adapter (LoRA) settings. Rank, alpha and target modules are required. The
        adapter&apos;s scale is alpha divided by rank.
      </div>
      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-2">
        <Field label="Rank" hint="The adapter's inner dimension.">
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="numeric"
              value={rank}
              onChange={(event) => onChange({ adapterRank: event.target.value })}
              entry="number"
              data-testid="adapter-rank"
            />
          )}
        </Field>
        <Field label="Alpha" hint="The scale is alpha divided by rank.">
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="decimal"
              value={alpha}
              onChange={(event) => onChange({ adapterAlpha: event.target.value })}
              entry="number"
              data-testid="adapter-alpha"
            />
          )}
        </Field>
      </div>
      <Field label="Target modules" hint="Comma-separated module names, such as q_proj, v_proj.">
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            value={targetModules}
            onChange={(event) => onChange({ adapterTargetModules: event.target.value })}
            entry="id-fallback"
            entryReason="module names depend on the model's architecture."
            data-testid="adapter-target-modules"
          />
        )}
      </Field>
      <Field label="Dropout" hint="Optional. Left empty, it is left out rather than sent as zero.">
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            inputMode="decimal"
            value={dropout}
            onChange={(event) => onChange({ adapterDropout: event.target.value })}
            entry="number"
            data-testid="adapter-dropout"
          />
        )}
      </Field>
    </Card>
  );
}
