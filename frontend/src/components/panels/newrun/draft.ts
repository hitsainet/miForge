/**
 * The run definition the form collects, and the body it submits (FR-009.33, .39; M19).
 *
 * ## The body is `schemas/run_definition.py::RunDefinition`, read out of that module
 *
 * R-02.53's definition is **F6's envelope**, and its shape was read from the model rather
 * than derived from the requirement's prose: `{run_id?, model_source, run_type,
 * run_configuration, signal_source, pack | parts, scoring, retention_floor_override,
 * cadences, stop_conditions, refresh, adapter, domain, seed, gpu_request}`, closed to unknown
 * keys. Re-deriving a field list from a requirement whose owner has already tabulated it is
 * how two documents end up with two lists.
 *
 * **It posts to `POST /api/v1/runs/define`, not `POST /api/v1/runs`** - F1's route is the
 * acquisition half and R-02.53 forbids that shape for a run definition.
 *
 * ## No field has a default. Not one.
 *
 * Every value is a string that starts **empty**, and `buildBody` **omits** every empty one -
 * including the nested objects, which are omitted entirely rather than sent as `{}`. Not a
 * batch size, not a cadence, not a stop condition, not a rank, not even the acquisition mode
 * the backend marks `default: true`. ADR-018: *a default on a submission form is a value the
 * operator never chose, and the run log will record it as theirs* - worse than a refusal
 * because it is silent and permanent. That is mutation **M19**.
 *
 * **Two defaults are the backend's and are left to it**: `Cadences.test_every` and
 * `retention_every` carry declared defaults in F6's own model, on the stated ground that *"a
 * cadence may have a declared default, because it changes only when a recorded action happens
 * and the record says when it happened"* - while anything deciding *whether a run ends* has
 * none. This form supplies neither, so an unfilled cadence is F6's default recorded as F6's,
 * not this form's guess recorded as the operator's.
 *
 * ## There is no client-side validation here, and that is the design
 *
 * No required-field check, no range check, no "at least one stop condition", no
 * pack-or-parts check - even though F6's model refuses when both or neither are given. The
 * refusals are the backend's copy and *a second validator in the frontend is a second answer*.
 *
 * ## The definition is kept in this browser until it is submitted (task 13.0)
 *
 * *Until 2026-10-17 it was component state only, and a reload lost every answer.* It is now
 * held by `usePersistentState` under `DRAFT_KEYS.newRun`, read back through
 * {@link decodeDraft}, and removed on a successful submit and on "Clear every field". **That is
 * browser-local and is not the staged or server-side draft task 9A.7 forbids**: nothing reaches
 * the backend before the one `POST /runs/define`, so R-02.53's whole-definition guarantee is
 * untouched. A restored draft is announced on screen, so it is never read as a default.
 */

/** Every field is a string: empty means **not entered**, which is not the same as zero. */
export interface RunDefinitionDraft {
  readonly mode: string;
  readonly path: string;
  readonly repo: string;
  readonly revision: string;
  readonly configPath: string;
  /** A create run's tokenizer: a folder on this host, or a Hub repository and revision. */
  readonly tokenizerPath: string;
  readonly tokenizerRepo: string;
  readonly tokenizerRevision: string;
  readonly runType: string;
  readonly packId: string;
  readonly packVersion: string;
  readonly signalSource: string;
  readonly learningRate: string;
  readonly warmupSteps: string;
  readonly batchSize: string;
  readonly gradientAccumulation: string;
  readonly optimizerStatePrecision: string;
  readonly seed: string;
  readonly testEvery: string;
  readonly retentionEvery: string;
  readonly saveEvery: string;
  readonly retentionFloor: string;
  readonly stopConditions: readonly StopConditionEntry[];
  readonly adapterRank: string;
  readonly adapterAlpha: string;
  readonly adapterTargetModules: string;
  readonly adapterDropout: string;
  readonly runId: string;
  readonly gpuRequest: string;
}

export interface StopConditionEntry {
  readonly condition: string;
  /**
   * The condition's parameters, **keyed by the registry's own names** - `GET
   * /api/v1/stop-conditions`'s `required_parameters` (task 14.0). Empty is empty.
   *
   * *Until 2026-10-18 this was one `value`, filed under a hand-kept key table that was wrong
   * for two of the five conditions: `step_count` was sent `{steps}` while the condition reads
   * `total`, so no step-count run could be submitted from this form, and `retention_floor`
   * was sent a `floor` it does not take. The table's own test asserted the wrong keys.*
   */
  readonly params: Readonly<Record<string, string>>;
}

/** Every field empty. The form's initial state, and the reason M19 has something to break. */
export const EMPTY_DRAFT: RunDefinitionDraft = {
  mode: '',
  path: '',
  repo: '',
  revision: '',
  configPath: '',
  tokenizerPath: '',
  tokenizerRepo: '',
  tokenizerRevision: '',
  runType: '',
  packId: '',
  packVersion: '',
  signalSource: '',
  learningRate: '',
  warmupSteps: '',
  batchSize: '',
  gradientAccumulation: '',
  optimizerStatePrecision: '',
  seed: '',
  testEvery: '',
  retentionEvery: '',
  saveEvery: '',
  retentionFloor: '',
  stopConditions: [],
  adapterRank: '',
  adapterAlpha: '',
  adapterTargetModules: '',
  adapterDropout: '',
  runId: '',
  gpuRequest: '',
};

/** The stored shape's version. Bump when `RunDefinitionDraft` changes incompatibly. */
export const DRAFT_VERSION = 1;

/**
 * A stored draft, or `null` when it is not one (task 13.3). Lenient within a version: a key
 * that is missing or mistyped - a field the secret rule dropped, say - stays empty, and a key
 * this build does not know is ignored. Anything that is not an object is refused outright.
 */
export function decodeDraft(data: unknown): RunDefinitionDraft | null {
  if (data === null || typeof data !== 'object' || Array.isArray(data)) return null;
  const record = data as Readonly<Record<string, unknown>>;
  const out: Record<string, unknown> = { ...EMPTY_DRAFT };
  for (const key of Object.keys(EMPTY_DRAFT) as (keyof RunDefinitionDraft)[]) {
    if (key === 'stopConditions') continue;
    const value = record[key];
    if (typeof value === 'string') out[key] = value;
  }
  const stops = record['stopConditions'];
  if (Array.isArray(stops)) {
    out['stopConditions'] = stops
      .filter(
        (entry): entry is Readonly<Record<string, unknown>> =>
          entry !== null &&
          typeof entry === 'object' &&
          typeof (entry as Record<string, unknown>)['condition'] === 'string',
      )
      .map((entry) => decodeStopEntry(entry));
  }
  return out as unknown as RunDefinitionDraft;
}

/**
 * A stored stop condition, including one saved before task 14.0 as `{condition, value}`.
 *
 * The old single value is carried onto the parameter its condition **actually reads** - only
 * for the two conditions that take exactly one (`step_count` → `total`, `test_threshold` →
 * `threshold`). For any other condition it is dropped rather than filed under a guessed name,
 * which is the defect the old key table was.
 */
export const LEGACY_SINGLE_PARAMETER: Readonly<Record<string, string>> = {
  step_count: 'total',
  test_threshold: 'threshold',
};

function decodeStopEntry(entry: Readonly<Record<string, unknown>>): StopConditionEntry {
  const condition = String(entry['condition']);
  const params: Record<string, string> = {};
  const stored = entry['params'];
  if (stored !== null && typeof stored === 'object' && !Array.isArray(stored)) {
    for (const [key, value] of Object.entries(stored as Record<string, unknown>)) {
      if (typeof value === 'string') params[key] = value;
    }
  }
  const legacy = entry['value'];
  const legacyKey = LEGACY_SINGLE_PARAMETER[condition];
  if (typeof legacy === 'string' && legacy.length > 0 && legacyKey !== undefined) {
    params[legacyKey] = params[legacyKey] ?? legacy;
  }
  return { condition, params };
}

function num(value: string): number | undefined {
  if (value.trim().length === 0) return undefined;
  const parsed = Number(value);
  // NaN is **not** coerced away and not replaced: an unparseable number is submitted as the
  // string the operator typed, so the backend refuses it naming the field rather than this
  // form silently dropping an answer that was given.
  return Number.isFinite(parsed) ? parsed : undefined;
}

function text(value: string): string | undefined {
  const trimmed = value.trim();
  return trimmed.length === 0 ? undefined : trimmed;
}

/**
 * One body, carrying **only what was entered** (FR-009.34, M19).
 *
 * `undefined` values are dropped by `JSON.stringify`, and this function never substitutes a
 * value for one. The keys are those of the live `RunSubmit` model where it has them, plus
 * R-02.53's remaining parts under the names F6's FR-006.56 tabulates - submitted **in the
 * same body**, because R-02.53's guarantee is that the *whole* definition is validated before
 * anything starts and a form submitting in pieces breaks exactly that.
 *
 * **This build's `POST /api/v1/runs` is closed to unknown keys** (`RunSubmit` sets
 * `extra="forbid"` and carries none of the loop parts), so a complete definition is refused
 * with `validation_failed` naming each unexpected field until F6 extends the route. That
 * refusal is rendered. The alternative - sending only the subset the route accepts - would
 * discard answers the operator gave, which is the one thing this form may never do.
 */
export function buildBody(draft: RunDefinitionDraft): Readonly<Record<string, unknown>> {
  const stopConditions = draft.stopConditions
    .filter((entry) => entry.condition.trim().length > 0)
    .map((entry) => {
      // `config` is an opaque map on F6's side with **no shape and no defaults**, because
      // each condition validates its own parameters and refuses a missing one by name. Each
      // parameter goes under the name the registry gave it, and an empty one is omitted -
      // never filled with a guessed window or threshold, which is the fabricated value that
      // would be invisible in the record.
      const config = stopConfig(entry.params);
      return {
        condition: entry.condition.trim(),
        ...(Object.keys(config).length === 0 ? {} : { config }),
      };
    });

  // `model_source` is F1's AcquisitionSpec, handed over opaquely. Omitted entirely when the
  // operator entered nothing, rather than sent as `{}` - an empty object is a model source
  // that names nothing, and F1 would have to refuse it as though one had been given.
  //
  // The tokenizer goes **only with a create**, because only a create shows the fields: a load
  // brings its own tokenizer from its folder and the server refuses one named beside it. A
  // value entered under create and left behind on a switch to load is not sent - the same
  // rule as the adapter block: never hidden-but-submitted.
  const create = text(draft.mode) === 'create';
  const modelSource = dropUndefined({
    mode: text(draft.mode),
    path: text(draft.path),
    repo: text(draft.repo),
    revision: text(draft.revision),
    config_path: text(draft.configPath),
    tokenizer_path: create ? text(draft.tokenizerPath) : undefined,
    tokenizer_repo: create ? text(draft.tokenizerRepo) : undefined,
    tokenizer_revision: create ? text(draft.tokenizerRevision) : undefined,
  });

  // `run_configuration` is F1's RunConfig (R-02.13). Same rule.
  const runConfiguration = dropUndefined({
    learning_rate: num(draft.learningRate),
    warmup_steps: num(draft.warmupSteps),
    batch_size: num(draft.batchSize),
    gradient_accumulation: num(draft.gradientAccumulation),
    optimizer_state_precision: text(draft.optimizerStatePrecision),
    seed: num(draft.seed),
  });

  // `signal_source` is F2's SourceSpec, whose one required field is `source`.
  const signalSource = dropUndefined({ source: text(draft.signalSource) });

  const cadences = dropUndefined({
    test_every: num(draft.testEvery),
    retention_every: num(draft.retentionEvery),
    save_every: num(draft.saveEvery),
  });

  const adapter = dropUndefined({
    rank: num(draft.adapterRank),
    alpha: num(draft.adapterAlpha),
    target_modules: splitModules(draft.adapterTargetModules),
    dropout: num(draft.adapterDropout),
  });

  const pack =
    text(draft.packId) === undefined
      ? undefined
      : dropUndefined({ pack_id: text(draft.packId), version: text(draft.packVersion) });

  return dropUndefined({
    run_id: text(draft.runId),
    model_source: empty(modelSource),
    run_type: text(draft.runType),
    run_configuration: empty(runConfiguration),
    signal_source: empty(signalSource),
    pack,
    // `retention_floor_override`, not `retention_floor`: F6 names it an OVERRIDE because the
    // pack carries the canonical floor. Whether a run may override it is F7's OQ-8 and is
    // not decided here - the field is offered, empty, and the backend decides.
    retention_floor_override: num(draft.retentionFloor),
    cadences: empty(cadences),
    stop_conditions: stopConditions.length === 0 ? undefined : stopConditions,
    adapter: empty(adapter),
    gpu_request: text(draft.gpuRequest),
  });
}

/**
 * A stop condition's parameters as F6's `config`: a number where the operator typed one (a
 * total, a threshold, a window), the text otherwise (`loss_plateau`'s `metric`). Empty is
 * omitted. Nothing is defaulted, and no name is invented: the keys are the registry's.
 */
export function stopConfig(
  params: Readonly<Record<string, string>>,
): Readonly<Record<string, number | string>> {
  const out: Record<string, number | string> = {};
  for (const [key, raw] of Object.entries(params)) {
    const trimmed = raw.trim();
    if (trimmed.length === 0) continue;
    const parsed = Number(trimmed);
    out[key] = Number.isFinite(parsed) ? parsed : trimmed;
  }
  return out;
}

/** An object with no keys is omitted, never sent as `{}`. See `buildBody`. */
function empty(
  record: Readonly<Record<string, unknown>>,
): Readonly<Record<string, unknown>> | undefined {
  return Object.keys(record).length === 0 ? undefined : record;
}

function splitModules(value: string): readonly string[] | undefined {
  const parts = value
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  return parts.length === 0 ? undefined : parts;
}

function dropUndefined(
  record: Readonly<Record<string, unknown>>,
): Readonly<Record<string, unknown>> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    if (value !== undefined) out[key] = value;
  }
  return out;
}
