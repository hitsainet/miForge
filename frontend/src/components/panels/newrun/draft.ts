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
 * ## The definition is component state and is never persisted
 *
 * A persisted draft is **F7's** requirement (R-02.39), for pack authoring.
 */

/** Every field is a string: empty means **not entered**, which is not the same as zero. */
export interface RunDefinitionDraft {
  readonly mode: string;
  readonly path: string;
  readonly repo: string;
  readonly revision: string;
  readonly configPath: string;
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
  /** The threshold, step count or plateau window. Empty is empty. */
  readonly value: string;
}

/** Every field empty. The form's initial state, and the reason M19 has something to break. */
export const EMPTY_DRAFT: RunDefinitionDraft = {
  mode: '',
  path: '',
  repo: '',
  revision: '',
  configPath: '',
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
      const value = num(entry.value);
      // `config` is an opaque map on F6's side with **no shape and no defaults**, because
      // each condition validates its own parameters and refuses a missing one by name. The
      // single value this field collects is carried under the condition's own key when the
      // condition names one, and omitted otherwise - never filled with a guessed window or
      // threshold, which is the fabricated value that would be invisible in the record.
      const config =
        value === undefined ? {} : { [STOP_VALUE_KEYS[entry.condition.trim()] ?? 'value']: value };
      return { condition: entry.condition.trim(), ...(value === undefined ? {} : { config }) };
    });

  // `model_source` is F1's AcquisitionSpec, handed over opaquely. Omitted entirely when the
  // operator entered nothing, rather than sent as `{}` - an empty object is a model source
  // that names nothing, and F1 would have to refuse it as though one had been given.
  const modelSource = dropUndefined({
    mode: text(draft.mode),
    path: text(draft.path),
    repo: text(draft.repo),
    revision: text(draft.revision),
    config_path: text(draft.configPath),
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
 * Where a stop condition's single collected value belongs inside its own `config`.
 *
 * **Not a schema and not a default**: F6 declares no shape for `config` precisely so that no
 * module invents a `window` or a `threshold` with a value nobody chose. This table only says
 * which key the one number this field collects goes under, per condition, and a condition
 * absent from it gets the number under `value` so the backend can refuse it by name rather
 * than see nothing at all.
 */
export const STOP_VALUE_KEYS: Readonly<Record<string, string>> = {
  step_count: 'steps',
  loss_plateau: 'window',
  test_threshold: 'threshold',
  retention_floor: 'floor',
};

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

/**
 * The five stop conditions of R-02.25, in F6's own precedence order.
 *
 * **Not from a registry, and that is a gap rather than a shortcut.** `training.stop.names()`
 * exists in the backend and **no route publishes it** - the one stop-condition route is
 * `GET /api/v1/runs/{run_id}/stop-conditions`, which is run-scoped and so answers for a run
 * that already exists rather than for a definition being composed. Contrast run types, which
 * `GET /api/v1/run-types` serves from the live registry - which is why M22 is a mutation and
 * this is not. What F6 does publish is `details.available` **on its
 * `run_has_no_stop_condition` refusal**, so the field replaces this list with the refusal's
 * own the moment one arrives, and the list is marked in the UI as the documented order rather
 * than a live answer. Reported as a finding: a set an operator must choose from deserves an
 * endpoint, and `names()` is one line from being one.
 */
export const DOCUMENTED_STOP_CONDITIONS: readonly string[] = [
  'operator_stop',
  'retention_floor',
  'test_threshold',
  'loss_plateau',
  'step_count',
];
