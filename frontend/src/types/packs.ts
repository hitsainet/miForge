/**
 * Part references, packs and drafts (F7 task 12.1; FR-007.24, FR-007.27).
 *
 * **`PartRef` is a discriminated union on this side too**, and that is the point of the
 * file. `backend/src/packs/references.py` makes `ContractPartRef` and `ExternalPartRef`
 * two distinct models joined by `Field(discriminator="source")`, with **no** `promote()`
 * and no `as_contract_reference()`, because R-02.40 forbids miForge claiming miDataworks
 * provenance for anything miDataworks did not publish. A client type that flattened the
 * two into `{ source: string; type?: string; origin?: string }` would reintroduce exactly
 * the confusion the backend spent a type on preventing: a component could read `type` off
 * an external part and render a contract reference for it.
 *
 * So: narrowing on `source` is **required** to read any field, and `PartReference.tsx`
 * switches on it exhaustively.
 */

/** miDataworks' citation form, `{type, name, version}` (BRD-03 R-03.47). */
export interface ContractPartRef {
  readonly source: 'midataworks';
  readonly type: string;
  readonly name: string;
  readonly version: string;
}

/**
 * A part from outside the suite, carrying **its own** provenance (R-02.40).
 *
 * The content hash is the pin. There is no ordinal, because `ExternalProvenance` is closed
 * on `{source, content_sha256, retrieved_at, licence}` and a pin that lived only in one
 * installation's database would not survive the export AC10 requires.
 */
export interface ExternalPartRef {
  readonly source: 'external';
  readonly origin: string;
  readonly content_sha256: string;
  readonly taken_in_at: string;
}

export type PartRef = ContractPartRef | ExternalPartRef;

/** Every `source` value the union admits. `PartReference.test.tsx` enumerates it. */
export const PART_REF_SOURCES = ['midataworks', 'external'] as const;
export type PartRefSource = (typeof PART_REF_SOURCES)[number];

/**
 * A runtime narrowing for a value that came off the wire.
 *
 * Needed because an older or hand-edited document can carry a `source` the union does not
 * admit, and the honest rendering of that is *"a reference whose source this build does not
 * recognise"* - not a crash and not a guess at which branch it meant.
 */
export function partRefSource(ref: PartRef | { readonly source?: unknown }): PartRefSource | null {
  const source = (ref as { readonly source?: unknown }).source;
  return source === 'midataworks' || source === 'external' ? source : null;
}

/* ------------------------------------------------------------------------------------ *
 * Packs
 * ------------------------------------------------------------------------------------ */

/** The seven slots of R-02.38, in `backend/src/packs/slots.py`'s `SLOT_ORDER`. */
export const SLOT_ORDER = [
  'prompt_set',
  'corpus',
  'preference_pairs',
  'selector_config',
  'grader_config',
  'test_set',
  'retention_set',
] as const;
export type SlotKind = (typeof SLOT_ORDER)[number];

/** The three R-02.38 requires for validity, in the order the refusal lists them. */
export const REQUIRED_SLOT_ORDER: readonly SlotKind[] = ['prompt_set', 'test_set', 'retention_set'];

export const SLOT_LABELS: Readonly<Record<SlotKind, string>> = {
  prompt_set: 'Prompt set',
  corpus: 'Corpus',
  preference_pairs: 'Preference pairs',
  selector_config: 'Selector configuration',
  grader_config: 'Grader configuration',
  test_set: 'Test set',
  retention_set: 'Retention set',
};

/** `packs/validation.py::PackValidity.payload`. A **200 describing invalidity**. */
export interface PackValidity {
  readonly state: string;
  readonly missing_slots: readonly string[];
  readonly unresolved: readonly {
    readonly slot: string;
    readonly reference: unknown;
    readonly reason: string;
  }[];
  readonly service_unavailable: readonly {
    readonly slot: string;
    readonly reference: unknown;
    readonly service: string;
    readonly reason: string;
  }[];
  readonly retention_floor_missing: boolean;
}

/** One row of `GET /api/v1/packs` - `packs/registry.py::RegistryRow.payload`. */
export interface PackListingRow {
  readonly pack_id: string;
  readonly version: string;
  readonly name: string;
  readonly domain: string | null;
  readonly parts_filled: number;
  readonly parts_total: number;
  /** `null` when it cannot be resolved - **never 0** (ADR-018, FR-007.11). */
  readonly prompt_count: number | null;
  readonly runs_using: readonly string[];
  readonly validity: PackValidity;
  readonly derived_from: string | null;
  readonly is_example: boolean;
  readonly definition_sha256: string;
  readonly created_at: string;
}

/** `GET /api/v1/packs`. */
export interface PackListing {
  readonly packs: readonly PackListingRow[];
}

/* ------------------------------------------------------------------------------------ *
 * Drafts - the seven-step authoring flow (R-02.39)
 * ------------------------------------------------------------------------------------ */

/** `packs/authoring.py::Draft.payload`. The flow's whole state. */
export interface DraftRecord {
  readonly draft_id: string;
  readonly name: string | null;
  readonly version: string;
  readonly domain: string | null;
  readonly current_step: number;
  readonly step_count: number;
  /** Keyed by the step number **as a string**, which is what the payload sends. */
  readonly step_states: Readonly<Record<string, string>>;
  readonly slots: Readonly<Record<string, unknown>>;
  readonly created_at: string;
  readonly updated_at: string;
  readonly last_dry_run: Readonly<Record<string, unknown>> | null;
  /** Added by the route, not by the store: `GET`/`PUT` return `{...payload, subtitle}`. */
  readonly subtitle?: string;
}

/** The seven steps, from `packs/authoring.py::STEPS`. Names are that module's. */
export const DRAFT_STEPS = [
  { number: 1, name: 'describe', slots: [], skippable: false, panel: null },
  { number: 2, name: 'prompts', slots: ['prompt_set'], skippable: false, panel: 'sample' },
  {
    number: 3,
    name: 'answers',
    slots: ['corpus', 'preference_pairs'],
    skippable: true,
    panel: 'sample',
  },
  {
    number: 4,
    name: 'judgment',
    slots: ['selector_config', 'grader_config'],
    skippable: true,
    panel: 'dry_run',
  },
  { number: 5, name: 'improvement', slots: ['test_set'], skippable: false, panel: 'scored_sample' },
  {
    number: 6,
    name: 'retention',
    slots: ['retention_set'],
    skippable: false,
    panel: 'scored_sample',
  },
  { number: 7, name: 'review', slots: [], skippable: false, panel: null },
] as const satisfies readonly {
  readonly number: number;
  readonly name: string;
  readonly slots: readonly SlotKind[];
  readonly skippable: boolean;
  readonly panel: 'sample' | 'dry_run' | 'scored_sample' | null;
}[];

/* ------------------------------------------------------------------------------------ *
 * The try and dry-run panels
 * ------------------------------------------------------------------------------------ */

/**
 * F3's `SelectionHealth`, **passed through whole** (`services/selectors/types.py`).
 *
 * All three counts and their denominator travel together, so the panel can print
 * `Accepted 3, rejected 1, undecided 1 of 5` and **cannot** print a bare percentage.
 * `acceptance_rate` is the server's; the panel computes no rate of its own (F7 task 10.1a).
 */
export interface SelectionHealth {
  readonly n: number;
  readonly accepted: number;
  readonly rejected: number;
  readonly undecided: number;
  readonly acceptance_rate: number;
  readonly reason_histogram: Readonly<Record<string, number>>;
  readonly grader_stats: {
    readonly n: number;
    readonly min: number;
    readonly max: number;
    readonly mean: number;
    readonly variance: number;
  } | null;
  /** Each signal arrives **with its sentence**; see `DegenerateWarning.tsx`. */
  readonly signals: readonly SelectionSignal[];
}

/** One named degenerate outcome. The `message` is F3's, supplied once (F3 FR-003.10). */
export interface SelectionSignal {
  readonly name: string;
  readonly message: string;
  readonly details: Readonly<Record<string, unknown>>;
}

/** The three signal names, verbatim from `services/selectors/types.py`. */
export const SIGNAL_ACCEPTS_EVERYTHING = 'selection_accepts_everything';
export const SIGNAL_REJECTS_EVERYTHING = 'selection_rejects_everything';
export const SIGNAL_NO_SPREAD = 'grader_scores_have_no_spread';
export const SELECTION_SIGNAL_NAMES = [
  SIGNAL_ACCEPTS_EVERYTHING,
  SIGNAL_REJECTS_EVERYTHING,
  SIGNAL_NO_SPREAD,
] as const;

/** One row of a dry-run - `packs/dryrun.py::DryRunResult.payload`'s `verdicts`. */
export interface DryRunVerdict {
  readonly prompt: string;
  readonly completion: string;
  readonly verdict: string;
  readonly reason_code: string;
  readonly reason: string | null;
  readonly score: number | null;
  readonly admitted: boolean;
}

/** `POST /api/v1/pack-drafts/{id}/dry-run` - `packs/dryrun.py::DryRunResult.payload`. */
export interface DryRunResult {
  readonly provider: Readonly<Record<string, unknown>>;
  readonly judge: Readonly<Record<string, unknown>> | null;
  readonly config_sha256: string;
  readonly config_kind: string;
  readonly hash_provenance: string;
  readonly health: SelectionHealth;
  readonly verdicts: readonly DryRunVerdict[];
  /** Set when every verdict is `undecided` because the configuration answers later. */
  readonly asynchronous_note: string | null;
  readonly advisory: boolean;
}

/** `POST /api/v1/pack-drafts/{id}/samples/{slot}` - `packs/samples.py::SlotSample.payload`. */
export interface SlotSample {
  readonly slot: string;
  readonly set: string;
  readonly n: number;
  readonly population: number;
  readonly seed: number;
  readonly prompts: readonly { readonly prompt: string; readonly reference: string | null }[];
}

/** `.../samples/{slot}/score` - `packs/samples.py::ScoredSample.payload`. */
export interface ScoredSample extends SlotSample {
  readonly score: number;
  readonly provider: string;
  readonly aggregation: string;
  readonly floor: number | null;
  /** `null` when no floor was proposed - *"is it breached"* has no answer without one. */
  readonly floor_breached: boolean | null;
}
