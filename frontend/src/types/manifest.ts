/**
 * The two vendored manifest schemas, as a descriptor the manifest view can read
 * (FR-009.17, FR-009.18).
 *
 * ## Why a hand-written descriptor with a test, rather than an import
 *
 * `docs/schemas/` is outside `frontend/`, so neither Vite nor `resolveJsonModule` can reach
 * the files at build time without widening the dev server's filesystem allowance. The
 * descriptors below are therefore written out - and `manifest.test.ts` **parses the two
 * vendored files from disk and asserts this module agrees with them, key for key and
 * required-set for required-set.** That is the only arrangement in which a schema change
 * upstream turns this feature red instead of quietly making its "unrecognised key" marker
 * lie. A descriptor with no such test would be a second source of truth, which is the
 * failure this repository has recorded eleven times.
 *
 * ## Three things read out of the files that the documents get wrong or leave implicit
 *
 * 1. **`kind` is required in `adapter-manifest-v1.json` and NOT required in
 *    `checkpoint-manifest-v1.json`** (its `required` is the ten names in
 *    `CHECKPOINT_SCHEMA.required`, and `kind` is not among them). So a **conforming**
 *    checkpoint manifest may omit `kind` - and FR-009.17 rule 4 says the view "selects the
 *    schema from `kind`, and refuses rather than guessing when `kind` is unrecognised".
 *    An absent `kind` is therefore a third case, neither recognised nor unrecognised, and
 *    `selectSchema` returns a distinct outcome for it that says the omission is conforming.
 *    Reported as a finding; not resolved by guessing.
 * 2. **Both root objects set `additionalProperties: false`**, so an unrecognised key makes
 *    the document **nonconforming** - which strengthens rather than weakens FR-009.17
 *    rule 2's "render it and name it".
 * 3. **`ScoreRecord.required` is `[set, score, n]`** - `floor` and `provider` are optional -
 *    and **`MergedAdapter.required` is `[name, step, manifest_sha256]` with
 *    `manifest_sha256` nullable.** Required *and* nullable is three states, not two
 *    (E-12/E-12a), and several documents in this project collapsed the middle one.
 */

export type ManifestKind = 'miforge.checkpoint-manifest' | 'miforge.adapter-manifest';

export interface ObjectSchema {
  /** The schema's own property order. The view renders in this order, then the remainder. */
  readonly properties: readonly string[];
  readonly required: readonly string[];
  /** `false` on every object in both files; an extra key is nonconforming. */
  readonly closed: boolean;
  /** Properties whose own value is one of the nested objects below. */
  readonly nested?: Readonly<Record<string, keyof typeof NESTED_SCHEMAS>>;
}

export const NESTED_SCHEMAS = {
  // Property lists are in sorted order, which is a *rendering* order chosen here: the two
  // vendored files declare `ScoreRecord`'s members in two different orders (the checkpoint
  // manifest's `provider` is last, the adapter manifest's is third), so there is no single
  // declaration order to copy. `manifest.test.ts` therefore compares the property lists as
  // **sets** and the required lists as sets too, which is what JSON Schema actually means.
  ScoreRecord: {
    properties: ['floor', 'n', 'provider', 'score', 'set'],
    required: ['set', 'score', 'n'],
    closed: true,
  },
  MergedAdapter: {
    properties: ['manifest_sha256', 'name', 'step'],
    // `manifest_sha256` is required AND nullable - three states, not two: a hash
    // (verified), present-and-null (conforming, deliberately unverified), and absent (a
    // missing required key, nonconforming). See `MergedVerification` in `types/tree.ts`.
    required: ['name', 'step', 'manifest_sha256'],
    closed: true,
  },
  OptimizerIdentity: {
    properties: ['config_hash', 'name', 'state'],
    // `state`, not `config_hash`. Read out of the file rather than recalled: the obvious
    // guess is wrong and would have reported a conforming manifest's absent `config_hash`
    // as a missing required key.
    required: ['name', 'state'],
    closed: true,
  },
  AdapterBase: {
    properties: ['manifest_sha256', 'name'],
    required: ['name'],
    closed: true,
  },
  AdapterConfig: {
    properties: ['alpha', 'dropout', 'rank', 'scaling', 'target_modules'],
    required: ['rank', 'alpha', 'target_modules'],
    closed: true,
  },
} as const satisfies Readonly<Record<string, Omit<ObjectSchema, 'nested'>>>;

export const CHECKPOINT_SCHEMA: ObjectSchema = {
  // Alphabetical, because that is the order the vendored file declares them in.
  properties: [
    'base_model',
    'created_at',
    'domain',
    'grader_hash',
    'kind',
    'merged_adapter',
    'name',
    'optimizer',
    'parent',
    'retention',
    'run_id',
    'schema_version',
    'selector_hash',
    'signal_source',
    'step',
    'test',
    'weights_sha256',
  ],
  // `kind` is NOT here. See hint 1 in the module docstring.
  required: [
    'run_id',
    'step',
    'domain',
    'signal_source',
    'parent',
    'optimizer',
    'weights_sha256',
    'base_model',
    'created_at',
    'name',
  ],
  closed: true,
  nested: {
    test: 'ScoreRecord',
    retention: 'ScoreRecord',
    merged_adapter: 'MergedAdapter',
    optimizer: 'OptimizerIdentity',
  },
};

export const ADAPTER_SCHEMA: ObjectSchema = {
  properties: [
    'adapter',
    'base',
    'created_at',
    'kind',
    'name',
    'optimizer',
    'parts',
    'retention',
    'run_id',
    'schema_version',
    'step',
    'test',
    'weights_sha256',
  ],
  required: [
    'kind',
    'schema_version',
    'name',
    'run_id',
    'step',
    'base',
    'adapter',
    'weights_sha256',
    'created_at',
  ],
  closed: true,
  nested: {
    test: 'ScoreRecord',
    retention: 'ScoreRecord',
    optimizer: 'OptimizerIdentity',
    base: 'AdapterBase',
    adapter: 'AdapterConfig',
  },
};

export const SCHEMAS_BY_KIND: Readonly<Record<ManifestKind, ObjectSchema>> = {
  'miforge.checkpoint-manifest': CHECKPOINT_SCHEMA,
  'miforge.adapter-manifest': ADAPTER_SCHEMA,
};

/** The file each `kind` is normative in, named in the view so the reader can go and look. */
export const SCHEMA_FILES: Readonly<Record<ManifestKind, string>> = {
  'miforge.checkpoint-manifest': 'docs/schemas/checkpoint-manifest-v1.json',
  'miforge.adapter-manifest': 'docs/schemas/adapter-manifest-v1.json',
};

/**
 * Which schema describes this document, or why none could be chosen.
 *
 * **Three outcomes, not two** (FR-009.17 rule 4). `guessed` is not among them: the view
 * renders the raw document and names the schema it could not pick, because a manifest
 * rendered against the wrong schema marks real keys unrecognised and real omissions fine -
 * which is worse than rendering it unschematised and saying so. **M14** is the mutation
 * that guesses.
 */
export type SchemaSelection =
  | { readonly outcome: 'selected'; readonly kind: ManifestKind; readonly schema: ObjectSchema }
  | { readonly outcome: 'kind_absent'; readonly reason: string }
  | { readonly outcome: 'kind_unrecognised'; readonly given: string; readonly reason: string };

export function selectSchema(document: Readonly<Record<string, unknown>>): SchemaSelection {
  const kind = document['kind'];
  if (kind === undefined || kind === null) {
    return {
      outcome: 'kind_absent',
      reason:
        'This document carries no "kind", so no schema can be selected for it. Omitting ' +
        '"kind" is conforming for a checkpoint manifest - it is not in that schema\'s ' +
        'required list - so this is not necessarily a defect. Every key is rendered below ' +
        'unschematised: nothing here is marked unrecognised and no omission is reported.',
    };
  }
  if (typeof kind !== 'string') {
    return {
      outcome: 'kind_unrecognised',
      given: JSON.stringify(kind),
      reason:
        'This document\'s "kind" is not a string, so no schema can be selected for it. ' +
        'Set "kind" to one of the two miForge manifest kinds, or read the document raw.',
    };
  }
  if (kind in SCHEMAS_BY_KIND) {
    const narrowed = kind as ManifestKind;
    const schema = SCHEMAS_BY_KIND[narrowed];
    return { outcome: 'selected', kind: narrowed, schema };
  }
  return {
    outcome: 'kind_unrecognised',
    given: kind,
    // Named, not guessed. A document whose kind this build does not know may be a newer
    // artifact kind, and picking the nearest schema would report its new keys as errors.
    reason:
      `"${kind}" is not a manifest kind miForge reads. The two it reads are ` +
      `${Object.keys(SCHEMAS_BY_KIND).join(' and ')}. Every key is rendered below ` +
      'unschematised; no key is marked unrecognised and no omission is reported.',
  };
}

export type KeyStatus = 'schema' | 'unrecognised' | 'missing_required';

export interface ManifestRowModel {
  readonly key: string;
  readonly status: KeyStatus;
  /** Absent for a `missing_required` row - there is no value to show. */
  readonly value?: unknown;
  /** Present when the key's value is itself a described object. */
  readonly nested?: readonly ManifestRowModel[];
}

/**
 * Every key on disk, plus every required key that is not.
 *
 * Order: the schema's own properties first (so two manifests read the same way), then the
 * document's remaining keys marked `unrecognised`, then the absent required keys marked
 * `missing_required`. **Nothing is dropped** - dropping an unrecognised key is **M13**, and
 * it is how a nonconforming manifest comes to look clean.
 *
 * A **present-and-null** value is a `schema` row whose value is `null`; an **absent required
 * key** is a `missing_required` row with no value at all. The two are different facts -
 * `domain` is required *and* nullable, so null is an assertion and absence is an omission -
 * and the row model keeps them apart rather than relying on a renderer to.
 */
export function manifestRows(
  document: Readonly<Record<string, unknown>>,
  schema: ObjectSchema | null,
): readonly ManifestRowModel[] {
  const present = Object.keys(document);
  if (schema === null) {
    // Unschematised: every key, in the document's own order, none judged.
    return present.map((key) => ({ key, status: 'schema' as const, value: document[key] }));
  }

  const rows: ManifestRowModel[] = [];
  const seen = new Set<string>();

  for (const key of schema.properties) {
    if (!(key in document)) continue;
    seen.add(key);
    rows.push({
      key,
      status: 'schema',
      value: document[key],
      ...nestedRows(key, document[key], schema),
    });
  }
  for (const key of present) {
    if (seen.has(key)) continue;
    rows.push({ key, status: 'unrecognised', value: document[key] });
  }
  for (const key of schema.required) {
    if (key in document) continue;
    rows.push({ key, status: 'missing_required' });
  }
  return rows;
}

function nestedRows(
  key: string,
  value: unknown,
  schema: ObjectSchema,
): { readonly nested?: readonly ManifestRowModel[] } {
  const nestedName = schema.nested?.[key];
  if (nestedName === undefined) return {};
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return {};
  const nestedSchema = NESTED_SCHEMAS[nestedName];
  return {
    nested: manifestRows(value as Readonly<Record<string, unknown>>, {
      properties: nestedSchema.properties,
      required: nestedSchema.required,
      closed: nestedSchema.closed,
    }),
  };
}

/**
 * The things an operator will look for on this screen that the manifest **does not carry**,
 * each with its real source named (FR-009.18 rule 4).
 *
 * The score's `provider` **left this list on 2026-10-13**: `ScoreRecord` gained `provider`
 * in both vendored schemas, so it renders as an ordinary key and is read from disk like any
 * other. Labelling it "joined" would now be false - which is the mistake this list exists to
 * prevent, and it would have been committed by the rule itself.
 */
export const NOT_IN_MANIFEST: readonly { readonly what: string; readonly where: string }[] = [
  {
    what: 'The pack this artifact was trained from',
    where:
      'Not a manifest field, deliberately: recording pack or part references ' +
      "there would be a schema change in two repositories. A run's pack is on its run " +
      'record; an adapter\'s part references are on the adapter manifest\'s "parts".',
  },
  {
    what: "An adapter's domain and signal source",
    where:
      'adapter-manifest-v1.json has no "domain" and no "signal_source" property. The ' +
      "base's domain is not the adapter's and is not shown as though it were.",
  },
];
