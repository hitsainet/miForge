/**
 * The descriptor against the **vendored files** (FR-009.17, FR-009.18).
 *
 * This is the test that makes `types/manifest.ts` safe to hand-write: it parses
 * `docs/schemas/checkpoint-manifest-v1.json` and `adapter-manifest-v1.json` from disk and
 * asserts the descriptor agrees with them. Without it the descriptor is a second source of
 * truth, and a schema change upstream would quietly make the view's "unrecognised key" marker
 * lie rather than turning anything red.
 *
 * Property lists are compared as **sets** and required lists as sets too, because that is what
 * JSON Schema means - and because the two files genuinely declare `ScoreRecord`'s members in
 * two different orders, so there is no single declaration order to copy.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  ADAPTER_SCHEMA,
  CHECKPOINT_SCHEMA,
  NESTED_SCHEMAS,
  manifestRows,
  selectSchema,
} from './manifest';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const SCHEMA_DIR = path.join(REPO_ROOT, 'docs', 'schemas');

interface JsonSchema {
  readonly $defs: Record<
    string,
    {
      readonly properties?: Record<string, unknown>;
      readonly required?: readonly string[];
      readonly additionalProperties?: unknown;
    }
  >;
}

function load(file: string): JsonSchema {
  return JSON.parse(readFileSync(path.join(SCHEMA_DIR, file), 'utf8')) as JsonSchema;
}

const checkpoint = load('checkpoint-manifest-v1.json');
const adapter = load('adapter-manifest-v1.json');

const sorted = (values: Iterable<string>): string[] => [...values].sort();

describe('[vendored schemas]', () => {
  it('should find both files, so a passing comparison is not a comparison of nothing', () => {
    // The construction-green guard. A wrong path would otherwise throw at import and look
    // like an environment problem rather than an assertion about the contract.
    expect(Object.keys(checkpoint.$defs)).toContain('CheckpointManifestV1');
    expect(Object.keys(adapter.$defs)).toContain('AdapterManifestV1');
  });

  it('should agree with checkpoint-manifest-v1.json, property for property', () => {
    const root = checkpoint.$defs['CheckpointManifestV1']!;
    expect(sorted(Object.keys(root.properties ?? {}))).toEqual(
      sorted(CHECKPOINT_SCHEMA.properties),
    );
    expect(sorted(root.required ?? [])).toEqual(sorted(CHECKPOINT_SCHEMA.required));
    expect(root.additionalProperties).toBe(false);
    expect(CHECKPOINT_SCHEMA.closed).toBe(true);
  });

  it('should agree with adapter-manifest-v1.json, property for property', () => {
    const root = adapter.$defs['AdapterManifestV1']!;
    expect(sorted(Object.keys(root.properties ?? {}))).toEqual(sorted(ADAPTER_SCHEMA.properties));
    expect(sorted(root.required ?? [])).toEqual(sorted(ADAPTER_SCHEMA.required));
    expect(root.additionalProperties).toBe(false);
  });

  it('should agree with every nested definition in both files', () => {
    const pairs: readonly [keyof typeof NESTED_SCHEMAS, JsonSchema][] = [
      ['ScoreRecord', checkpoint],
      ['MergedAdapter', checkpoint],
      ['OptimizerIdentity', checkpoint],
      ['AdapterBase', adapter],
      ['AdapterConfig', adapter],
    ];
    for (const [name, file] of pairs) {
      const theirs = file.$defs[name];
      expect(theirs, `${name} is not in the vendored file`).toBeDefined();
      const ours = NESTED_SCHEMAS[name];
      expect(sorted(Object.keys(theirs?.properties ?? {})), `${name} properties`).toEqual(
        sorted(ours.properties),
      );
      expect(sorted(theirs?.required ?? []), `${name} required`).toEqual(sorted(ours.required));
      expect(theirs?.additionalProperties, `${name} is not closed`).toBe(false);
    }
  });

  it('should record that kind is required for an adapter manifest and NOT for a checkpoint', () => {
    // The finding, pinned. FR-009.17 rule 4 says the view selects the schema from `kind`, and
    // a CONFORMING checkpoint manifest may omit it - which is why `selectSchema` has a third
    // outcome. If a later re-vendor adds `kind` to the checkpoint's required list, this goes
    // red and says so, and the third outcome can then be reconsidered.
    expect(checkpoint.$defs['CheckpointManifestV1']?.required).not.toContain('kind');
    expect(adapter.$defs['AdapterManifestV1']?.required).toContain('kind');
  });

  it('should record that ScoreRecord carries provider and does NOT require it', () => {
    // F-8 closed: `provider` is on `ScoreRecord`. It is optional, so a score written before
    // the change has none - rendered through AbsentValue, never filled from F4's current
    // provider. A test is the only thing that catches the omission; the schema cannot.
    const score = checkpoint.$defs['ScoreRecord']!;
    expect(Object.keys(score.properties ?? {})).toContain('provider');
    expect(score.required).not.toContain('provider');
    expect(score.required).toEqual(['set', 'score', 'n']);
  });

  it('should record that MergedAdapter.manifest_sha256 is required AND nullable', () => {
    // Three states, not two (E-12 / E-12a). Several documents in this project collapsed the
    // middle one, so it is asserted rather than remembered.
    const merged = checkpoint.$defs['MergedAdapter']!;
    expect(merged.required).toContain('manifest_sha256');
    const property = (merged.properties ?? {})['manifest_sha256'] as {
      readonly anyOf?: readonly { readonly type?: string }[];
    };
    expect(property.anyOf?.map((entry) => entry.type)).toContain('null');
  });
});

describe('[selectSchema]', () => {
  it('should select the checkpoint schema from its kind', () => {
    const selection = selectSchema({ kind: 'miforge.checkpoint-manifest' });
    expect(selection.outcome).toBe('selected');
    expect(selection.outcome === 'selected' && selection.schema).toBe(CHECKPOINT_SCHEMA);
  });

  it('should select the adapter schema from its kind', () => {
    const selection = selectSchema({ kind: 'miforge.adapter-manifest' });
    expect(selection.outcome === 'selected' && selection.kind).toBe('miforge.adapter-manifest');
  });

  it('should refuse to pick one for an unrecognised kind, naming it', () => {
    // M14: guessing here renders a manifest against the wrong schema, which marks real keys
    // unrecognised and reports real omissions as fine.
    const selection = selectSchema({ kind: 'miforge.something-new' });
    expect(selection.outcome).toBe('kind_unrecognised');
    expect(selection.outcome === 'kind_unrecognised' && selection.given).toBe(
      'miforge.something-new',
    );
    expect(selection.outcome !== 'selected' && selection.reason).toContain(
      'miforge.checkpoint-manifest',
    );
  });

  it('should treat an absent kind as its own outcome and say the omission is conforming', () => {
    const selection = selectSchema({ run_id: 'r1', step: 1 });
    expect(selection.outcome).toBe('kind_absent');
    expect(selection.outcome !== 'selected' && selection.reason).toContain('conforming');
  });
});

describe('[manifestRows]', () => {
  const document = {
    run_id: 'r1',
    step: 400,
    domain: null,
    weights_sha256: 'a'.repeat(64),
    invented_key: 'hello',
  };

  it('should render every key present, including one the schema does not describe', () => {
    // M13: dropping an unrecognised key is how a nonconforming manifest comes to look clean.
    const rows = manifestRows(document, CHECKPOINT_SCHEMA);
    const unrecognised = rows.filter((row) => row.status === 'unrecognised');
    expect(unrecognised.map((row) => row.key)).toEqual(['invented_key']);
    expect(unrecognised[0]?.value).toBe('hello');
  });

  it('should distinguish a present-and-null key from an absent required one', () => {
    // `domain` is required AND nullable: null is an assertion, absence is an omission.
    const rows = manifestRows(document, CHECKPOINT_SCHEMA);
    const domain = rows.find((row) => row.key === 'domain');
    expect(domain?.status).toBe('schema');
    expect(domain?.value).toBeNull();

    const parent = rows.find((row) => row.key === 'parent');
    expect(parent?.status).toBe('missing_required');
    expect(parent && 'value' in parent).toBe(false);
  });

  it('should order the schemas own properties first and the remainder after', () => {
    const rows = manifestRows(document, CHECKPOINT_SCHEMA);
    const statuses = rows.map((row) => row.status);
    expect(statuses.indexOf('unrecognised')).toBeGreaterThan(statuses.indexOf('schema'));
    expect(statuses.indexOf('missing_required')).toBeGreaterThan(statuses.indexOf('unrecognised'));
  });

  it('should judge nothing when no schema could be selected', () => {
    const rows = manifestRows(document, null);
    expect(rows.every((row) => row.status === 'schema')).toBe(true);
    expect(rows.map((row) => row.key)).toEqual(Object.keys(document));
  });

  it('should describe a nested ScoreRecord against its own definition', () => {
    const rows = manifestRows(
      { test: { set: 'held-out@3', score: 0.84, n: 320, nonsense: 1 } },
      CHECKPOINT_SCHEMA,
    );
    const nested = rows.find((row) => row.key === 'test')?.nested ?? [];
    expect(nested.find((row) => row.key === 'nonsense')?.status).toBe('unrecognised');
    // `floor` and `provider` are optional, so their absence is NOT reported as missing.
    expect(nested.filter((row) => row.status === 'missing_required')).toEqual([]);
  });
});
