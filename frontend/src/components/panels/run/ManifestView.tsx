/**
 * A manifest **as written on disk** (FR-009.17, FR-009.18).
 *
 * Fetched through F5's manifest endpoint, which reads the file and states in its own docstring
 * that *"the index is not consulted for this answer at all"* - which is what makes "every key
 * present is rendered" mean something: the view is looking at the artifact.
 *
 * ## The schema is selected from `kind`, and not guessed
 *
 * Three outcomes, not two (`types/manifest.ts::selectSchema`). Guessing when `kind` is
 * unrecognised is mutation **M14**, and it is worse than rendering unschematised: a manifest
 * read against the wrong schema marks real keys unrecognised and reports real omissions as
 * fine.
 *
 * **`kind` is not required in `checkpoint-manifest-v1.json`** - parsed, not recalled; its
 * `required` list is the ten names in `CHECKPOINT_SCHEMA.required` and `kind` is not among
 * them - so a **conforming** checkpoint manifest may omit it and then no schema can be
 * selected for it. That is a third case FR-009.17 rule 4 does not anticipate, and it is
 * rendered as what it is rather than forced into one of the other two.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { Callout } from '@/components/common/Callout';
import { MonoId } from '@/components/common/MonoId';
import { FailureNotice } from '@/components/common/FailureNotice';
import { fetchCheckpointManifest } from '@/api/checkpoints';
import type { ApiFailure } from '@/api/errors';
import {
  NOT_IN_MANIFEST,
  SCHEMA_FILES,
  manifestRows,
  selectSchema,
  type ManifestKind,
} from '@/types/manifest';
import { ManifestRow } from './ManifestRow';

export interface ManifestViewProps {
  readonly document: Readonly<Record<string, unknown>> | null;
  readonly manifestSha256?: string | null;
  /** Names a divergence between the index row and this document (FR-009.29, ADR-004). */
  readonly indexStale?: boolean;
}

export function ManifestView({
  document,
  manifestSha256,
  indexStale = false,
}: ManifestViewProps): ReactNode {
  if (document === null) {
    return (
      <Card data-testid="manifest-view-empty">
        <div className="text-hint text-dim">
          Select a save point or a tree node to read its manifest.
        </div>
      </Card>
    );
  }

  const selection = selectSchema(document);
  const schema = selection.outcome === 'selected' ? selection.schema : null;
  const rows = manifestRows(document, schema);

  return (
    <Card data-testid="manifest-view">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-heading">Manifest, as written on disk</div>
        {manifestSha256 === undefined || manifestSha256 === null ? null : (
          <MonoId value={manifestSha256} truncateTo={8} className="text-dim" />
        )}
      </div>

      {indexStale ? (
        <Callout tone="warning" title="The index row disagrees with this file" className="mb-3">
          The values below are the manifest&apos;s, which wins. The index row for this artifact is
          stale; the disagreement is logged. Run reindex to rebuild the index from the manifests.
        </Callout>
      ) : null}

      {selection.outcome === 'selected' ? (
        <div className="mb-3 text-hint text-dim" data-testid="manifest-schema">
          Read against <MonoId value={SCHEMA_FILES[selection.kind]} />.
        </div>
      ) : (
        <Callout
          tone="warning"
          title={
            selection.outcome === 'kind_absent'
              ? 'No schema could be selected: this document has no "kind"'
              : `No schema could be selected: "kind" is ${selection.given}`
          }
          className="mb-3"
          data-testid="manifest-schema-unselected"
        >
          {selection.reason}
        </Callout>
      )}

      <div className="rounded-control border border-line">
        {rows.map((row) => (
          <ManifestRow key={row.key} row={row} />
        ))}
      </div>

      <div className="mt-3">
        <div className="mb-1 text-hint text-dim">Not in the manifest</div>
        <ul className="stack-y-1 text-hint text-dim" data-testid="manifest-not-in">
          {NOT_IN_MANIFEST.map((entry) => (
            <li key={entry.what}>
              <span className="text-text">{entry.what}.</span> {entry.where}
            </li>
          ))}
        </ul>
      </div>
    </Card>
  );
}

/** Load one checkpoint's manifest. Thin: the view formats, this fetches. */
export function CheckpointManifestPanel({
  runId,
  step,
}: {
  readonly runId: string;
  readonly step: number | null;
}): ReactNode {
  const [document, setDocument] = useState<Readonly<Record<string, unknown>> | null>(null);
  const [sha, setSha] = useState<string | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  useEffect(() => {
    if (step === null) {
      setDocument(null);
      return;
    }
    let cancelled = false;
    void fetchCheckpointManifest(runId, step).then((result) => {
      if (cancelled) return;
      if (result.ok) {
        const manifest: unknown = result.value?.manifest;
        if (manifest === null || typeof manifest !== 'object' || Array.isArray(manifest)) {
          // A 200 with no manifest object is malformed, named - not a document of `undefined`,
          // which `selectSchema` dereferenced and took the whole app down with (found by
          // budgets.spec.ts, whose stub answers every `/checkpoints` URL with a listing).
          setDocument(null);
          setFailure({
            kind: 'malformed',
            status: 200,
            reason: 'the response carries no manifest object',
            body: JSON.stringify(result.value).slice(0, 2000),
          });
          return;
        }
        setDocument(manifest as Readonly<Record<string, unknown>>);
        setSha(result.value.manifest_sha256);
        setFailure(null);
      } else {
        setFailure(result.failure);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [runId, step]);

  if (failure !== null) {
    return <FailureNotice failure={failure} surface={`the manifest for step ${step}`} />;
  }
  return <ManifestView document={document} manifestSha256={sha} />;
}

export type { ManifestKind };
