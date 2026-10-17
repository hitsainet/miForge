/**
 * Import an exported pack document (F7 task 12.8; FR-007.15).
 *
 * **Schema errors are shown by name.** The backend validates and refuses with the failing
 * JSON Pointer and the schema keyword, and those are rendered rather than summarised - a
 * "could not import" with no pointer is the generic failure AC10 exists to forbid.
 *
 * **An imported pack whose parts this installation cannot resolve is still imported**, stored
 * and *reported* unresolvable: the backend never rewrites a reference to make it resolve,
 * because rewriting would make the imported pack a different pack, and AC10's claim is that
 * the imported version yields the **same** run inputs. So a 201 here is not "ready" and this
 * dialog renders the validity rather than closing on success.
 */
import { useState, type ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Field } from '@/components/common/Field';
import { ValidityPill } from '@/components/common/ValidityPill';
import { FailureNotice } from '@/components/common/FailureNotice';
import { MonoId } from '@/components/common/MonoId';
import { importPack } from '@/api/packs';
import type { ApiFailure } from '@/api/errors';
import type { PackListingRow } from '@/types/packs';

export interface ImportPackDialogProps {
  readonly onImported: () => void;
  readonly onClose: () => void;
}

export function ImportPackDialog({ onImported, onClose }: ImportPackDialogProps): ReactNode {
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [imported, setImported] = useState<PackListingRow | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  const take = async (file: File): Promise<void> => {
    const text = await file.text();
    let document: unknown;
    try {
      document = JSON.parse(text);
    } catch (cause) {
      // Named locally, because the file never reached the backend and reporting a server
      // refusal for a local parse failure would send the reader to the wrong place.
      setParseError(
        `${file.name} is not JSON (${cause instanceof Error ? cause.message : String(cause)}). ` +
          'An exported pack is the canonical bytes of the pack document; export it again if ' +
          'the file was edited.',
      );
      return;
    }
    setParseError(null);
    const result = await importPack(document);
    if (result.ok) {
      setImported(result.value);
      setFailure(null);
      onImported();
    } else {
      setImported(null);
      setFailure(result.failure);
    }
  };

  return (
    <Card className="mb-4" data-testid="import-pack">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-heading">Import a pack</div>
        <Button onClick={onClose}>Close this panel</Button>
      </div>
      <Field
        label="An exported pack document"
        hint="The canonical bytes definition_sha256 was taken over, so an export is verifiable by hashing what was received."
      >
        {({ id, describedBy }) => (
          <input
            id={id}
            aria-describedby={describedBy}
            type="file"
            accept="application/json,.json"
            className="w-full rounded-control border border-line bg-panel px-3 py-2.5 text-body"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file !== undefined) void take(file);
            }}
            data-testid="import-pack-file"
          />
        )}
      </Field>

      {parseError === null ? null : (
        <div role="alert" className="text-hint text-red" data-testid="import-parse-error">
          {parseError}
        </div>
      )}
      {failure === null ? null : <FailureNotice failure={failure} surface="importing that pack" />}
      {imported === null ? null : (
        <div
          className="mt-2 flex flex-wrap items-center gap-2 text-hint"
          data-testid="import-result"
        >
          <MonoId value={imported.pack_id} />
          <ValidityPill validity={imported.validity} />
          <span className="text-dim">
            Imported. References were not rewritten to make them resolve — a pack whose parts this
            installation cannot resolve is stored and reported unresolvable, because rewriting would
            make it a different pack.
          </span>
        </div>
      )}
    </Card>
  );
}
