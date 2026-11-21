/**
 * The part index on the Packs screen, with Archive / Unarchive (operator decision 2026-10-20).
 *
 * Parts are what a pack's slots pin. Until this section the only place an operator saw them
 * was inside a slot picker, so a test part could not be put away. **Archive hides a part from
 * this list and from every slot picker; it deletes nothing** - the files stay under the parts
 * root, `GET /parts/{id}` still answers, and a pack that already pins the part is untouched.
 * "Show archived parts" brings archived ones back, each marked, with Unarchive.
 */
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { MonoId } from '@/components/common/MonoId';
import { Pill } from '@/components/common/Pill';
import { FailureNotice } from '@/components/common/FailureNotice';
import { ShowArchivedToggle } from '@/components/common/ShowArchivedToggle';
import { formatEt } from '@/components/common/time';
import {
  PART_SOURCE_EXTERNAL,
  archivePart,
  fetchParts,
  unarchivePart,
  type PartRow,
} from '@/api/packs';
import type { ApiFailure } from '@/api/errors';
import { usePersistentFlag } from '@/hooks/usePersistentFlag';
import { DRAFT_KEYS } from '@/stores/draftStorage';
import {
  PurgeBlockedNote,
  PurgeDialog,
  PurgeHeader,
  PurgeItemCheckbox,
  usePurgeSelection,
} from '@/components/common/PurgeSelection';
import { previewPurge, purgeSelection } from '@/api/purge';

export interface PartIndexProps {
  /** Injected in tests. */
  readonly load?: typeof fetchParts;
  readonly archive?: typeof archivePart;
  readonly unarchive?: typeof unarchivePart;
  readonly preview?: typeof previewPurge;
  readonly purge?: typeof purgeSelection;
}

/** A part's name as the slot picker shows it: its reference, else its dataset and version. */
export function partName(row: PartRow): string {
  if (row.rendered !== null) return row.rendered;
  if (row.dataset === null) return row.id;
  return `${row.dataset}${row.version_number === null ? '' : ` v${row.version_number}`}`;
}

export function PartIndex({
  load = fetchParts,
  archive = archivePart,
  unarchive = unarchivePart,
  preview = previewPurge,
  purge = purgeSelection,
}: PartIndexProps = {}): ReactNode {
  const [showArchived, setShowArchived] = usePersistentFlag(DRAFT_KEYS.partsIndexArchived);
  const [rows, setRows] = useState<readonly PartRow[] | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const reload = useCallback(async () => {
    const result = await load({ includeArchived: showArchived });
    if (result.ok) {
      setRows(result.value.items);
      setFailure(null);
    } else {
      setFailure(result.failure);
    }
  }, [load, showArchived]);

  useEffect(() => {
    void reload();
  }, [reload]);

  // Permanent deletion of archived parts, shown with "Show archived parts" (operator
  // decision and request, 2026-10-20).
  const archivedIds = useMemo(
    () =>
      showArchived && rows !== null
        ? rows.filter((row) => (row.archived_at ?? null) !== null).map((row) => row.id)
        : [],
    [rows, showArchived],
  );
  const selection = usePurgeSelection('part', archivedIds, preview);
  const [purging, setPurging] = useState<readonly string[] | null>(null);

  const toggle = async (row: PartRow): Promise<void> => {
    const archived = (row.archived_at ?? null) !== null;
    setBusy(row.id);
    const result = await (archived ? unarchive(row.id) : archive(row.id));
    setBusy(null);
    if (!result.ok) {
      setFailure(result.failure);
      return;
    }
    await reload();
  };

  return (
    <section className="mt-6" aria-label="Parts" data-testid="part-index">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 className="font-semibold">Parts</h2>
        <ShowArchivedToggle
          checked={showArchived}
          onChange={setShowArchived}
          noun="parts"
          data-testid="part-show-archived"
        />
      </div>
      {failure === null ? null : (
        <FailureNotice
          failure={failure}
          surface="the part index"
          className="mb-3"
          data-testid="part-index-failure"
        />
      )}
      {rows === null ? (
        failure === null ? (
          <Card data-testid="part-index-loading">Reading the part index…</Card>
        ) : null
      ) : rows.length === 0 ? (
        <Card data-testid="part-index-empty">
          {showArchived
            ? 'No part is indexed. Import one (POST /api/v1/parts/import) or publish one from miDataworks.'
            : 'No part is listed. Import one (POST /api/v1/parts/import), publish one from miDataworks, or tick "Show archived parts".'}
        </Card>
      ) : (
        <>
          <PurgeHeader
            selection={selection}
            noun="parts"
            onDelete={() => setPurging(archivedIds.filter((id) => selection.selected.has(id)))}
          />
          <ul className="stack-y-2" data-testid="part-index-rows">
            {rows.map((row) => {
              const archivedAt = row.archived_at ?? null;
              return (
                <li
                  key={row.id}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-card border border-line bg-card p-card"
                  data-testid={`part-row-${row.id}`}
                >
                  {archivedAt === null ? null : (
                    <PurgeItemCheckbox selection={selection} id={row.id} label={partName(row)} />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="font-medium">{partName(row)}</div>
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-hint text-dim">
                      <MonoId value={row.id} />
                      <span>{row.part_kind}</span>
                      <span>
                        {row.source === PART_SOURCE_EXTERNAL ? 'external part' : 'from miDataworks'}
                      </span>
                      <span>
                        {row.row_count === null
                          ? 'row count not recorded'
                          : `${row.row_count} rows`}
                      </span>
                      <span title={row.imported_at ?? undefined}>
                        {row.imported_at === null
                          ? 'import time not recorded'
                          : `imported ${formatEt(row.imported_at)}`}
                      </span>
                      {archivedAt === null ? null : (
                        <Pill tone="neutral" title={archivedAt} data-testid="part-archived">
                          {`Archived ${formatEt(archivedAt)}`}
                        </Pill>
                      )}
                    </div>
                    {archivedAt === null ? null : (
                      <PurgeBlockedNote selection={selection} id={row.id} />
                    )}
                  </div>
                  <Button
                    onClick={() => void toggle(row)}
                    disabled={busy === row.id}
                    title={
                      archivedAt === null
                        ? 'Hide this part from this list and the slot pickers. Deletes nothing; a pack pinning it is untouched.'
                        : 'Return this part to this list and the slot pickers'
                    }
                    data-testid={`part-archive-action-${row.id}`}
                  >
                    {archivedAt === null ? 'Archive' : 'Unarchive'}
                  </Button>
                </li>
              );
            })}
          </ul>
        </>
      )}
      {purging === null ? null : (
        <PurgeDialog
          kind="part"
          ids={purging}
          noun="parts"
          preview={preview}
          purge={purge}
          onClose={() => setPurging(null)}
          onDone={() => {
            selection.deselectAll();
            selection.refresh();
            void reload();
          }}
        />
      )}
    </section>
  );
}
