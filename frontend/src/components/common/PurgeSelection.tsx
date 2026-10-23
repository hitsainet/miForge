/**
 * Select archived items and delete them permanently (operator request 2026-10-20: *"I should
 * be able to select/de-select all or select/de-select individual items for permanent
 * deletion."*). Shared by the Runs list and the part index, shown only with "Show archived".
 *
 * - **Every archived item has a checkbox.** An item the backend says cannot be purged shows
 *   why, inline, and its checkbox is disabled - it can never be selected.
 * - **One header checkbox selects or deselects every selectable item**; it is indeterminate
 *   when some are selected, and the count beside it is live ("3 selected").
 * - **"Delete permanently" opens a confirmation** that re-reads the preview, lists every
 *   selected item with the rows, files and bytes that go with it, and lists separately any
 *   that became blocked since - those are skipped. Confirming sends **exactly** the remaining
 *   selected ids; the backend purges each on its own and answers per item.
 *
 * Nothing here decides what is purgeable: the preview route is the authority
 * (`services/purge/service.py`), so a blocker the backend knows is a blocker the screen shows.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Button } from '@/components/common/Button';
import { Card } from '@/components/common/Card';
import { Input } from '@/components/common/Field';
import { FailureNotice } from '@/components/common/FailureNotice';
import {
  formatBytes,
  previewPurge,
  purgeSelection,
  type PurgeBatch,
  type PurgeKind,
  type PurgePlan,
} from '@/api/purge';
import type { ApiFailure } from '@/api/errors';

/** Why an item cannot be selected, in one line; `null` when it can. */
export function blockedReason(plan: PurgePlan | undefined): string | null {
  if (plan === undefined) return null;
  if (plan.purgeable) return null;
  if (plan.not_archived !== null) return plan.not_archived.message;
  const [first, ...rest] = plan.blockers;
  if (first === undefined) return 'The backend reports this item cannot be purged.';
  return rest.length === 0 ? first.message : `${first.message} (+${rest.length} more)`;
}

export interface PurgeSelectionState {
  readonly kind: PurgeKind;
  /** Plans by id, from the preview. Missing while it loads. */
  readonly plans: Readonly<Record<string, PurgePlan>>;
  readonly loading: boolean;
  readonly failure: ApiFailure | null;
  /** The archived ids on screen, in order. */
  readonly ids: readonly string[];
  /** The ones that may be selected: on screen, previewed, and purgeable. */
  readonly selectable: readonly string[];
  readonly selected: ReadonlySet<string>;
  readonly toggle: (id: string) => void;
  readonly selectAll: () => void;
  readonly deselectAll: () => void;
  readonly refresh: () => void;
}

/**
 * Selection over the archived ids on screen, with the backend's preview for each.
 * `preview` is injected in tests.
 */
export function usePurgeSelection(
  kind: PurgeKind,
  ids: readonly string[],
  preview: typeof previewPurge = previewPurge,
): PurgeSelectionState {
  const [plans, setPlans] = useState<Readonly<Record<string, PurgePlan>>>({});
  const [loading, setLoading] = useState(false);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [revision, setRevision] = useState(0);
  const key = ids.join('\n');
  const latest = useRef(0);

  useEffect(() => {
    const wanted = key.length === 0 ? [] : key.split('\n');
    if (wanted.length === 0) {
      setPlans({});
      setFailure(null);
      return;
    }
    const ticket = latest.current + 1;
    latest.current = ticket;
    setLoading(true);
    void preview(kind, wanted).then((result) => {
      if (ticket !== latest.current) return;
      setLoading(false);
      if (!result.ok) {
        setFailure(result.failure);
        return;
      }
      setFailure(null);
      setPlans(Object.fromEntries(result.value.map((plan) => [plan.id, plan])));
    });
  }, [kind, key, preview, revision]);

  const selectable = useMemo(() => ids.filter((id) => plans[id]?.purgeable === true), [ids, plans]);

  // Never keep a selection the screen no longer offers.
  useEffect(() => {
    const allowed = new Set(selectable);
    setSelected((current) => {
      const kept = [...current].filter((id) => allowed.has(id));
      return kept.length === current.size ? current : new Set(kept);
    });
  }, [selectable]);

  const toggle = useCallback(
    (id: string) => {
      if (!selectable.includes(id)) return;
      setSelected((current) => {
        const next = new Set(current);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    },
    [selectable],
  );
  const selectAll = useCallback(() => setSelected(new Set(selectable)), [selectable]);
  const deselectAll = useCallback(() => setSelected(new Set()), []);
  const refresh = useCallback(() => setRevision((current) => current + 1), []);

  return {
    kind,
    plans,
    loading,
    failure,
    ids,
    selectable,
    selected,
    toggle,
    selectAll,
    deselectAll,
    refresh,
  };
}

export interface PurgeHeaderProps {
  readonly selection: PurgeSelectionState;
  /** "runs", "parts". */
  readonly noun: string;
  readonly onDelete: () => void;
}

/** The header row: select all / deselect all (indeterminate when partial), the count, and
 * the action. Renders nothing when no archived item is on screen. */
export function PurgeHeader({ selection, noun, onDelete }: PurgeHeaderProps): ReactNode {
  const box = useRef<HTMLInputElement>(null);
  const count = selection.selected.size;
  const total = selection.selectable.length;
  const all = total > 0 && count === total;
  const some = count > 0 && count < total;
  useEffect(() => {
    if (box.current !== null) box.current.indeterminate = some;
  }, [some]);
  if (selection.ids.length === 0) return null;
  const prefix = selection.kind;
  return (
    <div
      className="mb-2 flex flex-wrap items-center justify-between gap-3 rounded-card border border-line bg-panel px-3 py-2"
      data-testid={`${prefix}-purge-header`}
    >
      <label className="inline-flex cursor-pointer items-center gap-2 text-hint">
        <input
          ref={box}
          type="checkbox"
          checked={all}
          disabled={total === 0}
          aria-checked={some ? 'mixed' : all}
          onChange={() => (all || some ? selection.deselectAll() : selection.selectAll())}
          data-testid={`${prefix}-purge-select-all`}
        />
        <span data-testid={`${prefix}-purge-select-all-label`}>
          {all || some ? 'Deselect all' : 'Select all'}
        </span>
        <span className="text-dim">
          {total === selection.ids.length
            ? `archived ${noun}`
            : `${total} of ${selection.ids.length} archived ${noun} can be deleted`}
        </span>
      </label>
      <div className="flex items-center gap-3">
        <span
          className="text-hint text-dim"
          aria-live="polite"
          data-testid={`${prefix}-purge-count`}
        >
          {`${count} selected`}
        </span>
        <Button
          onClick={onDelete}
          disabled={count === 0}
          className="border-red text-red"
          title="Delete the selected archived items and their files for good. You confirm first."
          data-testid={`${prefix}-purge-delete`}
        >
          Delete permanently
        </Button>
      </div>
      {selection.failure === null ? null : (
        <FailureNotice
          failure={selection.failure}
          surface="the purge preview"
          className="w-full"
          data-testid={`${prefix}-purge-preview-failure`}
        />
      )}
    </div>
  );
}

export interface PurgeItemCheckboxProps {
  readonly selection: PurgeSelectionState;
  readonly id: string;
  /** What the checkbox selects, for its accessible name. */
  readonly label: string;
}

/** The checkbox on one archived item, and the blocker inline when it cannot be purged. */
export function PurgeItemCheckbox({ selection, id, label }: PurgeItemCheckboxProps): ReactNode {
  const plan = selection.plans[id];
  const reason = blockedReason(plan);
  const prefix = selection.kind;
  return (
    <div className="flex min-w-0 flex-col items-start gap-1">
      <input
        type="checkbox"
        className="mt-1"
        checked={selection.selected.has(id)}
        disabled={plan === undefined || reason !== null}
        onChange={() => selection.toggle(id)}
        aria-label={`Select ${label} for permanent deletion`}
        data-testid={`${prefix}-purge-check-${id}`}
      />
      {reason === null ? null : (
        <span className="sr-only" data-testid={`${prefix}-purge-blocked-sr-${id}`}>
          {reason}
        </span>
      )}
    </div>
  );
}

/** The inline blocker text, placed by the row beside the item's own details. */
export function PurgeBlockedNote({
  selection,
  id,
}: {
  readonly selection: PurgeSelectionState;
  readonly id: string;
}): ReactNode {
  const reason = blockedReason(selection.plans[id]);
  if (reason === null) return null;
  return (
    <div
      className="mt-1 text-hint text-amber"
      data-testid={`${selection.kind}-purge-blocked-${id}`}
    >
      {`Cannot be deleted: ${reason}`}
    </div>
  );
}

export interface PurgeDialogProps {
  readonly kind: PurgeKind;
  /** The selection, in screen order. */
  readonly ids: readonly string[];
  readonly noun: string;
  readonly onClose: () => void;
  /** After a purge answered: the screen re-reads its list. */
  readonly onDone: (batch: PurgeBatch) => void;
  readonly preview?: typeof previewPurge;
  readonly purge?: typeof purgeSelection;
}

/** The confirmation: what goes with each item, what is blocked and skipped, then the purge. */
export function PurgeDialog({
  kind,
  ids,
  noun,
  onClose,
  onDone,
  preview = previewPurge,
  purge = purgeSelection,
}: PurgeDialogProps): ReactNode {
  const [plans, setPlans] = useState<readonly PurgePlan[] | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [batch, setBatch] = useState<PurgeBatch | null>(null);
  const dialog = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    void preview(kind, ids).then((result) => {
      if (!live) return;
      if (result.ok) setPlans(result.value);
      else setFailure(result.failure);
    });
    return () => {
      live = false;
    };
  }, [kind, ids, preview]);

  useEffect(() => {
    dialog.current?.querySelector<HTMLButtonElement>('[data-purge-cancel]')?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const going = (plans ?? []).filter((plan) => plan.purgeable);
  const skipped = (plans ?? []).filter((plan) => !plan.purgeable);
  const bytes = going.reduce((sum, plan) => sum + plan.removes.bytes, 0);

  const confirm = async (): Promise<void> => {
    setBusy(true);
    const result = await purge(
      kind,
      going.map((plan) => plan.id),
      reason,
    );
    setBusy(false);
    if (!result.ok) {
      setFailure(result.failure);
      return;
    }
    setBatch(result.value);
    onDone(result.value);
  };

  const prefix = kind;
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/50 p-4"
      data-testid={`${prefix}-purge-dialog-backdrop`}
    >
      <div
        ref={dialog}
        className="mt-12 w-full max-w-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${prefix}-purge-dialog-title`}
        data-testid={`${prefix}-purge-dialog`}
      >
        <Card>
          <h2 id={`${prefix}-purge-dialog-title`} className="mb-2 text-heading">
            {batch === null
              ? `Delete ${ids.length} archived ${ids.length === 1 ? noun.replace(/s$/, '') : noun} permanently?`
              : 'Purge finished'}
          </h2>
          {failure === null ? null : (
            <FailureNotice failure={failure} surface="the purge" className="mb-3" />
          )}
          {batch !== null ? (
            <div data-testid={`${prefix}-purge-outcome`}>
              <p className="mb-2 text-body">
                {`Deleted ${batch.purged}, freed ${formatBytes(batch.bytes_removed)}${batch.blocked === 0 ? '.' : `; ${batch.blocked} could not be deleted.`}`}
              </p>
              <ul className="mb-3 stack-y-1 text-hint">
                {batch.results.map((result) => (
                  <li key={result.id} data-testid={`${prefix}-purge-result-${result.id}`}>
                    {result.outcome === 'purged'
                      ? `${result.name}: deleted, ${formatBytes(result.bytes_removed)} freed${result.files_left.length === 0 ? '' : ` (${result.files_left.length} directories could not be removed; they are hidden)`}`
                      : `${result.id}: not deleted - ${result.reason}`}
                  </li>
                ))}
              </ul>
              <div className="flex justify-end">
                <Button onClick={onClose} data-testid={`${prefix}-purge-close`}>
                  Close
                </Button>
              </div>
            </div>
          ) : plans === null ? (
            failure === null ? (
              <p className="text-hint text-dim">Checking what each selected item would remove…</p>
            ) : null
          ) : (
            <>
              <p className="mb-3 text-hint text-dim">
                This cannot be undone. Each item is removed with everything that belongs only to it,
                files on disk included, and a permanent record of the deletion is kept.
              </p>
              {going.length === 0 ? null : (
                <ul className="mb-3 stack-y-2" data-testid={`${prefix}-purge-going`}>
                  {going.map((plan) => (
                    <li
                      key={plan.id}
                      className="rounded-card border border-line p-2 text-hint"
                      data-testid={`${prefix}-purge-going-${plan.id}`}
                    >
                      <div className="font-medium text-text">{plan.name}</div>
                      <div className="text-dim">
                        {`${plan.removes.row_total} database rows, ${plan.removes.files} files, ${formatBytes(plan.removes.bytes)}`}
                      </div>
                      {plan.kept.length === 0 ? null : (
                        <div className="text-dim">
                          {`Kept (not miForge's to delete): ${plan.kept.map((k) => k.path).join(', ')}`}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              )}
              {skipped.length === 0 ? null : (
                <div className="mb-3" data-testid={`${prefix}-purge-skipped`}>
                  <div className="mb-1 text-hint font-medium text-amber">
                    {`Blocked since you selected ${skipped.length === 1 ? 'it' : 'them'} - skipped:`}
                  </div>
                  <ul className="stack-y-1 text-hint">
                    {skipped.map((plan) => (
                      <li key={plan.id} data-testid={`${prefix}-purge-skipped-${plan.id}`}>
                        {`${plan.name}: ${blockedReason(plan) ?? 'cannot be purged'}`}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <label className="mb-3 block text-hint">
                <span className="mb-1 block text-dim">Why (optional, kept in the record)</span>
                <Input
                  entry="prose"
                  value={reason}
                  maxLength={2000}
                  onChange={(event) => setReason(event.target.value)}
                  data-testid={`${prefix}-purge-reason`}
                />
              </label>
              <div className="flex flex-wrap items-center justify-end gap-2">
                <Button
                  onClick={onClose}
                  data-purge-cancel=""
                  data-testid={`${prefix}-purge-cancel`}
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => void confirm()}
                  disabled={busy || going.length === 0}
                  className="border-red text-red"
                  data-testid={`${prefix}-purge-confirm`}
                >
                  {busy
                    ? 'Deleting…'
                    : `Delete ${going.length} permanently (${formatBytes(bytes)})`}
                </Button>
              </div>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
