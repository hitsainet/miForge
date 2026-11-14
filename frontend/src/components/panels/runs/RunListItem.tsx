/**
 * One row of the run list (task 14.0, AC-009.43): enough to tell runs apart and open one.
 *
 * The row is a **button**, so it is reachable and operable by keyboard (FR-009.7) - the
 * mockup's list rows were `div`s with `onClick`, which no focus ring can rescue.
 *
 * It carries what `GET /api/v1/runs` serves and nothing it does not: no liveness (that is the
 * run view's, from the heartbeat block), no scores, no progress fraction. A terminal run shows
 * **why it ended** - the stop reason, or the failure code - because that is the one thing an
 * operator scanning a list of finished runs wants; a missing one is named, never blank.
 */
import type { ReactNode } from 'react';
import { MonoId } from '@/components/common/MonoId';
import { Pill, StatePill } from '@/components/common/Pill';
import type { RunArchiveOutcome, RunListRow } from '@/api/runs';
import { formatEt } from '@/components/common/time';
import { RunArchiveAction } from './RunArchiveAction';
import {
  PurgeBlockedNote,
  PurgeItemCheckbox,
  type PurgeSelectionState,
} from '@/components/common/PurgeSelection';

const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

/** A timestamp in US Eastern via the shared formatter, with the ISO form one hover away. */
export function formatWhen(iso: string | null): string {
  if (iso === null) return 'not recorded';
  return formatEt(iso);
}

export interface RunListItemProps {
  readonly row: RunListRow;
  readonly onOpen: (runId: string) => void;
  /**
   * Archive / Unarchive answered (operator decision 2026-10-20); the list re-reads. Absent,
   * the row offers no archive action. The action sits **beside** the row's open button, not
   * inside it: a button inside a button is invalid and unreachable by keyboard.
   */
  readonly onArchived?: (outcome: RunArchiveOutcome) => void;
  /**
   * Permanent deletion (operator request 2026-10-20). Given, an ARCHIVED row carries a
   * selection checkbox - disabled, with the reason inline, when the backend says it cannot be
   * purged. A listed row never does: only an archived run can be purged.
   */
  readonly purge?: PurgeSelectionState;
}

export function RunListItem({ row, onOpen, onArchived, purge }: RunListItemProps): ReactNode {
  const ended = TERMINAL.has(row.status);
  const archivedAt = row.archived_at ?? null;
  return (
    <li className="flex items-start gap-2">
      {purge === undefined || archivedAt === null ? null : (
        <PurgeItemCheckbox selection={purge} id={row.run_id} label={`run ${row.run_id}`} />
      )}
      <button
        type="button"
        onClick={() => onOpen(row.run_id)}
        className="min-w-0 flex-1 rounded-card border border-line bg-card p-card text-left hover:bg-panel"
        data-testid={`run-list-row-${row.run_id}`}
        aria-label={`Open run ${row.run_id}`}
      >
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <MonoId value={row.run_id} />
            <div className="mt-1 flex flex-wrap items-center gap-2 text-hint text-dim">
              <StatePill state={row.status} />
              {archivedAt === null ? null : (
                <Pill tone="neutral" title={archivedAt} data-testid="run-list-archived">
                  {`Archived ${formatWhen(archivedAt)}`}
                </Pill>
              )}
              <span data-testid="run-list-type">{row.run_type}</span>
              <span title={row.created_at ?? undefined} data-testid="run-list-created">
                created {formatWhen(row.created_at)}
              </span>
            </div>
            {purge === undefined || archivedAt === null ? null : (
              <PurgeBlockedNote selection={purge} id={row.run_id} />
            )}
          </div>
          <div className="text-right text-hint text-dim">
            <div data-testid="run-list-step">
              {row.last_step === null ? 'no step recorded' : `step ${row.last_step}`}
            </div>
            {ended ? (
              <div data-testid="run-list-ended">
                {row.stop_reason !== null
                  ? `stopped: ${row.stop_reason}`
                  : row.failure_code !== null
                    ? `failed: ${row.failure_code}`
                    : 'ended; no stop reason recorded on the row'}
              </div>
            ) : null}
          </div>
        </div>
      </button>
      {onArchived === undefined ? null : (
        <RunArchiveAction
          runId={row.run_id}
          status={row.status}
          archivedAt={archivedAt}
          onDone={onArchived}
        />
      )}
    </li>
  );
}
