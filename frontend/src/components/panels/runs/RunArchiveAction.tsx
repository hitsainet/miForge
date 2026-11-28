/**
 * Archive / Unarchive one run (operator decision 2026-10-20).
 *
 * **Visibility only**: archiving hides a finished run from the Runs list and the run dropdown
 * and deletes nothing - its checkpoints, trainer state and history stay, and the run view
 * still opens it. **A pending or running run offers no Archive at all**, because the backend
 * refuses it (`run_not_archivable`) and an action that can only fail is not an action. An
 * archived run always offers Unarchive, whatever its status.
 */
import { useState, type ReactNode } from 'react';
import { Button } from '@/components/common/Button';
import { FailureNotice } from '@/components/common/FailureNotice';
import { archiveRun, unarchiveRun, type RunArchiveOutcome } from '@/api/runs';
import type { ApiFailure } from '@/api/errors';
import { isTerminal } from '@/stores/runStore';

export interface RunArchiveActionProps {
  readonly runId: string;
  readonly status: string;
  readonly archivedAt: string | null | undefined;
  /** Called with the backend's answer, so the caller re-reads what it lists. */
  readonly onDone: (outcome: RunArchiveOutcome) => void;
  /** Injected in tests. */
  readonly archive?: typeof archiveRun;
  readonly unarchive?: typeof unarchiveRun;
}

/** Whether a run offers an archive action at all. */
export function offersArchiveAction(
  status: string,
  archivedAt: string | null | undefined,
): boolean {
  return (archivedAt ?? null) !== null || isTerminal(status);
}

export function RunArchiveAction({
  runId,
  status,
  archivedAt,
  onDone,
  archive = archiveRun,
  unarchive = unarchiveRun,
}: RunArchiveActionProps): ReactNode {
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  if (!offersArchiveAction(status, archivedAt)) return null;
  const archived = (archivedAt ?? null) !== null;

  const act = async (): Promise<void> => {
    setBusy(true);
    const result = await (archived ? unarchive(runId) : archive(runId));
    setBusy(false);
    if (!result.ok) {
      setFailure(result.failure);
      return;
    }
    setFailure(null);
    onDone(result.value);
  };

  return (
    <div className="flex flex-col items-end gap-2">
      <Button
        onClick={() => void act()}
        disabled={busy}
        title={
          archived
            ? 'Return this run to the Runs list and the run dropdown'
            : 'Hide this run from the Runs list and the run dropdown. Deletes nothing: its checkpoints and history stay.'
        }
        data-testid={`run-archive-action-${runId}`}
      >
        {busy ? (archived ? 'Unarchiving…' : 'Archiving…') : archived ? 'Unarchive' : 'Archive'}
      </Button>
      {failure === null ? null : (
        <FailureNotice
          failure={failure}
          surface={archived ? 'the unarchive request' : 'the archive request'}
          data-testid={`run-archive-failure-${runId}`}
        />
      )}
    </div>
  );
}
