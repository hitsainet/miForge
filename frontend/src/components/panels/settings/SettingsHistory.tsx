/**
 * Recent changes to the settings (ENH-003 section 3.4, gap 3; decision D12: the last 50).
 *
 * Reads `GET /api/v1/settings/history`, the backend's append-only record: every change made
 * from this screen or by an agent's approved request, with the value before and after, who made
 * it (and the approval, for an agent) and the reason, when one was given. Nothing here can edit
 * the record - the table refuses edits in the database itself.
 *
 * `version` is bumped by the Settings screen after each save or reset, so the list re-reads
 * without the operator reloading the page.
 */
import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { FailureNotice } from '@/components/common/FailureNotice';
import { formatEt } from '@/components/common/time';
import type { ApiFailure } from '@/api/errors';
import { HISTORY_SHOWN, readSettingsHistory, type SettingChange } from '@/api/settings';
import { showActor, showSetting } from './settingsText';

export interface SettingsHistoryProps {
  /** key -> the plain label the screen shows for it. */
  readonly labels: Readonly<Record<string, string>>;
  /** Re-read when this changes. */
  readonly version: number;
}

function ChangeRow({
  change,
  label,
}: {
  readonly change: SettingChange;
  readonly label: string;
}): ReactNode {
  const verb = change.action === 'reset' ? 'back to the environment value' : 'set';
  return (
    <li className="flex flex-col gap-0.5 py-2" data-testid={`settings-change-${change.id}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-semibold">{label}</span>
        <span className="text-hint text-dim" title={change.changed_at}>
          {formatEt(change.changed_at)}
        </span>
      </div>
      <div className="text-body">
        {showSetting(change.previous)} → {showSetting(change.value)}{' '}
        <span className="text-dim">({verb})</span>
      </div>
      <div className="text-hint text-dim">
        By {showActor(change.actor, change.approval_id)}
        {change.reason ? ` - "${change.reason}"` : ''}
      </div>
    </li>
  );
}

export function SettingsHistory({ labels, version }: SettingsHistoryProps): ReactNode {
  const [changes, setChanges] = useState<readonly SettingChange[] | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  const load = useCallback(async () => {
    const result = await readSettingsHistory(HISTORY_SHOWN);
    if (result.ok) {
      setChanges(result.value.changes);
      setFailure(null);
    } else {
      setFailure(result.failure);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, version]);

  return (
    <section aria-labelledby="settings-history-title" data-testid="settings-history">
      <h2 id="settings-history-title" className="mb-1 mt-8 font-semibold">
        Recent changes
      </h2>
      <p className="mb-3 text-body text-dim">
        The last {HISTORY_SHOWN} changes, newest first. The record cannot be edited.
      </p>
      {failure === null ? null : <FailureNotice failure={failure} surface="The change history" />}
      {changes === null ? null : (
        <Card>
          {changes.length === 0 ? (
            <div className="text-body text-dim" data-testid="settings-history-empty">
              No setting has been changed here yet.
            </div>
          ) : (
            <ol className="divide-y divide-line" aria-label="Recent settings changes">
              {changes.map((change) => (
                <ChangeRow
                  key={change.id}
                  change={change}
                  label={labels[change.key] ?? change.key}
                />
              ))}
            </ol>
          )}
        </Card>
      )}
    </section>
  );
}
