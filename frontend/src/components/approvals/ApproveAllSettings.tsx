/**
 * **Approve all settings changes (N)** (ENH-006 D79, operator 2026-10-29): "When approving one
 * or more settings changes, I want to approve all outstanding settings changes with one
 * approval request."
 *
 * Shown in the banner and on the Approvals screen **only when two or more settings-change
 * requests wait** (one waiting: its own card's "Approve setting change" is the action). The
 * button names its action (ADR-012). It opens a confirm that lists every request it will
 * approve - the setting's label, from → to, who asked - and **the ids sent are exactly the
 * ones listed**, taken when the confirm opened, so a request that arrives while it is open is
 * never approved unseen. Confirming sends one `POST /approvals/approve-batch`; each request is
 * still approved, run and recorded on its own, and History shows each. There is no reject-all.
 *
 * Every value from a request is a React text node, as on the card.
 */
import { useState, type ReactNode } from 'react';
import { Button } from '@/components/common/Button';
import { MonoId } from '@/components/common/MonoId';
import { describeFailure } from '@/api/errors';
import { isSettingsChange, type Approval } from '@/api/approvals';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { askedBy, show } from './ApprovalCard';

/** The button appears from this many waiting settings changes. */
export const BATCH_MIN = 2;

export function approveAllLabel(count: number): string {
  return `Approve all settings changes (${count})`;
}

export function confirmAllLabel(count: number): string {
  return `Approve these ${count} settings changes`;
}

/** The setting's label as the Settings screen shows it; the key for a request made before. */
export function settingLabel(approval: Approval): string {
  const { label, key } = approval.summary;
  if (typeof label === 'string' && label !== '') return label;
  return typeof key === 'string' ? key : show(key);
}

export function fromTo(approval: Approval): string {
  return `${show(approval.summary.old)} → ${show(approval.summary.new)}`;
}

/** What the last approve-all did, in one sentence. */
export function batchOutcome(results: readonly { readonly ok: boolean }[]): string {
  const done = results.filter((item) => item.ok).length;
  if (done === results.length) {
    return `Approved ${done} settings changes together.`;
  }
  return `Approved ${done} of ${results.length} settings changes; ${results.length - done} did not run. History shows each.`;
}

export interface ApproveAllSettingsProps {
  /** The waiting requests; the settings changes among them are counted here. */
  readonly waiting: readonly Approval[];
  /** Distinguishes the banner's copy from the screen's in test ids. */
  readonly where: 'banner' | 'page';
}

export function ApproveAllSettings({ waiting, where }: ApproveAllSettingsProps): ReactNode {
  const approveSettingsBatch = useApprovalsStore((state) => state.approveSettingsBatch);
  const busy = useApprovalsStore((state) => state.batchInFlight);
  const lastBatch = useApprovalsStore((state) => state.lastBatch);
  const batchFailure = useApprovalsStore((state) => state.batchFailure);
  const [confirming, setConfirming] = useState<readonly Approval[] | null>(null);
  const [answered, setAnswered] = useState(false);

  const settings = waiting.filter(isSettingsChange);
  const testId = (name: string): string => `approve-all-${name}-${where}`;

  const outcome = !answered ? null : batchFailure !== null ? (
    <p className="text-body text-red" role="alert" data-testid={testId('failure')}>
      Nothing was approved. {describeFailure(batchFailure)} The list has been refreshed.
    </p>
  ) : lastBatch !== null ? (
    <p className="text-body" role="status" data-testid={testId('outcome')}>
      {batchOutcome(lastBatch.results)}
    </p>
  ) : null;

  if (confirming === null) {
    if (settings.length < BATCH_MIN) return outcome;
    return (
      <div className="flex flex-col gap-2">
        <div>
          <Button
            variant="primary"
            disabled={busy}
            onClick={() => {
              setAnswered(false);
              setConfirming(settings);
            }}
            data-testid={testId('button')}
          >
            {approveAllLabel(settings.length)}
          </Button>
        </div>
        {outcome}
      </div>
    );
  }

  const send = async (): Promise<void> => {
    await approveSettingsBatch(confirming.map((row) => row.approval_id));
    setConfirming(null);
    setAnswered(true);
  };

  return (
    <div
      role="group"
      aria-label="Confirm approving all settings changes"
      className="flex flex-col gap-3 rounded-card border border-line bg-card p-card"
      data-testid={testId('confirm')}
    >
      <p className="font-semibold">
        Approve these {confirming.length} settings changes? Each one is applied and recorded on its
        own.
      </p>
      <ul className="flex flex-col gap-2 text-body" aria-label="Settings changes to approve">
        {confirming.map((row) => (
          <li
            key={row.approval_id}
            className="flex flex-col gap-0.5"
            data-testid={`approve-all-item-${row.approval_id}`}
          >
            <span className="font-medium wrap-break-word">{settingLabel(row)}</span>
            <span className="wrap-break-word whitespace-pre-wrap">{fromTo(row)}</span>
            <span className="text-hint text-dim wrap-break-word">
              {askedBy(row.requested_by)} · <MonoId value={row.approval_id} />
            </span>
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap gap-2">
        <Button
          variant="primary"
          disabled={busy}
          onClick={() => void send()}
          data-testid={testId('confirm-button')}
        >
          {confirmAllLabel(confirming.length)}
        </Button>
        <Button disabled={busy} onClick={() => setConfirming(null)}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
