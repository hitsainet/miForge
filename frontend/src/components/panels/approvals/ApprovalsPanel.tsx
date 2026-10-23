/**
 * The Approvals screen (F12; FR-012.17, .18; MR-9).
 *
 * An agent's request that would spend the GPU, attach to live miLLM, delete, or change a
 * setting waits here until the operator decides. Pending first, newest first; each row says
 * **who asked, what it will do in the operator's terms, how old it is and when it expires**,
 * and offers one primary action named for what it starts ("Approve and start run") and a
 * neutral "Reject". Decided rows keep their outcome in one line - an executed GPU start links
 * to its run.
 *
 * The operator's clicks are never gated: this screen sends no agent header (nothing in the
 * frontend does), which is what the backend requires of a decision.
 *
 * Timestamps are US Eastern through `components/common/time.ts`, the raw ISO on hover. Counts
 * name their scale ("3 pending of 12 shown"). Orange appears only on the primary button.
 */
import { useEffect, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { FailureNotice } from '@/components/common/FailureNotice';
import { MonoId } from '@/components/common/MonoId';
import { Pill, type PillTone } from '@/components/common/Pill';
import { formatAge, formatEt } from '@/components/common/time';
import { useNavigation } from '@/contexts/NavigationContext';
import type { Approval, ApprovalAction, ApprovalStatus } from '@/api/approvals';
import { useApprovalsStore } from '@/stores/approvalsStore';

export const APPROVALS_TITLE = 'Approvals';
export const APPROVALS_SUBTITLE =
  'Agent requests that start GPU work, attach to miLLM, delete, or change a setting wait here for you';

/** What each action does, in one line the operator reads before deciding. */
export const ACTION_LABEL: Record<ApprovalAction, string> = {
  gpu_start: 'Start GPU work',
  hotupdate_attach: 'Attach to miLLM and train its served weights',
  delete: 'Delete',
  config_write: 'Change a setting',
};

/** The primary button names its action (ADR-012): never "Approve" alone. */
export const APPROVE_LABEL: Record<ApprovalAction, string> = {
  gpu_start: 'Approve and start run',
  hotupdate_attach: 'Approve attach and start',
  delete: 'Approve delete',
  config_write: 'Approve setting change',
};

const STATUS_TONE: Record<ApprovalStatus, PillTone> = {
  pending: 'amber',
  executing: 'cyan',
  executed: 'green',
  rejected: 'neutral',
  expired: 'neutral',
  failed: 'red',
};

/** The summary keys, in the order and the words an operator reads them (FR-012.18). */
const FIELD_LABEL: Readonly<Record<string, string>> = {
  run_id: 'Run',
  run_type: 'Run type',
  model: 'Model',
  gpu_uuid: 'GPU',
  gpu_request: 'GPU requested',
  step_budget: 'Step budget',
  pack: 'Pack',
  chain: 'Chain',
  stage_count: 'Stages',
  stage_0: 'First stage',
  from_checkpoint: 'From checkpoint',
  step: 'Step',
  optimizer_state: 'Optimizer state',
  at_step: 'At step',
  kinds: 'Evaluates',
  adapter: 'Adapter',
  into_run_id: 'Into run',
  target: 'Target',
  name: 'Name',
  current_step: 'Draft step',
  key: 'Setting',
  old: 'Current value',
  new: 'New value',
  attach: 'miLLM target',
  millm_base_url: 'miLLM URL',
  served_model: 'Served model',
  served_model_unreadable: 'Served model unreadable',
  route: 'Route',
};

/** Keys that are plumbing, not decision material. */
const HIDDEN = new Set(['admitted']);

function show(value: unknown): string {
  if (value === null || value === undefined) return 'not set';
  if (Array.isArray(value)) return value.map(show).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function SummaryList({
  summary,
  testId,
}: {
  readonly summary: Readonly<Record<string, unknown>>;
  readonly testId?: string;
}): ReactNode {
  const keys = Object.keys(summary).filter((key) => !HIDDEN.has(key));
  const ordered = [
    ...Object.keys(FIELD_LABEL).filter((key) => keys.includes(key)),
    ...keys.filter((key) => !(key in FIELD_LABEL)),
  ];
  return (
    <dl className="grid grid-cols-[10rem_1fr] gap-x-4 gap-y-1 text-body" data-testid={testId}>
      {ordered.map((key) => {
        const value = summary[key];
        const nested =
          value !== null && typeof value === 'object' && !Array.isArray(value)
            ? (value as Record<string, unknown>)
            : null;
        return (
          <div key={key} className="contents">
            <dt className="text-dim">{FIELD_LABEL[key] ?? key}</dt>
            <dd className="min-w-0 wrap-break-word">
              {nested !== null ? (
                <SummaryList summary={nested} />
              ) : key === 'run_id' || key === 'gpu_uuid' || key === 'key' ? (
                <MonoId value={show(value)} />
              ) : (
                show(value)
              )}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}

function runIdOf(result: unknown): string | null {
  if (result !== null && typeof result === 'object' && 'run_id' in result) {
    const id = (result as { run_id: unknown }).run_id;
    return typeof id === 'string' ? id : null;
  }
  return null;
}

function Outcome({ approval }: { readonly approval: Approval }): ReactNode {
  const navigation = useNavigation();
  const runId = runIdOf(approval.result);
  switch (approval.status) {
    case 'executed':
      return (
        <p className="text-body" data-testid={`approval-outcome-${approval.approval_id}`}>
          Executed {formatEt(approval.decided_at)}.{' '}
          {runId === null ? null : (
            <button
              type="button"
              className="underline"
              onClick={() => navigation.go('run', { runId })}
            >
              Open run {runId}
            </button>
          )}
        </p>
      );
    case 'failed':
      return (
        <p className="text-body" data-testid={`approval-outcome-${approval.approval_id}`}>
          Failed: <span className="font-mono text-ident">{approval.error?.code}</span> —{' '}
          {approval.error?.message}. The agent sees this refusal and must ask again.
        </p>
      );
    case 'rejected':
      return (
        <p className="text-body" data-testid={`approval-outcome-${approval.approval_id}`}>
          Rejected {formatEt(approval.decided_at)}
          {approval.reason ? `: ${approval.reason}` : ''}. The agent sees "rejected".
        </p>
      );
    case 'expired':
      return (
        <p className="text-body" data-testid={`approval-outcome-${approval.approval_id}`}>
          Expired {formatEt(approval.expires_at)} without a decision. It will never run.
        </p>
      );
    default:
      return null;
  }
}

function ApprovalRow({ approval }: { readonly approval: Approval }): ReactNode {
  const { approve, reject, inFlight } = useApprovalsStore();
  const busy = inFlight.has(approval.approval_id);
  return (
    <Card
      as="li"
      className="flex flex-col gap-3"
      data-testid={`approval-${approval.approval_id}`}
      aria-label={`${ACTION_LABEL[approval.action]} requested by ${approval.requested_by}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <Pill tone={STATUS_TONE[approval.status]}>{approval.status}</Pill>
          <span className="font-semibold">{ACTION_LABEL[approval.action]}</span>
          <span className="text-dim">
            requested by <MonoId value={approval.requested_by} />
          </span>
        </div>
        <span className="text-hint text-dim" title={approval.created_at}>
          asked {formatAge(approval.created_at)} · expires {formatEt(approval.expires_at)}
        </span>
      </div>
      <SummaryList summary={approval.summary} testId={`approval-summary-${approval.approval_id}`} />
      {approval.status === 'pending' ? (
        <div className="flex gap-2">
          <Button
            variant="primary"
            disabled={busy}
            onClick={() => void approve(approval.approval_id)}
            data-testid={`approve-${approval.approval_id}`}
          >
            {APPROVE_LABEL[approval.action]}
          </Button>
          <Button
            variant="ghost"
            disabled={busy}
            onClick={() => void reject(approval.approval_id)}
            data-testid={`reject-${approval.approval_id}`}
          >
            Reject
          </Button>
        </div>
      ) : (
        <Outcome approval={approval} />
      )}
    </Card>
  );
}

export function ApprovalsPanel(): ReactNode {
  const { listing, failure, decisionFailure, load } = useApprovalsStore();

  useEffect(() => {
    void load();
  }, [load]);

  const rows = listing?.approvals ?? [];
  const pending = rows.filter((row) => row.status === 'pending');
  const decided = rows.filter((row) => row.status !== 'pending');

  return (
    <>
      <PageHead title={APPROVALS_TITLE} subtitle={APPROVALS_SUBTITLE} />
      {failure === null ? null : (
        <FailureNotice failure={failure} surface="The approval queue" className="mb-4" />
      )}
      {decisionFailure === null ? null : (
        <FailureNotice failure={decisionFailure} surface="That decision" className="mb-4" />
      )}
      {listing === null ? null : (
        <p className="mb-4 text-body text-dim" data-testid="approvals-count">
          {pending.length} pending of {rows.length} shown
        </p>
      )}
      {listing !== null && rows.length === 0 ? (
        <p className="text-body text-dim" data-testid="approvals-empty">
          No agent has asked for anything that needs your approval.
        </p>
      ) : null}
      <ul className="flex flex-col gap-3" aria-label="Pending approvals">
        {pending.map((approval) => (
          <ApprovalRow key={approval.approval_id} approval={approval} />
        ))}
      </ul>
      {decided.length === 0 ? null : (
        <>
          <h2 className="mb-3 mt-8 font-semibold">Decided</h2>
          <ul className="flex flex-col gap-3" aria-label="Decided approvals">
            {decided.map((approval) => (
              <ApprovalRow key={approval.approval_id} approval={approval} />
            ))}
          </ul>
        </>
      )}
    </>
  );
}
