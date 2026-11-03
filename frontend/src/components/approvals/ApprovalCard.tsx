/**
 * One agent request, as the suite shows it (ENH-008 §2.1), and the words every approvals
 * surface shares (§2.2, §2.5). Used by the banner on every screen and by the Approvals screen,
 * so approving or rejecting from either is the same call and the same state.
 *
 * The waiting card's fields, in the suite's order:
 *
 * 1. the action in plain words, then the **Waiting** pill;
 * 2. who asked ("asked by agent forge-mcp");
 * 3. when ("asked {local time}", "expires in N h"), the exact instant on hover;
 * 4. the facts: the stored summary in the operator's field order, unknown keys after;
 * 5. (no extra card in miForge);
 * 6. **All arguments the agent sent**, collapsed;
 * 7. the request id, with a copy button;
 * 8. the named approve button ("Approve and start run", ADR-012: never a bare "Approve") and
 *    **Reject**, which opens an optional reason field (D8c).
 *
 * **Every value from a request is a React text node.** Objects go through `JSON.stringify` and
 * are still text; nothing here sets inner HTML. `ApprovalCard.hostile.test.tsx` proves it by
 * rendering markup in every field and counting elements.
 */
import { useRef, useState, type ReactNode } from 'react';
import { Copy } from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { MonoId } from '@/components/common/MonoId';
import { Pill, type PillTone } from '@/components/common/Pill';
import { LocalTimeText } from '@/components/common/LocalTime';
import { formatExpiresIn, formatLocal } from '@/components/common/time';
import { describeFailure, type ApiFailure } from '@/api/errors';
import { REJECT_REASON_MAX, type Approval } from '@/api/approvals';
import { useApprovalsStore } from '@/stores/approvalsStore';

/** What each action does, in one line the operator reads before deciding. */
export const ACTION_LABEL: Readonly<Record<string, string>> = {
  gpu_start: 'Start GPU work',
  hotupdate_attach: 'Attach to miLLM and train its served weights',
  delete: 'Delete',
  config_write: 'Change a setting',
};

/** The primary button names its action (ADR-012): never "Approve" alone. */
export const APPROVE_LABEL: Readonly<Record<string, string>> = {
  gpu_start: 'Approve and start run',
  hotupdate_attach: 'Approve attach and start',
  delete: 'Approve delete',
  config_write: 'Approve setting change',
};

export function actionLabel(action: string): string {
  return ACTION_LABEL[action] ?? action;
}

export function approveLabel(action: string): string {
  return APPROVE_LABEL[action] ?? `Approve ${actionLabel(action).toLowerCase()}`;
}

/** ENH-008 §2.2: the one outcome vocabulary, mapped from miForge's statuses. */
export const OUTCOME_LABEL: Readonly<Record<string, string>> = {
  pending: 'Waiting',
  executing: 'Running',
  executed: 'Approved and run',
  failed: 'Approved, then failed',
  rejected: 'Rejected',
  expired: 'Expired',
};

const OUTCOME_TONE: Readonly<Record<string, PillTone>> = {
  pending: 'amber',
  executing: 'cyan',
  executed: 'green',
  rejected: 'neutral',
  expired: 'neutral',
  failed: 'red',
};

export function OutcomePill({ status }: { readonly status: string }): ReactNode {
  return <Pill tone={OUTCOME_TONE[status] ?? 'neutral'}>{OUTCOME_LABEL[status] ?? status}</Pill>;
}

/** "asked by agent forge-mcp", or "asked by an agent" when no name was recorded. */
export function askedBy(requestedBy: string | null | undefined): string {
  const name = (requestedBy ?? '').replace(/^agent:/, '').trim();
  return name === '' ? 'asked by an agent' : `asked by agent ${name}`;
}

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
  label: 'Setting',
  key: 'Setting key',
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

/** A value as text. Objects are `JSON.stringify`'d - and are still rendered as text. */
export function show(value: unknown): string {
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
            <dd className="min-w-0 wrap-break-word whitespace-pre-wrap">
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

/** §2.1 item 6: every key and value of the stored payload, collapsed. */
function AllArguments({ approval }: { readonly approval: Approval }): ReactNode {
  const payload = approval.payload;
  const entries: [string, unknown][] =
    payload !== null && typeof payload === 'object' && !Array.isArray(payload)
      ? Object.entries(payload as Record<string, unknown>)
      : payload === undefined
        ? []
        : [['value', payload]];
  return (
    <details className="text-body" data-testid={`approval-arguments-${approval.approval_id}`}>
      <summary className="cursor-pointer text-dim">All arguments the agent sent</summary>
      {entries.length === 0 ? (
        <p className="mt-2 text-dim">No arguments were recorded.</p>
      ) : (
        <dl className="mt-2 grid grid-cols-[10rem_1fr] gap-x-4 gap-y-1">
          {entries.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="font-mono text-ident text-dim">{key}</dt>
              <dd className="min-w-0 font-mono text-ident wrap-break-word whitespace-pre-wrap">
                {typeof value === 'string' ? value : JSON.stringify(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </details>
  );
}

/** §2.1 item 7: the request id, small and monospace, with a copy button. */
export function RequestId({ id }: { readonly id: string }): ReactNode {
  const [copied, setCopied] = useState(false);
  const copy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };
  return (
    <span className="inline-flex items-center gap-2 text-hint text-dim">
      <MonoId value={id} data-testid={`approval-id-${id}`} />
      <button
        type="button"
        className="inline-flex items-center gap-1 hover:underline"
        aria-label="Copy request id"
        onClick={() => void copy()}
      >
        <Copy size={12} aria-hidden="true" />
        {copied ? 'Copied' : 'Copy'}
      </button>
    </span>
  );
}

/** §2.1: the error a decision met, in the card that asked. */
export function decisionMessage(failure: ApiFailure): string {
  if (failure.kind === 'refusal' && failure.code === 'approval_expired') {
    return 'This request expired. Ask the agent to try again.';
  }
  if (failure.status === 409 && failure.kind === 'refusal') {
    return `${failure.message} The list has been refreshed.`;
  }
  return describeFailure(failure);
}

export interface ApprovalCardProps {
  readonly approval: Approval;
  /** Called after a decision the server accepted, so the list can move focus on (§2.7). */
  readonly onDecided?: (id: string) => void;
  /** Lets the list focus this card. */
  readonly cardRef?: (element: HTMLElement | null) => void;
}

export function ApprovalCard({ approval, onDecided, cardRef }: ApprovalCardProps): ReactNode {
  const id = approval.approval_id;
  const approve = useApprovalsStore((state) => state.approve);
  const reject = useApprovalsStore((state) => state.reject);
  const busy = useApprovalsStore((state) => state.inFlight.has(id));
  const failure = useApprovalsStore((state) => state.decisionFailures[id]);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const own = useRef<HTMLElement | null>(null);
  const label = actionLabel(approval.action);
  const expires = formatLocal(approval.expires_at);

  const settle = (result: Approval | null): void => {
    if (result === null) {
      own.current?.focus();
      return;
    }
    onDecided?.(id);
  };

  return (
    <Card
      as="li"
      className="flex flex-col gap-3 outline-none"
      data-testid={`approval-${id}`}
      aria-label={`${label}, ${askedBy(approval.requested_by)}`}
    >
      <div
        ref={(element) => {
          own.current = element;
          cardRef?.(element);
        }}
        tabIndex={-1}
        className="flex flex-col gap-3 outline-none"
      >
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-semibold wrap-break-word">{label}</span>
          <OutcomePill status={approval.status} />
        </div>
        <div className="text-body text-dim wrap-break-word">{askedBy(approval.requested_by)}</div>
        <div className="flex flex-wrap gap-x-3 text-hint text-dim">
          <LocalTimeText iso={approval.created_at} prefix="asked " />
          <span aria-hidden="true">·</span>
          <span title={expires.title === '' ? undefined : expires.title}>
            {formatExpiresIn(approval.expires_at)}
          </span>
        </div>
        <SummaryList summary={approval.summary} testId={`approval-summary-${id}`} />
        <AllArguments approval={approval} />
        <RequestId id={id} />
        {failure === undefined ? null : (
          <p className="text-body text-red" role="alert" data-testid={`approval-error-${id}`}>
            {decisionMessage(failure)}
          </p>
        )}
        {approval.status !== 'pending' ? null : (
          // The decision row never scrolls out of reach (operator 2026-10-28): inside the
          // height-capped banner a long card put Approve below the fold. Sticky at the bottom
          // of the nearest scroller, on the card's own opaque background.
          <div
            className="sticky bottom-0 -mx-1 bg-card px-1 py-2"
            data-testid={`approval-actions-${id}`}
          >
            {rejecting ? (
              <div className="flex flex-wrap items-center gap-2">
                <input
                  type="text"
                  className="min-w-0 flex-1 rounded-control border border-line bg-panel px-3 py-2 text-body"
                  placeholder="Reason (optional)"
                  aria-label="Reason (optional)"
                  maxLength={REJECT_REASON_MAX}
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                  data-testid={`reject-reason-${id}`}
                />
                <Button
                  disabled={busy}
                  onClick={() => void reject(id, reason).then(settle)}
                  data-testid={`confirm-reject-${id}`}
                >
                  Confirm reject
                </Button>
                <Button
                  disabled={busy}
                  onClick={() => {
                    setRejecting(false);
                    setReason('');
                  }}
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="primary"
                  disabled={busy}
                  onClick={() => void approve(id).then(settle)}
                  data-testid={`approve-${id}`}
                >
                  {approveLabel(approval.action)}
                </Button>
                <Button
                  disabled={busy}
                  onClick={() => setRejecting(true)}
                  data-testid={`reject-${id}`}
                >
                  Reject
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
