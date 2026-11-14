/**
 * History (ENH-008 D6, §2.2): every decided request, the most recent decision first, in the
 * suite's columns:
 *
 * | Outcome | Action | Asked by | Asked | Decided by | Decided | Reason | Result | Id |
 *
 * A table on a wide screen; each row stacks into a card on a narrow one, in the same order (the
 * cell labels shown there are hidden from the table layout). Every value is text.
 */
import type { ReactNode } from 'react';
import { MonoId } from '@/components/common/MonoId';
import { LocalTimeText } from '@/components/common/LocalTime';
import { useOptionalNavigation } from '@/contexts/NavigationContext';
import type { Approval } from '@/api/approvals';
import { OutcomePill, actionLabel, askedBy } from './ApprovalCard';

export const HISTORY_COLUMNS = [
  'Outcome',
  'Action',
  'Asked by',
  'Asked',
  'Decided by',
  'Decided',
  'Reason',
  'Result',
  'Id',
] as const;

function runIdOf(result: unknown): string | null {
  if (result !== null && typeof result === 'object' && 'run_id' in result) {
    const id = (result as { run_id: unknown }).run_id;
    return typeof id === 'string' ? id : null;
  }
  return null;
}

/** "operator" for a decision, "nobody (expired)" for an expiry (§2.2; no login in the suite). */
export function decidedBy(approval: Approval): string {
  if (approval.status === 'expired') return 'nobody (expired)';
  return approval.decided_by ?? 'operator';
}

function Result({ approval }: { readonly approval: Approval }): ReactNode {
  const navigation = useOptionalNavigation();
  const runId = runIdOf(approval.result);
  if (approval.status === 'executed' && runId !== null && navigation !== null) {
    return (
      <button type="button" className="underline" onClick={() => navigation.go('run', { runId })}>
        Open run {runId}
      </button>
    );
  }
  if (approval.status === 'failed' && approval.error !== null) {
    return (
      <span>
        <span className="font-mono text-ident">{approval.error.code}</span> —{' '}
        {approval.error.message}
      </span>
    );
  }
  if (approval.status === 'expired') return <span>expired without a decision</span>;
  return <span>—</span>;
}

function Cell({
  label,
  children,
  testId,
}: {
  readonly label: string;
  readonly children: ReactNode;
  readonly testId?: string;
}): ReactNode {
  return (
    <td className="block px-0 py-1 align-top md:table-cell md:px-2 md:py-2" data-testid={testId}>
      <span className="text-hint text-dim md:hidden">{label}: </span>
      {children}
    </td>
  );
}

export function ApprovalHistory({ rows }: { readonly rows: readonly Approval[] }): ReactNode {
  return (
    <table className="block w-full text-left text-body md:table" data-testid="approvals-history">
      <thead className="hidden md:table-header-group">
        <tr className="border-b border-line text-hint text-dim">
          {HISTORY_COLUMNS.map((column) => (
            <th key={column} scope="col" className="px-2 py-2 font-medium">
              {column}
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="flex flex-col gap-3 md:table-row-group">
        {rows.map((approval) => {
          const id = approval.approval_id;
          return (
            <tr
              key={id}
              className="block rounded-card border border-line bg-card p-card md:table-row md:rounded-none md:border-0 md:border-b md:bg-transparent md:p-0"
              data-testid={`history-${id}`}
            >
              <Cell label="Outcome">
                <OutcomePill status={approval.status} />
              </Cell>
              <Cell label="Action">
                <span className="wrap-break-word">{actionLabel(approval.action)}</span>
              </Cell>
              <Cell label="Asked by">
                <span className="wrap-break-word">
                  {askedBy(approval.requested_by).replace(/^asked by /, '')}
                </span>
              </Cell>
              <Cell label="Asked">
                <LocalTimeText iso={approval.created_at} />
              </Cell>
              <Cell label="Decided by" testId={`history-decided-by-${id}`}>
                {decidedBy(approval)}
                {approval.batch_id === null || approval.batch_id === undefined ? null : (
                  <span className="block text-hint text-dim">
                    with other settings changes, in one approval{' '}
                    <MonoId value={approval.batch_id} />
                  </span>
                )}
              </Cell>
              <Cell label="Decided">
                <LocalTimeText iso={approval.decided_at ?? approval.expires_at} />
              </Cell>
              <Cell label="Reason">
                <span className="wrap-break-word whitespace-pre-wrap">
                  {approval.reason === null || approval.reason === '' ? '—' : approval.reason}
                </span>
              </Cell>
              <Cell label="Result" testId={`approval-outcome-${id}`}>
                <Result approval={approval} />
              </Cell>
              <Cell label="Id">
                <MonoId value={id} className="text-hint" />
              </Cell>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
