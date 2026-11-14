/**
 * The Approvals screen (F12; FR-012.17, .18; MR-9; ENH-008 D6).
 *
 * An agent's request that would spend the GPU, attach to live miLLM, delete, or change a
 * setting waits until the operator decides. Two sections, in the words all four suite apps use
 * (ENH-008 §2.5):
 *
 * - **Waiting for you (n)**: the waiting cards (`components/approvals/ApprovalCard.tsx`, the
 *   same cards the banner shows on every other screen), the soonest to expire first;
 *   With two or more settings changes waiting, **Approve all settings changes (N)** above them
 *   (ENH-006 D79);
 * - **History**: the decided requests, the most recent decision first, in the §2.2 columns,
 *   with **Show older**.
 *
 * The operator's clicks are never gated: this screen sends no agent header (nothing in the
 * frontend does), which is what the backend requires of a decision.
 *
 * The waiting list is the shell's poll (`App.tsx`); this screen reads History when it opens,
 * on the same interval while it is open, and after every decision.
 */
import { useEffect, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Button } from '@/components/common/Button';
import { FailureNotice } from '@/components/common/FailureNotice';
import { WaitingList } from '@/components/approvals/WaitingList';
import { ApproveAllSettings } from '@/components/approvals/ApproveAllSettings';
import { ApprovalHistory } from '@/components/approvals/ApprovalHistory';
import { APPROVALS_POLL_MS, useApprovalsStore } from '@/stores/approvalsStore';

export const APPROVALS_TITLE = 'Approvals';
export const APPROVALS_SUBTITLE =
  'Requests from AI agents that wait for your decision, and the ones already decided.';

export const WAITING_INTRO =
  'An AI agent asked to do each of these. Nothing happens until you approve it; a request you do not decide expires.';
export const WAITING_EMPTY = 'No agent requests are waiting.';
export const HISTORY_INTRO = 'Requests you decided, and the ones that expired.';
export const HISTORY_EMPTY = 'No decided requests yet.';

export function ApprovalsPanel(): ReactNode {
  const waiting = useApprovalsStore((state) => state.waiting);
  const failure = useApprovalsStore((state) => state.failure);
  const history = useApprovalsStore((state) => state.history);
  const historyFailure = useApprovalsStore((state) => state.historyFailure);
  const historyMore = useApprovalsStore((state) => state.historyMore);
  const load = useApprovalsStore((state) => state.load);
  const loadHistory = useApprovalsStore((state) => state.loadHistory);
  const showOlder = useApprovalsStore((state) => state.showOlder);

  useEffect(() => {
    void load();
    void loadHistory();
    const timer = setInterval(() => void loadHistory(), APPROVALS_POLL_MS);
    return () => clearInterval(timer);
  }, [load, loadHistory]);

  const rows = waiting ?? [];

  return (
    <>
      <PageHead title={APPROVALS_TITLE} subtitle={APPROVALS_SUBTITLE} />
      <section aria-labelledby="approvals-waiting-heading" className="mb-10">
        <h2 id="approvals-waiting-heading" className="mb-1 font-semibold" aria-live="polite">
          Waiting for you ({rows.length})
        </h2>
        <p className="mb-4 text-body text-dim">{WAITING_INTRO}</p>
        {failure === null ? null : (
          <FailureNotice failure={failure} surface="The approval queue" className="mb-4" />
        )}
        <div className="mb-4 empty:hidden">
          <ApproveAllSettings waiting={rows} where="page" />
        </div>
        {waiting !== null && rows.length === 0 ? (
          <p className="text-body text-dim" data-testid="approvals-empty">
            {WAITING_EMPTY}
          </p>
        ) : (
          <WaitingList approvals={rows} label="Waiting for you" data-testid="approvals-waiting" />
        )}
      </section>
      <section aria-labelledby="approvals-history-heading">
        <h2 id="approvals-history-heading" className="mb-1 font-semibold">
          History
        </h2>
        <p className="mb-4 text-body text-dim">{HISTORY_INTRO}</p>
        {historyFailure === null ? null : (
          <FailureNotice failure={historyFailure} surface="The approval history" className="mb-4" />
        )}
        {history === null ? null : history.length === 0 ? (
          <p className="text-body text-dim" data-testid="approvals-history-empty">
            {HISTORY_EMPTY}
          </p>
        ) : (
          <ApprovalHistory rows={history} />
        )}
        {historyMore ? (
          <Button className="mt-4" onClick={() => void showOlder()} data-testid="approvals-older">
            Show older
          </Button>
        ) : null}
      </section>
    </>
  );
}
