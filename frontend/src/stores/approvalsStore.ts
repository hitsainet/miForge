/**
 * The approval queue as the shell sees it (F12; FR-012.17).
 *
 * One store, read by two places: the sidebar's badge (the pending count) and the Approvals
 * screen (the list). Polled every {@link APPROVALS_POLL_MS} while either is mounted - there is
 * no socket event for an approval in v1 (FTDD section 6.3).
 *
 * **A decision is sent once.** `approve`/`reject` add the id to `inFlight` before the request
 * and the button reads it, so a double click sends one request; the server's claim is the real
 * guard (exactly-once is `mf_approvals`' conditional update), this is courtesy. After a
 * decision the list is re-read, because the outcome (executed, failed) is the server's.
 */
import { create } from 'zustand';
import type { ApiFailure } from '@/api/errors';
import {
  approveApproval,
  listApprovals,
  rejectApproval,
  type Approval,
  type ApprovalListing,
} from '@/api/approvals';

export const APPROVALS_POLL_MS = 10_000;

export interface ApprovalsState {
  readonly listing: ApprovalListing | null;
  readonly failure: ApiFailure | null;
  /** The last decision's refusal, shown beside the list (a 409 for an expired request). */
  readonly decisionFailure: ApiFailure | null;
  readonly inFlight: ReadonlySet<string>;
  readonly load: () => Promise<void>;
  readonly approve: (id: string) => Promise<Approval | null>;
  readonly reject: (id: string) => Promise<Approval | null>;
}

export const useApprovalsStore = create<ApprovalsState>((set, get) => {
  const decide = async (
    id: string,
    send: (id: string) => ReturnType<typeof approveApproval>,
  ): Promise<Approval | null> => {
    if (get().inFlight.has(id)) return null;
    set({ inFlight: new Set([...get().inFlight, id]), decisionFailure: null });
    const result = await send(id);
    const remaining = new Set(get().inFlight);
    remaining.delete(id);
    set({
      inFlight: remaining,
      decisionFailure: result.ok ? null : result.failure,
    });
    await get().load();
    return result.ok ? result.value : null;
  };

  return {
    listing: null,
    failure: null,
    decisionFailure: null,
    inFlight: new Set<string>(),
    load: async () => {
      const result = await listApprovals();
      set(result.ok ? { listing: result.value, failure: null } : { failure: result.failure });
    },
    approve: (id) => decide(id, approveApproval),
    reject: (id) => decide(id, rejectApproval),
  };
});
