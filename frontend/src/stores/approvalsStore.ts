/**
 * The approval queue as the shell sees it (F12; FR-012.17; ENH-008 D6, D7).
 *
 * One store, read by three places: the sidebar's badge (the waiting count), the banner on every
 * screen, and the Approvals screen. **The shell polls the waiting list every
 * {@link APPROVALS_POLL_MS}** - there is no socket event for an approval in v1 (FTDD section
 * 6.3). The History list is read by the Approvals screen while it is open, and after every
 * decision.
 *
 * - `waiting`: `GET /approvals?status=pending`, the soonest to expire first.
 * - `history`: `GET /approvals?limit=N` minus the pending rows (the History filter), the most
 *   recent decision first. **Show older** raises `N` by {@link HISTORY_PAGE} up to the API's
 *   ceiling of {@link APPROVALS_MAX_LIMIT}.
 *
 * **A decision is sent once.** `approve`/`reject` add the id to `inFlight` before the request
 * and the buttons read it, so a double click sends one request; the server's claim is the real
 * guard (exactly-once is `mf_approvals`' conditional update), this is courtesy. After a
 * decision both lists are re-read, because the outcome (executed, failed) is the server's. A
 * refused decision is kept per request (`decisionFailures`) so the card that asked shows it.
 *
 * **Approve all settings changes** (ENH-006 D79): `approveSettingsBatch(ids)` sends the ids the
 * operator confirmed in one request, with every one of them in flight, and keeps the answer
 * (`lastBatch`) or the refusal (`batchFailure`) for the confirm to show.
 */
import { create } from 'zustand';
import type { ApiFailure } from '@/api/errors';
import {
  APPROVALS_MAX_LIMIT,
  approveApproval,
  approveBatch,
  type ApprovalBatch,
  listApprovals,
  rejectApproval,
  type Approval,
} from '@/api/approvals';

/** ENH-008 D7: one poll, every 15 seconds, in all four apps. */
export const APPROVALS_POLL_MS = 15_000;
/** D6: History loads the most recent 200 decided requests; Show older loads 200 more. */
export const HISTORY_PAGE = 200;

export interface ApprovalsState {
  /** The waiting requests, soonest to expire first. `null` until the first read returns. */
  readonly waiting: readonly Approval[] | null;
  readonly failure: ApiFailure | null;
  /** The decided requests, most recent decision first. `null` until History is first read. */
  readonly history: readonly Approval[] | null;
  readonly historyFailure: ApiFailure | null;
  readonly historyLimit: number;
  /** True when the last History read filled its limit, so older rows may exist. */
  readonly historyMore: boolean;
  /** The last refusal of a decision, by request id: shown inside that request's card. */
  readonly decisionFailures: Readonly<Record<string, ApiFailure>>;
  readonly inFlight: ReadonlySet<string>;
  /** The last approve-all answer, or `null`. */
  readonly lastBatch: ApprovalBatch | null;
  /** Why the last approve-all was refused (nothing was approved), or `null`. */
  readonly batchFailure: ApiFailure | null;
  readonly batchInFlight: boolean;
  readonly load: () => Promise<void>;
  readonly loadHistory: () => Promise<void>;
  readonly showOlder: () => Promise<void>;
  readonly approve: (id: string) => Promise<Approval | null>;
  readonly reject: (id: string, reason?: string) => Promise<Approval | null>;
  /** Approve these waiting settings changes with one action (D79). Sent once. */
  readonly approveSettingsBatch: (ids: readonly string[]) => Promise<ApprovalBatch | null>;
}

function timeOf(iso: string | null | undefined): number {
  if (iso === null || iso === undefined) return Number.NaN;
  return Date.parse(iso);
}

/** Soonest to expire first; an unreadable expiry sorts last. */
export function bySoonestExpiry(a: Approval, b: Approval): number {
  const left = timeOf(a.expires_at);
  const right = timeOf(b.expires_at);
  if (Number.isNaN(left)) return Number.isNaN(right) ? 0 : 1;
  if (Number.isNaN(right)) return -1;
  return left - right;
}

/** When a decided request was decided: its decision, else its expiry, else when it was asked. */
export function decidedTime(approval: Approval): number {
  for (const value of [approval.decided_at, approval.expires_at, approval.created_at]) {
    const ms = timeOf(value);
    if (!Number.isNaN(ms)) return ms;
  }
  return 0;
}

/** The History filter: every row that is no longer waiting, most recent decision first. */
export function historyOf(rows: readonly Approval[]): Approval[] {
  return rows
    .filter((row) => row.status !== 'pending')
    .sort((a, b) => decidedTime(b) - decidedTime(a));
}

export const INITIAL_APPROVALS_STATE = {
  waiting: null,
  failure: null,
  history: null,
  historyFailure: null,
  historyLimit: HISTORY_PAGE,
  historyMore: false,
  decisionFailures: {},
  inFlight: new Set<string>(),
  lastBatch: null,
  batchFailure: null,
  batchInFlight: false,
} as const;

export const useApprovalsStore = create<ApprovalsState>((set, get) => {
  const decide = async (
    id: string,
    send: () => ReturnType<typeof approveApproval>,
  ): Promise<Approval | null> => {
    if (get().inFlight.has(id)) return null;
    const cleared = { ...get().decisionFailures };
    delete cleared[id];
    set({ inFlight: new Set([...get().inFlight, id]), decisionFailures: cleared });
    const result = await send();
    const remaining = new Set(get().inFlight);
    remaining.delete(id);
    const failures = { ...get().decisionFailures };
    if (result.ok) delete failures[id];
    else failures[id] = result.failure;
    set({ inFlight: remaining, decisionFailures: failures });
    await Promise.all([get().load(), get().loadHistory()]);
    return result.ok ? result.value : null;
  };

  /** The rows of a list answer; a body without a list reads as an empty queue, never a crash. */
  const rowsOf = (body: { readonly approvals?: unknown } | null | undefined) =>
    (Array.isArray(body?.approvals) ? body.approvals : []) as Approval[];

  return {
    ...INITIAL_APPROVALS_STATE,
    load: async () => {
      const result = await listApprovals({ status: 'pending', limit: APPROVALS_MAX_LIMIT });
      set(
        result.ok
          ? {
              waiting: rowsOf(result.value)
                .filter((row) => row.status === 'pending')
                .sort(bySoonestExpiry),
              failure: null,
            }
          : { failure: result.failure },
      );
    },
    loadHistory: async () => {
      const limit = get().historyLimit;
      const result = await listApprovals({ limit });
      set(
        result.ok
          ? {
              history: historyOf(rowsOf(result.value)),
              historyMore: rowsOf(result.value).length >= limit && limit < APPROVALS_MAX_LIMIT,
              historyFailure: null,
            }
          : { historyFailure: result.failure },
      );
    },
    showOlder: async () => {
      set({ historyLimit: Math.min(APPROVALS_MAX_LIMIT, get().historyLimit + HISTORY_PAGE) });
      await get().loadHistory();
    },
    approve: (id) => decide(id, () => approveApproval(id)),
    reject: (id, reason) => decide(id, () => rejectApproval(id, reason)),
    approveSettingsBatch: async (ids) => {
      if (get().batchInFlight || ids.length === 0) return null;
      set({
        batchInFlight: true,
        batchFailure: null,
        lastBatch: null,
        inFlight: new Set([...get().inFlight, ...ids]),
      });
      const result = await approveBatch(ids);
      const remaining = new Set(get().inFlight);
      for (const id of ids) remaining.delete(id);
      set(
        result.ok
          ? { batchInFlight: false, inFlight: remaining, lastBatch: result.value }
          : { batchInFlight: false, inFlight: remaining, batchFailure: result.failure },
      );
      await Promise.all([get().load(), get().loadHistory()]);
      return result.ok ? result.value : null;
    },
  };
});
