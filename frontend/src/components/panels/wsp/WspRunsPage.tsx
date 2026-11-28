/**
 * WSP runs: every weight-space programming run, newest first, read-only (fix 2026-11-20).
 *
 * The operator looked in Runs for the WSP runs an agent had started and found none: the Runs tab
 * lists training runs (`GET /api/v1/runs`), and nothing on the frontend read `/wsp/` at all.
 * This screen is `GET /api/v1/wsp/runs`, one row per run, each opening `WspRunPage`.
 *
 * A row carries what the list route serves and nothing it does not: the run's provider and
 * version, kind and mode, state, **why it stopped** (the stop reason, the outcome and the recorded
 * cause, which until 2026-11-20 lived only in the worker's result file), the metered usage, the
 * forward passes it spent, when it was created and finished, and the caller's reference. Every
 * value is a text node.
 *
 * **Live (2026-11-20).** The list re-reads itself through `usePolledRead`: every 5 s while any
 * row is live (not completed, failed or cancelled) or a WSP request waits for approval, every 30 s
 * otherwise (`progress.tsx` gives the reasoning). A live row carries its compact progress (elapsed
 * clock, calls, tokens, heartbeat age, budgets); a failed read keeps the last good list and says
 * so. Above the runs, each WSP request still waiting in Approvals is a row of its own (the shell's
 * approvals store, already polled every 15 s), linking to Approvals.
 *
 * **The pool (feature 019 P3).** Above the runs, a card reads `GET /api/v1/wsp/pool` on the same
 * cadence: how many workers are busy, idle or not answering, the pool size against the workers
 * provisioned, what waits per editor endpoint, and a notice when active runs share one server. A
 * failed pool read says so in its own notice and never hides the run list.
 *
 * **Cost and cache (feature 020, FR-020.21).** Each row's second usage line is its `metered`
 * figures, summed from the run's calls by the runs route: "USD 0.4210 · 1,204,800 read from cache ·
 * 41,000 written to cache", or "cache not recorded" for a run from before 020 (`meteredText`).
 *
 * **The pinned header (operator request 2026-11-25).** Everything from the title down to the
 * "N runs, newest first." label stays at the top of `shell-main`, the shell's one scroll
 * container, while the run rows scroll beneath it: the WSP run page's pattern (`position: sticky`,
 * below the approvals dock, opaque `bg-bg`), not a second scroll region for the list. The pinned
 * block is capped at 60 % of the viewport less 4rem (the resource strip's room) and less the
 * approvals dock, never under 4rem, so the list keeps at least about 40 % of the viewport: on a
 * 600 px window, 42 % without the dock and 41 % with it expanded. Past the cap the part above the
 * count label scrolls inside the block, and the label itself never scrolls away. Capping and
 * scrolling inside, rather than unpinning on short windows, keeps the operator's request (all of
 * it stays put) and leaves every item reachable.
 * The block registers as pinned (`pinned.ts`), so `shell-main`'s scroll padding counts it with the
 * approvals dock and a row reached by Tab is scrolled into view below both rather than behind them. The DOM order, so the focus order, and the
 * landmarks are unchanged: the block is a plain `div`.
 *
 * No write action: the screen starts, cancels and approves nothing.
 */
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { pinnedRef } from '@/components/layout/pinned';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { MonoId } from '@/components/common/MonoId';
import { Pill, StatePill, type StateTone } from '@/components/common/Pill';
import { FailureNotice } from '@/components/common/FailureNotice';
import { useNavigation } from '@/contexts/NavigationContext';
import { formatEt } from '@/components/common/time';
import {
  WSP_APPROVAL_ACTIONS,
  getWspPool,
  isLiveWspStatus,
  listWspRuns,
  type WspMetered,
  type WspPool,
  type WspRunListing,
  type WspRunRow,
  type WspUsage,
} from '@/api/wsp';
import type { ApiResult } from '@/api/client';
import type { Approval } from '@/api/approvals';
import { usePolledRead } from '@/hooks/usePolledRead';
import { useClock } from '@/hooks/useClock';
import { useApprovalsStore } from '@/stores/approvalsStore';
import {
  CompactProgress,
  PendingApprovalItem,
  cacheText,
  UpdatedAgo,
  WSP_IDLE_POLL_MS,
  WSP_LIVE_POLL_MS,
} from './progress';
import { countOf } from '@/components/common/plural';

export const WSP_RUNS_TITLE = 'WSP runs';
export const WSP_RUNS_SUBTITLE =
  'Weight-space programming sessions: an editor model reads and changes a probe’s weights';

/**
 * The pinned block: sticky at the top of `shell-main`, under the approvals dock (z-30) and as far
 * down as that dock is tall, opaque in both themes, and capped so the run list keeps about 40 %
 * of a short window (`max(4rem, 60dvh - 4rem - dock)`). A flex column, so only its upper part
 * shrinks and scrolls; the count label below it stays in view.
 */
export const WSP_RUNS_PINNED_CLASS =
  'sticky top-(--approvals-dock-height,0px) z-20 -mt-2 mb-3 flex flex-col border-b border-line ' +
  'bg-bg py-2 max-h-[max(4rem,calc(60dvh_-_4rem_-_var(--approvals-dock-height,0px)))]';

/** The part of the pinned block above the count label: it scrolls only past the cap. */
export const WSP_RUNS_PINNED_SCROLL_CLASS =
  'min-h-0 shrink overflow-y-auto [&>[data-testid=shell-page-head]]:mb-3';

/** The WSP run states (`miforge-wsp-run-v1`), which are not the training run's. */
export const WSP_STATE_TONES: Readonly<Record<string, StateTone>> = {
  queued: { tone: 'neutral', label: 'Queued' },
  pushing: { tone: 'neutral', label: 'Pushing' },
  running: { tone: 'accent', label: 'Running' },
  collecting: { tone: 'neutral', label: 'Collecting' },
  completed: { tone: 'green', label: 'Completed' },
  failed: { tone: 'red', label: 'Failed' },
  cancelled: { tone: 'neutral', label: 'Cancelled' },
};

/** Each pool word, as the card shows it (`wsp/pool.py`). */
export const WSP_POOL_WORDS: Readonly<
  Record<string, { tone: 'neutral' | 'amber' | 'green'; label: string }>
> = {
  busy: { tone: 'neutral', label: 'busy' },
  idle: { tone: 'green', label: 'idle' },
  // Operator 2026-11-26: its session ended and its pod is being replaced; it takes work again
  // once a fresh pod answers.
  restarting: { tone: 'neutral', label: 'restarting' },
  not_answering: { tone: 'amber', label: 'not answering' },
};

function poolWord(state: string): { tone: 'neutral' | 'amber' | 'green'; label: string } {
  return WSP_POOL_WORDS[state] ?? { tone: 'neutral', label: state };
}

export function PoolCard({ pool }: { readonly pool: WspPool }): ReactNode {
  const waiting =
    pool.queued_by_endpoint.reduce((n, row) => n + row.queued, 0) + pool.queued_without_endpoint;
  return (
    <Card className="mb-4" data-testid="wsp-pool">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="font-medium">WSP workers</span>
        <span className="text-hint text-dim" data-testid="wsp-pool-size">
          {pool.effective} in use of {pool.maximum} provisioned (setting: {pool.pool_size})
        </span>
      </div>
      <div className="mb-2 flex flex-wrap items-center gap-2" data-testid="wsp-pool-counts">
        <Pill tone="neutral">{pool.busy} busy</Pill>
        <Pill tone="green">{pool.idle} idle</Pill>
        {pool.restarting ? <Pill tone="neutral">{pool.restarting} restarting</Pill> : null}
        <Pill tone="amber">{pool.not_answering} not answering</Pill>
      </div>
      {pool.workers.length === 0 ? (
        <p className="text-hint text-dim" data-testid="wsp-pool-unread">
          No worker has been read yet: the driver reads each worker every pass.
        </p>
      ) : (
        <ul className="stack-y-1" data-testid="wsp-pool-workers">
          {pool.workers.map((worker) => {
            const word = poolWord(worker.state);
            return (
              <li key={worker.index} className="flex flex-wrap items-center gap-2 text-hint">
                <span>Worker {worker.index}</span>
                <Pill tone={word.tone} data-testid={`wsp-pool-worker-${worker.index}`}>
                  {word.label}
                </Pill>
                {worker.run_id === null ? null : <MonoId value={worker.run_id} />}
                <span className="text-dim">read {when(worker.read_at)}</span>
              </li>
            );
          })}
        </ul>
      )}
      <p className="mt-2 text-hint text-dim" data-testid="wsp-pool-queue">
        {waiting === 0
          ? 'Nothing waits for a worker.'
          : `${waiting} queued: ${[
              ...pool.queued_by_endpoint.map((row) => `${row.queued} for ${row.endpoint_id}`),
              ...(pool.queued_without_endpoint > 0
                ? [countOf(pool.queued_without_endpoint, 'replay or rerun', 'replays or reruns')]
                : []),
            ].join(', ')}. One run is active per editor endpoint, and one per miServer server.`}
      </p>
      {pool.shared_base_urls.length === 0 ? null : (
        <div className="mt-2 text-hint" data-testid="wsp-pool-shared">
          {pool.shared_base_urls.map((group) => (
            <p key={group.base_url}>
              {group.run_ids.length} active runs share one server, {group.base_url} (endpoints{' '}
              {group.endpoint_ids.join(', ')}): they compete for it, and each may run slower.
            </p>
          ))}
        </div>
      )}
    </Card>
  );
}

export function when(iso: string | null | undefined): string {
  return iso === null || iso === undefined ? 'not recorded' : formatEt(iso);
}

export function usageText(usage: WspUsage | undefined): string {
  if (usage === undefined) return 'usage not recorded';
  const tokens = usage.input_tokens + usage.output_tokens;
  return `${usage.requests} ${usage.requests === 1 ? 'request' : 'requests'} · ${tokens.toLocaleString('en-US')} tokens (${usage.input_tokens.toLocaleString('en-US')} in, ${usage.output_tokens.toLocaleString('en-US')} out)`;
}

/**
 * A run's cost and cache totals (FR-020.21): "USD 0.4210 · 1,204,800 read from cache · 41,000
 * written to cache", or "USD 0.0000 · cache not recorded"; "cost and cache not recorded" when
 * the backend served no `metered` (an older backend).
 */
export function meteredText(metered: WspMetered | null | undefined): string {
  if (metered === null || metered === undefined) return 'cost and cache not recorded';
  return `USD ${metered.cost_usd.toFixed(4)} · ${cacheText(metered.cache_read_tokens, metered.cache_write_tokens)}`;
}

export function WspRunItem({
  row,
  onOpen,
  readAt = null,
  now = 0,
}: {
  readonly row: WspRunRow;
  readonly onOpen: (runId: string) => void;
  /** When the list holding this row was read, and the clock now, for the row's live clock
   * (without them the row shows the figures as served). */
  readonly readAt?: number | null;
  readonly now?: number;
}): ReactNode {
  const failure = row.failure ?? null;
  return (
    <li>
      <button
        type="button"
        onClick={() => onOpen(row.run_id)}
        className="w-full min-w-0 rounded-card border border-line bg-card p-card text-left hover:bg-panel"
        data-testid={`wsp-run-row-${row.run_id}`}
        aria-label={`Open WSP run ${row.run_id}`}
      >
        <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <MonoId value={row.run_id} />
            <div className="mt-1 flex flex-wrap items-center gap-2 text-hint text-dim">
              <StatePill state={row.status} tones={WSP_STATE_TONES} />
              <span data-testid="wsp-run-kind">
                {row.kind} · {row.mode}
              </span>
              <span data-testid="wsp-run-provider">
                {row.provider ?? 'provider not recorded'}
                {row.provider_version === null || row.provider_version === undefined
                  ? ''
                  : ` ${row.provider_version}`}
              </span>
              {row.caller_ref === null ? null : (
                <span className="min-w-0 wrap-anywhere" data-testid="wsp-run-caller-ref">
                  for {row.caller_ref}
                </span>
              )}
            </div>
            <div className="mt-1 text-hint" data-testid="wsp-run-stop">
              {row.stop_reason === null ? 'no stop reason yet' : `stopped: ${row.stop_reason}`}
              {row.outcome === null ? '' : ` · outcome: ${row.outcome}`}
              {row.outcome_reason ? ` (${row.outcome_reason})` : ''}
            </div>
            {row.progress === null || row.progress === undefined ? null : (
              <CompactProgress
                progress={row.progress}
                status={row.status}
                readAt={readAt}
                now={now}
              />
            )}
            {failure === null ? null : (
              <div className="mt-1 text-hint text-red wrap-break-word" data-testid="wsp-run-cause">
                <MonoId value={failure.code} /> {failure.message}
              </div>
            )}
          </div>
          <div className="text-right text-hint text-dim">
            <div data-testid="wsp-run-usage">{usageText(row.usage)}</div>
            <div data-testid="wsp-run-metered">{meteredText(row.metered)}</div>
            <div data-testid="wsp-run-forward">
              {row.forward_passes_used === null
                ? 'forward passes not recorded'
                : `${row.forward_passes_used} forward ${row.forward_passes_used === 1 ? 'pass' : 'passes'}`}
            </div>
            <div data-testid="wsp-run-created">created {when(row.created_at)}</div>
            <div data-testid="wsp-run-finished">
              {row.finished_at === null || row.finished_at === undefined
                ? 'not finished'
                : `finished ${when(row.finished_at)}`}
            </div>
          </div>
        </div>
      </button>
    </li>
  );
}

export interface WspRunsPageProps {
  /** Injected in tests. The real one is `GET /api/v1/wsp/runs`. */
  readonly list?: typeof listWspRuns;
  /** Injected in tests. The real one is `GET /api/v1/wsp/pool`. */
  readonly readPool?: typeof getWspPool;
}

interface Read {
  readonly listing: WspRunListing;
  readonly readAt: number;
  /** Which "Re-read the list" press this read answers. */
  readonly asked: number;
}

/** The WSP requests waiting for the operator: filed `wsp_session` approvals with no run yet. */
export function pendingWspApprovals(waiting: readonly Approval[] | null): Approval[] {
  return (waiting ?? []).filter(
    (approval) => approval.status === 'pending' && WSP_APPROVAL_ACTIONS.has(approval.action),
  );
}

export function WspRunsPage({
  list = listWspRuns,
  readPool = getWspPool,
}: WspRunsPageProps = {}): ReactNode {
  const navigation = useNavigation();
  const now = useClock();
  // "Re-read the list" changes the read, which re-reads at once (usePolledRead restarts).
  const [asked, setAsked] = useState(0);
  const load = useCallback(async (): Promise<ApiResult<Read>> => {
    const result = await list();
    return result.ok
      ? { ok: true, value: { listing: result.value, readAt: Date.now(), asked } }
      : result;
  }, [list, asked]);
  const waiting = useApprovalsStore((state) => state.waiting);
  const pending = useMemo(() => pendingWspApprovals(waiting), [waiting]);
  // The cadence follows what the last read showed (adjusted during render, React's pattern for
  // state derived from a previous render): 5 s while anything is live, 30 s otherwise.
  const [pollMs, setPollMs] = useState(WSP_LIVE_POLL_MS);
  const { value, failure } = usePolledRead(load, pollMs);
  // Feature 019 P3: the pool, on the same cadence; its own failure never hides the run list.
  const { value: pool, failure: poolFailure } = usePolledRead(readPool, pollMs);
  const listing = value?.listing ?? null;
  const rows = listing?.runs ?? [];
  const anyLive = rows.some((row) => isLiveWspStatus(row.status)) || pending.length > 0;
  const wanted = listing === null || anyLive ? WSP_LIVE_POLL_MS : WSP_IDLE_POLL_MS;
  if (wanted !== pollMs) setPollMs(wanted);

  const live = rows.some((row) => isLiveWspStatus(row.status));
  return (
    <>
      {/* Pinned (operator request 2026-11-25): everything down to the count label. */}
      <div ref={pinnedRef} className={WSP_RUNS_PINNED_CLASS} data-testid="wsp-runs-pinned">
        <div className={WSP_RUNS_PINNED_SCROLL_CLASS} data-testid="wsp-runs-pinned-scroll">
          <PageHead
            title={WSP_RUNS_TITLE}
            subtitle={WSP_RUNS_SUBTITLE}
            actions={
              <Button onClick={() => setAsked((n) => n + 1)} data-testid="wsp-runs-reread">
                Re-read the list
              </Button>
            }
          />
          <p className="mb-4 text-hint text-dim" data-testid="wsp-runs-explanation">
            Training runs are on Runs. A weight-space programming (WSP) run starts when an agent
            asks and you approve it in Approvals; this list only reads, and keeps itself current.
          </p>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <UpdatedAgo
              readAt={value?.readAt ?? null}
              now={now}
              failure={failure}
              live={anyLive}
              testId="wsp-runs-updated"
            />
            {anyLive ? null : (
              <span className="text-hint text-dim" data-testid="wsp-runs-idle">
                · nothing live: reading every {Math.round(WSP_IDLE_POLL_MS / 1000)} s
              </span>
            )}
          </div>
          {failure === null ? null : (
            <FailureNotice
              failure={failure}
              surface="the WSP run list"
              className="mb-4"
              data-testid="wsp-runs-failure"
            />
          )}
          {pool === null ? null : <PoolCard pool={pool} />}
          {poolFailure === null ? null : (
            <FailureNotice
              failure={poolFailure}
              surface="the WSP worker pool"
              className="mb-4"
              data-testid="wsp-pool-failure"
            />
          )}
          {pending.length === 0 ? null : (
            <ul className="mb-4 stack-y-2" data-testid="wsp-pending-rows">
              {pending.map((approval) => (
                <PendingApprovalItem
                  key={approval.approval_id}
                  approval={approval}
                  onOpen={() => navigation.go('approvals')}
                />
              ))}
            </ul>
          )}
        </div>
        {rows.length === 0 ? null : (
          <div className="shrink-0 text-hint text-dim" data-testid="wsp-runs-count">
            {rows.length} {rows.length === 1 ? 'run' : 'runs'}, newest first.
            {live ? (
              <>
                {' '}
                <Pill tone="neutral">live rows update on their own</Pill>
              </>
            ) : null}
          </div>
        )}
      </div>
      {listing === null ? (
        failure === null ? (
          <Card data-testid="wsp-runs-loading">Reading the WSP run list…</Card>
        ) : null
      ) : rows.length === 0 ? (
        <Card data-testid="wsp-runs-empty">
          No WSP run has been started yet. An agent starts one through miForge&apos;s tools, and it
          waits in Approvals until you approve it.
        </Card>
      ) : (
        <ul className="stack-y-2" data-testid="wsp-runs-rows">
          {rows.map((row) => (
            <WspRunItem
              key={row.run_id}
              row={row}
              readAt={value?.readAt ?? null}
              now={now}
              onOpen={(runId) => navigation.go('wsp-run', { runId })}
            />
          ))}
        </ul>
      )}
    </>
  );
}
