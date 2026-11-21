/**
 * Live progress for WSP runs (2026-11-20): the cadence the two WSP screens read at, and the pieces
 * both draw - the "updated N s ago" line, a run's elapsed clock, its budget bars, the detail
 * page's progress panel and call timeline, a list row's one line of progress, and a waiting
 * approval's row.
 *
 * **Counts and codes only.** Nothing here renders an editor message: the progress and calls
 * routes serve none (`wsp/progress.py`), and every value below is a text node.
 *
 * **The cadence** follows `stores/transport.ts`. A WSP run has no socket, so its transport is
 * permanently `disconnected`, and the record is read at that rule's fallback interval, 5 s: an
 * editor call takes seconds to minutes and a session up to its time budget, so five seconds is
 * finer than the thing it measures without loading the API. The list relaxes to 30 s when no
 * visible run is live and no WSP approval is waiting: then nothing it shows can change except by
 * an agent filing a new request, which the shell's approvals poll (15 s) already watches.
 */
import type { ReactNode } from 'react';
import { MonoId } from '@/components/common/MonoId';
import { Pill } from '@/components/common/Pill';
import { formatEt } from '@/components/common/time';
import type { ApiFailure } from '@/api/errors';
import type { Approval } from '@/api/approvals';
import type { WspBudget, WspCall, WspProgress, WspProgressCompact } from '@/api/wsp';
import { pollIntervalMs } from '@/stores/transport';

/** A live run's read interval: the transport rule's no-socket cadence. */
export const WSP_LIVE_POLL_MS = pollIntervalMs('disconnected');
/** The list's interval when nothing it shows is live. */
export const WSP_IDLE_POLL_MS = 30_000;

/** `"45 s"`, `"3 m 12 s"`, `"1 h 02 m"`; an absent value is `"—"`. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—';
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h} h ${String(m).padStart(2, '0')} m`;
  if (m > 0) return `${m} m ${String(s).padStart(2, '0')} s`;
  return `${s} s`;
}

/**
 * A server-measured duration, carried forward by the local clock since it was read. The server
 * measured `base` at its own `now`; adding only the time since the read keeps a skewed browser
 * clock out of the figure.
 */
export function liveSeconds(
  base: number | null | undefined,
  readAt: number | null,
  now: number,
  live: boolean,
): number | null {
  if (base === null || base === undefined) return null;
  if (!live || readAt === null) return base;
  return base + Math.max(0, (now - readAt) / 1000);
}

/** "updated N s ago", and when the last read failed, that it failed and how old the data is. */
export function UpdatedAgo({
  readAt,
  now,
  failure,
  live,
  testId,
}: {
  readonly readAt: number | null;
  readonly now: number;
  readonly failure: ApiFailure | null;
  readonly live: boolean;
  readonly testId: string;
}): ReactNode {
  const age = readAt === null ? null : formatDuration((now - readAt) / 1000);
  return (
    <span
      className="text-hint text-dim"
      data-testid={testId}
      data-stale={failure ? 'true' : 'false'}
    >
      {readAt === null ? 'not read yet' : `updated ${age} ago`}
      {failure === null
        ? live
          ? ` · reading every ${Math.round(WSP_LIVE_POLL_MS / 1000)} s`
          : ''
        : ` · the last read failed: showing what was read ${age ?? '—'} ago`}
    </span>
  );
}

const BUDGET_LABEL: Readonly<Record<string, string>> = {
  tokens: 'Tokens',
  cost_usd: 'Cost',
  time_s: 'Time',
  steps: 'Steps',
};

function amount(budget: WspBudget, value: number | null): string {
  if (value === null) return '—';
  switch (budget.name) {
    case 'cost_usd':
      return `$${value.toFixed(4)}`;
    case 'time_s':
      return formatDuration(value);
    default:
      return Math.round(value).toLocaleString('en-US');
  }
}

/** One budget: `spent / cap`, and a bar when it has a cap. Steps are labelled an estimate. */
export function BudgetBar({
  budget,
  spent,
}: {
  readonly budget: WspBudget;
  /** The spend to show, when the caller carries it forward (time); else the budget's own. */
  readonly spent?: number | null | undefined;
}): ReactNode {
  const value = spent === undefined ? budget.spent : spent;
  const fraction =
    budget.cap === null || budget.cap <= 0 || value === null
      ? null
      : Math.min(1, value / budget.cap);
  const label = BUDGET_LABEL[budget.name] ?? budget.name;
  const tone =
    fraction === null
      ? 'bg-dim'
      : fraction >= 1
        ? 'bg-red'
        : fraction >= 0.8
          ? 'bg-amber'
          : 'bg-dim';
  return (
    <div className="min-w-0" data-testid={`wsp-budget-${budget.name}`}>
      <div className="flex flex-wrap justify-between gap-2 text-hint">
        <span>
          {label}
          {budget.estimate ? (
            <span className="text-dim" data-testid="wsp-budget-estimate">
              {' '}
              (estimate: editor calls)
            </span>
          ) : null}
        </span>
        <span className="text-dim">
          {amount(budget, value)}
          {budget.cap === null ? ' · no cap' : ` of ${amount(budget, budget.cap)}`}
        </span>
      </div>
      {fraction === null ? null : (
        <div
          className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-panel"
          role="progressbar"
          aria-label={`${label} spent`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(fraction * 100)}
        >
          <div className={`h-full ${tone}`} style={{ width: `${Math.round(fraction * 100)}%` }} />
        </div>
      )}
    </div>
  );
}

function when(iso: string | null | undefined): string {
  return iso === null || iso === undefined ? 'none yet' : formatEt(iso);
}

/** The detail page's progress block. */
export function ProgressPanel({
  progress,
  readAt,
  now,
}: {
  readonly progress: WspProgress;
  readonly readAt: number | null;
  readonly now: number;
}): ReactNode {
  const live = progress.live;
  const elapsed = liveSeconds(progress.elapsed_s, readAt, now, live);
  const worker = progress.worker;
  const heartbeat = liveSeconds(worker?.heartbeat_age_s, readAt, now, live);
  const m = progress.metered;
  return (
    <div data-testid="wsp-progress" data-live={live ? 'true' : 'false'}>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span data-testid="wsp-progress-elapsed">
          {progress.clock_from === 'created_at'
            ? 'Waiting for '
            : live
              ? 'Running for '
              : 'Ran for '}
          {formatDuration(elapsed)}
        </span>
        {live ? <Pill tone="neutral">live</Pill> : <Pill tone="neutral">final figures</Pill>}
      </div>
      {progress.queue === null ? null : (
        <p className="mb-2" data-testid="wsp-progress-queue">
          Place in the queue: {progress.queue.position} of {progress.queue.queued}
          {progress.queue.active_run_id === null ? (
            '; no run is active, so it starts at the driver’s next pass.'
          ) : (
            <>
              ; the worker is busy with <MonoId value={progress.queue.active_run_id} />.
            </>
          )}
        </p>
      )}
      <dl className="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <div className="min-w-0">
          <dt className="text-hint text-dim">Worker</dt>
          <dd data-testid="wsp-progress-worker">
            {worker === null
              ? 'no heartbeat yet'
              : `${worker.state ?? 'state not reported'} · reports ${formatDuration(worker.elapsed_s)} elapsed`}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="text-hint text-dim">Last heartbeat</dt>
          <dd data-testid="wsp-progress-heartbeat">
            {worker === null || worker.last_heartbeat_at === null
              ? 'none yet'
              : live
                ? `${formatDuration(heartbeat)} ago`
                : when(worker.last_heartbeat_at)}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="text-hint text-dim">Editor calls</dt>
          <dd data-testid="wsp-progress-calls">
            {m.calls} metered
            {m.refused === 0 ? '' : ` · ${m.refused} refused`}
            {progress.in_flight === null ? '' : ' · one in flight'}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="text-hint text-dim">Tokens</dt>
          <dd data-testid="wsp-progress-tokens">
            {m.tokens.toLocaleString('en-US')} ({m.input_tokens.toLocaleString('en-US')} in,{' '}
            {m.output_tokens.toLocaleString('en-US')} out)
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="text-hint text-dim">Cost</dt>
          <dd>${m.cost_usd.toFixed(4)}</dd>
        </div>
        <div className="min-w-0">
          <dt className="text-hint text-dim">Last call</dt>
          <dd data-testid="wsp-progress-last-call">{when(m.last_call_at)}</dd>
        </div>
      </dl>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2" data-testid="wsp-progress-budgets">
        {progress.budgets.map((budget) => (
          <BudgetBar
            key={budget.name}
            budget={budget}
            spent={
              budget.name === 'time_s' ? liveSeconds(budget.spent, readAt, now, live) : undefined
            }
          />
        ))}
      </div>
    </div>
  );
}

const OUTCOME_LABEL: Readonly<
  Record<string, { label: string; tone: 'green' | 'red' | 'amber' | 'neutral' }>
> = {
  served: { label: 'served', tone: 'green' },
  upstream_error: { label: 'upstream error', tone: 'amber' },
  upstream_failed: { label: 'no answer', tone: 'red' },
  session_ended: { label: 'session ended', tone: 'red' },
  refused: { label: 'refused', tone: 'amber' },
  closed_at_worst: { label: 'closed at worst case', tone: 'neutral' },
};

/** The detail page's per-call timeline: one row per call the gateway saw, counts and codes. */
export function CallTimeline({ calls }: { readonly calls: readonly WspCall[] }): ReactNode {
  if (calls.length === 0) {
    return (
      <p className="text-dim" data-testid="wsp-calls-none">
        No editor call yet.
      </p>
    );
  }
  return (
    <ol className="stack-y-1" data-testid="wsp-calls">
      {calls.map((call) => {
        const outcome =
          call.outcome === null
            ? { label: 'metered', tone: 'neutral' as const }
            : (OUTCOME_LABEL[call.outcome] ?? { label: call.outcome, tone: 'neutral' as const });
        return (
          <li
            key={call.seq}
            className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line py-1 text-hint"
            data-testid="wsp-call"
            data-outcome={call.outcome ?? 'metered'}
          >
            <span className="text-dim">#{call.seq + 1}</span>
            <span className="text-dim">{call.at === null ? '—' : formatEt(call.at)}</span>
            <Pill tone={outcome.tone}>{outcome.label}</Pill>
            {call.http_status === null ? null : <span>HTTP {call.http_status}</span>}
            {call.latency_ms === null ? null : (
              <span data-testid="wsp-call-latency">{(call.latency_ms / 1000).toFixed(1)} s</span>
            )}
            <span data-testid="wsp-call-tokens">
              {call.input_tokens.toLocaleString('en-US')} in ·{' '}
              {call.output_tokens.toLocaleString('en-US')} out
            </span>
            {call.finish_reason === null ? null : (
              <span className="text-dim">finish: {call.finish_reason}</span>
            )}
            {call.refusal_code === null ? null : <MonoId value={call.refusal_code} />}
          </li>
        );
      })}
    </ol>
  );
}

/** A list row's one line of progress. */
export function CompactProgress({
  progress,
  status,
  readAt,
  now,
}: {
  readonly progress: WspProgressCompact;
  readonly status: string;
  readonly readAt: number | null;
  readonly now: number;
}): ReactNode {
  const elapsed = liveSeconds(progress.elapsed_s, readAt, now, true);
  const heartbeat = liveSeconds(progress.heartbeat_age_s, readAt, now, true);
  const capped = progress.budgets.filter((b) => b.cap !== null && b.fraction !== null);
  return (
    <div className="mt-1 text-hint" data-testid="wsp-run-progress">
      <span data-testid="wsp-run-elapsed">
        {progress.clock_from === 'created_at' ? 'waiting' : status} for {formatDuration(elapsed)}
      </span>
      {progress.queue_position === null ? '' : ` · place ${progress.queue_position} in the queue`}
      {` · ${progress.calls} ${progress.calls === 1 ? 'call' : 'calls'}`}
      {progress.refused === 0 ? '' : ` (${progress.refused} refused)`}
      {` · ${progress.tokens.toLocaleString('en-US')} tokens`}
      {heartbeat === null ? '' : ` · heartbeat ${formatDuration(heartbeat)} ago`}
      {capped.length === 0 ? null : (
        <span className="text-dim" data-testid="wsp-run-budgets">
          {' · '}
          {capped
            .map(
              (b) =>
                `${(BUDGET_LABEL[b.name] ?? b.name).toLowerCase()} ${b.estimate ? '~' : ''}${Math.round((b.fraction ?? 0) * 100)}%`,
            )
            .join(', ')}
        </span>
      )}
    </div>
  );
}

function summaryText(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

/** A WSP run waiting for the operator: a filed `wsp_session` approval with no run yet. */
export function PendingApprovalItem({
  approval,
  onOpen,
}: {
  readonly approval: Approval;
  readonly onOpen: () => void;
}): ReactNode {
  const summary = approval.summary;
  const endpoint = (summary['endpoint'] ?? {}) as Readonly<Record<string, unknown>>;
  const bundle = (summary['bundle'] ?? {}) as Readonly<Record<string, unknown>>;
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className="w-full min-w-0 rounded-card border border-amber/60 bg-card p-card text-left hover:bg-panel"
        data-testid={`wsp-pending-${approval.approval_id}`}
        aria-label={`Open Approvals for waiting WSP request ${approval.approval_id}`}
      >
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Pill tone="amber">Waiting for your approval</Pill>
          <MonoId value={approval.approval_id} />
          <span className="text-hint text-dim">
            {summaryText(summary['kind']) ?? 'kind not stated'} ·{' '}
            {summaryText(summary['mode']) ?? 'mode not stated'}
            {(approval.action as string) === 'wsp_session_external' ? ' · external endpoint' : ''}
          </span>
        </div>
        <div className="mt-1 text-hint text-dim">
          {summaryText(endpoint['model']) === null ? '' : `editor ${String(endpoint['model'])} · `}
          {summaryText(bundle['bundle_id']) === null
            ? ''
            : `bundle ${String(bundle['bundle_id'])} · `}
          asked by {approval.requested_by} · filed {formatEt(approval.created_at)} · open Approvals
          to decide
        </div>
      </button>
    </li>
  );
}
