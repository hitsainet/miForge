/**
 * One WSP run, read-only (fix 2026-11-20): `GET /api/v1/wsp/runs/{id}` (`miforge-wsp-run-v1`)
 * and its transcript, `GET /api/v1/wsp/runs/{id}/transcript`.
 *
 * - **Why it stopped** first: the state, the stop reason, the outcome and the recorded cause
 *   (`failure`), which is what an operator opens a finished run for.
 * - Then what it ran on and what it used: the provider and version, the request's kind and task,
 *   the metered usage, the forward passes spent, the approval it ran under.
 * - Then what it produced: the candidate, its patch and content identities, its lineage, and its
 *   verdict when one is stored.
 * - **Copy and the floating header** (operator request 2026-11-22): every step's text box has a
 *   Copy button, "Copy page" copies the whole screen as text (`pageText`), and the header stays
 *   at the top of the scroll, below the approvals dock (`--approvals-dock-height`). It registers as
 *   pinned (`pinned.ts`, 2026-11-26), so a control reached by Tab lands below it, not behind it.
 * - **Hosted runs (feature 020, FR-020.21).** The usage row's second line is the run's cost in USD
 *   and its cache totals, from its progress read (`metered`; the contract's `usage` never carries
 *   them). The history mode, the handoffs and a refusal with its category are read from the
 *   transcript (its header and events, `sessionFacts`), so they appear once the run has ended; the
 *   effort is the thinking frozen in the request.
 * - Then the transcript, **as data**: every entry is text in a `<pre>` (a text node; React escapes
 *   it), the engine's JSON turns pretty-printed through a guarded parse (`transcript.ts`).
 *
 * The reads fail separately, each naming what it could not read, so a transcript that cannot be
 * read never hides the run.
 *
 * **Live (2026-11-20).** While the record says the run can still change, the page re-reads the
 * run, its progress (`GET .../progress`) and its NEW editor calls (`GET .../calls?after_seq=`)
 * through `useRecordPolling`, at the transport rule's no-socket cadence (5 s; `progress.tsx`).
 * The record decides the stop: once its status is completed, failed or cancelled the polling
 * pauses, and the transcript, which the worker writes only at the end, is read once. A failed
 * read keeps the last good values and says how old they are.
 */
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { pinnedRef } from '@/components/layout/pinned';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { CopyButton } from '@/components/common/CopyButton';
import { pageText } from '@/components/common/pageText';
import { Callout } from '@/components/common/Callout';
import { MonoId } from '@/components/common/MonoId';
import { StatePill } from '@/components/common/Pill';
import { FailureNotice } from '@/components/common/FailureNotice';
import { useNavigation } from '@/contexts/NavigationContext';
import type { ApiFailure } from '@/api/errors';
import {
  fetchWspCalls,
  fetchWspProgress,
  fetchWspRun,
  fetchWspTranscript,
  isLiveWspStatus,
  type WspCall,
  type WspProgress,
  type WspRunDocument,
  type WspRunWindow,
  type WspTranscript,
} from '@/api/wsp';
import { useRecordPolling } from '@/hooks/useRecordPolling';
import { useClock } from '@/hooks/useClock';
import { CallTimeline, ProgressPanel, UpdatedAgo } from './progress';
import { WSP_STATE_TONES, meteredText, usageText, when } from './WspRunsPage';
import {
  answerLabel,
  claimsLine,
  entryView,
  refusalText,
  sessionFacts,
  type SessionFacts,
} from './transcript';
import { UNCONFIRMED, verdictStanding } from './standing';

export const WSP_RUN_SUBTITLE =
  'One weight-space programming run: why it stopped, what it used, what it produced, and its transcript';

/** Opaque, under the approvals dock (z-30) and as far down as that dock is tall. */
const HEADER_CLASS =
  'sticky top-(--approvals-dock-height,0px) z-20 -mt-2 mb-4 border-b border-line bg-bg py-2 ' +
  '[&>[data-testid=shell-page-head]]:mb-1';

function Item({
  label,
  children,
  testId,
}: {
  readonly label: string;
  readonly children: ReactNode;
  readonly testId?: string;
}): ReactNode {
  return (
    <div className="min-w-0">
      <dt className="text-hint text-dim">{label}</dt>
      <dd className="wrap-break-word" data-testid={testId}>
        {children}
      </dd>
    </div>
  );
}

function text(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return JSON.stringify(value);
}

/**
 * The thinking frozen in the run's request (2026-11-21): the value in effect for the session,
 * whatever the endpoint says now. `null` is the server's own default (nothing was sent).
 */
export function runThinkingText(request: Readonly<Record<string, unknown>>): string {
  const thinking = request['thinking'];
  const allowance = request['thinking_tokens'];
  if (thinking === 'on') {
    return typeof allowance === 'number'
      ? `on, ${allowance.toLocaleString('en-US')} reasoning tokens per turn`
      : 'on';
  }
  if (thinking === null || thinking === undefined) return "not set (the server's default)";
  return text(thinking);
}

/**
 * The paging the run froze (2026-11-21): `off` when frozen off; absent means on for
 * `miwsp_engine`, and does not apply to a run of the retired `weightspace` (018-P4), which did
 * not page.
 */
export function pagingText(request: Readonly<Record<string, unknown>>): string {
  if (request['paging'] === 'off') return 'off (every result kept in the conversation)';
  const provider = request['provider'];
  const id =
    provider !== null && typeof provider === 'object'
      ? (provider as Record<string, unknown>)['id']
      : undefined;
  if (id === 'miwsp_engine') return 'on (bulky results paged out; experiment ledger kept)';
  return 'not applicable (this provider does not page)';
}

const EFFORT_VALUES: readonly unknown[] = ['low', 'medium', 'high'];

/**
 * The effort in effect (feature 020, FR-020.21): the thinking frozen in the request when it is an
 * effort level; `default` is the model's own effort (none named); anything else names no effort.
 */
export function effortText(request: Readonly<Record<string, unknown>>): string {
  const thinking = request['thinking'];
  if (EFFORT_VALUES.includes(thinking)) return String(thinking);
  if (thinking === 'default') return "none named: the model's or the server's own";
  return 'none (this run names no effort)';
}

/**
 * The history mode the session ran in (FR-020.6, FR-020.21), from its transcript's header: there
 * is no other record of it. Before the transcript is read, says when it will be.
 */
export function historyText(facts: SessionFacts | null, live: boolean): string {
  if (facts === null) {
    return live ? 'recorded in the transcript, read when the run ends' : 'not read yet';
  }
  if (facts.historyMode === null) return 'not recorded (no engine header in the transcript)';
  if (facts.historyMode === 'append_only') return 'append_only (nothing already sent changed)';
  return `${facts.historyMode} (older tool results may be paged out)`;
}

/** The handoffs the transcript records (FR-020.10, FR-020.21): none, or each by number. */
export function handoffsText(facts: SessionFacts | null, live: boolean): string {
  if (facts === null) {
    return live ? 'recorded in the transcript, read when the run ends' : 'not read yet';
  }
  if (facts.handoffs.length === 0) return 'none';
  const each = facts.handoffs.map((h) =>
    h.conversation === null ? `#${h.n}` : `#${h.n} (conversation ${h.conversation})`,
  );
  return `${facts.handoffs.length}: ${each.join(', ')}`;
}

const tokenCount = (value: number): string => `${value.toLocaleString('en-US')} tokens`;

/**
 * The window the run used and its source (feature 019, FR-019.17), from its progress read: "not
 * recorded" for a run from before 019 (no source), never a number.
 */
export function windowText(window: WspRunWindow | null | undefined): string {
  const source = window?.window_source;
  const size = window?.window_tokens;
  if (source === null || source === undefined || typeof size !== 'number') return 'not recorded';
  if (source === 'stated') return `${tokenCount(size)} (stated on the endpoint)`;
  if (source === 'reported') return `${tokenCount(size)} (reported by the server)`;
  if (source === 'assumed') return `${tokenCount(size)} (assumed: the server reported none)`;
  return `${tokenCount(size)} (${source})`;
}

/**
 * The output ceiling the run used (FR-019.17, F19-D12): "lowered to the window" when the push
 * lowered it to the window the server reported. Since 2026-11-28 the engine also lowers an
 * uncapped session's ceiling after a reply was cut off (step 5d, run wsp_3417a18bc85d); that
 * figure is an eighth of the window or less, so it never equals the window, which tells the two
 * apart without another field.
 */
export function ceilingText(window: WspRunWindow | null | undefined): string {
  const ceiling = window?.output_ceiling_tokens;
  if (window?.window_source == null || typeof ceiling !== 'number') return 'not recorded';
  if (window.output_ceiling_lowered !== true) return tokenCount(ceiling);
  const toWindow = ceiling === window.window_reported_tokens || ceiling === window.window_tokens;
  return toWindow
    ? `${tokenCount(ceiling)}, lowered to the window the server reported`
    : `${tokenCount(ceiling)}, lowered after a reply was cut off`;
}

/** The run's two window warnings (FR-019.17): each only when the backend served its code. */
function WindowWarnings({
  window,
}: {
  readonly window: WspRunWindow | null | undefined;
}): ReactNode {
  const warnings = window?.window_warnings ?? [];
  return (
    <>
      {warnings.includes('window_assumed') ? (
        <Callout
          tone="warning"
          title="The context window was assumed"
          className="mt-3"
          data-testid="wsp-run-window-assumed"
        >
          The server reported no window, so this run assumed{' '}
          {tokenCount(window?.window_tokens ?? 8192)} and may have been starved of context. State
          the window on the endpoint (Editor endpoints, Change, Context window) and start the run
          again.
        </Callout>
      ) : null}
      {warnings.includes('window_stated_above_reported') ? (
        <Callout
          tone="warning"
          title="The stated window is larger than the server reported"
          className="mt-3"
          data-testid="wsp-run-window-mismatch"
        >
          The endpoint states {tokenCount(window?.window_tokens ?? 0)}, and the server reported{' '}
          {tokenCount(window?.window_reported_tokens ?? 0)}. The run used the stated window, so a
          long session can overflow the server. Lower the endpoint's context window, or check the
          server's configuration.
        </Callout>
      ) : null}
    </>
  );
}

function Summary({
  run,
  window,
  progress,
  facts,
  live,
}: {
  readonly run: WspRunDocument;
  readonly window?: WspRunWindow | null;
  readonly progress: WspProgress | null;
  readonly facts: SessionFacts | null;
  readonly live: boolean;
}): ReactNode {
  const request = run.request;
  const m = progress?.metered;
  return (
    <Card className="mb-4" data-testid="wsp-run-summary">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <StatePill state={run.status} tones={WSP_STATE_TONES} data-testid="wsp-run-state" />
        <span data-testid="wsp-run-stop-reason">
          {run.stop_reason === null ? 'no stop reason yet' : `stopped: ${run.stop_reason}`}
        </span>
        <span className="text-dim" data-testid="wsp-run-outcome">
          {run.outcome === null ? '' : `outcome: ${run.outcome}`}
        </span>
      </div>
      {run.failure === null ? null : (
        <Callout tone="error" title="Why it stopped" data-testid="wsp-run-failure">
          <MonoId value={run.failure.code} />{' '}
          <span className="whitespace-pre-wrap wrap-anywhere">{run.failure.message}</span>
        </Callout>
      )}
      {facts?.refusal === null || facts?.refusal === undefined ? null : (
        <Callout
          tone="warning"
          title={refusalText(facts.refusal.category)}
          className="mt-3"
          data-testid="wsp-run-refusal"
        >
          On turn {facts.refusal.turn}. The session ended there, and the run keeps none of its
          edits.
        </Callout>
      )}
      <dl className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Item label="Provider" testId="wsp-run-provider">
          {run.lineage.provider} {run.lineage.provider_version}
        </Item>
        <Item label="Kind and mode">
          {text(request['kind'])} · {text(request['mode'])}
        </Item>
        <Item label="Created">{when(run.created_at)}</Item>
        <Item label="Usage" testId="wsp-run-usage">
          {usageText(run.usage)}
          <br />
          <span data-testid="wsp-run-metered">
            {m === undefined
              ? 'cost and cache: reading the progress…'
              : meteredText({
                  cost_usd: m.cost_usd,
                  cache_read_tokens: m.cache_read_tokens ?? null,
                  cache_write_tokens: m.cache_write_tokens ?? null,
                })}
          </span>
        </Item>
        <Item label="Forward passes used">{text(run.forward_passes_used)}</Item>
        <Item label="Thinking" testId="wsp-run-thinking">
          {runThinkingText(request)}
        </Item>
        <Item label="Output cap" testId="wsp-run-output-cap">
          {request['output_cap'] === 'off' ? 'off (no per-turn cap named)' : 'on'}
        </Item>
        <Item label="Effort" testId="wsp-run-effort">
          {effortText(request)}
        </Item>
        <Item label="History mode" testId="wsp-run-history-mode">
          {historyText(facts, live)}
        </Item>
        <Item label="Handoffs" testId="wsp-run-handoffs">
          {handoffsText(facts, live)}
        </Item>
        <Item label="Paging" testId="wsp-run-paging">
          {pagingText(request)}
        </Item>
        <Item label="Context window" testId="wsp-run-window">
          {windowText(window)}
        </Item>
        <Item label="Output ceiling" testId="wsp-run-ceiling">
          {ceilingText(window)}
        </Item>
        <Item label="Caller reference">{text(run.caller_ref)}</Item>
        <Item label="Approval">
          {run.approval === null ? (
            'none'
          ) : (
            <>
              <MonoId value={run.approval.approval_id} /> ({run.approval.action})
            </>
          )}
        </Item>
        <Item label="Bundle">
          <MonoId value={run.lineage.bundle_sha256} />
        </Item>
      </dl>
      <WindowWarnings window={window} />
      {typeof request['task'] === 'string' && request['task'].length > 0 ? (
        <div className="mt-3">
          <div className="text-hint text-dim">Task</div>
          <pre className="whitespace-pre-wrap wrap-break-word" data-testid="wsp-run-task">
            {request['task']}
          </pre>
        </div>
      ) : null}
    </Card>
  );
}

/**
 * Why a run has no candidate. A `no_change` run with a `content_sha256` changed weights: its
 * verified patch is kept as evidence and was not made a candidate (fix 2026-11-23, D128 round 11:
 * this said "left the weights as they were" for one that had changed 12 of them). The reason comes
 * from the progress read, because `miforge.wsp-run/v1` is closed.
 */
export function noCandidateText(run: WspRunDocument, reason: string | null | undefined): string {
  if (run.outcome === 'no_change') {
    if (run.content_sha256 === null)
      return 'No candidate: the session left the weights as they were.';
    return reason
      ? `No candidate: the edit changed weights, but it was kept as evidence, not as a candidate: ${reason}.`
      : 'No candidate: the edit changed weights, but it was kept as evidence, not as a candidate (no reason was recorded: runs finished before 2026-11-23 did not store one).';
  }
  if (run.outcome === 'report') return 'No candidate: this run reports and never changes weights.';
  return 'No candidate.';
}

function Produced({
  run,
  reason,
}: {
  readonly run: WspRunDocument;
  readonly reason: string | null | undefined;
}): ReactNode {
  return (
    <Card className="mb-4" data-testid="wsp-run-produced">
      <h2 className="mb-2 font-medium">What it produced</h2>
      {run.candidate_id === null ? (
        <p className="text-dim" data-testid="wsp-run-no-candidate">
          {noCandidateText(run, reason)}
        </p>
      ) : (
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Item label="Candidate" testId="wsp-run-candidate">
            <MonoId value={run.candidate_id} />
          </Item>
          <Item label="Patch">
            <MonoId value={run.patch_sha256 ?? '—'} />
          </Item>
          <Item label="Content">
            <MonoId value={run.content_sha256 ?? '—'} />
          </Item>
          <Item label="Lineage">{run.lineage.chain.length} patch(es) from the bundle</Item>
        </dl>
      )}
      {run.verdict === null ? (
        run.candidate_id === null ? null : (
          <p className="mt-2 text-dim">No verdict is stored yet.</p>
        )
      ) : (
        <div className="mt-3" data-testid="wsp-run-verdict">
          <div className="text-hint text-dim">Verdict: {text(run.verdict['value'])}</div>
          <VerdictStanding verdict={run.verdict} />
          <IdenticalBaselines verdict={run.verdict} />
          <pre className="mt-1 max-h-96 overflow-auto whitespace-pre-wrap wrap-break-word text-hint">
            {JSON.stringify(run.verdict, null, 2)}
          </pre>
        </div>
      )}
    </Card>
  );
}

/**
 * Fix 8 (operator, 2026-11-27): the baselines whose scores are the parent's own on a set
 * (`heads[].identical_to_parent`). They stay among the baselines `improved` must beat: beating
 * one is beating the parent. An older verdict has no flag and shows no line.
 */
export function identicalBaselines(verdict: Readonly<Record<string, unknown>>): string[] {
  const sets = Array.isArray(verdict['per_set']) ? (verdict['per_set'] as unknown[]) : [];
  const out: string[] = [];
  for (const entry of sets) {
    if (typeof entry !== 'object' || entry === null) continue;
    const set = entry as Readonly<Record<string, unknown>>;
    const heads = Array.isArray(set['heads']) ? (set['heads'] as unknown[]) : [];
    for (const head of heads) {
      if (typeof head !== 'object' || head === null) continue;
      const h = head as Readonly<Record<string, unknown>>;
      if (h['identical_to_parent'] === true)
        out.push(`${String(h['label'])} on ${String(set['name'])}`);
    }
  }
  return out;
}

function IdenticalBaselines({
  verdict,
}: {
  readonly verdict: Readonly<Record<string, unknown>>;
}): ReactNode {
  const found = identicalBaselines(verdict);
  if (found.length === 0) return null;
  return (
    <div className="text-hint text-dim" data-testid="wsp-run-identical-baselines">
      Identical to the parent (still counted; beating one is beating the parent): {found.join(', ')}
    </div>
  );
}

/**
 * FU-7b: what the verdict's value means for acting on it. `improved` stands only when a primary
 * held-out set scored for the first time by this verdict confirms it (`standing.ts`).
 */
function VerdictStanding({
  verdict,
}: {
  readonly verdict: Readonly<Record<string, unknown>>;
}): ReactNode {
  const reading = verdictStanding(verdict);
  if (reading.standing === UNCONFIRMED) {
    return (
      <Callout
        tone="warning"
        title="Standing: improved_unconfirmed"
        className="mt-2"
        data-testid="wsp-run-verdict-standing"
      >
        Every primary held-out set this candidate beat the control on had been scored before, so the
        gain may fit those sets. Do not act on it until a held-out set never scored before confirms
        it.
      </Callout>
    );
  }
  if (reading.confirmedBy.length > 0) {
    return (
      <div className="text-hint text-dim" data-testid="wsp-run-verdict-standing">
        Standing: improved, confirmed on {reading.confirmedBy.join(', ')} (scored for the first time
        by this verdict)
      </div>
    );
  }
  return null;
}

export function TranscriptView({
  transcript,
  submittedCheck,
}: {
  readonly transcript: WspTranscript;
  /** The run document's `submitted_check` (debt D4): the flag the kept answer's label reads. */
  readonly submittedCheck?: WspRunDocument['submitted_check'];
}): ReactNode {
  const views = transcript.entries.map(entryView);
  // Plan A2: the worker's figures sit beside the last submit, the answer the run kept.
  const lastAnswer = views.reduce((last, view, index) => (view.answer ? index : last), -1);
  const measured = views.find((view) => view.role === 'event: submitted_measured')?.text;
  // FU-2: the editor's claimed figures against that measurement (the last such line).
  const measuredEvent = [...transcript.entries]
    .reverse()
    .find(
      (entry): entry is Record<string, unknown> =>
        typeof entry === 'object' &&
        entry !== null &&
        (entry as Record<string, unknown>)['event'] === 'submitted_measured',
    );
  const claims = measuredEvent === undefined ? null : claimsLine(measuredEvent);
  // Debt D4: the kept answer's label reads the run document's flag, not one fixed sentence.
  const kept = answerLabel(submittedCheck, measuredEvent !== undefined);
  const generic = answerLabel(undefined, false);
  if (transcript.transcript_sha256 === null) {
    return (
      <p className="text-dim" data-testid="wsp-transcript-none">
        No transcript yet: the run has no result.
      </p>
    );
  }
  return (
    <>
      {transcript.partial ? (
        <Callout tone="warning" title="Part of this transcript is not JSON">
          The session ended abnormally, so a line may be cut. Such lines are shown as written.
        </Callout>
      ) : null}
      <ol className="stack-y-2" data-testid="wsp-transcript">
        {views.map((view, index) => {
          return (
            <li
              // Entries have no id of their own; their position is their identity.
              key={index}
              className="rounded-card border border-line p-card"
              data-testid="wsp-transcript-entry"
              data-role={view.role}
            >
              <div className="mb-1 flex items-center justify-between gap-3 text-hint text-dim">
                <span>
                  {index + 1}. {view.role}
                  {view.structured ? ' (JSON, formatted)' : ''}
                </span>
                {view.text.length === 0 ? null : (
                  <CopyButton getText={() => view.text} name={`Copy step ${index + 1} content`} />
                )}
              </div>
              {view.answer ? (
                <div className="mb-1 text-hint">
                  <p
                    className="text-amber"
                    data-testid="wsp-answer-label"
                    data-measured={(index === lastAnswer ? kept : generic).measured}
                  >
                    {(index === lastAnswer ? kept : generic).text}
                  </p>
                  {index === lastAnswer && measured !== undefined ? (
                    <p data-testid="wsp-answer-measured">{measured}</p>
                  ) : null}
                  {index === lastAnswer && claims !== null ? (
                    <p
                      className={claims.differs === true ? 'font-semibold text-red' : undefined}
                      data-testid="wsp-answer-claims"
                      data-differs={String(claims.differs)}
                    >
                      {claims.text}
                    </p>
                  ) : null}
                </div>
              ) : null}
              {view.text.length === 0 ? null : (
                <pre className="max-h-96 overflow-auto whitespace-pre-wrap wrap-break-word text-hint">
                  {view.text}
                </pre>
              )}
              {view.reasoning === null ? null : (
                <details className="mt-1" data-testid="wsp-transcript-reasoning">
                  <summary className="cursor-pointer text-hint text-dim">
                    The editor&apos;s reasoning ({view.reasoning.source}; recorded, never run)
                  </summary>
                  <div className="flex justify-end">
                    <CopyButton
                      getText={() => view.reasoning?.text ?? ''}
                      name={`Copy step ${index + 1} reasoning`}
                    />
                  </div>
                  <pre className="max-h-96 overflow-auto whitespace-pre-wrap wrap-break-word text-hint">
                    {view.reasoning.text}
                  </pre>
                </details>
              )}
              {view.calls.map((call, n) => (
                <div key={n} className="mt-1">
                  <div className="flex items-center justify-between gap-3 text-hint text-dim">
                    <span>calls {call.name}</span>
                    <CopyButton
                      getText={() => call.args}
                      name={`Copy step ${index + 1} call ${n + 1} arguments`}
                    />
                  </div>
                  <pre className="max-h-96 overflow-auto whitespace-pre-wrap wrap-break-word text-hint">
                    {call.args}
                  </pre>
                </div>
              ))}
            </li>
          );
        })}
      </ol>
    </>
  );
}

interface Read {
  readonly runId: string;
  readonly run: WspRunDocument | null;
  readonly runFailure: ApiFailure | null;
  /** When the run document was last read successfully (`Date.now()`). */
  readonly readAt: number | null;
  readonly progress: WspProgress | null;
  readonly progressFailure: ApiFailure | null;
  readonly calls: readonly WspCall[];
  readonly callsFailure: ApiFailure | null;
}

interface TranscriptRead {
  readonly runId: string;
  readonly transcript: WspTranscript | null;
  readonly failure: ApiFailure | null;
}

export interface WspRunPageProps {
  /** Injected in tests. The real ones are the GET routes. */
  readonly fetchRun?: typeof fetchWspRun;
  readonly fetchTranscript?: typeof fetchWspTranscript;
  readonly fetchProgress?: typeof fetchWspProgress;
  readonly fetchCalls?: typeof fetchWspCalls;
}

/** Calls merged by sequence number, oldest first: a re-read page never duplicates a call. */
function mergeCalls(kept: readonly WspCall[], fresh: readonly WspCall[]): WspCall[] {
  const bySeq = new Map<number, WspCall>();
  for (const call of [...kept, ...fresh]) bySeq.set(call.seq, call);
  return [...bySeq.values()].sort((a, b) => a.seq - b.seq);
}

export function WspRunPage({
  fetchRun = fetchWspRun,
  fetchTranscript = fetchWspTranscript,
  fetchProgress = fetchWspProgress,
  fetchCalls = fetchWspCalls,
}: WspRunPageProps = {}): ReactNode {
  const navigation = useNavigation();
  const pageRef = useRef<HTMLDivElement>(null);
  const runId = navigation.params['runId'] ?? null;
  const now = useClock();
  // Each answer is kept with the id it answers, so a second run opened in this screen never
  // shows the first one's values while its own are read.
  const [read_, setRead] = useState<Read | null>(null);
  const [transcriptRead, setTranscriptRead] = useState<TranscriptRead | null>(null);
  // The last call seen, so each poll asks only for newer ones. A ref, so `load` keeps one
  // identity across polls (a new identity would restart the polling effect).
  const lastSeq = useRef<{ runId: string; seq: number } | null>(null);

  const load = useCallback(async () => {
    if (runId === null) return;
    const after = lastSeq.current?.runId === runId ? lastSeq.current.seq : null;
    const [document, progress, page] = await Promise.all([
      fetchRun(runId),
      fetchProgress(runId),
      fetchCalls(runId, after),
    ]);
    if (page.ok && page.value.calls.length > 0) {
      lastSeq.current = { runId, seq: page.value.calls[page.value.calls.length - 1]!.seq };
    }
    setRead((previous) => {
      const kept = previous?.runId === runId ? previous : null;
      return {
        runId,
        run: document.ok ? document.value : (kept?.run ?? null),
        runFailure: document.ok ? null : document.failure,
        readAt: document.ok ? Date.now() : (kept?.readAt ?? null),
        progress: progress.ok ? progress.value : (kept?.progress ?? null),
        progressFailure: progress.ok ? null : progress.failure,
        calls: page.ok
          ? mergeCalls(after === null ? [] : (kept?.calls ?? []), page.value.calls)
          : (kept?.calls ?? []),
        callsFailure: page.ok ? null : page.failure,
      };
    });
  }, [runId, fetchRun, fetchProgress, fetchCalls]);

  const current = read_ !== null && read_.runId === runId ? read_ : null;
  const run = current?.run ?? null;
  // The RECORD decides the stop (useRecordPolling's rule): a run whose status is terminal has
  // nothing left to observe. Until the first answer, the run is treated as live.
  const terminal = run !== null && !isLiveWspStatus(run.status);

  // No socket serves WSP runs, so the transport is permanently `disconnected`: the record is
  // read at that rule's fallback cadence (5 s).
  useRecordPolling({ load, transport: 'disconnected', paused: runId === null || terminal });

  const readTranscript = useCallback(async () => {
    if (runId === null) return;
    const lines = await fetchTranscript(runId);
    setTranscriptRead({
      runId,
      transcript: lines.ok ? lines.value : null,
      failure: lines.ok ? null : lines.failure,
    });
  }, [runId, fetchTranscript]);

  // The transcript exists only once the run has ended: read it once, at that point.
  const transcriptFor = transcriptRead?.runId ?? null;
  useEffect(() => {
    if (terminal && transcriptFor !== runId) void readTranscript();
  }, [terminal, transcriptFor, runId, readTranscript]);

  const runFailure = current?.runFailure ?? null;
  const progress = current?.progress ?? null;
  const ownTranscript = transcriptRead !== null && transcriptRead.runId === runId;
  const transcript = ownTranscript ? transcriptRead.transcript : null;
  const transcriptFailure = ownTranscript ? transcriptRead.failure : null;
  const live = !terminal;

  if (runId === null) {
    return (
      <>
        <PageHead
          title="WSP run"
          subtitle={WSP_RUN_SUBTITLE}
          back={{ screen: 'wsp-runs', label: 'All WSP runs' }}
        />
        <Callout tone="info" title="No WSP run is open">
          Open one from WSP runs. This screen needs a run identifier.
        </Callout>
      </>
    );
  }

  // An ended run is copied only once its transcript is here (or could not be read).
  const transcriptDue = terminal && transcript === null && transcriptFailure === null;

  return (
    <div ref={pageRef} data-testid="wsp-run-page">
      <div ref={pinnedRef} className={HEADER_CLASS} data-testid="wsp-run-header">
        <PageHead
          title={runId}
          subtitle={WSP_RUN_SUBTITLE}
          back={{ screen: 'wsp-runs', label: 'All WSP runs' }}
          actions={
            <>
              <CopyButton
                size="action"
                label="Copy page"
                getText={() => (pageRef.current === null ? '' : pageText(pageRef.current))}
                disabled={run === null || transcriptDue}
                title={transcriptDue ? 'Waiting for the transcript to be read' : undefined}
                data-testid="wsp-copy-page"
              />
              <Button
                onClick={() => {
                  void load();
                  if (terminal) void readTranscript();
                }}
              >
                Re-read the run
              </Button>
            </>
          }
        />
        <UpdatedAgo
          readAt={current?.readAt ?? null}
          now={now}
          failure={runFailure ?? current?.progressFailure ?? null}
          live={live && run !== null}
          testId="wsp-run-updated"
        />
      </div>
      {runFailure === null ? null : (
        <FailureNotice
          failure={runFailure}
          surface="this WSP run"
          className="mb-4"
          data-testid="wsp-run-read-failure"
        />
      )}
      {run === null ? (
        runFailure === null ? (
          <Card data-testid="wsp-run-loading">Reading the WSP run…</Card>
        ) : null
      ) : (
        <>
          <Summary
            run={run}
            window={progress}
            progress={progress}
            facts={transcript === null ? null : sessionFacts(transcript.entries)}
            live={live}
          />
          <Card className="mb-4" data-testid="wsp-run-progress-card">
            <h2 className="mb-2 font-medium">Progress</h2>
            {current?.progressFailure ? (
              <FailureNotice failure={current.progressFailure} surface="this run's progress" />
            ) : null}
            {progress === null ? (
              current?.progressFailure ? null : (
                <p className="text-dim">Reading the progress…</p>
              )
            ) : (
              <ProgressPanel progress={progress} readAt={current?.readAt ?? null} now={now} />
            )}
          </Card>
          <Card className="mb-4" data-testid="wsp-run-calls">
            <h2 className="mb-1 font-medium">Editor calls</h2>
            <p className="mb-2 text-hint text-dim">
              One row per call the gateway saw: its outcome, status, latency and tokens. Never the
              prompt or the answer.
              {live ? ' New calls appear while the run is live.' : ''}
            </p>
            {current?.callsFailure ? (
              <FailureNotice failure={current.callsFailure} surface="this run's editor calls" />
            ) : null}
            <CallTimeline calls={current?.calls ?? []} />
          </Card>
          <Produced run={run} reason={progress?.outcome_reason} />
        </>
      )}
      <Card data-testid="wsp-run-transcript">
        <h2 className="mb-2 font-medium">Transcript</h2>
        {transcriptFailure === null ? null : (
          <FailureNotice failure={transcriptFailure} surface="this run's transcript" />
        )}
        {!terminal ? (
          <p className="text-dim" data-testid="wsp-transcript-pending">
            The worker writes the transcript when the run ends; it is read once then.
          </p>
        ) : transcript === null ? (
          transcriptFailure === null ? (
            <p className="text-dim">Reading the transcript…</p>
          ) : null
        ) : (
          <TranscriptView transcript={transcript} submittedCheck={run?.submitted_check} />
        )}
      </Card>
    </div>
  );
}
