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
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
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
  type WspTranscript,
} from '@/api/wsp';
import { useRecordPolling } from '@/hooks/useRecordPolling';
import { useClock } from '@/hooks/useClock';
import { CallTimeline, ProgressPanel, UpdatedAgo } from './progress';
import { WSP_STATE_TONES, usageText, when } from './WspRunsPage';
import { entryView } from './transcript';

export const WSP_RUN_SUBTITLE =
  'One weight-space programming run: why it stopped, what it used, what it produced, and its transcript';

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
 * `miwsp_engine` and does not apply to `weightspace`, which does not page.
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

function Summary({ run }: { readonly run: WspRunDocument }): ReactNode {
  const request = run.request;
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
          <span className="whitespace-pre-wrap">{run.failure.message}</span>
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
        </Item>
        <Item label="Forward passes used">{text(run.forward_passes_used)}</Item>
        <Item label="Thinking" testId="wsp-run-thinking">
          {runThinkingText(request)}
        </Item>
        <Item label="Output cap" testId="wsp-run-output-cap">
          {request['output_cap'] === 'off' ? 'off (no per-turn cap named)' : 'on'}
        </Item>
        <Item label="Paging" testId="wsp-run-paging">
          {pagingText(request)}
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

function Produced({ run }: { readonly run: WspRunDocument }): ReactNode {
  return (
    <Card className="mb-4" data-testid="wsp-run-produced">
      <h2 className="mb-2 font-medium">What it produced</h2>
      {run.candidate_id === null ? (
        <p className="text-dim" data-testid="wsp-run-no-candidate">
          No candidate
          {run.outcome === 'no_change'
            ? ': the session left the weights as they were.'
            : run.outcome === 'report'
              ? ': this run reports and never changes weights.'
              : '.'}
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
          <pre className="mt-1 max-h-96 overflow-auto whitespace-pre-wrap wrap-break-word text-hint">
            {JSON.stringify(run.verdict, null, 2)}
          </pre>
        </div>
      )}
    </Card>
  );
}

export function TranscriptView({ transcript }: { readonly transcript: WspTranscript }): ReactNode {
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
        {transcript.entries.map((entry, index) => {
          const view = entryView(entry);
          return (
            <li
              // Entries have no id of their own; their position is their identity.
              key={index}
              className="rounded-card border border-line p-card"
              data-testid="wsp-transcript-entry"
              data-role={view.role}
            >
              <div className="mb-1 text-hint text-dim">
                {index + 1}. {view.role}
                {view.structured ? ' (JSON, formatted)' : ''}
              </div>
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
                  <pre className="max-h-96 overflow-auto whitespace-pre-wrap wrap-break-word text-hint">
                    {view.reasoning.text}
                  </pre>
                </details>
              )}
              {view.calls.map((call, n) => (
                <div key={n} className="mt-1">
                  <div className="text-hint text-dim">calls {call.name}</div>
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

  return (
    <>
      <PageHead
        title={runId}
        subtitle={WSP_RUN_SUBTITLE}
        back={{ screen: 'wsp-runs', label: 'All WSP runs' }}
        actions={
          <Button
            onClick={() => {
              void load();
              if (terminal) void readTranscript();
            }}
          >
            Re-read the run
          </Button>
        }
      />
      <div className="mb-3">
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
          <Summary run={run} />
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
          <Produced run={run} />
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
          <TranscriptView transcript={transcript} />
        )}
      </Card>
    </>
  );
}
