/**
 * The WSP screens keep themselves current (live progress, 2026-11-20).
 *
 * Before, both pages read once on mount and the operator refreshed by hand to see a new run or
 * any progress. Each test here drives the real page with injected reads and fake timers, so the
 * polling start, the terminal stop, the transcript-at-the-end read, failure retention, the
 * progress panel, the call timeline, the list's cadence and the waiting-approval rows are each
 * red when their wiring is removed.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, within } from '@testing-library/react';
import { NavigationProvider, useNavigation } from '@/contexts/NavigationContext';
import type { ApiResult } from '@/api/client';
import type { Approval } from '@/api/approvals';
import type { WspCall, WspCallPage, WspProgress, WspRunDocument, WspRunRow } from '@/api/wsp';
import {
  WSP_RUN_CALLS,
  WSP_RUN_DOCUMENT,
  WSP_RUN_ID,
  WSP_RUN_PROGRESS,
  WSP_RUN_ROW,
} from '@/testing/stubBackend';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';
import { WspRunPage } from './WspRunPage';
import { WspRunsPage } from './WspRunsPage';
import { WSP_IDLE_POLL_MS, WSP_LIVE_POLL_MS, formatDuration } from './progress';

const ok = <T,>(value: T): Promise<ApiResult<T>> => Promise.resolve({ ok: true, value });
const down = <T,>(): Promise<ApiResult<T>> =>
  Promise.resolve({
    ok: false,
    failure: { kind: 'transport', status: 0, cause: 'connection refused' },
  });

const flush = () => act(() => vi.advanceTimersByTimeAsync(0));
const wait = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

function liveDocument(status: string): WspRunDocument {
  return {
    ...(WSP_RUN_DOCUMENT as unknown as WspRunDocument),
    status,
    stop_reason: null,
    outcome: null,
    failure: null,
  };
}

const LIVE_PROGRESS: WspProgress = {
  ...(WSP_RUN_PROGRESS as unknown as WspProgress),
  status: 'running',
  live: true,
  finished_at: null,
  elapsed_s: 125,
  worker: {
    state: 'running',
    elapsed_s: 120,
    last_heartbeat_at: '2026-11-20T15:02:00Z',
    heartbeat_age_s: 4,
  },
  metered: {
    calls: 2,
    refused: 1,
    input_tokens: 900,
    output_tokens: 300,
    tokens: 1200,
    cost_usd: 0.0123,
    last_call_at: '2026-11-20T15:02:01Z',
  },
  budgets: [
    { name: 'tokens', unit: 'tokens', spent: 1200, cap: 10000, fraction: 0.12, estimate: false },
    { name: 'cost_usd', unit: 'USD', spent: 0.0123, cap: 1, fraction: 0.0123, estimate: false },
    { name: 'time_s', unit: 's', spent: 120, cap: 600, fraction: 0.2, estimate: false },
    { name: 'steps', unit: 'editor calls', spent: 2, cap: 50, fraction: 0.04, estimate: true },
  ],
};

function call(seq: number, extra: Partial<WspCall> = {}): WspCall {
  return {
    seq,
    at: '2026-11-20T15:02:01Z',
    outcome: 'served',
    http_status: 200,
    latency_ms: 1500,
    finish_reason: 'tool_calls',
    refusal_code: null,
    input_tokens: 450,
    output_tokens: 150,
    cost_usd: 0,
    ...extra,
  };
}

function page(calls: readonly WspCall[], status = 'running'): WspCallPage {
  return { run_id: WSP_RUN_ID, status, live: status !== 'completed', calls, more: false };
}

function Where(): React.ReactNode {
  const navigation = useNavigation();
  return <span data-testid="where">{navigation.screen}</span>;
}

beforeEach(() => {
  vi.useFakeTimers();
  useApprovalsStore.setState(INITIAL_APPROVALS_STATE);
});

afterEach(() => {
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------- the detail page

interface Reads {
  status: string;
  runFails: boolean;
  calls: WspCall[];
}

function mountRun(reads: Reads) {
  const fetchRun = vi.fn(() => (reads.runFails ? down<never>() : ok(liveDocument(reads.status))));
  const fetchProgress = vi.fn(() =>
    ok({
      ...LIVE_PROGRESS,
      status: reads.status,
      live: !['completed', 'failed', 'cancelled'].includes(reads.status),
    }),
  );
  const fetchCalls = vi.fn((_id: string, after: number | null = null) =>
    ok(
      page(
        reads.calls.filter((c) => after === null || c.seq > after),
        reads.status,
      ),
    ),
  );
  const fetchTranscript = vi.fn(() =>
    ok({
      run_id: WSP_RUN_ID,
      transcript_sha256: 'c'.repeat(64),
      entries: [{ role: 'user', content: 'Repair the probe.' }],
      partial: false,
    }),
  );
  render(
    <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
      <WspRunPage
        fetchRun={fetchRun}
        fetchProgress={fetchProgress}
        fetchCalls={fetchCalls}
        fetchTranscript={fetchTranscript}
      />
    </NavigationProvider>,
  );
  return { fetchRun, fetchProgress, fetchCalls, fetchTranscript };
}

describe('[WSP run] live polling', () => {
  it('should poll a live run every 5 s, stop when the record is terminal, then read the transcript once', async () => {
    const reads: Reads = { status: 'running', runFails: false, calls: [call(0)] };
    const f = mountRun(reads);
    await flush();
    expect(f.fetchRun).toHaveBeenCalledTimes(1);
    expect(f.fetchProgress).toHaveBeenCalledTimes(1);
    expect(f.fetchCalls).toHaveBeenCalledTimes(1);
    // A live run has no transcript yet: it is not read.
    expect(f.fetchTranscript).not.toHaveBeenCalled();
    expect(screen.getByTestId('wsp-transcript-pending')).toBeTruthy();
    expect(WSP_LIVE_POLL_MS).toBe(5_000);
    await wait(WSP_LIVE_POLL_MS);
    expect(f.fetchRun).toHaveBeenCalledTimes(2);
    expect(f.fetchProgress).toHaveBeenCalledTimes(2);
    expect(f.fetchCalls).toHaveBeenCalledTimes(2);
    reads.status = 'completed';
    await wait(WSP_LIVE_POLL_MS);
    expect(f.fetchRun).toHaveBeenCalledTimes(3);
    expect(f.fetchTranscript).toHaveBeenCalledTimes(1);
    // Terminal: nothing more is read, however long the page stays open.
    await wait(10 * WSP_LIVE_POLL_MS);
    expect(f.fetchRun).toHaveBeenCalledTimes(3);
    expect(f.fetchProgress).toHaveBeenCalledTimes(3);
    expect(f.fetchCalls).toHaveBeenCalledTimes(3);
    expect(f.fetchTranscript).toHaveBeenCalledTimes(1);
    expect(screen.getAllByTestId('wsp-transcript-entry')).toHaveLength(1);
  });

  it('should read a run that is already terminal once, and its transcript once', async () => {
    const f = mountRun({ status: 'failed', runFails: false, calls: [] });
    await flush();
    await wait(6 * WSP_LIVE_POLL_MS);
    expect(f.fetchRun).toHaveBeenCalledTimes(1);
    expect(f.fetchTranscript).toHaveBeenCalledTimes(1);
  });

  it('should ask only for calls after the last one seen, and add them to the timeline', async () => {
    const reads: Reads = { status: 'running', runFails: false, calls: [call(0)] };
    const f = mountRun(reads);
    await flush();
    expect(f.fetchCalls.mock.calls[0]).toEqual([WSP_RUN_ID, null]);
    expect(screen.getAllByTestId('wsp-call')).toHaveLength(1);
    reads.calls = [call(0), call(1, { outcome: 'refused', refusal_code: 'wsp_budget_exhausted' })];
    await wait(WSP_LIVE_POLL_MS);
    expect(f.fetchCalls.mock.calls[1]).toEqual([WSP_RUN_ID, 0]);
    const rows = screen.getAllByTestId('wsp-call');
    expect(rows).toHaveLength(2);
    expect(rows[1]).toHaveAttribute('data-outcome', 'refused');
    expect(rows[1]).toHaveTextContent('wsp_budget_exhausted');
    await wait(WSP_LIVE_POLL_MS);
    expect(f.fetchCalls.mock.calls[2]).toEqual([WSP_RUN_ID, 1]);
    expect(screen.getAllByTestId('wsp-call')).toHaveLength(2); // no duplicate
  });

  it('should keep the last good run on screen when a read fails, and say so', async () => {
    const reads: Reads = { status: 'running', runFails: false, calls: [] };
    mountRun(reads);
    await flush();
    expect(screen.getByTestId('wsp-run-summary')).toBeTruthy();
    expect(screen.getByTestId('wsp-run-updated')).toHaveTextContent('updated 0 s ago');
    reads.runFails = true;
    await wait(WSP_LIVE_POLL_MS);
    // Still there, beside a named failure and the age of what is shown.
    expect(screen.getByTestId('wsp-run-summary')).toBeTruthy();
    expect(screen.getByTestId('wsp-run-read-failure')).toBeTruthy();
    const updated = screen.getByTestId('wsp-run-updated');
    expect(updated).toHaveAttribute('data-stale', 'true');
    expect(updated).toHaveTextContent('the last read failed: showing what was read 5 s ago');
    // Polling goes on after a failure, and a good read clears it.
    reads.runFails = false;
    await wait(WSP_LIVE_POLL_MS);
    expect(screen.queryByTestId('wsp-run-read-failure')).toBeNull();
    expect(screen.getByTestId('wsp-run-updated')).toHaveAttribute('data-stale', 'false');
  });
});

describe('[WSP run] the progress panel and the timeline', () => {
  it('should show the heartbeat, the counts and each budget against its cap', async () => {
    mountRun({ status: 'running', runFails: false, calls: [call(0)] });
    await flush();
    const panel = screen.getByTestId('wsp-progress');
    expect(panel).toHaveAttribute('data-live', 'true');
    expect(within(panel).getByTestId('wsp-progress-elapsed')).toHaveTextContent(
      'Running for 2 m 05 s',
    );
    expect(within(panel).getByTestId('wsp-progress-worker')).toHaveTextContent(
      'running · reports 2 m 00 s elapsed',
    );
    expect(within(panel).getByTestId('wsp-progress-heartbeat')).toHaveTextContent('4 s ago');
    expect(within(panel).getByTestId('wsp-progress-calls')).toHaveTextContent(
      '2 metered · 1 refused',
    );
    expect(within(panel).getByTestId('wsp-progress-tokens')).toHaveTextContent(
      '1,200 (900 in, 300 out)',
    );
    const bars = within(panel).getAllByRole('progressbar');
    expect(bars.map((b) => b.getAttribute('aria-valuenow'))).toEqual(['12', '1', '20', '4']);
    expect(within(panel).getByTestId('wsp-budget-steps')).toHaveTextContent(
      'Steps (estimate: editor calls)2 of 50',
    );
    expect(within(panel).getByTestId('wsp-budget-tokens')).toHaveTextContent('1,200 of 10,000');
    // The clock moves between reads: the elapsed time and the heartbeat age carry forward.
    await wait(3_000);
    expect(within(panel).getByTestId('wsp-progress-elapsed')).toHaveTextContent('2 m 08 s');
    expect(within(panel).getByTestId('wsp-progress-heartbeat')).toHaveTextContent('7 s ago');
  });

  it('should say where a queued run stands in the queue', async () => {
    const fetchProgress = () =>
      ok({
        ...LIVE_PROGRESS,
        status: 'queued',
        clock_from: 'created_at' as const,
        worker: null,
        queue: { position: 2, queued: 3, active_run_id: 'wsp_aaaaaaaaaaaa' },
      });
    render(
      <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
        <WspRunPage
          fetchRun={() => ok(liveDocument('queued'))}
          fetchProgress={fetchProgress}
          fetchCalls={() => ok(page([], 'queued'))}
        />
      </NavigationProvider>,
    );
    await flush();
    expect(screen.getByTestId('wsp-progress-queue')).toHaveTextContent(
      'Place in the queue: 2 of 3; the worker is busy with wsp_aaaaaaaaaaaa',
    );
    expect(screen.getByTestId('wsp-progress-elapsed')).toHaveTextContent('Waiting for');
    expect(screen.getByTestId('wsp-progress-worker')).toHaveTextContent('no heartbeat yet');
    expect(screen.getByTestId('wsp-calls-none')).toBeTruthy();
  });

  it('should draw each call with its outcome, status, latency and tokens, and no text', async () => {
    mountRun({
      status: 'running',
      runFails: false,
      calls: [
        call(0),
        call(1, {
          outcome: 'upstream_failed',
          http_status: null,
          refusal_code: 'wsp_gateway_upstream_failed',
          finish_reason: null,
        }),
        call(2, { outcome: null, latency_ms: null, http_status: null, finish_reason: null }),
      ],
    });
    await flush();
    const [served, failed, old] = screen.getAllByTestId('wsp-call');
    expect(served).toHaveTextContent('#1');
    expect(served).toHaveTextContent('served');
    expect(served).toHaveTextContent('HTTP 200');
    expect(served).toHaveTextContent('1.5 s');
    expect(served).toHaveTextContent('450 in · 150 out');
    expect(served).toHaveTextContent('finish: tool_calls');
    expect(failed).toHaveTextContent('no answer');
    expect(failed).toHaveTextContent('wsp_gateway_upstream_failed');
    expect(old).toHaveAttribute('data-outcome', 'metered'); // recorded before outcomes were
  });

  it('should show the final figures of the stub run through its own reads', async () => {
    render(
      <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
        <WspRunPage
          fetchRun={() => ok(WSP_RUN_DOCUMENT as unknown as WspRunDocument)}
          fetchProgress={() => ok(WSP_RUN_PROGRESS as unknown as WspProgress)}
          fetchCalls={() => ok(WSP_RUN_CALLS as unknown as WspCallPage)}
          fetchTranscript={() =>
            ok({ run_id: WSP_RUN_ID, transcript_sha256: null, entries: [], partial: false })
          }
        />
      </NavigationProvider>,
    );
    await flush();
    expect(screen.getByTestId('wsp-progress')).toHaveAttribute('data-live', 'false');
    expect(screen.getByTestId('wsp-progress-elapsed')).toHaveTextContent('Ran for 4 s');
    expect(screen.getByTestId('wsp-call')).toHaveAttribute('data-outcome', 'session_ended');
  });
});

// --------------------------------------------------- the window and the ceiling (feature 019)

function mountWindow(window: Partial<WspProgress>) {
  render(
    <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
      <WspRunPage
        fetchRun={() => ok(WSP_RUN_DOCUMENT as unknown as WspRunDocument)}
        fetchProgress={() => ok({ ...(WSP_RUN_PROGRESS as unknown as WspProgress), ...window })}
        fetchCalls={() => ok(WSP_RUN_CALLS as unknown as WspCallPage)}
        fetchTranscript={() =>
          ok({ run_id: WSP_RUN_ID, transcript_sha256: null, entries: [], partial: false })
        }
      />
    </NavigationProvider>,
  );
}

describe('[WSP run] the window and the output ceiling (FR-019.17)', () => {
  it.each([
    ['reported', 32768, '32,768 tokens (reported by the server)'],
    ['stated', 65536, '65,536 tokens (stated on the endpoint)'],
    ['assumed', 8192, '8,192 tokens (assumed: the server reported none)'],
  ])('should name a %s window and its size', async (source, tokens, text) => {
    mountWindow({ window_source: source, window_tokens: tokens, window_warnings: [] });
    await flush();
    expect(screen.getByTestId('wsp-run-window')).toHaveTextContent(text);
  });

  it('should read "not recorded" for a run from before 019, never a number', async () => {
    mountWindow({
      window_source: null,
      window_tokens: null,
      output_ceiling_tokens: null,
      window_warnings: [],
    });
    await flush();
    expect(screen.getByTestId('wsp-run-window')).toHaveTextContent(/^not recorded$/);
    expect(screen.getByTestId('wsp-run-ceiling')).toHaveTextContent(/^not recorded$/);
    expect(screen.queryByTestId('wsp-run-window-assumed')).toBeNull();
    expect(screen.queryByTestId('wsp-run-window-mismatch')).toBeNull();
  });

  it('should warn of an assumed window only when the backend says so', async () => {
    mountWindow({
      window_source: 'assumed',
      window_tokens: 8192,
      window_warnings: ['window_assumed'],
    });
    await flush();
    expect(screen.getByTestId('wsp-run-window-assumed')).toHaveTextContent('8,192 tokens');
    expect(screen.getByTestId('wsp-run-window-assumed')).toHaveTextContent('State the window');
    expect(screen.queryByTestId('wsp-run-window-mismatch')).toBeNull();
  });

  it('should warn of a stated window above the report, naming both, only for a mismatch', async () => {
    mountWindow({
      window_source: 'stated',
      window_tokens: 65536,
      window_reported_tokens: 32768,
      window_warnings: ['window_stated_above_reported'],
    });
    await flush();
    const warning = screen.getByTestId('wsp-run-window-mismatch');
    expect(warning).toHaveTextContent('65,536 tokens');
    expect(warning).toHaveTextContent('32,768 tokens');
    expect(screen.queryByTestId('wsp-run-window-assumed')).toBeNull();
  });

  it('should show the ceiling, and when it was lowered to the window', async () => {
    mountWindow({
      window_source: 'reported',
      window_tokens: 16384,
      output_ceiling_tokens: 16384,
      output_ceiling_lowered: true,
    });
    await flush();
    expect(screen.getByTestId('wsp-run-ceiling')).toHaveTextContent(
      '16,384 tokens, lowered to the window the server reported',
    );
  });

  it('should say a ceiling lowered after a cut-off reply was lowered for that', async () => {
    mountWindow({
      window_source: 'reported',
      window_tokens: 126976,
      window_reported_tokens: 126976,
      output_ceiling_tokens: 4096,
      output_ceiling_lowered: true,
    });
    await flush();
    expect(screen.getByTestId('wsp-run-ceiling')).toHaveTextContent(
      '4,096 tokens, lowered after a reply was cut off',
    );
  });

  it('should show a ceiling that was not lowered as its size alone', async () => {
    mountWindow({
      window_source: 'reported',
      window_tokens: 65536,
      output_ceiling_tokens: 32000,
      output_ceiling_lowered: false,
    });
    await flush();
    expect(screen.getByTestId('wsp-run-ceiling')).toHaveTextContent(/^32,000 tokens$/);
  });
});

// ------------------------------------------------------------------------------- the list page

function row(status: string, extra: Partial<WspRunRow> = {}): WspRunRow {
  return {
    ...(WSP_RUN_ROW as unknown as WspRunRow),
    status,
    stop_reason: null,
    outcome: null,
    failure: null,
    finished_at: null,
    progress:
      status === 'completed' || status === 'failed' || status === 'cancelled'
        ? null
        : {
            elapsed_s: 65,
            clock_from: status === 'queued' ? 'created_at' : 'started_at',
            worker_state: status === 'queued' ? null : 'running',
            heartbeat_age_s: status === 'queued' ? null : 3,
            calls: 4,
            refused: 1,
            tokens: 5200,
            cost_usd: 0,
            last_call_at: '2026-11-20T15:02:01Z',
            queue_position: status === 'queued' ? 1 : null,
            budgets: [
              { name: 'tokens', spent: 5200, cap: 10000, fraction: 0.52, estimate: false },
              { name: 'cost_usd', spent: 0, cap: null, fraction: null, estimate: false },
              { name: 'time_s', spent: 60, cap: 600, fraction: 0.1, estimate: false },
              { name: 'steps', spent: 4, cap: 50, fraction: 0.08, estimate: true },
            ],
          },
    ...extra,
  };
}

function mountList(state: { rows: WspRunRow[]; fails: boolean }) {
  const list = vi.fn(() => (state.fails ? down<never>() : ok({ runs: state.rows })));
  render(
    <NavigationProvider initialScreen="wsp-runs">
      <WspRunsPage list={list} />
      <Where />
    </NavigationProvider>,
  );
  return list;
}

describe('[WSP runs] the list keeps itself current', () => {
  it('should poll every 5 s while a row is live and relax to 30 s when none is', async () => {
    const state = { rows: [row('running')], fails: false };
    const list = mountList(state);
    await flush();
    expect(list).toHaveBeenCalledTimes(1);
    await wait(WSP_LIVE_POLL_MS);
    expect(list).toHaveBeenCalledTimes(2);
    await wait(WSP_LIVE_POLL_MS);
    expect(list).toHaveBeenCalledTimes(3);
    state.rows = [row('completed')];
    await wait(WSP_LIVE_POLL_MS); // this read finds nothing live
    expect(list).toHaveBeenCalledTimes(4);
    expect(screen.getByTestId('wsp-runs-idle')).toHaveTextContent('reading every 30 s');
    await wait(WSP_LIVE_POLL_MS); // the read already scheduled at 5 s
    const settled = list.mock.calls.length;
    await wait(WSP_IDLE_POLL_MS - 1);
    expect(list).toHaveBeenCalledTimes(settled); // relaxed: nothing for 30 s
    await wait(1);
    expect(list).toHaveBeenCalledTimes(settled + 1);
  });

  it('should show a live row with its elapsed clock, calls, tokens, heartbeat and budgets', async () => {
    mountList({
      rows: [row('running'), row('queued', { run_id: 'wsp_bbbbbbbbbbbb' })],
      fails: false,
    });
    await flush();
    const live = screen.getByTestId(`wsp-run-row-${WSP_RUN_ID}`);
    const progress = within(live).getByTestId('wsp-run-progress');
    expect(progress).toHaveTextContent('running for 1 m 05 s');
    expect(progress).toHaveTextContent('4 calls (1 refused)');
    expect(progress).toHaveTextContent('5,200 tokens');
    expect(progress).toHaveTextContent('heartbeat 3 s ago');
    expect(within(live).getByTestId('wsp-run-budgets')).toHaveTextContent(
      'tokens 52%, time 10%, steps ~8%',
    );
    await wait(2_000);
    expect(within(live).getByTestId('wsp-run-elapsed')).toHaveTextContent('1 m 07 s');
    const queued = screen.getByTestId('wsp-run-row-wsp_bbbbbbbbbbbb');
    expect(within(queued).getByTestId('wsp-run-progress')).toHaveTextContent(
      'waiting for 1 m 07 s · place 1 in the queue',
    );
  });

  it('should keep the last good list when a read fails, and say so', async () => {
    const state = { rows: [row('running')], fails: false };
    mountList(state);
    await flush();
    expect(screen.getByTestId('wsp-runs-updated')).toHaveTextContent('updated 0 s ago');
    state.fails = true;
    await wait(WSP_LIVE_POLL_MS);
    expect(screen.getByTestId(`wsp-run-row-${WSP_RUN_ID}`)).toBeTruthy();
    expect(screen.getByTestId('wsp-runs-failure')).toBeTruthy();
    expect(screen.getByTestId('wsp-runs-updated')).toHaveTextContent(
      'the last read failed: showing what was read 5 s ago',
    );
  });

  it('should show a finished row with no progress line', async () => {
    mountList({ rows: [row('completed')], fails: false });
    await flush();
    expect(screen.queryByTestId('wsp-run-progress')).toBeNull();
  });
});

describe('[WSP runs] requests waiting for approval', () => {
  const approval = (id: string, action: string, status = 'pending'): Approval =>
    ({
      approval_id: id,
      action,
      status,
      summary: {
        kind: 'repair',
        mode: 'session',
        endpoint: { model: 'qwen-editor' },
        bundle: { bundle_id: 'probe-17' },
      },
      requested_by: 'agent:forge-mcp',
      decided_by: null,
      reason: null,
      result: null,
      error: null,
      created_at: '2026-11-20T15:00:00Z',
      decided_at: null,
      expires_at: '2026-11-21T15:00:00Z',
    }) as unknown as Approval;

  it('should list each waiting WSP request, and only those, linking to Approvals', async () => {
    useApprovalsStore.setState({
      waiting: [
        approval('appr-wsp', 'wsp_session'),
        approval('appr-ext', 'wsp_session_external'),
        approval('appr-gpu', 'gpu_start'),
      ],
    });
    mountList({ rows: [], fails: false });
    await flush();
    const rows = within(screen.getByTestId('wsp-pending-rows')).getAllByRole('button');
    expect(rows).toHaveLength(2);
    expect(screen.getByTestId('wsp-pending-appr-wsp')).toHaveTextContent(
      'Waiting for your approval',
    );
    expect(screen.getByTestId('wsp-pending-appr-wsp')).toHaveTextContent('repair · session');
    expect(screen.getByTestId('wsp-pending-appr-wsp')).toHaveTextContent('editor qwen-editor');
    expect(screen.getByTestId('wsp-pending-appr-ext')).toHaveTextContent('external endpoint');
    expect(screen.queryByTestId('wsp-pending-appr-gpu')).toBeNull();
    // With a request waiting the list reads at the live cadence (its run is about to appear).
    expect(screen.queryByTestId('wsp-runs-idle')).toBeNull();
    await act(async () => {
      screen.getByTestId('wsp-pending-appr-wsp').click();
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByTestId('where')).toHaveTextContent('approvals');
  });
});

describe('[WSP] formatDuration', () => {
  it('should read seconds, minutes and hours', () => {
    expect(formatDuration(45)).toBe('45 s');
    expect(formatDuration(192)).toBe('3 m 12 s');
    expect(formatDuration(3720)).toBe('1 h 02 m');
    expect(formatDuration(null)).toBe('—');
  });
});

// ------------------------------------------------------------ hosted runs (feature 020, U-2)

const HOSTED_DOCUMENT = {
  ...(WSP_RUN_DOCUMENT as unknown as WspRunDocument),
  status: 'failed',
  stop_reason: 'editor_refused',
  outcome: 'failed',
  failure: { code: 'editor_refused', message: 'the editor refused (category: cyber)' },
  request: {
    kind: 'repair',
    mode: 'session',
    task: 'Repair the probe.',
    thinking: 'medium',
    paging: 'off',
    provider: { id: 'miwsp_engine' },
  },
} as WspRunDocument;

const HOSTED_PROGRESS: WspProgress = {
  ...(WSP_RUN_PROGRESS as unknown as WspProgress),
  status: 'failed',
  metered: {
    ...(WSP_RUN_PROGRESS as unknown as WspProgress).metered,
    cost_usd: 0.421,
    cache_read_tokens: 1_204_800,
    cache_write_tokens: 41_000,
  },
};

const HOSTED_CALLS: WspCall[] = [
  call(0, { cache_read_tokens: 0, cache_write_tokens: 41_000 }),
  call(1, { cache_read_tokens: 1_204_800, cache_write_tokens: 0 }),
  call(2, { cache_read_tokens: null, cache_write_tokens: null }),
  call(3, { cache_read_tokens: 500, cache_write_tokens: null }),
];

const HOSTED_TRANSCRIPT = [
  { miwsp_transcript: 1, provider: 'miwsp_engine', history_mode: 'append_only' },
  { role: 'system', content: 'You edit weights.' },
  { event: 'timing', turn: 1, gap_s: null, duration_s: 2.4 },
  {
    event: 'handoff',
    n: 1,
    conversation: 2,
    messages: [
      { role: 'system', content: 'You edit weights.' },
      { role: 'user', content: 'Where things stand: two edits kept.' },
    ],
  },
  { event: 'timing', turn: 4, gap_s: 3.25, duration_s: 1.0 },
  { event: 'refusal', turn: 4, category: 'cyber' },
];

function mountHosted(
  document: WspRunDocument,
  progress: WspProgress,
  calls: readonly WspCall[],
  entries: unknown[],
) {
  render(
    <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
      <WspRunPage
        fetchRun={() => ok(document)}
        fetchProgress={() => ok(progress)}
        fetchCalls={() => ok(page(calls, document.status))}
        fetchTranscript={() =>
          ok({ run_id: WSP_RUN_ID, transcript_sha256: 'c'.repeat(64), entries, partial: false })
        }
      />
    </NavigationProvider>,
  );
}

describe('[WSP run] a hosted run: cost, cache, effort, history mode, handoffs, refusal (U-2)', () => {
  it("should show each call's cache reads and writes beside its tokens, and 'not recorded' for an older call", async () => {
    mountHosted(HOSTED_DOCUMENT, HOSTED_PROGRESS, HOSTED_CALLS, HOSTED_TRANSCRIPT);
    await flush();
    const cache = screen.getAllByTestId('wsp-call-cache').map((span) => span.textContent);
    expect(cache).toEqual([
      '0 read from cache · 41,000 written to cache',
      '1,204,800 read from cache · 0 written to cache',
      'cache not recorded',
      '500 read from cache · not recorded written to cache',
    ]);
    // Beside the tokens, in the same row.
    const first = screen.getAllByTestId('wsp-call')[0] as HTMLElement;
    expect(within(first).getByTestId('wsp-call-tokens')).toHaveTextContent('450 in · 150 out');
    expect(within(first).getByTestId('wsp-call-cache')).toBeInTheDocument();
  });

  it("should show the run's cost in USD and its cache totals on the usage row and in the progress panel", async () => {
    mountHosted(HOSTED_DOCUMENT, HOSTED_PROGRESS, HOSTED_CALLS, HOSTED_TRANSCRIPT);
    await flush();
    expect(screen.getByTestId('wsp-run-metered')).toHaveTextContent(
      /^USD 0\.4210 · 1,204,800 read from cache · 41,000 written to cache$/,
    );
    expect(screen.getByTestId('wsp-progress-cost')).toHaveTextContent(/^USD 0\.4210$/);
    expect(screen.getByTestId('wsp-progress-cache')).toHaveTextContent(
      /^1,204,800 read from cache · 41,000 written to cache$/,
    );
  });

  it('should show the effort, the history mode, each handoff and the refusal with its category', async () => {
    mountHosted(HOSTED_DOCUMENT, HOSTED_PROGRESS, HOSTED_CALLS, HOSTED_TRANSCRIPT);
    await flush();
    expect(screen.getByTestId('wsp-run-effort')).toHaveTextContent(/^medium$/);
    expect(screen.getByTestId('wsp-run-history-mode')).toHaveTextContent(
      /^append_only \(nothing already sent changed\)$/,
    );
    expect(screen.getByTestId('wsp-run-handoffs')).toHaveTextContent(/^1: #1 \(conversation 2\)$/);
    const refusal = screen.getByTestId('wsp-run-refusal');
    expect(refusal).toHaveTextContent('The editor refused (category: cyber)');
    expect(refusal).toHaveTextContent('On turn 4.');
  });

  it("should say 'no category given' for a refusal with none", async () => {
    mountHosted(HOSTED_DOCUMENT, HOSTED_PROGRESS, HOSTED_CALLS, [
      HOSTED_TRANSCRIPT[0],
      { event: 'refusal', turn: 2, category: null },
    ]);
    await flush();
    expect(screen.getByTestId('wsp-run-refusal')).toHaveTextContent(
      'The editor refused (category: no category given)',
    );
    expect(screen.getByTestId('wsp-run-handoffs')).toHaveTextContent(/^none$/);
  });

  it("should read 'cache not recorded' and no refusal for a run from before 020, and paging from its header", async () => {
    mountHosted(
      WSP_RUN_DOCUMENT as unknown as WspRunDocument,
      WSP_RUN_PROGRESS as unknown as WspProgress,
      WSP_RUN_CALLS.calls as unknown as WspCall[],
      [{ miwsp_transcript: 1, provider: 'miwsp_engine' }],
    );
    await flush();
    expect(screen.getByTestId('wsp-run-metered')).toHaveTextContent(
      /^USD 0\.0000 · cache not recorded$/,
    );
    expect(screen.getByTestId('wsp-progress-cache')).toHaveTextContent(/^cache not recorded$/);
    expect(screen.getByTestId('wsp-call-cache')).toHaveTextContent(/^cache not recorded$/);
    expect(screen.getByTestId('wsp-run-history-mode')).toHaveTextContent(/^paging/);
    expect(screen.getByTestId('wsp-run-effort')).toHaveTextContent(/^none/);
    expect(screen.queryByTestId('wsp-run-refusal')).toBeNull();
  });

  it('should say a live run records its history mode and handoffs in the transcript, read when it ends', async () => {
    mountRun({ status: 'running', runFails: false, calls: [] });
    await flush();
    expect(screen.getByTestId('wsp-run-history-mode')).toHaveTextContent(
      'recorded in the transcript, read when the run ends',
    );
    expect(screen.getByTestId('wsp-run-handoffs')).toHaveTextContent(
      'recorded in the transcript, read when the run ends',
    );
  });
});

describe('[WSP runs] a row shows its cost and cache totals (U-2)', () => {
  it.each([
    [
      { cost_usd: 0.421, cache_read_tokens: 1_204_800, cache_write_tokens: 41_000 },
      'USD 0.4210 · 1,204,800 read from cache · 41,000 written to cache',
    ],
    [
      { cost_usd: 0, cache_read_tokens: null, cache_write_tokens: null },
      'USD 0.0000 · cache not recorded',
    ],
    [undefined, 'cost and cache not recorded'],
  ])('should read %j as its line', async (metered, text) => {
    // An older backend serves no `metered` at all: the stub row's is removed, not set undefined.
    const { metered: _served, ...older } = row('completed');
    void _served;
    mountList({
      rows: [metered === undefined ? (older as WspRunRow) : row('completed', { metered })],
      fails: false,
    });
    await flush();
    expect(screen.getByTestId('wsp-run-metered')).toHaveTextContent(text);
  });
});
