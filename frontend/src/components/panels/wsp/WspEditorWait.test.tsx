/**
 * Debt D3 (2026-11-28): a run waiting for its editor model says so on the run page. Until now the
 * wait (`editor_wait_since`, set when the driver's claim reads no window: a miServer endpoint with
 * no model resident, or a server that gives no answer) lived only in the driver's log and pass
 * actions, and the page read "no run is active, so it starts at the driver's next pass".
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { NavigationProvider } from '@/contexts/NavigationContext';
import type { ApiResult } from '@/api/client';
import type { WspProgress, WspRunDocument } from '@/api/wsp';
import { WSP_RUN_DOCUMENT, WSP_RUN_ID, WSP_RUN_PROGRESS } from '@/testing/stubBackend';
import { WspRunPage } from './WspRunPage';

const ok = <T,>(value: T): Promise<ApiResult<T>> => Promise.resolve({ ok: true, value });
const flush = () => act(() => vi.advanceTimersByTimeAsync(0));

const SINCE = '2026-11-28T14:00:00Z';
const QUEUED: WspProgress = {
  ...(WSP_RUN_PROGRESS as unknown as WspProgress),
  status: 'queued',
  live: true,
  now: '2026-11-28T14:02:05Z',
  started_at: null,
  finished_at: null,
  clock_from: 'created_at',
  worker: null,
  queue: { position: 1, queued: 1, active_run_id: null },
  editor_wait_since: SINCE,
};

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

async function mount(progress: WspProgress): Promise<void> {
  render(
    <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
      <WspRunPage
        fetchRun={() =>
          ok({ ...(WSP_RUN_DOCUMENT as unknown as WspRunDocument), status: progress.status })
        }
        fetchProgress={() => ok(progress)}
        fetchCalls={() =>
          ok({
            run_id: WSP_RUN_ID,
            status: progress.status,
            live: progress.live,
            calls: [],
            more: false,
          })
        }
        fetchTranscript={() =>
          ok({ run_id: WSP_RUN_ID, transcript_sha256: null, entries: [], partial: false })
        }
      />
    </NavigationProvider>,
  );
  for (let i = 0; i < 4; i += 1) await flush();
}

describe('[WSP run] waiting for the editor model (debt D3)', () => {
  it('should say a queued run waits for its editor model, since when and for how long', async () => {
    await mount(QUEUED);
    const line = screen.getByTestId('wsp-progress-editor-wait');
    expect(line).toHaveTextContent('Waiting for the editor model since');
    expect(line).toHaveTextContent('(2 m 05 s)');
    expect(line).toHaveTextContent('editor_not_resident');
    expect(screen.getByTestId('wsp-progress-queue')).toHaveTextContent(
      'it waits for its editor model',
    );
    expect(screen.getByTestId('wsp-progress-queue')).not.toHaveTextContent(
      'starts at the driver’s next pass',
    );
  });

  it('should keep the record of a wait once the run started', async () => {
    await mount({
      ...(WSP_RUN_PROGRESS as unknown as WspProgress),
      editor_wait_since: SINCE,
    });
    expect(screen.queryByTestId('wsp-progress-editor-wait')).toBeNull();
    expect(screen.getByTestId('wsp-progress-editor-waited')).toHaveTextContent(
      'Waited for the editor model from',
    );
  });

  it('the control: a queued run that never waited shows no wait', async () => {
    await mount({ ...QUEUED, editor_wait_since: null });
    expect(screen.queryByTestId('wsp-progress-editor-wait')).toBeNull();
    expect(screen.queryByTestId('wsp-progress-editor-waited')).toBeNull();
    expect(screen.getByTestId('wsp-progress-queue')).toHaveTextContent(
      'starts at the driver’s next pass',
    );
  });
});
