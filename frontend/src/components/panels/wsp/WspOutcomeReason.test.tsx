/**
 * Why a run has no candidate (fix 2026-11-23, D128 round 11, run `wsp_8482873d2714`).
 *
 * The editor's edit changed 12 weights and lowered the practice-validation AUROC below the
 * parent's, so miForge kept the patch as evidence and recorded `no_change` (FR-016.30). The run
 * page said "the session left the weights as they were", and nothing said why. Now the reason is
 * stored on the run, served in the list row and the progress read, and shown in both screens; a
 * run that truly changed nothing (`content_sha256` null) still says it left the weights.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, within } from '@testing-library/react';
import { NavigationProvider } from '@/contexts/NavigationContext';
import type { ApiResult } from '@/api/client';
import type { WspProgress, WspRunDocument, WspRunListing, WspRunRow } from '@/api/wsp';
import { WSP_RUN_DOCUMENT, WSP_RUN_ID, WSP_RUN_PROGRESS, WSP_RUN_ROW } from '@/testing/stubBackend';
import { WspRunPage, noCandidateText } from './WspRunPage';
import { WspRunsPage } from './WspRunsPage';

const ok = <T,>(value: T): Promise<ApiResult<T>> => Promise.resolve({ ok: true, value });
const flush = () => act(() => vi.advanceTimersByTimeAsync(0));

const REASON = "practice-validation AUROC 0.990293 is below the parent's 0.990332";
const DOC = WSP_RUN_DOCUMENT as unknown as WspRunDocument;
const CHANGED = { ...DOC, outcome: 'no_change', content_sha256: 'c'.repeat(64) };

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

async function mountRun(doc: WspRunDocument, reason: string | null): Promise<void> {
  render(
    <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
      <WspRunPage
        fetchRun={() => ok({ ...doc, status: 'completed' })}
        fetchProgress={() =>
          ok({
            ...(WSP_RUN_PROGRESS as unknown as WspProgress),
            status: 'completed',
            live: false,
            outcome_reason: reason,
          })
        }
        fetchCalls={() =>
          ok({ run_id: WSP_RUN_ID, status: 'completed', live: false, calls: [], more: false })
        }
        fetchTranscript={() =>
          ok({ run_id: WSP_RUN_ID, transcript_sha256: null, entries: [], partial: false })
        }
      />
    </NavigationProvider>,
  );
  await flush();
  await flush();
}

describe('[WSP run] a no_change run says whether its weights changed, and why', () => {
  it('should show the stored reason for a changed patch kept as evidence', async () => {
    await mountRun(CHANGED, REASON);
    const text = screen.getByTestId('wsp-run-no-candidate').textContent ?? '';
    expect(text).toBe(
      `No candidate: the edit changed weights, but it was kept as evidence, not as a candidate: ${REASON}.`,
    );
    expect(text).not.toContain('left the weights as they were');
  });

  it('should say no reason was recorded for a changed patch from before the fix', async () => {
    await mountRun(CHANGED, null);
    expect(screen.getByTestId('wsp-run-no-candidate')).toHaveTextContent(
      'no reason was recorded: runs finished before 2026-11-23 did not store one',
    );
  });

  it('the control: a run that changed nothing still says it left the weights', async () => {
    await mountRun({ ...DOC, outcome: 'no_change', content_sha256: null }, null);
    expect(screen.getByTestId('wsp-run-no-candidate').textContent).toBe(
      'No candidate: the session left the weights as they were.',
    );
  });

  it('should keep the report and the plain texts', () => {
    expect(noCandidateText({ ...DOC, outcome: 'report' }, null)).toBe(
      'No candidate: this run reports and never changes weights.',
    );
    expect(noCandidateText({ ...DOC, outcome: 'failed' }, REASON)).toBe('No candidate.');
  });
});

describe('[WSP runs] the list row shows the reason beside the outcome', () => {
  async function mountList(row: WspRunRow): Promise<HTMLElement> {
    const listing: WspRunListing = { runs: [row] };
    render(
      <NavigationProvider initialScreen="wsp-runs">
        <WspRunsPage list={() => ok(listing)} />
      </NavigationProvider>,
    );
    await flush();
    await flush();
    return screen.getByTestId(`wsp-run-row-${row.run_id}`);
  }

  it('should append the reason to the stop line', async () => {
    const row = await mountList({
      ...(WSP_RUN_ROW as unknown as WspRunRow),
      stop_reason: 'submitted',
      outcome: 'no_change',
      outcome_reason: REASON,
    });
    expect(within(row).getByTestId('wsp-run-stop').textContent).toBe(
      `stopped: submitted · outcome: no_change (${REASON})`,
    );
  });

  it('the control: a row without a reason is as before', async () => {
    const row = await mountList({
      ...(WSP_RUN_ROW as unknown as WspRunRow),
      stop_reason: 'submitted',
      outcome: 'no_change',
      outcome_reason: null,
    });
    expect(within(row).getByTestId('wsp-run-stop').textContent).toBe(
      'stopped: submitted · outcome: no_change',
    );
  });
});
