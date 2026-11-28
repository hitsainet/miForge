/**
 * Debt D4 (2026-11-28): the label beside the answer the run kept reads the run document's
 * `submitted_check`, not one fixed sentence. Fix 2 (2026-11-27, run `wsp_0971152b6843`): an
 * inspect that submitted before any observation tool ran carries `measured: false` and its
 * transcript has no `submitted_measured` line, so the page said "The editor's own account, not
 * checked." exactly as for every other run. The flag reaches the page through the run read.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, within } from '@testing-library/react';
import { NavigationProvider } from '@/contexts/NavigationContext';
import type { ApiResult } from '@/api/client';
import type { WspProgress, WspRunDocument, WspSubmittedCheck } from '@/api/wsp';
import { WSP_RUN_DOCUMENT, WSP_RUN_ID, WSP_RUN_PROGRESS } from '@/testing/stubBackend';
import { answerLabel } from './transcript';
import { TranscriptView, WspRunPage } from './WspRunPage';

const ok = <T,>(value: T): Promise<ApiResult<T>> => Promise.resolve({ ok: true, value });
const flush = () => act(() => vi.advanceTimersByTimeAsync(0));

/** `runs.UNOBSERVED_CHECK`, as the backend stores it for fix 2. */
const UNOBSERVED: WspSubmittedCheck = {
  checks: [],
  compared: false,
  differs: null,
  measured: false,
  split: null,
  stated: false,
  stated_split: null,
};
const MEASURED_CHECK: WspSubmittedCheck = { ...UNOBSERVED, measured: true, split: 'validation' };
const MEASURED_LINE = {
  event: 'submitted_measured',
  measured: true,
  changed: true,
  split: 'validation',
  now: { auroc: 0.99, recall: 0.2, false_alarms: 6, false_alarm_rate: 0.001 },
  parent: { auroc: 0.99, recall: 0.2, false_alarms: 6, false_alarm_rate: 0.001 },
};
const NOT_MEASURED_LINE = { event: 'submitted_measured', measured: false, reason: 'no_time' };
const submit = (answer: string) => ({
  role: 'assistant',
  content: null,
  tool_calls: [{ id: 'c', name: 'submit', arguments: { answer } }],
});
const GENERIC = "The editor's own account, not checked.";

function transcriptOf(entries: readonly unknown[]) {
  return { run_id: WSP_RUN_ID, transcript_sha256: 'c'.repeat(64), partial: false, entries };
}

describe('answerLabel', () => {
  it('should read the generic label when the document has no check', () => {
    expect(answerLabel(undefined, false)).toEqual({ measured: 'unknown', text: GENERIC });
    expect(answerLabel({ measured: 'yes' }, false).measured).toBe('unknown');
  });

  it("should say fix 2's inspect answered before any observation", () => {
    const label = answerLabel(UNOBSERVED, false);
    expect(label.measured).toBe('unobserved');
    expect(label.text).toContain('before any observation tool ran');
    expect(label.text).not.toBe(GENERIC);
  });

  it('should tell an unmeasured edit (a submitted_measured line) from the unobserved inspect', () => {
    expect(answerLabel({ ...UNOBSERVED }, true).measured).toBe('false');
  });

  it('should say a measured answer was measured, its words still unchecked', () => {
    const label = answerLabel(MEASURED_CHECK, true);
    expect(label.measured).toBe('true');
    expect(label.text).toContain('measured the submitted weights');
    expect(label.text).toContain('words are not checked');
  });
});

describe('the transcript view', () => {
  it('should put the flag on the kept answer only', () => {
    render(
      <TranscriptView
        transcript={transcriptOf([submit('refused first'), submit('kept')])}
        submittedCheck={UNOBSERVED}
      />,
    );
    const [first, kept] = screen.getAllByTestId('wsp-transcript-entry');
    expect(within(first!).getByTestId('wsp-answer-label')).toHaveAttribute(
      'data-measured',
      'unknown',
    );
    expect(within(kept!).getByTestId('wsp-answer-label')).toHaveAttribute(
      'data-measured',
      'unobserved',
    );
  });

  it('should read the unmeasured edit as not measured', () => {
    render(
      <TranscriptView
        transcript={transcriptOf([submit('x'), NOT_MEASURED_LINE])}
        submittedCheck={UNOBSERVED}
      />,
    );
    expect(screen.getByTestId('wsp-answer-label')).toHaveAttribute('data-measured', 'false');
  });
});

describe('[WSP run] the page shows the run document flag beside the answer (debt D4)', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  async function mount(check: WspSubmittedCheck | undefined, entries: readonly unknown[]) {
    const doc = {
      ...(WSP_RUN_DOCUMENT as unknown as WspRunDocument),
      status: 'completed',
      ...(check === undefined ? {} : { submitted_check: check }),
    };
    render(
      <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
        <WspRunPage
          fetchRun={() => ok(doc)}
          fetchProgress={() =>
            ok({
              ...(WSP_RUN_PROGRESS as unknown as WspProgress),
              status: 'completed',
              live: false,
            })
          }
          fetchCalls={() =>
            ok({ run_id: WSP_RUN_ID, status: 'completed', live: false, calls: [], more: false })
          }
          fetchTranscript={() => ok(transcriptOf(entries))}
        />
      </NavigationProvider>,
    );
    for (let i = 0; i < 4; i += 1) await flush();
  }

  it("should show fix 2's flag for an inspect that answered unobserved", async () => {
    await mount(UNOBSERVED, [{ role: 'user', content: 'go' }, submit('ready')]);
    const label = screen.getByTestId('wsp-answer-label');
    expect(label).toHaveAttribute('data-measured', 'unobserved');
    expect(label).toHaveTextContent('before any observation tool ran');
  });

  it('should show a measured answer as measured', async () => {
    await mount(MEASURED_CHECK, [submit('better'), MEASURED_LINE]);
    expect(screen.getByTestId('wsp-answer-label')).toHaveAttribute('data-measured', 'true');
  });

  it('the control: a run with no check keeps the generic label', async () => {
    await mount(undefined, [submit('ready')]);
    const label = screen.getByTestId('wsp-answer-label');
    expect(label).toHaveAttribute('data-measured', 'unknown');
    expect(label).toHaveTextContent(GENERIC);
  });
});
