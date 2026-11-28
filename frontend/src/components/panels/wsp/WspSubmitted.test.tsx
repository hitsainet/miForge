/**
 * The editor's answer is its own account; the worker's measurement sits beside it (plan A2,
 * 2026-11-22: run wsp_4bd06d60acf0 claimed figures it never measured).
 */
import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { entryView, eventText } from './transcript';
import { TranscriptView } from './WspRunPage';

const MEASURED = {
  event: 'submitted_measured',
  measured: true,
  changed: true,
  split: 'validation',
  now: { auroc: 0.990531, recall: 0.191, false_alarms: 6, false_alarm_rate: 0.00127 },
  parent: { auroc: 0.990332, recall: 0.164794, false_alarms: 6, false_alarm_rate: 0.00127 },
  auroc_against_parent: { delta: 0.000199, low: -0.000122, high: 0.000526, within_noise: true },
};
const LINE =
  'measured by the worker on practice validation: AUROC 0.990531 (parent 0.990332), ' +
  'difference +0.000199, 95% interval -0.000122 to +0.000526: within noise; ' +
  'recall 0.1910 (parent 0.1648); false alarms 6 (parent 6)';

const native = (answer: string) => ({
  role: 'assistant',
  content: null,
  tool_calls: [{ id: 'c', name: 'submit', arguments: { answer } }],
});
const structured = (answer: string) => ({
  role: 'assistant',
  content: JSON.stringify({ tool: 'submit', arguments: { answer } }),
});

function show(entries: readonly unknown[]) {
  render(
    <TranscriptView
      transcript={{
        run_id: 'wsp_0123456789ab',
        transcript_sha256: 'c'.repeat(64),
        partial: false,
        entries,
      }}
    />,
  );
  return screen.getAllByTestId('wsp-transcript-entry');
}

describe('eventText', () => {
  it('should read the measurement out', () => {
    expect(eventText('submitted_measured', MEASURED)).toBe(LINE);
  });

  it('should say when a difference is real, and which way', () => {
    const real = (low: number, high: number) =>
      eventText('submitted_measured', {
        ...MEASURED,
        auroc_against_parent: { delta: low, low, high, within_noise: false },
      });
    expect(real(0.001, 0.002)).toContain(": above the parent's;");
    expect(real(-0.002, -0.001)).toContain(": below the parent's;");
  });

  it("should say when the submitted weights are the parent's", () => {
    const same = eventText('submitted_measured', {
      ...MEASURED,
      changed: false,
      now: MEASURED.parent,
    });
    expect(same).toContain("the submitted weights are the parent's, unchanged; ");
  });

  it('should say why nothing was measured', () => {
    const none = { event: 'submitted_measured', measured: false };
    expect(eventText('submitted_measured', { ...none, reason: 'no_time' })).toBe(
      'not measured by the worker: no time was left before the deadline',
    );
    expect(
      eventText('submitted_measured', { ...none, reason: 'failed', error: 'MemoryError' }),
    ).toBe('not measured by the worker: the measurement failed (MemoryError)');
  });

  it('should show an unexpected shape as its JSON', () => {
    expect(eventText('submitted_measured', { measured: true, now: 'x' })).toBeNull();
  });
});

describe('the answer', () => {
  it.each([
    ['native', native],
    ['structured', structured],
  ])('should be labelled, with the figures beside it (%s)', (_, submit) => {
    const steps = show([{ role: 'user', content: 'go' }, submit('AUROC 0.9999'), MEASURED]);
    const answer = within(steps[1]!);
    expect(answer.getByTestId('wsp-answer-label')).toHaveTextContent(
      "The editor's own account, not checked",
    );
    expect(answer.getByTestId('wsp-answer-measured')).toHaveTextContent(LINE);
    expect(within(steps[0]!).queryByTestId('wsp-answer-label')).toBeNull();
  });

  it('should put the figures beside the last submit only', () => {
    const steps = show([native('first, refused'), native('second'), MEASURED]);
    expect(within(steps[0]!).getByTestId('wsp-answer-label')).toBeInTheDocument();
    expect(within(steps[0]!).queryByTestId('wsp-answer-measured')).toBeNull();
    expect(within(steps[1]!).getByTestId('wsp-answer-measured')).toBeInTheDocument();
  });

  it('should still label an older run, which has no measurement', () => {
    const steps = show([native('AUROC 0.9999')]);
    expect(within(steps[0]!).getByTestId('wsp-answer-label')).toBeInTheDocument();
    expect(screen.queryByTestId('wsp-answer-measured')).toBeNull();
  });

  it('should not take other JSON for an answer', () => {
    expect(entryView(structured('x')).answer).toBe(true);
    expect(entryView({ role: 'user', content: '{"tool":"submit"}' }).answer).toBe(false);
    expect(entryView({ role: 'assistant', content: '{"tool":"python"}' }).answer).toBe(false);
  });
});
