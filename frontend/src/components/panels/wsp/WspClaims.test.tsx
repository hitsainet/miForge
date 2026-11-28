/**
 * FU-2 (operator, 2026-11-25): the editor's claimed figures (`submit`'s `figures`) against the
 * worker's measurement of the submitted weights, beside the last submit. Run wsp_4bd06d60acf0
 * claimed AUROC 0.9904 where 0.990304 was measured. A line with no `claims` (a run from before
 * FU-2) reads "no figures stated". Prose is never parsed.
 */
import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { claimsLine } from './transcript';
import { TranscriptView } from './WspRunPage';

const BASE = {
  event: 'submitted_measured',
  measured: true,
  changed: true,
  split: 'validation',
  now: { auroc: 0.990304, recall: 0.2109, false_alarms: 6, false_alarm_rate: 0.00127 },
  parent: { auroc: 0.990332, recall: 0.164794, false_alarms: 6, false_alarm_rate: 0.00127 },
  auroc_against_parent: { delta: -0.000028, low: -0.0003, high: 0.0002, within_noise: true },
};
const DIFFERS = {
  ...BASE,
  claims: {
    stated: true,
    stated_split: 'validation',
    compared: true,
    differs: true,
    checks: [
      { figure: 'auroc', claimed: 0.9904, measured: 0.990304, matches: false },
      { figure: 'recall', claimed: 0.2109, measured: 0.2109, matches: true },
    ],
  },
};
const MATCHES = {
  ...BASE,
  claims: {
    ...DIFFERS.claims,
    differs: false,
    checks: [{ figure: 'auroc', claimed: 0.9903, measured: 0.990304, matches: true }],
  },
};
const OTHER_SPLIT = {
  ...BASE,
  claims: {
    stated: true,
    stated_split: 'train',
    compared: false,
    differs: null,
    checks: [{ figure: 'auroc', claimed: 0.99, measured: null, matches: null }],
  },
};
const NONE = {
  ...BASE,
  claims: { stated: false, stated_split: null, compared: false, differs: null, checks: [] },
};

const submit = {
  role: 'assistant',
  content: null,
  tool_calls: [{ id: 'c', name: 'submit', arguments: { answer: 'better' } }],
};

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

describe('claimsLine', () => {
  it('should name each figure that differs, with both values', () => {
    expect(claimsLine(DIFFERS)).toEqual({
      differs: true,
      text: "The editor's figures differ from the measurement: AUROC stated 0.9904, measured 0.990304.",
    });
  });

  it('should say the stated figures match', () => {
    expect(claimsLine(MATCHES)).toEqual({
      differs: false,
      text: "The editor's stated figures match the measurement: AUROC 0.9903.",
    });
  });

  it('should say why figures on another split were not compared', () => {
    expect(claimsLine(OTHER_SPLIT)).toEqual({
      differs: null,
      text: 'The editor stated AUROC 0.99; not compared: they name the train split, and the worker measured validation.',
    });
  });

  it('should read no figures for a submit that stated none, and for a line from before FU-2', () => {
    const none = 'The editor stated no figures in its submit.';
    expect(claimsLine(NONE)).toEqual({ differs: null, text: none });
    expect(claimsLine(BASE)).toEqual({ differs: null, text: none });
  });
});

describe('the run page', () => {
  it('should flag differing figures beside the last submit', () => {
    const steps = show([{ role: 'user', content: 'go' }, submit, DIFFERS]);
    const flag = within(steps[1]!).getByTestId('wsp-answer-claims');
    expect(flag).toHaveAttribute('data-differs', 'true');
    expect(flag).toHaveClass('text-red');
    expect(flag).toHaveTextContent('AUROC stated 0.9904, measured 0.990304');
    expect(within(steps[0]!).queryByTestId('wsp-answer-claims')).toBeNull();
  });

  it('should not flag matching figures', () => {
    const steps = show([submit, MATCHES]);
    const line = within(steps[0]!).getByTestId('wsp-answer-claims');
    expect(line).toHaveAttribute('data-differs', 'false');
    expect(line).not.toHaveClass('text-red');
  });

  it('should read "no figures stated" for an older run that has a measurement', () => {
    const steps = show([submit, BASE]);
    expect(within(steps[0]!).getByTestId('wsp-answer-claims')).toHaveTextContent(
      'The editor stated no figures in its submit.',
    );
  });

  it('should show no claims line for a run with no measurement', () => {
    show([submit]);
    expect(screen.queryByTestId('wsp-answer-claims')).toBeNull();
  });
});
