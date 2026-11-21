/**
 * The degenerate-selection warning (FR-009.16; F3 FR-003.10).
 *
 * **The sentence rendered is F3's, from the record.** `SelectionSignal.message` exists so that
 * *"F7 and F9 render one sentence rather than inventing two"*, and *"a signal shown under a
 * sentence that does not describe it is worse than no sentence: the operator reads the wrong
 * failure."* So this component renders `signal.message` and the test proves it renders what it
 * was given - including a sentence this file does not contain.
 *
 * The constants in `EXPECTED_SENTENCES` are a **fixture**: they assert that what F3 ships is
 * what this screen expects, so a reworded sentence upstream is a visible change here rather
 * than two screens saying different things.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DegenerateWarning, EXPECTED_SENTENCES, KNOWN_SIGNAL_NAMES } from './DegenerateWarning';
import {
  SIGNAL_ACCEPTS_EVERYTHING,
  SIGNAL_NO_SPREAD,
  SIGNAL_REJECTS_EVERYTHING,
} from '@/types/packs';

const health = { n: 64, accepted: 64, rejected: 0, undecided: 0 };

describe('[DegenerateWarning]', () => {
  it('should use F3s three signal names verbatim', () => {
    expect([...KNOWN_SIGNAL_NAMES]).toEqual([
      'selection_accepts_everything',
      'selection_rejects_everything',
      'grader_scores_have_no_spread',
    ]);
  });

  it('should render the sentence the record carried, not a local copy', () => {
    // The proof that this component holds no copy: it renders a sentence that appears nowhere
    // in its own source.
    render(
      <DegenerateWarning
        signals={[
          { name: SIGNAL_ACCEPTS_EVERYTHING, message: 'A bespoke upstream sentence.', details: {} },
        ]}
        health={health}
        atStep={412}
      />,
    );
    expect(screen.getByTestId(`signal-${SIGNAL_ACCEPTS_EVERYTHING}`)).toHaveTextContent(
      'A bespoke upstream sentence.',
    );
    expect(DegenerateWarning.toString()).not.toContain('is not selecting');
  });

  it('should precede the sentence with the measurement and its sample', () => {
    // FR-009.16: `Accepted 64 of 64 (step 412).` - the measurement that triggered it, with
    // its sample, and three-valued because `undecided` is not a reject.
    render(
      <DegenerateWarning
        signals={[{ name: SIGNAL_ACCEPTS_EVERYTHING, message: 'x', details: {} }]}
        health={health}
        atStep={412}
      />,
    );
    const callout = screen.getByTestId(`signal-${SIGNAL_ACCEPTS_EVERYTHING}`);
    expect(callout).toHaveTextContent('Accepted 64, rejected 0, undecided 0 of 64 (step 412).');
  });

  it('should expect the accept/reject sentence F3 ships, as a fixture', () => {
    // R-02.39 quotes this verbatim as the rationale for requiring the dry-run; it is compared
    // as a whole string, not a substring.
    expect(EXPECTED_SENTENCES[SIGNAL_ACCEPTS_EVERYTHING]).toBe(
      'A selector that accepts everything is not selecting; one that rejects everything will stall the run.',
    );
    expect(EXPECTED_SENTENCES[SIGNAL_REJECTS_EVERYTHING]).toBe(
      EXPECTED_SENTENCES[SIGNAL_ACCEPTS_EVERYTHING],
    );
  });

  it('should expect a DIFFERENT sentence for the no-spread signal', () => {
    // F3's task 8.2 attached the accept/reject sentence to all three signals, including this
    // one - and it does not describe this failure: a grader with no spread is not accepting
    // everything, it is giving every candidate the same advantage. Reported to F3; this is
    // the sentence that describes it.
    expect(EXPECTED_SENTENCES[SIGNAL_NO_SPREAD]).not.toBe(
      EXPECTED_SENTENCES[SIGNAL_ACCEPTS_EVERYTHING],
    );
    expect(EXPECTED_SENTENCES[SIGNAL_NO_SPREAD]).toContain('same advantage');
  });

  it('should render nothing when no signal fired', () => {
    const { container } = render(<DegenerateWarning signals={[]} health={health} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('should render an unrecognised signal name as it arrived, naming it', () => {
    render(
      <DegenerateWarning
        signals={[{ name: 'selector_is_haunted', message: 'A new upstream signal.', details: {} }]}
        health={health}
      />,
    );
    const callout = screen.getByTestId('signal-selector_is_haunted');
    expect(callout).toHaveTextContent('A new upstream signal.');
    expect(callout).toHaveTextContent('an unrecognised signal name');
  });

  it('should render every signal that fired, not the first', () => {
    render(
      <DegenerateWarning
        signals={[
          { name: SIGNAL_ACCEPTS_EVERYTHING, message: 'one', details: {} },
          { name: SIGNAL_NO_SPREAD, message: 'two', details: {} },
        ]}
        health={health}
      />,
    );
    expect(screen.getByTestId(`signal-${SIGNAL_ACCEPTS_EVERYTHING}`)).toBeInTheDocument();
    expect(screen.getByTestId(`signal-${SIGNAL_NO_SPREAD}`)).toBeInTheDocument();
  });
});
