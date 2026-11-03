/**
 * The degenerate-selection warning (FR-009.16; F3 FR-003.10).
 *
 * ## Whose sentence is whose, checked against F3 rather than assumed
 *
 * **F3 supplies the accept/reject sentence with the signal.** `SelectionSignal.message` exists
 * in `backend/src/services/selectors/types.py` with the docstring *"The sentence is supplied
 * here, once, so F7 and F9 render one sentence rather than inventing two. A signal shown
 * under a sentence that does not describe it is worse than no sentence: the operator reads
 * the wrong failure."* So **this component renders `signal.message`** and holds no copy of it.
 * A second copy here is a second string to drift, which is the thing F3 supplying it was
 * meant to prevent.
 *
 * F9's own FR-009.16 instructs this feature to hold the two sentences "in one place". That
 * instruction is **superseded by the record carrying them**, and the divergence is reported
 * rather than resolved by keeping both: the constants below exist only so the test can assert
 * that what arrives is what F3 ships, and `EXPECTED_SENTENCES` is a **test fixture, not a
 * rendering source** - nothing in this file reads it.
 *
 * ## What the panel adds, which is the measurement
 *
 * FR-009.16 requires the sentence be *"preceded by the measurement that triggered it, with
 * its sample"*: `Accepted 64 of 64 (step 412).` The counts come from the record and this
 * component computes no rate of its own.
 *
 * Amber, which is this design system's warning colour. Never accent.
 */
import type { ReactNode } from 'react';
import { Callout } from '@/components/common/Callout';
import { selectionCountsText } from '@/components/common/SelectionCounts';
import {
  SELECTION_SIGNAL_NAMES,
  SIGNAL_ACCEPTS_EVERYTHING,
  SIGNAL_NO_SPREAD,
  SIGNAL_REJECTS_EVERYTHING,
  type SelectionHealth,
  type SelectionSignal,
} from '@/types/packs';

/**
 * The strings F3 ships, **as a test fixture**.
 *
 * `DegenerateWarning.test.tsx` asserts the component renders the message it is given and that
 * these are the strings F3's own constants hold, so a reworded sentence upstream is a visible
 * change here rather than two screens saying different things. **Nothing in this module reads
 * them to render with.**
 *
 * The no-spread sentence is **new copy and F9's**, because F3's own task 8.2 attaches the
 * accept/reject sentence to all three signals - including `grader_scores_have_no_spread` -
 * and that sentence does not describe that failure: a grader with no spread is not accepting
 * everything, it is giving every candidate the same advantage. `services/selectors/health.py`
 * has since adopted a distinct sentence for it, marked there as *"New copy, not from any
 * requirement."* These two values are copies of what that module holds today.
 */
export const EXPECTED_SENTENCES: Readonly<Record<string, string>> = {
  [SIGNAL_ACCEPTS_EVERYTHING]:
    'A selector that accepts everything is not selecting; one that rejects everything will stall the run.',
  [SIGNAL_REJECTS_EVERYTHING]:
    'A selector that accepts everything is not selecting; one that rejects everything will stall the run.',
  [SIGNAL_NO_SPREAD]:
    'A grader whose scores have no spread gives every candidate the same advantage; the step runs and learns nothing from it.',
};

/** The three names F3 publishes, re-exported so a caller does not spell them again. */
export const KNOWN_SIGNAL_NAMES = SELECTION_SIGNAL_NAMES;

export interface DegenerateWarningProps {
  readonly signals: readonly SelectionSignal[];
  readonly health: Pick<SelectionHealth, 'n' | 'accepted' | 'rejected' | 'undecided'>;
  readonly atStep?: number;
  readonly 'data-testid'?: string;
}

export function DegenerateWarning({
  signals,
  health,
  atStep,
  ...rest
}: DegenerateWarningProps): ReactNode {
  if (signals.length === 0) return null;
  const measurement = `${selectionCountsText(health)}${atStep === undefined ? '' : ` (step ${atStep})`}.`;

  return (
    <div className="stack-y-2" {...rest}>
      {signals.map((signal) => (
        <Callout
          key={signal.name}
          tone="warning"
          title={measurement}
          data-testid={`signal-${signal.name}`}
        >
          {/* F3's own sentence, rendered. Not a local copy of it. */}
          <span data-signal-name={signal.name}>{signal.message}</span>
          {KNOWN_SIGNAL_NAMES.includes(
            signal.name as (typeof KNOWN_SIGNAL_NAMES)[number],
          ) ? null : (
            <span className="ml-1 text-dim">
              (an unrecognised signal name: {signal.name} — rendered as it arrived)
            </span>
          )}
        </Callout>
      ))}
    </div>
  );
}
