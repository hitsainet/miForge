/**
 * A three-valued verdict (F7 task 12.7; BRD section 7's wave-1 list; F3's own review).
 *
 * `accept` · `reject` · `undecided`. **`undecided` is neutral `dim`** - not amber, which is
 * this design system's warning colour, and never accent. An undecided candidate is not a
 * warning: it is a candidate at a human-review queue that nobody has answered yet, which is
 * an ordinary state of an asynchronous selector.
 *
 * **There is no approve button here, no "send for review" and no pending count.** Reviewing
 * is miDataworks' queue (R-02.36, BRD-03 R-03.41); this renders what came back.
 *
 * The default branch is not optional: *"a two-valued verdict cannot express one of R-02.9's
 * own five selector types"*, the vocabulary has already had four spellings in this project,
 * and a pill that throws on an unknown value takes the panel down with it.
 */
import type { ReactNode } from 'react';
import { Pill, type PillTone } from './Pill';

export const VERDICT_VALUES = ['accept', 'reject', 'undecided'] as const;
export type VerdictValue = (typeof VERDICT_VALUES)[number];

const VERDICTS = {
  accept: { tone: 'green', label: 'Accept' },
  reject: { tone: 'red', label: 'Reject' },
  // Neutral. See the module docstring - amber is the warning colour and this is not a warning.
  undecided: { tone: 'neutral', label: 'Undecided' },
} as const satisfies Readonly<
  Record<VerdictValue, { readonly tone: PillTone; readonly label: string }>
>;

export interface VerdictPillProps {
  readonly verdict: string;
  readonly 'data-testid'?: string;
}

export function VerdictPill({ verdict, ...rest }: VerdictPillProps): ReactNode {
  const known = VERDICTS[verdict as VerdictValue];
  if (known === undefined) {
    return (
      <Pill tone="neutral" title={`Unrecognised verdict: ${verdict}`} {...rest}>
        {verdict} (unrecognised)
      </Pill>
    );
  }
  return (
    <Pill tone={known.tone} {...rest}>
      {known.label}
    </Pill>
  );
}
