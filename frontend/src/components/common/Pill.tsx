/**
 * Status pills (foundation tasks 7.2, 7.3; ADR-012).
 *
 * The handoff's shape rule: *"Status pills fully rounded, 12 px text, `color` at 22
 * percent alpha for the background"* - the mockup writes that as `${color}22`, which is
 * the hex byte `0x22`, so the alpha is `0x22/0xff ≈ 0.133`. `SHELL.pillBackgroundAlpha`
 * carries the number and says so, because "22 percent" read as 0.22 is a different pill.
 *
 * **The accent invariant lives here, not at the call sites.** ADR-012: the accent may not
 * appear on *"any status pill except 'Running'"*. `StatePill` enforces that: a tone map
 * that assigns `accent` to anything whose label is not `Running` is downgraded to
 * `neutral` and warned about. The design rule is one line of code rather than a sentence
 * every future feature has to remember - and `accent.test.ts` asserts it, because a
 * runtime downgrade that nobody tests is a rule that erodes quietly.
 */

import type { ReactNode } from 'react';
import { SHELL } from '@/config/tokens';

/**
 * The tones a pill may take. Deliberately not "any colour": `cyan` is reserved for miLLM
 * state, and `emerald`/`indigo` for miStudio and miDataworks, so a pill cannot be given
 * a colour that already means "another application" by accident (ADR-012).
 */
export type PillTone = 'accent' | 'green' | 'cyan' | 'red' | 'amber' | 'neutral';

/** The only label ADR-012 permits an accent pill. Compared, not pattern-matched. */
export const ACCENT_PILL_LABEL = 'Running';

const TONE_VARIABLE: Record<PillTone, string> = {
  // accent-ok:running-progress - the "Running" pill is the running state's own surface;
  // `StatePill` is what guarantees no other label reaches this entry.
  accent: '--mf-accent',
  green: '--mf-green',
  cyan: '--mf-cyan',
  red: '--mf-red',
  amber: '--mf-amber',
  neutral: '--mf-dim',
};

export interface PillProps {
  readonly tone: PillTone;
  readonly children: ReactNode;
  readonly title?: string;
  readonly 'data-testid'?: string;
}

/**
 * A pill in one tone.
 *
 * The colour is applied through `color-mix` over the token's variable rather than by
 * string-concatenating an alpha onto a hex, which is what the mockup does and what breaks
 * the moment a token is an `rgba()` rather than a `#rrggbb` - `accentTint` already is.
 */
export function Pill({ tone, children, title, ...rest }: PillProps): ReactNode {
  const variable = TONE_VARIABLE[tone];
  const alphaPercent = (SHELL.pillBackgroundAlpha * 100).toFixed(1);
  return (
    <span
      className="inline-flex items-center rounded-full px-2.5 py-1 text-hint font-medium"
      style={{
        color: `var(${variable})`,
        background: `color-mix(in srgb, var(${variable}) ${alphaPercent}%, transparent)`,
      }}
      title={title}
      data-tone={tone}
      {...rest}
    >
      {children}
    </span>
  );
}

/** A state's presentation: a tone and the sentence-case label the operator reads. */
export interface StateTone {
  readonly tone: PillTone;
  readonly label: string;
}

/**
 * The run states, as F6 defines them: `queued → running → { completed | failed }`
 * (F6 FR-006.13, quoted in F9's FTID 3.6). **No `held`, no `paused`** - the mockup has
 * both and neither is in the enum.
 *
 * Only the four are here. A feature with states of its own passes its own map; it does
 * not extend this one, because a shared map that every feature appends to is a registry
 * nobody owns.
 */
/**
 * The run states this build renders, from **two sources that disagree** - and the divergence
 * is carried rather than resolved here.
 *
 * - `backend/src/models/run.py` is the live column: `RUN_STATUSES` is
 *   `{pending, running, completed, failed, cancelled}`, and a submitted run's `status`
 *   **defaults to `pending`**.
 * - F6's FR-006.13 declares the enum as `queued -> running -> { completed | failed }`.
 *
 * So `pending` and `cancelled` are here because they are what the database can hold today -
 * without `pending`, the most common state of a freshly submitted run would render as
 * *"pending (unrecognised)"* on the run index - and `queued` stays because it is F6's
 * spelling and a renamed column would otherwise render unrecognised on the day it lands.
 * **Reported as a finding**: one concept, two vocabularies, which is the drift that produced
 * four spellings of the verdict value and three of the signal vocabulary in this project.
 *
 * The default branch below stays regardless of which spelling wins. An enum grows.
 */
export const RUN_STATE_TONES: Readonly<Record<string, StateTone>> = {
  pending: { tone: 'neutral', label: 'Pending' },
  queued: { tone: 'neutral', label: 'Queued' },
  running: { tone: 'accent', label: ACCENT_PILL_LABEL },
  completed: { tone: 'green', label: 'Completed' },
  failed: { tone: 'red', label: 'Failed' },
  // Neutral, not red: a run an operator stopped did what it was told. R-02.25's operator
  // stop and ADR-007's cooperative cancellation are not failures.
  cancelled: { tone: 'neutral', label: 'Cancelled' },
};

export interface StatePillProps {
  readonly state: string;
  /** Defaults to `RUN_STATE_TONES`. A feature with its own states supplies them. */
  readonly tones?: Readonly<Record<string, StateTone>>;
  readonly 'data-testid'?: string;
}

/**
 * A pill for a state value, with a default branch and the accent invariant.
 *
 * **The default branch is the point** (FR-009.8). The mockup's version destructures
 * `map[state]` with no fallback and **throws** on an unrecognised state - its own
 * 2026-09-19 review found this - and a real enum grows: F6's did, between the mockup
 * being written and the reframe. An unknown state renders the raw value, named as
 * unrecognised, in `neutral`. A screen that degrades is recoverable; a screen that
 * white-screens on a value the backend legitimately added is not.
 *
 * Under `noUncheckedIndexedAccess` the lookup is `StateTone | undefined`, so TypeScript
 * will not let the fallback be deleted without a compile error - the same arrangement as
 * `tsconfig.app.json`'s note about it.
 */
export function StatePill({ state, tones = RUN_STATE_TONES, ...rest }: StatePillProps): ReactNode {
  const known = tones[state];
  if (known === undefined) {
    return (
      <Pill tone="neutral" title={`Unrecognised state: ${state}`} {...rest}>
        {state} (unrecognised)
      </Pill>
    );
  }
  const tone = enforceAccentOnlyForRunning(known);
  return (
    <Pill tone={tone} {...rest}>
      {known.label}
    </Pill>
  );
}

/**
 * ADR-012's "no status pill but Running" in executable form.
 *
 * Downgrades rather than throws, for FR-009.8's reason - a pill must not take a screen
 * down - and warns, so the violation is visible in development. `accent.test.ts` is the
 * gate; this is the net under it.
 */
export function enforceAccentOnlyForRunning(state: StateTone): PillTone {
  if (state.tone === 'accent' && state.label !== ACCENT_PILL_LABEL) {
    console.warn(
      `StatePill: accent is reserved for the "${ACCENT_PILL_LABEL}" pill (ADR-012); ` +
        `"${state.label}" rendered in neutral instead.`,
    );
    return 'neutral';
  }
  return state.tone;
}
