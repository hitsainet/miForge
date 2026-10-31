/**
 * Where the accent token is allowed to appear (foundation task 7.2; ADR-012).
 *
 * **One entry per file per surface, and `accent.test.ts` enforces it in both
 * directions**: a file that paints with the accent and is not listed here fails, and an
 * entry listed here that nothing exercises fails too.
 *
 * The second direction is the one that is usually left out, and it is what stops this
 * list becoming a rubber stamp. An allowlist that only ever grows records the history of
 * everyone who wanted an exception; one that must stay exact records the design.
 *
 * **Adding an entry is a design decision, not a fix for a red test.** The question to
 * answer first is whether the new surface is one of ADR-012's - *"If a screen starts to
 * look orange, something is wrong."*
 */

import type { AccentSurface } from './tokens';

export interface AccentUse {
  /** Repo-relative path from `frontend/`, forward slashes. */
  readonly file: string;
  readonly surface: AccentSurface;
}

export const ACCENT_ALLOWLIST: readonly AccentUse[] = [
  // ADR-012: "the active nav item (tint background, orange text and icon)".
  { file: 'src/components/layout/NavItem.tsx', surface: 'active-nav' },
  // ADR-012: "primary buttons".
  { file: 'src/components/common/Button.tsx', surface: 'primary-button' },
  // ADR-012: "the app logo tile" and "the icon tile on list rows".
  { file: 'src/components/common/IconTile.tsx', surface: 'logo-tile' },
  { file: 'src/components/common/IconTile.tsx', surface: 'row-icon-tile' },
  // ADR-012: "the callout banner tint".
  { file: 'src/components/common/Callout.tsx', surface: 'callout-tint' },
  // ADR-012: "running progress bars", and the `"Running"` pill its next sentence excepts.
  // See the note on ADR_012_ACCENT_SURFACES for why one name covers both.
  { file: 'src/components/common/Pill.tsx', surface: 'running-progress' },
  // F9's two: ADR-012 names "the loss curve" and "running progress bars" among its seven.
  // Both are enumerated rather than inferred, and both are the ONLY accent in their file -
  // `RunCurves` draws test green and retention cyan, and `RunRow`'s bar is `dim` for every
  // state but running, which is the mockup bug FR-009.3 fixes.
  { file: 'src/components/panels/run/RunCurves.tsx', surface: 'loss-curve' },
  { file: 'src/components/panels/runs/RunRow.tsx', surface: 'running-progress' },
];

/**
 * Files excluded from the scan, with the reason each is excluded.
 *
 * Kept as a table rather than a glob so that an exclusion has to be argued for. A glob
 * like `src/config/**` would quietly exempt whatever is put in that directory next.
 */
export const ACCENT_SCAN_EXCLUSIONS: readonly { readonly file: string; readonly why: string }[] = [
  { file: 'src/config/tokens.ts', why: 'defines the token and the rule' },
  { file: 'src/config/accentAllowlist.ts', why: 'is the allowlist' },
  // `*.test.ts(x)` is skipped by the scanner's own pattern, not listed here: a
  // component's test may legitimately assert that the accent IS present.
];

/**
 * The marker a painting line must carry, in a comment on or just above it.
 *
 * **Why a marker and not only a per-file allowlist.** A file is allowed *one* surface -
 * `Button.tsx` may paint a primary button - and without a marker it could paint anything
 * else and still pass, because the allowlist cannot see what a line is for. The marker is
 * the author stating which of ADR-012's surfaces this line is, where a reviewer and a
 * test can both read it.
 */
export const ACCENT_MARKER_PREFIX = 'accent-ok:';

/**
 * How many lines above a painting line the marker may sit.
 *
 * Six, because the house comment style puts the reasoning in the comment and the
 * reasoning is usually two or three lines - see `Button.tsx`. A window of one would push
 * the citation off the line that needs it, and a marker with no citation is a token.
 */
export const ACCENT_MARKER_WINDOW = 6;
