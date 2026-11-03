/**
 * miForge's design tokens, and the accent rule as data (foundation task 7.2; ADR-012).
 *
 * The dark values are **the mockup's, byte for byte** - `0xcc/prds/miforge-mockup.jsx`'s
 * `const T = {...}`, which ADR-012 then restates. They are not re-derived here and must
 * not be "tidied": the three suite apps share a shell and the shared part is the hex.
 *
 * Two things in this module are load-bearing beyond colour:
 *
 * 1. **`ACCENT_SURFACES` is the accent rule in a form a test can read.** ADR-012 says
 *    orange is accent only and names where it may appear; `accent.test.ts` enforces that
 *    list over the source tree. A convention nobody checks is how a design system erodes
 *    - and this project has already shipped the equivalent mistake in prose four times
 *    (a claim whose cited source did not support it), so the rule is encoded rather than
 *    remembered.
 * 2. **Light mode is derived here and nowhere else.** The mockup is dark-only and ADR-012
 *    lists only the dark palette, but ADR-011 requires Tailwind `darkMode: 'class'` and
 *    foundation 7.6 requires a smoke spec in **light and dark**. So a light set has to
 *    exist. It is marked as this module's own work, not the mockup's - see LIGHT.
 */

/** The dark palette, copied from the mockup's `T`. Do not re-derive. */
export const DARK = {
  bg: '#0d1420',
  panel: '#131b28',
  card: '#18212f',
  line: '#243044',
  text: '#e6ebf2',
  dim: '#8b97a8',
  accent: '#F97316',
  accentHover: '#FB923C',
  accentTint: 'rgba(249,115,22,0.12)',
  green: '#22c55e',
  cyan: '#22d3ee',
  red: '#ef4444',
  amber: '#f59e0b',
} as const;

/**
 * The light palette.
 *
 * **This is not from the mockup.** `miforge-mockup-handoff.md` 2 gives one palette and
 * 6 lists its known gaps; a light theme is in neither, so nothing upstream specifies
 * these values. They are chosen here to hold the *relationships* the dark set encodes -
 * `panel` and `card` separate from `bg`, `line` visible against both, `dim` legible but
 * clearly secondary - at light-mode luminance.
 *
 * **The accent, and the four state colours, are identical in both themes on purpose.**
 * `accent` is miForge's identity and the suite reserves `cyan` for miLLM state, emerald
 * for miStudio and indigo for miDataworks (ADR-012); a theme that shifted those would
 * make "this chip means miLLM" theme-dependent. `accentTint` is an alpha over whatever
 * is behind it, so it composites correctly in both without a second value.
 */
export const LIGHT = {
  bg: '#f6f8fb',
  panel: '#ffffff',
  card: '#ffffff',
  line: '#d8dfe9',
  text: '#121923',
  dim: '#5b6675',
  accent: '#F97316',
  accentHover: '#EA6B06',
  accentTint: 'rgba(249,115,22,0.12)',
  green: '#16a34a',
  cyan: '#0891b2',
  red: '#dc2626',
  amber: '#b45309',
} as const;

export type TokenName = keyof typeof DARK;

/** Every token name, in declaration order. Used by `tokens.test.ts` and the CSS check. */
export const TOKEN_NAMES = Object.keys(DARK) as readonly TokenName[];

/** `accentTint` -> `--mf-accent-tint`. The one place the two spellings are bridged. */
export function cssVariableName(token: TokenName): string {
  return `--mf-${token.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`;
}

/* ------------------------------------------------------------------------------------ *
 * Shape and type, from the handoff's 2.
 * ------------------------------------------------------------------------------------ */

/**
 * The suite shell's measurements (ADR-012; handoff 2 and 3).
 *
 * These are px integers rather than Tailwind class names because three of them are
 * **quoted as numbers** in the documents that bind this shell - the 224 px sidebar in
 * ADR-012 and FR-009.2, the 12 px card radius and 20 px padding in the card grammar -
 * and a test can compare a number to a document. `sidebarWidthPx: 224` is `w-56`; the
 * mockup writes `w-56` and a reader has to know the scale to check it.
 */
export const SHELL = {
  sidebarWidthPx: 224,
  /** Collapsed to icons, as miStudio's and miLLM's sidebars collapse (`w-16`). */
  sidebarCollapsedWidthPx: 64,
  /** Content column. The handoff says "max-width ~72rem"; the mockup's `max-w-6xl`. */
  contentMaxWidthPx: 1152,
  cardRadiusPx: 12,
  cardPaddingPx: 20,
  cardBorderPx: 1,
  /** Inputs and buttons. */
  controlRadiusPx: 8,
  /** Icon tiles are square. */
  iconTilePx: 36,
  /** Checkpoint-tree indent per level of depth. The mockup's `c.depth * 28`. */
  treeIndentPx: 28,
  /** Status-pill background alpha over the pill's own colour - the mockup's `${color}22`. */
  pillBackgroundAlpha: 0.133, // 0x22 / 0xff; "22 percent" in the handoff is the hex.
} as const;

/**
 * The type scale (handoff 2, "Type").
 *
 * Sentence case everywhere, no all-caps labels - that is copy, not type, and is enforced
 * by review rather than by this table.
 */
export const TYPE = {
  pageTitlePx: 24,
  sectionHeadingPx: 16,
  bodyPx: 14,
  hintPx: 12,
  monoPx: 12,
} as const;

/**
 * Font stacks. Inter for text, JetBrains Mono for **identifiers, hashes and step logs**
 * (ADR-012). The fallbacks matter: the webfont is fetched from Google Fonts in
 * `index.css`, the dev node is frequently offline, and a mono stack that degrades to a
 * proportional font makes a 64-character hash unreadable rather than merely plainer.
 */
export const FONTS = {
  sans: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  mono: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
} as const;

/* ------------------------------------------------------------------------------------ *
 * The accent rule, as data.
 * ------------------------------------------------------------------------------------ */

/**
 * The seven surfaces ADR-012 permits the accent on, quoted from it:
 *
 * > orange appears on the active nav item, primary buttons, the logo tile, list-row icon
 * > tiles, the loss curve, running progress bars, and the callout banner tint. **Not** on
 * > body text, borders, backgrounds, or any status pill except "Running."
 *
 * Frozen as its own constant so that widening the rule is a visible edit to a list that
 * cites a document, not a new line in an allowlist. `accent.test.ts` asserts that
 * `ACCENT_SURFACES` is exactly this set plus the one documented addition below.
 *
 * **One name covers two things, and it is `running-progress`.** ADR-012's enumeration
 * lists "running progress bars", and the *same sentence* that forbids the accent on
 * status pills excepts `"Running"` - so the Running pill is permitted by ADR-012 without
 * appearing in its list of seven. Rather than invent an eighth name for it and lose the
 * ability to say "these are exactly the ADR's seven", `running-progress` is read as **the
 * running state's accent**: a running run's progress bar and its `"Running"` pill. Both
 * are the same fact rendered twice, and `Pill.tsx`'s `enforceAccentOnlyForRunning` is
 * what keeps the second from spreading to any other state.
 */
export const ADR_012_ACCENT_SURFACES = [
  'active-nav',
  'primary-button',
  'logo-tile',
  'row-icon-tile',
  'loss-curve',
  'running-progress',
  'callout-tint',
] as const;

/**
 * The one surface added to ADR-012's seven, and why.
 *
 * **`focus-ring`.** Foundation 7.5 requires "the sibling apps' focus ring" because the
 * mockup has none (its own handoff 6 says so). Both siblings draw that ring in **their
 * own accent** - miStudio `focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2
 * focus:ring-offset-slate-900`, miLLM `focus:border-primary-400/50` - so copying the
 * pattern and not the colour would make miForge the only app in the suite whose focus
 * indicator is not its accent.
 *
 * It is listed separately rather than folded into the seven because **it is an extension
 * of ADR-012, decided here, and the operator may reverse it.** The argument for it: a
 * focus ring is not any of the four things ADR-012 forbids - it is not body text, not a
 * border (Tailwind's ring is a box-shadow, and `line` remains the only border colour),
 * not a background, and not a status pill - and it is transient, on exactly one element,
 * which is the opposite of "the screen starts to look orange". The argument against it:
 * the seven were enumerated and this is an eighth.
 */
export const ADDED_ACCENT_SURFACES = ['focus-ring'] as const;

/** Every surface the accent token may appear on. Read by `accent.test.ts`. */
export const ACCENT_SURFACES = [...ADR_012_ACCENT_SURFACES, ...ADDED_ACCENT_SURFACES] as const;

export type AccentSurface = (typeof ACCENT_SURFACES)[number];

/**
 * Suite state colours, reserved (ADR-012). A chip or curve that means "miLLM" is cyan in
 * every app in the suite; these are not miForge's to repurpose.
 *
 * `miStudio` and `miDataworks` have no token in `DARK`/`LIGHT` because nothing in the
 * shell renders their state yet. They are named here so that the next person reaches for
 * the reserved colour rather than inventing one, and so the reservation is greppable.
 */
export const SUITE_STATE_COLOURS = {
  miLLM: 'cyan',
  miStudio: 'emerald',
  miDataworks: 'indigo',
} as const;
