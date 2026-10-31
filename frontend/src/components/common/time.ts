/**
 * The one way this UI shows a moment in time: US Eastern, human-readable, ISO on hover.
 *
 * The API, the database and the pods all stamp UTC ISO strings
 * (`2026-10-18T11:35:22.843344+00:00`). The operator reads Eastern time, and until 2026-10-19
 * screens printed the raw string, or the browser's own zone via `toLocaleString()`, depending
 * on which file rendered it. Every timestamp now goes through {@link formatEt}, and callers
 * put the raw value in a `title` so the exact instant stays one hover away.
 */

const ZONE = 'America/New_York';

const DAY_AND_TIME = new Intl.DateTimeFormat('en-US', {
  timeZone: ZONE,
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
});

const WITH_YEAR = new Intl.DateTimeFormat('en-US', {
  timeZone: ZONE,
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
});

const YEAR = new Intl.DateTimeFormat('en-US', { timeZone: ZONE, year: 'numeric' });

function parse(iso: string | null | undefined): Date | null {
  if (iso === null || iso === undefined || iso === '') return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : date;
}

/**
 * `"Oct 18, 7:35 AM ET"`, with the year added when it is not the current one. An unparseable
 * value is returned unchanged rather than rendered as "Invalid Date"; `null` becomes `""`.
 */
export function formatEt(iso: string | null | undefined, nowMs: number = Date.now()): string {
  if (iso === null || iso === undefined) return '';
  const date = parse(iso);
  if (date === null) return iso;
  const sameYear = YEAR.format(date) === YEAR.format(new Date(nowMs));
  return `${(sameYear ? DAY_AND_TIME : WITH_YEAR).format(date)} ET`;
}

/** `"just now"`, `"4 min ago"`, `"3 h ago"`, `"2 days ago"`; `""` for an unreadable value. */
export function formatAge(iso: string | null | undefined, nowMs: number = Date.now()): string {
  const date = parse(iso);
  if (date === null) return '';
  const seconds = Math.max(0, Math.round((nowMs - date.getTime()) / 1000));
  if (seconds < 45) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 36) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? '' : 's'} ago`;
}

/**
 * ENH-008 D8b: the approvals surfaces show a moment in the **browser's own time zone**, with
 * its abbreviation, and the exact instant (ISO 8601 in UTC) as the tooltip. The suite's four
 * apps share this format, so a browser set to Eastern time reads the same thing it did under
 * {@link formatEt}.
 *
 * The build spec names `{dateStyle: 'medium', timeStyle: 'short', timeZoneName: 'short'}`, but
 * ECMA-402 refuses `timeZoneName` beside `dateStyle`/`timeStyle` (a `TypeError`, "Invalid
 * option"). The explicit fields below are what `dateStyle: 'medium'` + `timeStyle: 'short'`
 * expand to, plus the zone, so the text is the one the spec means.
 *
 * A missing or unparseable value is `"unknown"`, never `"Invalid Date"`, and has no tooltip.
 */
export interface LocalTime {
  readonly text: string;
  /** The exact instant, ISO 8601 in UTC; `''` when the value could not be read. */
  readonly title: string;
}

export const UNKNOWN_TIME = 'unknown';

export function formatLocal(iso: string | null | undefined, timeZone?: string): LocalTime {
  const date = parse(iso);
  if (date === null) return { text: UNKNOWN_TIME, title: '' };
  const format = new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short',
    ...(timeZone === undefined ? {} : { timeZone }),
  });
  return { text: format.format(date), title: date.toISOString() };
}

/**
 * `"expires in 23 h"`, `"expires in 40 min"`, `"expires in less than 1 min"`; `"expired"` once
 * the moment has passed (the server marks it at the next read); `"never expires"` for no value.
 */
export function formatExpiresIn(
  iso: string | null | undefined,
  nowMs: number = Date.now(),
): string {
  if (iso === null || iso === undefined || iso === '') return 'never expires';
  const date = parse(iso);
  if (date === null) return `expires ${UNKNOWN_TIME}`;
  const seconds = Math.round((date.getTime() - nowMs) / 1000);
  if (seconds <= 0) return 'expired';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 1) return 'expires in less than 1 min';
  if (minutes < 60) return `expires in ${minutes} min`;
  return `expires in ${Math.round(minutes / 60)} h`;
}
