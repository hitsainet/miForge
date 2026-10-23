/**
 * Unsaved entries that survive a reload (F9 task 13.3; operator request 2026-10-17).
 *
 * The one module that reads and writes form drafts in browser storage. Forms reach it only
 * through `hooks/usePersistentState.ts`, so there is one envelope, one secret rule and one
 * failure policy - not one per form.
 *
 * ## `localStorage`, not `sessionStorage` (D-13d)
 *
 * The operator asked for entries that **stick**. `sessionStorage` is per tab: it is gone when
 * the tab closes, absent in a tab opened from a bookmark, and *copied* into a duplicated tab.
 * `localStorage` survives all three. Its costs are accepted and bounded: two tabs on one form
 * share one draft (PPRD section 7 is one operator), and a draft lingers - so every entry
 * carries `savedAt` and is discarded after {@link DRAFT_MAX_AGE_MS}, is removed on a
 * successful submit, and is announced when restored so it is never mistaken for a default.
 *
 * ## The envelope
 *
 * `miforge.draft.<form>` holds `{"v": <version>, "savedAt": <ms>, "data": ...}`. A wrong
 * version, unparseable JSON, an expired entry or data the form's own `decode` rejects is
 * **removed and ignored** - a draft from an older build is discarded, never crashed on.
 *
 * ## Storage that throws
 *
 * Every call is wrapped. Private windows, blocked site data, a full quota and sandboxed frames
 * all throw from `localStorage`; the form then works and simply does not persist - the same
 * rule `ThemeContext` follows for the theme.
 *
 * ## No secret is ever written (D-13f)
 *
 * {@link stripSecrets} runs inside {@link writeDraft} on every write, so no caller can opt
 * out. It drops any key whose words include a credential word and any string shaped like a
 * credential. No form has such a field today (ADR-020 keeps credentials in the environment);
 * the rule exists so the first one added cannot land in a browser profile. M25's target.
 */

export const DRAFT_PREFIX = 'miforge.draft.';

/** Fourteen days. A draft older than this is more likely forgotten than wanted. */
export const DRAFT_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000;

/**
 * Every persisted form's storage key, in one table. A form not named here has no draft;
 * `persistence.test.tsx` derives which forms exist from the screens, not from this table.
 */
export const DRAFT_KEYS = {
  newRun: 'new-run',
  runsIndex: 'runs-index',
  onboard: (draftId: string) => `onboard.${draftId}`,
  /** "Show archived parts" on the Packs screen's part index (operator 2026-10-20). */
  partsIndexArchived: 'parts-index.show-archived',
  /** "Show archived parts" beside the Onboard slot pickers (operator 2026-10-20). */
  slotPickersArchived: 'slot-pickers.show-archived',
  /** The Settings screen's unsaved values, by setting name (F12; operator 2026-10-20). */
  settings: 'settings',
} as const;

export interface StoredDraft<T> {
  readonly data: T;
  readonly savedAt: number;
}

interface Envelope {
  readonly v: number;
  readonly savedAt: number;
  readonly data: unknown;
}

/** Words that make a key a credential. Matched against the key's words, not substrings. */
const SECRET_WORDS: ReadonlySet<string> = new Set([
  'token',
  'tokens',
  'password',
  'passwd',
  'passphrase',
  'pwd',
  'secret',
  'secrets',
  'key',
  'apikey',
  'credential',
  'credentials',
  'auth',
  'authorization',
  'bearer',
  'cookie',
]);

/**
 * The words of an identifier: `hfToken` -> hf, token; `api_key` -> api, key; `x-auth` -> x,
 * auth. Whole words, so `tokenizer` is not `token` and `monkey` is not `key`.
 */
function words(name: string): string[] {
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .split(/[^A-Za-z0-9]+/)
    .filter((part) => part.length > 0)
    .map((part) => part.toLowerCase());
}

/** True for a field name that holds a credential. */
export function isSecretKey(name: string): boolean {
  return words(name).some((word) => SECRET_WORDS.has(word));
}

const SECRET_VALUE_PATTERNS: readonly RegExp[] = [
  /\bhf_[A-Za-z0-9]{20,}\b/, // Hugging Face access token
  /\bsk-[A-Za-z0-9_-]{16,}\b/, // OpenAI-style API key
  /\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b/, // GitHub token
  /\bgithub_pat_[A-Za-z0-9_]{20,}\b/,
  /\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/i,
  /[a-z][a-z0-9+.-]*:\/\/[^\s/:@]+:[^\s/@]+@/i, // a URL carrying user:password@
];

/** True for a string shaped like a credential, whatever the field is called. */
export function isSecretValue(value: string): boolean {
  return SECRET_VALUE_PATTERNS.some((pattern) => pattern.test(value));
}

/**
 * A copy of `value` with every secret key removed and every credential-shaped string
 * replaced by `''` (in an object) or dropped (in an array). Recursive.
 */
export function stripSecrets(value: unknown): unknown {
  if (typeof value === 'string') return isSecretValue(value) ? '' : value;
  if (Array.isArray(value)) {
    return value
      .filter((item) => !(typeof item === 'string' && isSecretValue(item)))
      .map((item) => stripSecrets(item));
  }
  if (value !== null && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [key, inner] of Object.entries(value as Record<string, unknown>)) {
      if (isSecretKey(key)) continue;
      out[key] = stripSecrets(inner);
    }
    return out;
  }
  return value;
}

function storage(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : window.localStorage;
  } catch {
    return null;
  }
}

export function removeDraft(form: string): void {
  try {
    storage()?.removeItem(DRAFT_PREFIX + form);
  } catch {
    // Storage unavailable: there is nothing to remove that this page could read either.
  }
}

/**
 * The stored draft, or `null`. Anything unreadable is removed so it is not re-read forever.
 *
 * `decode` is the form's own shape check, returning `null` for a value it cannot use.
 */
export function readDraft<T>(
  form: string,
  version: number,
  decode: (data: unknown) => T | null,
  now: number = Date.now(),
): StoredDraft<T> | null {
  let raw: string | null;
  try {
    raw = storage()?.getItem(DRAFT_PREFIX + form) ?? null;
  } catch {
    return null;
  }
  if (raw === null) return null;
  let envelope: Envelope;
  try {
    envelope = JSON.parse(raw) as Envelope;
  } catch {
    removeDraft(form);
    return null;
  }
  if (
    envelope === null ||
    typeof envelope !== 'object' ||
    envelope.v !== version ||
    typeof envelope.savedAt !== 'number' ||
    !Number.isFinite(envelope.savedAt) ||
    now - envelope.savedAt > DRAFT_MAX_AGE_MS
  ) {
    removeDraft(form);
    return null;
  }
  let data: T | null;
  try {
    data = decode(envelope.data);
  } catch {
    data = null;
  }
  if (data === null) {
    removeDraft(form);
    return null;
  }
  return { data, savedAt: envelope.savedAt };
}

/** Write a draft, secrets stripped. Returns `false` when storage refused it. */
export function writeDraft(
  form: string,
  version: number,
  data: unknown,
  now: number = Date.now(),
): boolean {
  const envelope: Envelope = { v: version, savedAt: now, data: stripSecrets(data) };
  try {
    const target = storage();
    if (target === null) return false;
    target.setItem(DRAFT_PREFIX + form, JSON.stringify(envelope));
    return true;
  } catch {
    return false;
  }
}

/** Helpers for `decode` functions: a plain object, and a string field of one. */
export function asRecord(value: unknown): Readonly<Record<string, unknown>> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : null;
}
