/**
 * The settings surface (F12; FR-012.7, .8; `backend/src/api/v1/endpoints/settings.py`).
 *
 * `GET /settings` serves every field of the backend's `Settings` with its effective value,
 * where it came from and whether it may change at runtime; secrets arrive already redacted.
 * `PUT` / `DELETE /settings/{key}` write or clear an override for a key on the backend's
 * `RUNTIME_KEYS`; any other key is refused there, not here. Both take an optional reason, and
 * `GET /settings/history` reads the append-only record of every change (ENH-003 D12).
 *
 * **Never sends `X-Miforge-Agent`** - nothing in the frontend does - so a write from this
 * screen is the operator's own and is not held for approval.
 */
import { apiDelete, apiGet, apiPut, type ApiResult } from './client';
import { V1 } from './v1';

export type SettingSource = 'env' | 'override';

/** When a change takes effect (`core/settings_catalog.py`; ENH-003 section 3.4, gap 4). */
export type SettingEffectWhen = 'now' | 'next_sweep' | 'next_attached_run' | 'restart';

export interface SettingEffect {
  readonly when: SettingEffectWhen;
  readonly text: string;
  /** The pod to restart, for `restart` only. */
  readonly pod?: string;
}

/** One recorded change (`GET /settings/history`; append-only). */
export interface SettingChange {
  readonly id: number;
  readonly key: string;
  readonly action: 'set' | 'reset';
  readonly previous: unknown;
  readonly previous_source: SettingSource;
  readonly value: unknown;
  readonly source: SettingSource;
  /** `operator`, or `agent:<name>` for an agent's approved request. */
  readonly actor: string;
  readonly approval_id: string | null;
  readonly reason: string | null;
  readonly changed_at: string;
}

export interface SettingRow {
  readonly value: unknown;
  readonly source: SettingSource;
  readonly runtime_writable: boolean;
  /** Plain-language name, description and effect timing. Absent from an older backend. */
  readonly label?: string;
  readonly description?: string;
  readonly effect?: SettingEffect;
  /** The environment's value, served redacted like `value`. */
  readonly env_value?: unknown;
  readonly updated_at?: string | null;
  readonly updated_by?: string | null;
  readonly last_change?: SettingChange | null;
}

export interface SettingsListing {
  readonly settings: Readonly<Record<string, SettingRow>>;
  readonly count: number;
  readonly runtime_keys: readonly string[];
  readonly overrides: readonly string[];
}

export interface SettingWritten {
  readonly key: string;
  readonly value: unknown;
  readonly previous?: unknown;
  readonly source: SettingSource;
}

export const SETTINGS_PATH = `${V1}/settings`;

export const readSettings = (): Promise<ApiResult<SettingsListing>> =>
  apiGet<SettingsListing>(SETTINGS_PATH);

export interface SettingsHistory {
  readonly changes: readonly SettingChange[];
  readonly count: number;
}

/** How many changes the screen shows (ENH-003 D12: the last 50). */
export const HISTORY_SHOWN = 50;

/** A reason is optional; an empty one is not sent. */
function reasonOf(reason: string | undefined): string | undefined {
  const text = reason?.trim() ?? '';
  return text === '' ? undefined : text;
}

export const writeSetting = (
  key: string,
  value: unknown,
  reason?: string,
): Promise<ApiResult<SettingWritten>> => {
  const why = reasonOf(reason);
  return apiPut<SettingWritten>(
    `${SETTINGS_PATH}/${encodeURIComponent(key)}`,
    why === undefined ? { value } : { value, reason: why },
  );
};

export const resetSetting = (key: string, reason?: string): Promise<ApiResult<SettingWritten>> => {
  const why = reasonOf(reason);
  const query = why === undefined ? '' : `?${new URLSearchParams({ reason: why }).toString()}`;
  return apiDelete<SettingWritten>(`${SETTINGS_PATH}/${encodeURIComponent(key)}${query}`);
};

export const readSettingsHistory = (
  limit: number = HISTORY_SHOWN,
): Promise<ApiResult<SettingsHistory>> =>
  apiGet<SettingsHistory>(`${SETTINGS_PATH}/history?limit=${limit}`);
