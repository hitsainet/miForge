/**
 * The Settings screen's groups (2026-10-28): which settings sit together, and in what order.
 *
 * **The groups are the backend's** (`core/settings_catalog.py::GROUPS`, `GROUP_OF`), served by
 * `GET /api/v1/settings` as `groups` (in screen order) and each row's `group`. The screen holds
 * no copy of the field list, so a new setting lands in its group the day the backend places it.
 *
 * Within a group the rows are **sorted by label**, the words the operator reads, not by the raw
 * field name - `millm_base_url` reads "miLLM address". A row whose group the listing does not
 * name, or an older backend that serves no groups at all, falls into a final **Other** group
 * rather than disappearing.
 */
import type { SettingRow, SettingsGroup } from '@/api/settings';

export const OTHER_GROUP: SettingsGroup = { id: 'other', label: 'Other' };

export interface GroupedSettings {
  readonly id: string;
  readonly label: string;
  readonly rows: readonly (readonly [string, SettingRow])[];
}

const labelOf = ([name, row]: readonly [string, SettingRow]): string => row.label ?? name;

/** Alphabetical by label, case- and accent-insensitive; the raw name breaks a tie. */
export function byLabel(
  a: readonly [string, SettingRow],
  b: readonly [string, SettingRow],
): number {
  return (
    labelOf(a).localeCompare(labelOf(b), 'en', { sensitivity: 'base' }) || a[0].localeCompare(b[0])
  );
}

/** `entries` in the listing's group order, sorted by label within each; empty groups dropped. */
export function groupSettings(
  entries: readonly (readonly [string, SettingRow])[],
  groups: readonly SettingsGroup[] = [],
): readonly GroupedSettings[] {
  const order = [...groups.filter((group) => group.id !== OTHER_GROUP.id), OTHER_GROUP];
  const known = new Set(order.map((group) => group.id));
  const buckets = new Map<string, (readonly [string, SettingRow])[]>();
  for (const entry of entries) {
    const id = entry[1].group !== undefined && known.has(entry[1].group) ? entry[1].group : 'other';
    const bucket = buckets.get(id) ?? [];
    bucket.push(entry);
    buckets.set(id, bucket);
  }
  return order
    .map((group) => ({
      id: group.id,
      label: group.label,
      rows: [...(buckets.get(group.id) ?? [])].sort(byLabel),
    }))
    .filter((group) => group.rows.length > 0);
}
