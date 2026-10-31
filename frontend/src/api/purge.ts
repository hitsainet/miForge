/**
 * Purge: permanently delete ARCHIVED runs and parts (operator decision 2026-10-20).
 *
 * `POST /runs|parts/purge/plan` says, per item, what a purge would remove and what blocks
 * it - it changes nothing. `POST /runs|parts/purge` purges a selection, **each item on its
 * own**: one blocked item never stops the rest, and the answer has a result per item.
 * Operator only: the backend refuses a request carrying the agent header, and no MCP tool
 * reaches these routes (`services/purge/service.py`).
 */
import { apiPost, type ApiResult } from './client';
import { V1 } from './v1';

export type PurgeKind = 'run' | 'part';

export interface PurgeBlocker {
  readonly code: string;
  readonly message: string;
  readonly [detail: string]: unknown;
}

/** One item of a preview: `services.purge.service._Plan.payload`. */
export interface PurgePlan {
  readonly kind: PurgeKind;
  readonly id: string;
  readonly found: boolean;
  readonly name: string;
  readonly archived: boolean;
  readonly purgeable: boolean;
  readonly not_archived: { readonly code: string; readonly message: string } | null;
  readonly blockers: readonly PurgeBlocker[];
  readonly removes: {
    readonly rows: Readonly<Record<string, number>>;
    readonly row_total: number;
    readonly files: number;
    readonly bytes: number;
    readonly directories: readonly string[];
  };
  readonly kept: readonly { readonly path: string; readonly why: string }[];
}

/** One item of a batch: purged (a tombstone), or blocked with the reason. */
export type PurgeResult =
  | {
      readonly outcome: 'purged';
      readonly id: string;
      readonly name: string;
      readonly tombstone_id: string;
      readonly bytes_removed: number;
      readonly files_removed: number;
      readonly rows_removed: Readonly<Record<string, number>>;
      readonly files_left: readonly string[];
    }
  | {
      readonly outcome: 'blocked';
      readonly id: string;
      readonly code: string;
      readonly reason: string;
      readonly blockers: readonly PurgeBlocker[];
    };

export interface PurgeBatch {
  readonly kind: PurgeKind;
  readonly results: readonly PurgeResult[];
  readonly purged: number;
  readonly blocked: number;
  readonly bytes_removed: number;
}

/** The route's own bound on one selection. */
export const PURGE_BATCH_LIMIT = 200;

const base = (kind: PurgeKind): string => `${V1}/${kind === 'run' ? 'runs' : 'parts'}/purge`;

/** What purging each item would remove, and what blocks it. Changes nothing. */
export async function previewPurge(
  kind: PurgeKind,
  ids: readonly string[],
): Promise<ApiResult<readonly PurgePlan[]>> {
  const items: PurgePlan[] = [];
  for (let start = 0; start < ids.length; start += PURGE_BATCH_LIMIT) {
    const chunk = ids.slice(start, start + PURGE_BATCH_LIMIT);
    const result = await apiPost<{ readonly items: readonly PurgePlan[] }>(`${base(kind)}/plan`, {
      ids: chunk,
    });
    if (!result.ok) return result;
    items.push(...result.value.items);
  }
  return { ok: true, value: items };
}

/** Purge exactly `ids`, each independently. */
export const purgeSelection = (
  kind: PurgeKind,
  ids: readonly string[],
  reason: string,
): Promise<ApiResult<PurgeBatch>> =>
  apiPost<PurgeBatch>(base(kind), {
    ids,
    ...(reason.trim().length === 0 ? {} : { reason: reason.trim() }),
  });

/** `12.3 MB`, `512 B`. Binary multiples, one decimal above a kilobyte. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}
