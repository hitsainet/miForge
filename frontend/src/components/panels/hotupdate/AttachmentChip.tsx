/**
 * The attachment chip, in the resource strip's right-hand slot (F11 task 16.4).
 *
 * **It shows miLLM's attachment state now**, from `GET /api/v1/attachment`, which the backend
 * reads live on each request. Until 2026-10-19 it rendered the open run's record - miLLM's
 * state as that run last observed it - and a run that failed while attached left that record
 * at "attached, weights clean" indefinitely, so the chip reported an attachment from the
 * morning while miLLM held nothing. A run's own last observation is shown on the run view
 * instead (`RunAttachmentRecord`), labelled as such.
 *
 * `ResourceStrip`'s `trailing` prop is the slot reserved for this, so the strip still knows
 * nothing about miLLM (ADR-022). The shell renders the chip only when the backend reports a
 * serving process configured; on the standalone default there is no chip at all.
 *
 * `weights_state` is cyan, never the miForge accent (ADR-012). A stale attachment adds the
 * copy naming the operator's `clear-stale`.
 */
import type { ReactNode } from 'react';
import { Link2, Unlink } from 'lucide-react';
import type { AttachmentStatus } from '@/api/attachment';
import { formatAge, formatEt } from '@/components/common/time';
import { WeightsStatePill } from './WeightsStatePill';
import { STALE_ATTACHMENT_COPY } from './weightsState';

export interface AttachmentChipProps {
  readonly status: AttachmentStatus;
  /**
   * `true` when the latest read of `/api/v1/attachment` failed and `status` is the last good
   * one. The chip then says the state is from an earlier check, and how old it is.
   */
  readonly outdated?: boolean;
  /** For tests: the clock the age is measured against. */
  readonly nowMs?: number;
}

/** The host of the serving base URL, or the URL itself when it does not parse. */
export function servingHost(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

function Checked({
  observedAt,
  outdated,
  nowMs,
}: {
  readonly observedAt: string | undefined;
  readonly outdated: boolean;
  readonly nowMs: number;
}): ReactNode {
  if (!observedAt) return null;
  return (
    <span data-testid="attachment-observed" title={observedAt}>
      {outdated
        ? `last checked ${formatEt(observedAt, nowMs)} (${formatAge(observedAt, nowMs)})`
        : `checked ${formatEt(observedAt, nowMs)}`}
    </span>
  );
}

export function AttachmentChip({
  status,
  outdated = false,
  nowMs = Date.now(),
}: AttachmentChipProps): ReactNode {
  const host = status.serving_base_url ? servingHost(status.serving_base_url) : 'miLLM';

  if (status.reachable === false) {
    return (
      <span
        className="inline-flex items-center gap-2 text-hint text-dim"
        data-testid="attachment-chip"
        data-attachment="unreachable"
        title={status.reason}
      >
        <Unlink size={13} aria-hidden="true" className="text-cyan" />
        <span>
          miLLM at <b className="font-normal text-text">{host}</b> not reachable
        </span>
        <Checked observedAt={status.observed_at} outdated={outdated} nowMs={nowMs} />
      </span>
    );
  }

  if (status.attached !== true) {
    return (
      <span
        className="inline-flex items-center gap-2 text-hint text-dim"
        data-testid="attachment-chip"
        data-attachment="detached"
      >
        <Unlink size={13} aria-hidden="true" className="text-cyan" />
        <span data-testid="attachment-detached">
          Not attached to miLLM at{' '}
          <b className="font-normal text-text" data-testid="attachment-serving">
            {host}
          </b>
        </span>
        <Checked observedAt={status.observed_at} outdated={outdated} nowMs={nowMs} />
      </span>
    );
  }

  return (
    <span
      className="inline-flex items-center gap-2 text-hint text-dim"
      data-testid="attachment-chip"
      data-attachment="attached"
      title={status.stale === true ? STALE_ATTACHMENT_COPY : undefined}
    >
      <Link2 size={13} aria-hidden="true" className="text-cyan" />
      <span>
        Attached to miLLM at{' '}
        <b className="font-normal text-text" data-testid="attachment-serving">
          {host}
        </b>
      </span>
      {status.holder ? <span data-testid="attachment-holder">held by {status.holder}</span> : null}
      {status.resident_model ? (
        <span data-testid="attachment-model">{status.resident_model}</span>
      ) : null}
      <WeightsStatePill state={status.weights_state ?? null} />
      <Checked observedAt={status.observed_at} outdated={outdated} nowMs={nowMs} />
      {status.stale === true ? (
        <span className="text-amber" data-testid="attachment-stale" role="status">
          {STALE_ATTACHMENT_COPY}
        </span>
      ) : null}
    </span>
  );
}
