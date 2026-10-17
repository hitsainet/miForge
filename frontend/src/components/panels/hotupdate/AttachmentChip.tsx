/**
 * The attachment chip, in the resource strip's right-hand slot (F11 task 16.4).
 *
 * `ResourceStrip`'s `trailing` prop is the slot F9 reserved for exactly this, so the strip
 * itself still knows nothing about miLLM (ADR-022): the chip is rendered by the shell only
 * when the open run carries an attachment block, and is absent on every other screen and
 * every other acquisition mode.
 *
 * What it shows is the mockup's strip content for this mode: the serving process, the
 * resident model, `weights_state` (cyan, never the miForge accent) and the holder when it
 * is not this run. A stale attachment adds the copy naming the operator's `clear-stale`.
 */
import type { ReactNode } from 'react';
import { Link2 } from 'lucide-react';
import { WeightsStatePill } from './WeightsStatePill';
import { STALE_ATTACHMENT_COPY, type AttachmentBlock } from './weightsState';

export interface AttachmentChipProps {
  readonly attachment: AttachmentBlock;
}

/** The host of the serving base URL, or the URL itself when it does not parse. */
export function servingHost(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export function AttachmentChip({ attachment }: AttachmentChipProps): ReactNode {
  return (
    <span
      className="inline-flex items-center gap-2 text-hint text-dim"
      data-testid="attachment-chip"
      title={attachment.stale === true ? STALE_ATTACHMENT_COPY : undefined}
    >
      <Link2 size={13} aria-hidden="true" className="text-cyan" />
      <span>
        Attached to miLLM at{' '}
        <b className="font-normal text-text" data-testid="attachment-serving">
          {servingHost(attachment.serving_base_url)}
        </b>
      </span>
      <span data-testid="attachment-model">
        {attachment.resident_model === null
          ? 'resident model not reported'
          : attachment.resident_model}
      </span>
      <WeightsStatePill state={attachment.weights_state} />
      {attachment.holder === null ? null : (
        <span data-testid="attachment-holder">held by {attachment.holder}</span>
      )}
      {attachment.attached === false ? (
        <span data-testid="attachment-detached">detached</span>
      ) : null}
      {attachment.observed_at ? (
        <span data-testid="attachment-observed" title={attachment.observed_at}>
          observed {attachment.observed_at}
        </span>
      ) : (
        <span data-testid="attachment-observed">state not yet observed</span>
      )}
      {attachment.stale === true ? (
        <span className="text-amber" data-testid="attachment-stale" role="status">
          {STALE_ATTACHMENT_COPY}
        </span>
      ) : null}
    </span>
  );
}
