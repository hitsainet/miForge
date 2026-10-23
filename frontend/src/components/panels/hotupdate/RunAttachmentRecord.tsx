/**
 * One attached run's own record of miLLM's state, on the run view (F11 task 16.4).
 *
 * This is `GET /api/v1/runs/{id}`'s `attachment` block: miLLM's status **as this run last
 * observed it**, written by the trainer. It is history, so it is labelled with when it was
 * observed and, once the run has ended, says plainly that it is the last state the run saw and
 * not miLLM's state now - that is the shell's attachment chip, which reads miLLM live. Until
 * 2026-10-19 this block fed the shell's chip, and a run that failed while attached kept the
 * chip reading "attached" long after miLLM had dropped the attachment.
 */
import type { ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { formatAge, formatEt } from '@/components/common/time';
import { isTerminal } from '@/stores/runStore';
import { servingHost } from './AttachmentChip';
import { WeightsStatePill } from './WeightsStatePill';
import { STALE_ATTACHMENT_COPY, type AttachmentBlock } from './weightsState';

export interface RunAttachmentRecordProps {
  readonly attachment: AttachmentBlock;
  readonly runStatus: string;
  readonly nowMs?: number;
}

export function RunAttachmentRecord({
  attachment,
  runStatus,
  nowMs = Date.now(),
}: RunAttachmentRecordProps): ReactNode {
  const ended = isTerminal(runStatus);
  const observed = attachment.observed_at ?? null;
  const attached = attachment.attached === true && !ended;

  return (
    <Card className="mb-4" data-testid="run-attachment">
      <div className="mb-1 text-heading">miLLM attachment</div>
      <div className="flex flex-wrap items-center gap-2 text-hint text-dim">
        <span>
          {attached ? 'Attached to' : ended ? 'Ran attached to' : 'Detached from'} miLLM at{' '}
          <b className="font-normal text-text" data-testid="run-attachment-serving">
            {servingHost(attachment.serving_base_url)}
          </b>
        </span>
        {attachment.resident_model ? (
          <span data-testid="run-attachment-model">{attachment.resident_model}</span>
        ) : null}
        {attachment.weights_state === null ? null : (
          <WeightsStatePill state={attachment.weights_state} />
        )}
        {attachment.holder ? (
          <span data-testid="run-attachment-holder">held by {attachment.holder}</span>
        ) : null}
      </div>
      <div className="mt-1 text-hint text-dim" data-testid="run-attachment-observed">
        {observed === null
          ? 'The run has not recorded miLLM’s state yet.'
          : `Last observed by this run ${formatEt(observed, nowMs)} (${formatAge(observed, nowMs)}).`}
        {ended
          ? ' The run has ended, so this is the last state it saw, not miLLM’s state now; the strip above shows that.'
          : ''}
      </div>
      {attachment.stale === true && !ended ? (
        <div className="mt-1 text-hint text-amber" data-testid="run-attachment-stale" role="status">
          {STALE_ATTACHMENT_COPY}
        </div>
      ) : null}
    </Card>
  );
}
