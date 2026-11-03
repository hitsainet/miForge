/**
 * "Continue from the last save" for an ATTACHED run that ended  (D52; F11 FTASKS 18.0).
 *
 * Shown only when the record says the run acquired by `attach` and is `failed` or
 * `completed`. The backend does the deciding: it verifies the checkpoint, refuses unless the
 * model miLLM serves IS that checkpoint, restores the optimizer state and numbers steps from
 * the checkpoint's step + 1. This control submits and reports - a refusal's code, its message
 * and, for a submission refused by the validator, **each** named failure (FR-006.59), because
 * the one an operator needs here (`resume_served_model_mismatch`) carries the operator action.
 */
import { useState, type ReactNode } from 'react';
import { submitContinue, type ContinueSubmitted } from '@/api/checkpoints';
import type { ApiFailure } from '@/api/errors';
import { Button } from '@/components/common/Button';
import { Callout } from '@/components/common/Callout';
import { FailureNotice, renderDetail } from '@/components/common/FailureNotice';
import { MonoId } from '@/components/common/MonoId';
import type { RunRecord } from '@/types/run';

const CONTINUABLE = new Set(['failed', 'completed']);

export function isContinuable(record: RunRecord): boolean {
  return record.acquisition_mode === 'attach' && CONTINUABLE.has(record.status);
}

interface NestedFailure {
  readonly code: string;
  readonly message: string;
  readonly details?: Readonly<Record<string, unknown>>;
}

function nestedFailures(failure: ApiFailure): readonly NestedFailure[] {
  if (failure.kind !== 'refusal') return [];
  const list = failure.details['failures'];
  return Array.isArray(list) ? (list as NestedFailure[]) : [];
}

export function ContinueRun({
  record,
  onSubmitted,
}: {
  readonly record: RunRecord;
  readonly onSubmitted: () => void;
}): ReactNode {
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [submitted, setSubmitted] = useState<ContinueSubmitted | null>(null);

  if (!isContinuable(record) && submitted === null) return null;

  const submit = async (): Promise<void> => {
    setBusy(true);
    setFailure(null);
    const result = await submitContinue(record.run_id);
    setBusy(false);
    if (result.ok) {
      setSubmitted(result.value);
      onSubmitted();
    } else {
      setFailure(result.failure);
    }
  };

  return (
    <div className="mt-3" data-testid="continue-run">
      {submitted === null ? (
        <Button variant="primary" disabled={busy} onClick={() => void submit()}>
          Continue from the last save
        </Button>
      ) : (
        <Callout tone="info" title="Continue submitted" data-testid="continue-submitted">
          Resuming from step {submitted.plan.step} (<MonoId value={submitted.plan.name} />) with
          optimizer state {submitted.plan.optimizer.state}; the first new step is{' '}
          {submitted.plan.step + 1}.
          {submitted.resume.superseded_steps === null
            ? null
            : ` Steps ${submitted.resume.superseded_steps.first}–${submitted.resume.superseded_steps.last} of the previous attempt are superseded.`}
        </Callout>
      )}
      {failure === null ? null : (
        <>
          <FailureNotice failure={failure} surface="The continue" className="mt-2" />
          {nestedFailures(failure).map((nested) => (
            <Callout
              key={nested.code}
              tone="error"
              title={nested.code}
              className="mt-2"
              data-testid="continue-failure"
            >
              {nested.message}
              {nested.details?.['operator_action'] === undefined ? null : (
                <div className="mt-1 text-hint">
                  Operator action:{' '}
                  <MonoId value={renderDetail(nested.details['operator_action'])} />
                </div>
              )}
            </Callout>
          ))}
        </>
      )}
    </div>
  );
}
