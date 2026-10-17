/**
 * F8's estimate, shown and **not obeyed** (FR-009.36; M20, written first).
 *
 * ## It returns no boolean, takes no `onValid`, and cannot disable anything
 *
 * There is no prop on this component that could reach the submit button, and nothing
 * downstream of it reads a verdict to decide anything. F8's estimator is *the same code path*
 * as the gate, which is what makes the preview truthful - and still **not** the decision. The
 * authoritative refusal is the create path's.
 *
 * **M20 is the most plausible improvement and the most damaging**: disabling submit on this
 * screen's reading of the budget moves R-02.53's refusal into the browser, where it can
 * disagree with the gate. It was run against this component before the preview was wired into
 * the form, which is the only order in which that control means anything.
 *
 * ## `unknown` is not `fits`
 *
 * On a machine with no card `free_bytes` is `null` and the verdict is `unknown` - *"I could
 * not read the limit, so it fits"* is the pass-through failure wearing the estimator's name,
 * and it would be the most misleading number on the screen. The verdict is rendered as it
 * arrives and `unknown` is never softened into a pass.
 *
 * Every figure goes through `NumberWithSample`, so none of them is a bare number.
 */
import { useCallback, useState, type ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { AbsentValue } from '@/components/common/AbsentValue';
import { NumberWithSample, formatGb } from '@/components/common/NumberWithSample';
import { FailureNotice } from '@/components/common/FailureNotice';
import { estimateBudget, type BudgetEstimate, type BudgetRequest } from '@/api/budget';
import type { ApiFailure } from '@/api/errors';

export interface BudgetPreviewProps {
  /** `null` while the operator has not entered enough for a request to mean anything. */
  readonly request: BudgetRequest | null;
}

export function BudgetPreview({ request }: BudgetPreviewProps): ReactNode {
  const [estimate, setEstimate] = useState<BudgetEstimate | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [pending, setPending] = useState(false);

  const ask = useCallback(async () => {
    if (request === null) return;
    setPending(true);
    const result = await estimateBudget(request);
    if (result.ok) {
      setEstimate(result.value);
      setFailure(null);
    } else {
      setEstimate(null);
      setFailure(result.failure);
    }
    setPending(false);
  }, [request]);

  const terms = estimate?.terms ?? [];
  const estimateBytes = estimate?.estimate_bytes;
  const freeBytes = estimate?.free_bytes ?? null;

  return (
    <Card className="mb-4" data-testid="budget-preview">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-heading">Budget estimate</div>
        <Button onClick={() => void ask()} disabled={request === null || pending}>
          {pending ? 'Estimating…' : 'Estimate the footprint'}
        </Button>
      </div>

      <div className="mb-2 text-hint text-dim">
        F8&apos;s estimator, which is the same code path the gate runs. It is a preview, not a
        permission: nothing on this form is enabled or disabled by what it says, and the run is
        refused or admitted by the create path.
      </div>

      {request === null ? (
        <div className="text-hint text-dim">
          Enter a run type and a batch configuration to estimate. The estimate needs only a model
          source and a batch configuration, which is why it is useful before the rest of this form
          is.
        </div>
      ) : null}

      {failure === null ? null : <FailureNotice failure={failure} surface="the budget estimate" />}

      {estimate === null ? null : (
        <>
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
            <div>
              <div className="text-hint text-dim">Estimated</div>
              {estimateBytes === undefined ? (
                <AbsentValue reason="missing_required_key" detail="estimate_bytes" />
              ) : (
                <NumberWithSample
                  value={formatGb(estimateBytes)}
                  unit="GB"
                  sample={terms.length}
                  sampleLabel="terms"
                  data-testid="budget-estimate"
                />
              )}
            </div>
            <div>
              <div className="text-hint text-dim">Free on the card</div>
              {freeBytes === null ? (
                /* Not zero and not "fits". No card means no limit was read. */
                <AbsentValue
                  reason="missing_required_key"
                  detail="free_bytes — no CUDA device was resolved, so no limit was read"
                  data-testid="budget-no-free"
                />
              ) : (
                <NumberWithSample
                  value={formatGb(freeBytes)}
                  unit="GB"
                  sample={1}
                  sampleLabel="card"
                  {...(estimate.gpu_name == null ? {} : { provider: estimate.gpu_name })}
                  data-testid="budget-free"
                />
              )}
            </div>
            <div>
              <div className="text-hint text-dim">Verdict</div>
              <span data-testid="budget-verdict">{estimate.verdict ?? 'not reported'}</span>
            </div>
          </div>

          {terms.length === 0 ? (
            <div className="mt-3 text-hint text-dim">
              The estimator returned no per-term breakdown, so the total cannot be attributed here.
              It is shown as the total it is rather than apportioned by guesswork.
            </div>
          ) : (
            <ul className="mt-3 space-y-1 text-hint" data-testid="budget-terms">
              {terms.map((term) => (
                <li key={term.name}>
                  <span className="text-dim">{term.name}: </span>
                  <NumberWithSample
                    value={formatGb(term.bytes)}
                    unit="GB"
                    sample={1}
                    sampleLabel="term"
                  />
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </Card>
  );
}
