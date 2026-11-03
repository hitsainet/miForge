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
import {
  estimateBudget,
  type BudgetCard,
  type BudgetEstimate,
  type BudgetRequest,
  type BudgetSource,
} from '@/api/budget';
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
          {pending ? 'Estimating…' : 'Estimate GPU memory'}
        </Button>
      </div>

      <div className="mb-2 text-hint text-dim">
        An estimate of the GPU memory this run needs, from the same calculation that runs when the
        run starts. It does not block or allow anything on this form; the start does that.
      </div>

      {request === null ? (
        <div className="text-hint text-dim">
          Choose a model source and a run type, and enter a learning rate and batch size to
          estimate. The rest of the form can still be empty.
        </div>
      ) : null}

      {failure === null ? null : <FailureNotice failure={failure} surface="the budget estimate" />}

      {estimate === null ? null : (
        <>
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
            <div>
              <div className="text-hint text-dim">Estimated</div>
              {estimateBytes === undefined ? (
                <AbsentValue
                  reason="missing_required_key"
                  detail="the estimator returned no total"
                />
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
                  detail="no GPU was resolved, so free memory was not read"
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

          <SourceLine source={estimate.source ?? null} />
          <CardList cards={estimate.cards ?? []} />

          {terms.length === 0 ? (
            <div className="mt-3 text-hint text-dim">
              The estimator returned a total with no breakdown, so only the total is shown.
            </div>
          ) : (
            <ul className="mt-3 stack-y-1 text-hint" data-testid="budget-terms">
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

/** What the source kinds are called on screen. A kind not listed is shown as it arrives. */
const SOURCE_KIND: Readonly<Record<string, string>> = {
  directory: 'the model directory',
  checkpoint: 'the checkpoint',
  hub: 'the Hub repository',
  config: 'the architecture config',
  attach: 'the model miLLM is serving',
};

/** Where the server read the model's size, so the figure is never a size from nowhere. */
function SourceLine({ source }: { readonly source: BudgetSource | null }): ReactNode {
  if (source === null) return null;
  const where = SOURCE_KIND[source.kind] ?? source.kind;
  const named = source.path ?? source.repo;
  const count = source.param_count;
  return (
    <div className="mt-3 text-hint text-dim" data-testid="budget-source">
      Size read from {where}
      {named === undefined ? '' : ` (${named})`}
      {typeof count === 'number'
        ? `: ${count.toLocaleString('en-US')} parameters.`
        : ': its architecture config.'}
    </div>
  );
}

/**
 * Every card the run could be placed on, each with its own verdict. Rendered for one card
 * or several alike: nothing here assumes a count.
 */
function CardList({ cards }: { readonly cards: readonly BudgetCard[] }): ReactNode {
  if (cards.length === 0) return null;
  return (
    <ul className="mt-3 stack-y-1 text-hint" data-testid="budget-cards">
      {cards.map((card) => (
        <li key={card.uuid ?? card.name ?? ''} data-testid={`budget-card-${card.uuid ?? ''}`}>
          <span className="text-dim">{card.name ?? card.uuid ?? 'GPU'}: </span>
          {card.free_bytes === null ? (
            <span>free memory not read</span>
          ) : (
            <NumberWithSample
              value={formatGb(card.free_bytes)}
              unit="GB free"
              sample={1}
              sampleLabel="card"
            />
          )}
          <span className="text-dim"> - </span>
          <span>{card.verdict}</span>
        </li>
      ))}
    </ul>
  );
}
