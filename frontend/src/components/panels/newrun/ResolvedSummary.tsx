/**
 * What the validator resolved the definition to (FR-009.35 rule 1, FR-009.36).
 *
 * `POST /api/v1/runs/validate` answers **200 either way** and carries a `resolved` block -
 * the pinned pack and part versions, the budget estimate beside free VRAM, the projected
 * footprint beside free disk, and every named-absent optional service. That is exactly what
 * an operator should see before committing, and it is **F6's figures**, not F8's asked
 * separately: *two numbers on one screen that could disagree is worse than one, and the one
 * that matters is the gate's.*
 *
 * **`may_be_more_to_fix` is rendered, not swallowed.** A step the validator could not run
 * means the failure list may be short, and a form that showed a tidy list of two while three
 * were waiting would teach an operator to trust a number that is not a count.
 *
 * It renders no verdict of its own and gates nothing: `ok: true` here is not permission,
 * because the create path runs the same validator again.
 */
import type { ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { Callout } from '@/components/common/Callout';
import { MonoId } from '@/components/common/MonoId';
import { AbsentValue } from '@/components/common/AbsentValue';
import { NumberWithSample, formatGb } from '@/components/common/NumberWithSample';
import type { ValidationOutcome } from '@/api/runs';

export interface ResolvedSummaryProps {
  readonly outcome: ValidationOutcome | null;
}

function bytes(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function record(value: unknown): Readonly<Record<string, unknown>> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : {};
}

export function ResolvedSummary({ outcome }: ResolvedSummaryProps): ReactNode {
  if (outcome === null) return null;

  const resolved = record(outcome.resolved);
  const budget = record(resolved['budget']);
  const footprint = record(outcome.footprint ?? resolved['footprint']);
  const estimate = bytes(budget['estimate_bytes']);
  const free = bytes(budget['free_bytes']);
  const projected = bytes(footprint['projected_bytes'] ?? footprint['bytes']);
  const freeDisk = bytes(footprint['free_bytes']);
  const services = Array.isArray(resolved['services_absent']) ? resolved['services_absent'] : [];

  return (
    <Card className="mb-4" data-testid="resolved-summary">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-heading">
          {outcome.ok ? 'This definition validates' : 'This definition has problems'}
        </div>
        <div className="text-hint text-dim">
          {outcome.steps_checked.length} check
          {outcome.steps_checked.length === 1 ? '' : 's'} ran:{' '}
          {outcome.steps_checked.join(', ') || 'none'}
        </div>
      </div>

      <div className="mb-2 text-hint text-dim">
        The same validator the create path runs, read-only. A green answer{' '}
        <strong>enables nothing</strong>: it runs again at submit, and a part can stop resolving or
        free VRAM change in between.
      </div>

      {outcome.may_be_more_to_fix ? (
        <Callout tone="warning" title="There may be more to fix" className="mb-3">
          Some checks could not run because an earlier part did not resolve. The list below is what
          could be checked, not necessarily everything that is wrong.
        </Callout>
      ) : null}

      <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-hint sm:grid-cols-2">
        <div>
          <dt className="inline text-dim">budget: </dt>
          <dd className="inline">
            {estimate === null ? (
              <AbsentValue
                reason="missing_required_key"
                detail="the validator reported no estimate"
              />
            ) : (
              <NumberWithSample
                value={formatGb(estimate)}
                unit="GB"
                sample={1}
                sampleLabel="estimate"
              />
            )}
          </dd>
        </div>
        <div>
          <dt className="inline text-dim">free VRAM: </dt>
          <dd className="inline">
            {free === null ? (
              /* Not zero and not "fits": no card means no limit was read. */
              <AbsentValue
                reason="missing_required_key"
                detail="no card was resolved, so no limit was read"
              />
            ) : (
              <NumberWithSample value={formatGb(free)} unit="GB" sample={1} sampleLabel="card" />
            )}
          </dd>
        </div>
        <div>
          <dt className="inline text-dim">projected on disk: </dt>
          <dd className="inline">
            {projected === null ? (
              <AbsentValue
                reason="missing_required_key"
                detail="the validator reported no footprint"
              />
            ) : (
              <NumberWithSample
                value={formatGb(projected)}
                unit="GB"
                sample={1}
                sampleLabel="projection"
              />
            )}
          </dd>
        </div>
        <div>
          <dt className="inline text-dim">free disk: </dt>
          <dd className="inline">
            {freeDisk === null ? (
              <AbsentValue reason="missing_required_key" detail="free disk was not reported" />
            ) : (
              <NumberWithSample
                value={formatGb(freeDisk)}
                unit="GB"
                sample={1}
                sampleLabel="volume"
              />
            )}
          </dd>
        </div>
      </dl>

      {services.length === 0 ? null : (
        <div className="mt-2 text-hint text-amber" data-testid="resolved-services-absent">
          Named absent, never substituted:{' '}
          {services.map((service) => (
            <MonoId key={String(service)} value={String(service)} className="mr-2" />
          ))}
        </div>
      )}

      {Object.keys(resolved).length === 0 ? (
        <div className="mt-2 text-hint text-dim">
          The validator returned no resolved block, so nothing is shown here rather than a summary
          assembled from the form&apos;s own values.
        </div>
      ) : (
        <details className="mt-3">
          <summary className="cursor-pointer text-hint text-dim">
            Everything the validator resolved
          </summary>
          <MonoId value={JSON.stringify(resolved, null, 2)} variant="block" className="mt-2" />
        </details>
      )}
    </Card>
  );
}
