/**
 * Which selector types can answer right now (FR-009.37's render-time courtesy).
 *
 * F3's `GET /api/v1/selectors` reports, **per call**, whether each type's requirement is
 * currently satisfied and names the unsatisfied service. An unavailable type is **shown,
 * named and still visible** - never hidden, never swapped for another, never a mute disabled
 * control, because *an option missing from a response is indistinguishable from one that is
 * broken* (ADR-013).
 *
 * **This is a courtesy and nothing else.** The authoritative availability answer is step 6 of
 * F6's validator (its FR-006.60), which refuses `<service>_not_configured` naming the service
 * **and the capability lost**. This panel performs no check of its own: a form that computed
 * availability would be F9 implementing another feature's requirement.
 *
 * **With every optional integration unset the form still submits a run** - a pack of external
 * parts and a rule-based selector need none (AC20). **A human-review selector is the one
 * genuinely sibling-dependent choice**, and it reports `midataworks_not_configured` rather
 * than appearing to work.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { Card } from '@/components/common/Card';
import { MonoId } from '@/components/common/MonoId';
import { unavailableSentence } from './unavailableReasons';
import { FailureNotice } from '@/components/common/FailureNotice';
import { fetchSelectors, type SelectorCapability } from '@/api/selectors';
import type { ApiFailure } from '@/api/errors';

export interface SelectorAvailabilityProps {
  readonly load?: typeof fetchSelectors;
}

export function SelectorAvailability({
  load = fetchSelectors,
}: SelectorAvailabilityProps = {}): ReactNode {
  const [rows, setRows] = useState<readonly SelectorCapability[] | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  useEffect(() => {
    let cancelled = false;
    void load().then((result) => {
      if (cancelled) return;
      if (result.ok) setRows(result.value);
      else setFailure(result.failure);
    });
    return () => {
      cancelled = true;
    };
  }, [load]);

  if (failure !== null) {
    return <FailureNotice failure={failure} surface="selector availability" className="mb-4" />;
  }

  return (
    <Card className="mb-4" data-testid="selector-availability">
      <div className="mb-1 text-heading">Selector types available now</div>
      <div className="mb-2 text-hint text-dim">
        A selector decides which generated answers become training examples. The pack sets which
        selector a run uses; this list shows which types can run right now. The server checks again
        when the run starts.
      </div>
      {rows === null ? (
        <div className="text-hint text-dim">Reading the selector types…</div>
      ) : (
        <ul className="stack-y-1 text-hint">
          {rows.map((row) => (
            <li
              key={row.name}
              className={row.available ? 'text-dim' : 'text-amber'}
              data-selector={row.name}
              data-available={row.available ? 'true' : 'false'}
            >
              <MonoId value={row.name} /> {row.summary}
              {row.available ? null : (
                <> — unavailable: {unavailableSentence(row.unavailable_reason)}</>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
