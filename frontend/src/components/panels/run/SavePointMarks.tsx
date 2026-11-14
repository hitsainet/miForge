/**
 * Save marks, from the saves (FR-009.11).
 *
 * **This component takes `steps: number[]` and cannot see run configuration**, so it cannot
 * compute a cadence. That is the whole design: `step % save_every === 0` is mutation **M11**
 * and would require giving this component a `saveEvery` prop it does not have.
 *
 * Two failures the cadence version produces, and the second is the dangerous one:
 *
 * - an **out-of-cadence** save (F5 serves `POST .../checkpoints` to request one at the next
 *   step boundary) would be invisible;
 * - a **cadence step whose save failed** would be marked as though it had succeeded.
 *
 * The marks are dashed `dim` verticals. Never accent: ADR-012 gives the accent to the loss
 * curve, and a chart with orange save marks reads as though the saves were the measurement.
 */
import type { ReactNode } from 'react';
import { ReferenceLine } from 'recharts';

export interface SavePointMarksProps {
  /** The steps at which a checkpoint **exists**. From F5's listing; never a cadence. */
  readonly steps: readonly number[];
}

/**
 * Recharts composes reference lines as children of the chart, so this returns a fragment
 * rather than an element with its own box. It is a component (not a plain function returning
 * an array) so that its props are the guard described above.
 */
export function SavePointMarks({ steps }: SavePointMarksProps): ReactNode {
  return (
    <>
      {steps.map((step) => (
        <ReferenceLine
          key={step}
          x={step}
          stroke="var(--mf-dim)"
          strokeDasharray="4 4"
          ifOverflow="extendDomain"
        />
      ))}
    </>
  );
}
