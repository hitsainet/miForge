/**
 * The resource strip (foundation task 7.3; ADR-012, FR-009.2, handoff 3).
 *
 * > Resource strip is the same as the sibling apps, plus one chip on the right.
 *
 * **What the right-hand chip is, after the standalone reframe.** The mockup's chip shows
 * BRD-01 attachment state - `Attached · serving`, `Attached · quiesced`, `Detached`,
 * `Stale` - and the handoff calls it *"the only place the user sees BRD-01's status
 * endpoint."* The handoff's own 6c withdraws that for the default product:
 *
 * > In the default path **there is nothing attached and no such status** - the strip
 * > should show what model miForge holds and where it came from. The attachment chip
 * > belongs to the hot-update mode only.
 *
 * So this component has **no attachment chip and no miLLM reference**. ADR-022 fences
 * 5.10 off the core path, and a chip in the shell is on every screen - it would be the
 * single most load-bearing place to reintroduce the dependency the reframe removed. The
 * right-hand slot is a generic `trailing` node, so F11 can put its chip there under the
 * hot-update mode without this file learning what miLLM is.
 *
 * **Every number names its scale and sample** (ADR-012). `ResourceReading` makes that
 * structural - `unit` is required and `of` carries the total - and a `null` value renders
 * **"not reported"**, never `0`. See `layout/types.ts`.
 */

import type { ReactNode } from 'react';
import type { ResourceReading } from './types';

export interface ResourceStripProps {
  readonly readings: readonly ResourceReading[];
  /** The right-hand slot. F11's attachment chip goes here; the core path leaves it empty. */
  readonly trailing?: ReactNode;
}

/**
 * Render one reading's text.
 *
 * Exported because this is the function the "every number names its scale" rule lives in,
 * and a test asserting the rule should call it directly rather than through a DOM query.
 */
export function formatReading(reading: ResourceReading): string {
  if (reading.value === null) {
    // Not `0`, not an em dash. A GPU at 0 % and a GPU that could not be read are
    // different facts; ADR-018's rule about a default that always firing being a
    // fabricated value applies with more force to a number than to a string.
    return 'not reported';
  }
  const precision = reading.precision ?? (reading.unit === '%' ? 0 : 1);
  const value = reading.value.toFixed(precision);
  const separator = reading.unit === '%' || reading.unit === '°C' ? '' : ' ';
  if (reading.of === undefined) {
    return `${value}${separator}${reading.unit}`;
  }
  return `${value}/${reading.of.toFixed(precision)}${separator}${reading.unit}`;
}

export function ResourceStrip({ readings, trailing }: ResourceStripProps): ReactNode {
  return (
    <div
      className="flex items-center justify-end gap-5 border-b border-line bg-panel px-8 py-3"
      data-testid="shell-resource-strip"
      aria-label="Host resources"
    >
      {readings.map((reading) => (
        <span
          key={reading.id}
          className="inline-flex items-center gap-1.5 text-hint text-dim"
          data-testid={`resource-${reading.id}`}
        >
          <reading.icon size={13} aria-hidden="true" />
          {reading.label}:{' '}
          <b className={reading.value === null ? 'font-normal text-dim' : 'text-text'}>
            {formatReading(reading)}
          </b>
        </span>
      ))}
      {trailing}
    </div>
  );
}
