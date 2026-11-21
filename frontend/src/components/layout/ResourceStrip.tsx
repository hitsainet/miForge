/**
 * The resource strip (foundation task 7.3; ADR-012, FR-009.2, handoff 3).
 *
 * > Resource strip is the same as the sibling apps, plus one chip on the right.
 *
 * Host readings on the left - CPU, RAM, and per card its utilisation, VRAM and temperature,
 * from `GET /api/v1/host/resources` (see `hostReadings.ts`) - and a generic `trailing` slot
 * on the right, where the shell puts the attachment chip. **This component has no attachment
 * chip and no sibling reference**: ADR-022 fences 5.10 off the core path, and a chip in the
 * shell is on every screen.
 *
 * **Every number names its scale and sample** (ADR-012). `ResourceReading` makes that
 * structural - `unit` is required and `of` carries the total - and a `null` value renders as
 * its `reason` ("no GPU visible"), never as `0`. See `layout/types.ts`.
 */

import type { ReactNode } from 'react';
import type { ResourceReading } from './types';

export interface ResourceStripProps {
  readonly readings: readonly ResourceReading[];
  /** The right-hand slot. The shell's attachment chip goes here. */
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
    // Not `0`, not an em dash. A GPU at 0 % and a GPU that could not be read are different
    // facts; the reason says which one this is.
    return reading.reason ?? 'unavailable';
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
      className="flex flex-wrap items-center justify-end gap-x-5 gap-y-1 border-b border-line bg-panel px-8 py-3"
      data-testid="shell-resource-strip"
      aria-label="Host resources"
    >
      {readings.map((reading) => (
        <span
          key={reading.id}
          className="inline-flex items-center gap-1.5 text-hint text-dim"
          data-testid={`resource-${reading.id}`}
          title={reading.title}
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
