/**
 * A pack version's validity (F7 task 12.5; FR-007.8, FR-007.9).
 *
 * Four states plus a default branch. The amber one **names the slot** - `Incomplete · no test
 * set` - because AC10's whole content is that an incomplete pack says which slot is missing,
 * and `Incomplete` on its own is the generic failure that criterion exists to forbid.
 *
 * The service state is separate from the invalid state on purpose: the backend orders its own
 * refusals so that *"a pack genuinely missing its test set"* is not reported as
 * `midataworks_not_configured`, and a pill that painted both red would undo that ordering on
 * the screen. An unconfigured sibling is not a fault in the pack.
 *
 * **A default branch, because the mockup's pill threw.** `map[state]` with no fallback, found
 * in its own 2026-09-19 review.
 */
import type { ReactNode } from 'react';
import { Pill, type PillTone } from './Pill';
import { SLOT_LABELS, type PackValidity, type SlotKind } from '@/types/packs';

const STATES = {
  valid: { tone: 'green', label: 'Valid' },
  incomplete: { tone: 'amber', label: 'Incomplete' },
  unresolved: { tone: 'red', label: 'Unresolved' },
  service_unavailable: { tone: 'neutral', label: 'Needs a service' },
} as const satisfies Readonly<Record<string, { readonly tone: PillTone; readonly label: string }>>;

export interface ValidityPillProps {
  readonly validity: PackValidity;
  readonly 'data-testid'?: string;
}

/** `Incomplete · no test set`, listing **every** missing slot rather than the first. */
export function validityLabel(validity: PackValidity): string {
  const known = STATES[validity.state as keyof typeof STATES];
  const base = known?.label ?? `${validity.state} (unrecognised)`;
  if (validity.missing_slots.length > 0) {
    const named = validity.missing_slots
      .map((slot) => SLOT_LABELS[slot as SlotKind] ?? slot)
      .map((label) => `no ${label.toLowerCase()}`)
      .join(', ');
    return `${base} · ${named}`;
  }
  if (validity.service_unavailable.length > 0) {
    const services = [...new Set(validity.service_unavailable.map((row) => row.service))];
    return `${base} · ${services.join(', ')} not configured`;
  }
  if (validity.unresolved.length > 0) {
    const slots = validity.unresolved
      .map((row) => SLOT_LABELS[row.slot as SlotKind] ?? row.slot)
      .join(', ');
    return `${base} · ${slots} did not resolve`;
  }
  if (validity.retention_floor_missing) return `${base} · no retention floor`;
  return base;
}

export function ValidityPill({ validity, ...rest }: ValidityPillProps): ReactNode {
  const known = STATES[validity.state as keyof typeof STATES];
  const tone: PillTone = known?.tone ?? 'neutral';
  return (
    <Pill tone={tone} title={validityLabel(validity)} {...rest}>
      {validityLabel(validity)}
    </Pill>
  );
}
