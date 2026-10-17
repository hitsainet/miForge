/**
 * The `weights_state` pill - cyan for every value (ADR-012). See `weightsState.ts`.
 *
 * `null` means the state was not reported, which renders as words, not as a pill: a pill
 * reading "clean" for a state nobody read would be a fabricated value (ADR-018).
 */
import type { ReactNode } from 'react';
import { StatePill } from '@/components/common/Pill';
import { WEIGHTS_STATE_TONES } from './weightsState';

export interface WeightsStatePillProps {
  readonly state: string | null;
}

export function WeightsStatePill({ state }: WeightsStatePillProps): ReactNode {
  if (state === null) {
    return (
      <span className="text-hint text-dim" data-testid="weights-state-unreported">
        weights_state not reported
      </span>
    );
  }
  return <StatePill state={state} tones={WEIGHTS_STATE_TONES} data-testid="weights-state-pill" />;
}
