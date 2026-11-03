/**
 * The seven steps, with done / current / todo markers (F7 task 12.6; FR-007.17).
 *
 * The states are the **server's** (`step_states`, keyed by the step number as a string), not
 * computed from "is this number less than the current one". That matters because the flow does
 * not discard later steps: an author can complete step 5, go back to step 3, and step 5 stays
 * `done` - which a less-than comparison would render as `todo` and make look lost.
 *
 * A skippable step is marked, because a step an author may legitimately leave empty should not
 * read as unfinished work.
 */
import type { ReactNode } from 'react';
import { Check, Circle, Dot } from 'lucide-react';
import { DRAFT_STEPS, type DraftRecord } from '@/types/packs';

export const STEP_TITLES: Readonly<Record<number, string>> = {
  1: 'Describe the domain',
  2: 'Choose the prompts',
  3: 'Choose the answers',
  4: 'Choose how judgment works',
  5: 'Choose what improvement means',
  6: 'Choose what must not regress',
  7: 'Review and save',
};

export interface StepListProps {
  readonly draft: DraftRecord;
  readonly onGo: (step: number) => void;
}

const MARKERS = {
  done: { icon: Check, tone: 'text-green' },
  current: { icon: Circle, tone: 'text-text' },
  todo: { icon: Dot, tone: 'text-dim' },
} as const;

export function StepList({ draft, onGo }: StepListProps): ReactNode {
  return (
    <nav aria-label="Onboarding steps" data-testid="step-list">
      <ol className="stack-y-1">
        {DRAFT_STEPS.map((step) => {
          const state = draft.step_states[String(step.number)] ?? 'todo';
          const marker = MARKERS[state as keyof typeof MARKERS] ?? MARKERS.todo;
          const Icon = marker.icon;
          const current = draft.current_step === step.number;
          return (
            <li key={step.number}>
              <button
                type="button"
                onClick={() => onGo(step.number)}
                aria-current={current ? 'step' : undefined}
                className={`flex w-full items-center gap-2 rounded-control px-3 py-2 text-left text-body ${
                  current ? 'bg-card' : 'hover:bg-card'
                }`}
                data-step={step.number}
                data-step-state={state}
              >
                <Icon size={15} className={marker.tone} aria-hidden="true" />
                <span className="min-w-0 truncate">
                  {step.number}. {STEP_TITLES[step.number] ?? step.name}
                </span>
                {step.skippable ? (
                  <span className="ml-auto shrink-0 text-hint text-dim">optional</span>
                ) : null}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
