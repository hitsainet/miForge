/**
 * An optional service that is not configured, rendered **by name** (foundation task 7.4;
 * ADR-010, ADR-013, ADR-018, AC20).
 *
 * ADR-010: an unset base URL means *absent and normal*, and a refusal is named
 * `<service>_not_configured`. ADR-013 and ADR-018: *name what failed*, and **never
 * substitute a different behaviour for a missing service**.
 *
 * **This component is the frontend half of that rule, and it is the half that is usually
 * missing.** A client that decodes the envelope correctly and then renders an empty
 * panel has obeyed the letter and broken the promise: the operator sees a blank where a
 * capability should be and cannot tell an absent sibling from a bug. So:
 *
 * - the **service** is named, from the refusal's code, not from a message string;
 * - the **capabilities lost** are listed, from `details.capabilities_unavailable`;
 * - the surface **stays visible** rather than disappearing (F9 FTID 3.5a: *"A type whose
 *   service is absent renders unavailable by name and stays visible"*);
 * - nothing is substituted, defaulted or zeroed.
 *
 * **Not an error.** `role="status"` and no red: a standalone miForge with no sibling
 * configured is the *supported* configuration (AC20), and painting that red teaches the
 * operator to ignore red.
 *
 * **Distinct from F9's `AbsentValue`.** That renders a *value* that is not recorded in an
 * artifact - `no_floor_recorded`, `not_recorded_for_kind`. This renders a *capability*
 * that is unreachable because a service is unconfigured. Collapsing them would say "no
 * data" where the truth is "no integration", and the remedy for the second is a base URL.
 */

import type { ReactNode } from 'react';
import { PlugZap } from 'lucide-react';
import type { NotConfiguredFailure } from '@/api/errors';

export interface UnavailableProps {
  readonly failure: NotConfiguredFailure;
  /** What the operator was looking at, so the sentence reads as a consequence. */
  readonly surface?: string;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function Unavailable({
  failure,
  surface,
  className = '',
  ...rest
}: UnavailableProps): ReactNode {
  const capabilities = failure.capabilitiesUnavailable;
  return (
    <div
      role="status"
      className={`rounded-card border border-line bg-panel p-card text-body ${className}`.trim()}
      data-service={failure.service}
      data-code={failure.code}
      {...rest}
    >
      <div className="flex items-start gap-3">
        <PlugZap size={18} className="mt-0.5 shrink-0 text-dim" aria-hidden="true" />
        <div className="min-w-0">
          <div className="font-medium">
            {failure.service} is not configured
            {surface === undefined ? '' : `, so ${surface} is unavailable`}.
          </div>
          {/* The backend's own message. It says what to do next (ADR-021) - typically
              which environment variable to set - and rewording it here would put this
              component in the business of guessing another module's remedy. */}
          <div className="mt-1 text-hint text-dim">{failure.message}</div>
          {capabilities.length > 0 ? (
            <ul className="mt-2 list-disc pl-5 text-hint text-dim">
              {capabilities.map((capability) => (
                <li key={capability}>{capability}</li>
              ))}
            </ul>
          ) : (
            // Never invented. An empty list says so rather than listing a plausible
            // capability, which would be the fabricated value ADR-018 forbids.
            <div className="mt-2 text-hint text-dim">
              The response did not state which capabilities this affects.
            </div>
          )}
          <div className="mt-2 text-hint text-dim">
            Everything that does not need {failure.service} continues to work.
          </div>
        </div>
      </div>
    </div>
  );
}
