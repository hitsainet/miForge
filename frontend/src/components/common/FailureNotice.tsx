/**
 * A decoded failure, rendered with what to do next (FR-009.5; ADR-013, ADR-021).
 *
 * The backend's envelope is `{"error": {code, message, details}}` and `api/errors.ts` decodes
 * it into four typed failures. This renders each of them **by kind**, and the one that
 * matters is `service_not_configured`: it goes to `Unavailable`, which names the service and
 * the capabilities lost rather than reporting a fault. An unconfigured optional sibling is
 * absent-and-normal (ADR-010), and rendering it as an error teaches an operator to distrust
 * the screen.
 *
 * **There is no generic branch over the code.** An unrecognised refusal renders **the code
 * itself**, because the code is the only diagnosable thing in the response and a friendly
 * sentence in its place discards it. The backend's `message` already says what to do next.
 */
import type { ReactNode } from 'react';
import type { ApiFailure } from '@/api/errors';
import { Callout } from './Callout';
import { MonoId } from './MonoId';
import { Unavailable } from './Unavailable';

export interface FailureNoticeProps {
  readonly failure: ApiFailure;
  /** What was being read, so the notice degrades **one named region** (FR-009.9, task 10.5). */
  readonly surface: string;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function FailureNotice({
  failure,
  surface,
  className = '',
  ...rest
}: FailureNoticeProps): ReactNode {
  if (failure.kind === 'service_not_configured') {
    return <Unavailable failure={failure} surface={surface} className={className} {...rest} />;
  }
  if (failure.kind === 'transport') {
    return (
      <Callout tone="warning" title={`${surface} is unavailable`} className={className} {...rest}>
        No response from miForge ({failure.cause}). The screen is showing the last values it read.
        Check that the backend is running, then reload.
      </Callout>
    );
  }
  if (failure.kind === 'malformed') {
    return (
      <Callout
        tone="error"
        title={`${surface} returned something unreadable`}
        className={className}
        {...rest}
      >
        HTTP {failure.status} with a body that is not an error envelope ({failure.reason}). The body
        is shown below so it can be reported rather than guessed at.
        <MonoId value={failure.body} variant="block" className="mt-2 block" />
      </Callout>
    );
  }
  return (
    <Callout tone="error" title={`${surface} was refused`} className={className} {...rest}>
      <div className="flex flex-wrap items-center gap-2">
        {/* The code, not a paraphrase of it. */}
        <MonoId value={failure.code} data-testid="failure-code" />
        <span>{failure.message}</span>
      </div>
      <DetailList details={failure.details} />
    </Callout>
  );
}

/** `details` **rendered, not summarised** (FR-009.35 rule 5). */
export function DetailList({
  details,
}: {
  readonly details: Readonly<Record<string, unknown>>;
}): ReactNode {
  const entries = Object.entries(details);
  if (entries.length === 0) return null;
  return (
    <ul className="mt-2 space-y-0.5 text-hint" data-testid="failure-details">
      {entries.map(([key, value]) => (
        <li key={key}>
          <span className="text-dim">{key}:</span>{' '}
          <span className="font-mono text-ident">{renderDetail(value)}</span>
        </li>
      ))}
    </ul>
  );
}

export function renderDetail(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === null) return 'null';
  if (Array.isArray(value)) return value.map((item) => renderDetail(item)).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}
