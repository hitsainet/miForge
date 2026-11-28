/**
 * The only way this feature renders a number (FR-009.6; ADR-012; ADR-018).
 *
 * **`sample` and `sampleLabel` are required props, so there is no prop combination that
 * renders a bare number.** That is the point: ADR-012's *"every number names its scale and
 * its sample"* is a rule a reviewer has to remember, and a required prop is a rule the
 * compiler remembers. `Test score 0.84 (320 prompts, in-process)`, never `0.84`.
 *
 * `atStep` exists because test and retention run on a cadence and loss does not, so the
 * latest test score is frequently **not** from the run's current step (FR-009.14). A tile
 * that showed `0.84` beside `step 590` when the score was taken at 500 is a number quoted
 * against the wrong sample.
 */
import type { ReactNode } from 'react';

export interface NumberWithSampleProps {
  readonly value: number | string;
  readonly unit?: string;
  /** The count the value was taken over. Required - see the module docstring. */
  readonly sample: number;
  /** What the count counts: `prompts`, `steps`, `candidates`. Required. */
  readonly sampleLabel: string;
  /** Who produced it (ADR-009: every score records its provider). */
  readonly provider?: string;
  /** The step the value is from, when it differs from the screen's current step. */
  readonly atStep?: number;
  /** A second figure that qualifies the first, e.g. a retention floor. */
  readonly qualifier?: string;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function NumberWithSample({
  value,
  unit,
  sample,
  sampleLabel,
  provider,
  atStep,
  qualifier,
  className = '',
  ...rest
}: NumberWithSampleProps): ReactNode {
  const parts: string[] = [];
  if (qualifier !== undefined) parts.push(qualifier);
  parts.push(`${formatCount(sample)} ${sampleLabel}`);
  if (atStep !== undefined) parts.push(`at step ${formatCount(atStep)}`);

  return (
    <span className={className} {...rest}>
      <span className="font-semibold tabular-nums">{value}</span>
      {unit === undefined ? null : <span className="ml-0.5 text-dim">{unit}</span>}
      <span className="ml-1.5 text-hint text-dim">({parts.join(', ')})</span>
      {/* The provider gets its own line: a provider id names a model and a 40-character
          revision, one unbreakable token that spilled out of a quarter-width tile when it sat
          inside the parenthetical. Shortened for reading, whole in the tooltip, and allowed to
          break anywhere if a short form is still wider than its box. */}
      {provider === undefined ? null : (
        <span
          className="mt-0.5 block font-mono text-hint text-dim wrap-anywhere"
          title={provider}
          data-testid="number-provider"
        >
          via {shortProvider(provider)}
        </span>
      )}
    </span>
  );
}

/**
 * `in_process:hf:LiquidAI/LFM2.5-1.2B-Instruct@0f604ada…` reads as
 * `in-process · LiquidAI/LFM2.5-1.2B-Instruct @0f604ad`. Anything not in that shape is shown
 * as written, because a guessed reading of an unfamiliar id is worse than the id.
 */
export function shortProvider(provider: string): string {
  const match = /^([a-z_]+):(?:hf:)?([^@\s/][^@\s]*?)(?:@([0-9a-f]{12,}))?$/.exec(provider);
  if (match === null) return provider;
  const [, kind, model, revision] = match;
  const where = (kind ?? '').replace(/_/g, '-');
  return revision === undefined
    ? `${where} · ${model}`
    : `${where} · ${model} @${revision.slice(0, 7)}`;
}

/** Thousands separators, so `50000 steps` does not read as `5000`. */
export function formatCount(value: number): string {
  return value.toLocaleString('en-GB');
}

/**
 * Bytes as GB with one decimal, for the budget preview.
 *
 * GB, decimal, because that is what `nvidia-smi` and every VRAM figure in this project's
 * documents use (`11.6 GB estimated · 21.3 GB free`). Named here rather than inline so the
 * two figures on that line cannot be computed two ways.
 */
export function formatGb(bytes: number): string {
  return (bytes / 1_000_000_000).toFixed(1);
}
