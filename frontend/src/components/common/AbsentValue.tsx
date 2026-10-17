/**
 * The only way this feature expresses a missing value (FR-009.6; ADR-018).
 *
 * **It accepts no value, so it cannot print a zero.** Eight reasons, eight sentences, in one
 * map - and the eight are the only absences this feature can express. If a ninth appears it
 * gets a reason; it does not get a blank. Mutation **M12** breaks this by writing
 * `floor ?? 0` at a call site, and the control only bites if the call sites come through
 * here.
 *
 * ADR-018's rule - *"a default that always fires is a fabricated value"* - applies with more
 * force on a chart than in an API, because a chart is read as a measurement. A retention
 * floor rendered as `0` says *"nothing is protected and that is the recorded policy"*, which
 * is the opposite of *"no floor was recorded"*.
 */
import type { ReactNode } from 'react';

export type AbsentReason =
  | 'no_floor_recorded'
  | 'no_test_at_this_step'
  | 'no_domain'
  | 'run_not_indexed'
  | 'not_recorded_for_kind'
  | 'lineage_stops_here'
  | 'target_not_loaded'
  | 'missing_required_key';

export const ABSENT_REASONS: readonly AbsentReason[] = [
  'no_floor_recorded',
  'no_test_at_this_step',
  'no_domain',
  'run_not_indexed',
  'not_recorded_for_kind',
  'lineage_stops_here',
  'target_not_loaded',
  'missing_required_key',
];

/**
 * One sentence per reason. Each says what is true, and where it matters, what to do next.
 *
 * `no_domain` reads **"No domain"** because R-02.29's own amendment says the tree *"shows
 * that as no domain rather than as a blank that reads like missing data"*, and the backend's
 * `models/domain_state.render_domain` renders the same state with the same two words.
 */
export const ABSENT_SENTENCES = {
  no_floor_recorded:
    'No floor recorded. This set was scored without one, so nothing was compared against it.',
  no_test_at_this_step:
    'No test at this step. Test and retention run on a cadence, so most steps have none; this is not a gap in the data.',
  no_domain: 'No domain. This artifact asserts that it is not domain-scoped.',
  run_not_indexed:
    'Run not indexed. The artifact is on disk and valid; no run row describes it. Run reindex to rebuild the index from the manifests.',
  not_recorded_for_kind:
    'Not recorded for this artifact kind. An adapter manifest has no field for it, so there is nothing to show and nothing is read across from its base.',
  lineage_stops_here:
    'Lineage stops here. The named ancestor does not resolve in this installation, so the chain above it is unknown — this is not a root.',
  target_not_loaded:
    'Not on this page. The named artifact exists but was outside this listing, so its row is not shown here.',
  missing_required_key:
    'Missing required key. The schema requires this key and the document on disk does not carry it, so the artifact does not conform.',
} as const satisfies Readonly<Record<AbsentReason, string>>;

export interface AbsentValueProps {
  readonly reason: AbsentReason;
  /** The specific thing, e.g. the parent name that did not resolve. Never a substitute value. */
  readonly detail?: string;
  /** `inline` for a table cell, `block` for a panel. */
  readonly variant?: 'inline' | 'block';
  readonly className?: string;
  readonly 'data-testid'?: string;
}

/** The short label, for a cell too narrow for the sentence. The sentence is the title. */
export const ABSENT_LABELS = {
  no_floor_recorded: 'no floor recorded',
  no_test_at_this_step: 'no test at this step',
  no_domain: 'no domain',
  run_not_indexed: 'run not indexed',
  not_recorded_for_kind: 'not recorded for this artifact kind',
  lineage_stops_here: 'lineage stops here',
  target_not_loaded: 'not on this page',
  missing_required_key: 'missing required key',
} as const satisfies Readonly<Record<AbsentReason, string>>;

export function AbsentValue({
  reason,
  detail,
  variant = 'inline',
  className = '',
  ...rest
}: AbsentValueProps): ReactNode {
  const sentence = ABSENT_SENTENCES[reason];
  const label = ABSENT_LABELS[reason];
  if (variant === 'block') {
    return (
      <div className={`text-hint text-dim ${className}`.trim()} data-absent={reason} {...rest}>
        {sentence}
        {detail === undefined ? null : <span className="ml-1 font-mono">{detail}</span>}
      </div>
    );
  }
  return (
    <span
      className={`text-dim ${className}`.trim()}
      data-absent={reason}
      title={sentence}
      {...rest}
    >
      {label}
      {detail === undefined ? null : <span className="ml-1 font-mono text-ident">{detail}</span>}
    </span>
  );
}
