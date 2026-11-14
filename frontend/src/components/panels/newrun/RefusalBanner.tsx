/**
 * Every refusal, on the field its own `details` names (FR-009.35; M21).
 *
 * ## It takes a *list*, not one refusal
 *
 * F6 collects failures across every part of the definition and returns them **together**,
 * explicitly *"because the one consumer of this path is a form (F9), and a validator that
 * stops at the first failure makes an operator resubmit once per mistake."* A component with
 * a single-refusal prop cannot express that response, so the prop is `refusals:
 * DecodedRefusal[]` and a banner showing only the first throws away what F6 built for this
 * screen.
 *
 * ## No generic branch
 *
 * An unrecognised code renders **the code itself**. The code is the only diagnosable thing in
 * the response, and a friendly sentence in its place discards it. The backend's `message`
 * already says what to do next (ADR-021).
 *
 * ## `details` are rendered, not summarised
 *
 * `budget_exceeded` without both figures is a refusal an operator cannot act on - F8's own
 * rule - and `pack_incomplete` without the slot names is the generic failure AC10 exists to
 * forbid. So every key of `details` is printed.
 *
 * ## A 422 keeps the form
 *
 * This component renders **beside** the fields and clears nothing. The form's values survive,
 * the named fields are marked, and the operator fixes one thing. Clearing the form on a 422,
 * or rendering the refusal as a page-level error instead of against the fields, is mutation
 * **M21**: *a form that clears itself on a refusal makes the operator re-enter nine things to
 * learn the tenth was wrong.*
 */
import type { ReactNode } from 'react';
import { Callout } from '@/components/common/Callout';
import { MonoId } from '@/components/common/MonoId';
import { DetailList } from '@/components/common/FailureNotice';
import type { ApiFailure, RefusalFailure } from '@/api/errors';

/** Which field a refusal belongs against. One of the form's own field ids. */
export type FormField =
  | 'model-source'
  | 'run-type'
  | 'pack'
  | 'signal-source'
  | 'optimizer'
  | 'cadences'
  | 'retention-floor'
  | 'stop-conditions'
  | 'save-cadence'
  | 'adapter'
  | 'budget'
  | 'form';

export interface DecodedRefusal {
  readonly code: string;
  readonly message: string;
  readonly details: Readonly<Record<string, unknown>>;
}

/**
 * Code to field, explicit and per code, from the owners' own `details` shapes (FTID 3.5a).
 *
 * Nothing is inferred from a code's prefix: `budget_input_missing` belongs against the model
 * source rather than the budget, because the input that is missing is the model's parameter
 * count, and a prefix rule would have put it on the preview where the operator can do nothing
 * about it.
 */
export const REFUSAL_FIELDS: Readonly<Record<string, FormField>> = {
  run_has_no_stop_condition: 'stop-conditions',
  stop_condition_unknown: 'stop-conditions',
  stop_condition_config_incomplete: 'stop-conditions',
  cadence_invalid: 'cadences',
  refresh_not_applicable: 'signal-source',
  refresh_mode_unknown: 'signal-source',
  refresh_rule_incomplete: 'signal-source',
  pack_incomplete: 'pack',
  part_unpinnable: 'pack',
  pack_not_found: 'pack',
  // The **service**, not the pack. F6's E-22: named, never silently substituted.
  midataworks_not_configured: 'pack',
  budget_exceeded: 'budget',
  budget_input_missing: 'model-source',
  budget_spec_incomplete: 'model-source',
  budget_spec_ambiguous: 'model-source',
  free_vram_unreadable: 'budget',
  tokenizer_required: 'model-source',
  // The create run's tokenizer, chosen in the model source card.
  tokenizer_source_outside_root: 'model-source',
  tokenizer_source_incomplete: 'model-source',
  tokenizer_not_at_root: 'model-source',
  tokenizer_vocab_mismatch: 'model-source',
  model_revision_not_found: 'model-source',
  model_fetch_failed: 'model-source',
  adapter_config_invalid: 'adapter',
  adapter_config_required: 'adapter',
  run_type_unknown: 'run-type',
  signal_source_unknown: 'signal-source',
};

/** `cadence_invalid` carries `details.field`; that field decides the row it lands on. */
const CADENCE_FIELDS: Readonly<Record<string, FormField>> = {
  save_every: 'save-cadence',
  retention_floor: 'retention-floor',
};

export function fieldFor(refusal: DecodedRefusal): FormField {
  if (refusal.code === 'cadence_invalid') {
    const named = refusal.details['field'];
    if (typeof named === 'string') {
      const mapped = CADENCE_FIELDS[named];
      if (mapped !== undefined) return mapped;
    }
    return 'cadences';
  }
  // No generic *message*, but a refusal whose code this build does not know still has to
  // appear somewhere, and the form is the honest place: attaching it to a guessed field would
  // mark a field that may be correct.
  return REFUSAL_FIELDS[refusal.code] ?? 'form';
}

/**
 * Decode one API failure into **one or more** refusals.
 *
 * Three envelopes reach this function, and all three carry a *list*:
 *
 * 1. **`run_definition_invalid`** from `POST /api/v1/runs/define` - F6's own, with
 *    `details.failures` as a list of `{step, code, message, details}` and `details.skipped`
 *    as the steps that could not run. *"A list rather than a merged message, so a form can
 *    put each refusal beside the field it belongs to."*
 * 2. **`validation_failed`** from Pydantic, when the *shape* of the body is wrong rather
 *    than its content - `details.fields` is FastAPI's `{loc, msg, type}` list. Expanded into
 *    one refusal per offending field, so a body wrong in four places renders four messages
 *    against four fields rather than one banner saying the request did not validate.
 * 3. Anything else - one refusal, passed through with its own code.
 *
 * `skipped` is carried as a refusal of its own rather than dropped: a step the validator
 * **could not run** means there may be more to fix, and a form that showed only `failures`
 * would tell the operator they had one problem when they may have three.
 */
export function decodeRefusals(failure: ApiFailure): readonly DecodedRefusal[] {
  if (failure.kind !== 'refusal' && failure.kind !== 'service_not_configured') {
    return [];
  }
  const refusal: RefusalFailure | typeof failure = failure;

  if (refusal.code === 'run_definition_invalid') {
    const collected = fromOutcome(refusal.details);
    if (collected.length > 0) return collected;
  }

  if (refusal.code === 'validation_failed') {
    const fields = refusal.details['fields'];
    if (Array.isArray(fields) && fields.length > 0) {
      return fields.map((entry) => {
        const row = (entry ?? {}) as Readonly<Record<string, unknown>>;
        const loc = Array.isArray(row['loc']) ? row['loc'].map((part) => String(part)) : [];
        // The LAST non-numeric segment, so `body.run_configuration.batch_size` lands on the
        // optimizer block rather than on an index. A `loc` of `['body', 0, 'condition']`
        // would otherwise name the field `condition`, which is not a form region.
        const named = [...loc].reverse().find((part) => part !== 'body' && !/^\d+$/.test(part));
        return {
          code: `validation_failed:${named ?? 'body'}`,
          // Pydantic prefixes a validator's own sentence with "Value error, "; the
          // sentence is the message.
          message:
            typeof row['msg'] === 'string'
              ? row['msg'].replace(/^Value error, /, '')
              : refusal.message,
          details: {
            field: named ?? 'body',
            location: loc.join('.'),
            type: row['type'] ?? 'unknown',
          },
        };
      });
    }
  }
  return [{ code: refusal.code, message: refusal.message, details: refusal.details }];
}

/**
 * The validator's failure list, from either envelope it arrives in.
 *
 * Used for `POST /runs/validate`'s **200 body** - which is a 200 whether or not the
 * definition is valid, because *asking whether a definition is valid is a successful
 * question* - and for `run_definition_invalid`'s `details`, which carries the same two lists.
 * One decoder, so the check path and the submit path cannot render the same failure two ways.
 */
export function fromOutcome(
  body: Readonly<Record<string, unknown>> | null | undefined,
): readonly DecodedRefusal[] {
  if (body === null || body === undefined) return [];
  const out: DecodedRefusal[] = [];

  const failures = body['failures'];
  if (Array.isArray(failures)) {
    for (const entry of failures) {
      const row = (entry ?? {}) as Readonly<Record<string, unknown>>;
      const code = typeof row['code'] === 'string' ? row['code'] : 'unnamed_failure';
      const details =
        row['details'] !== null &&
        typeof row['details'] === 'object' &&
        !Array.isArray(row['details'])
          ? (row['details'] as Readonly<Record<string, unknown>>)
          : {};
      out.push({
        code,
        message: typeof row['message'] === 'string' ? row['message'] : code,
        // The validator's own step is carried: it says WHICH part of the definition the
        // refusal came from, which is the thing that makes a code attributable at all.
        details: { ...details, ...(typeof row['step'] === 'string' ? { step: row['step'] } : {}) },
      });
    }
  }

  const skipped = body['skipped'];
  if (Array.isArray(skipped)) {
    for (const entry of skipped) {
      const row = (entry ?? {}) as Readonly<Record<string, unknown>>;
      const step = typeof row['step'] === 'string' ? row['step'] : 'a step';
      out.push({
        // Named `check_skipped:<step>` rather than folded into the failures: a step that
        // could not run is not a thing the operator did wrong, and rendering it as one would
        // send them to change a field that may be correct.
        code: `check_skipped:${step}`,
        message:
          typeof row['message'] === 'string'
            ? row['message']
            : `${step} could not be checked, so there may be more to fix.`,
        details: {
          step,
          waited_on: Array.isArray(row['waited_on']) ? row['waited_on'] : [],
        },
      });
    }
  }
  return out;
}

/** Which field a `validation_failed:<name>` refusal lands on, by the name FastAPI reported. */
export const VALIDATION_FIELDS: Readonly<Record<string, FormField>> = {
  // The nested objects of F6's `RunDefinition`, as `loc` names them…
  model_source: 'model-source',
  run_configuration: 'optimizer',
  signal_source: 'signal-source',
  cadences: 'cadences',
  retention_floor_override: 'retention-floor',
  // …and the leaves inside them, for a `loc` that reaches one.
  mode: 'model-source',
  path: 'model-source',
  repo: 'model-source',
  revision: 'model-source',
  config_path: 'model-source',
  tokenizer_path: 'model-source',
  tokenizer_repo: 'model-source',
  tokenizer_revision: 'model-source',
  learning_rate: 'optimizer',
  warmup_steps: 'optimizer',
  batch_size: 'optimizer',
  gradient_accumulation: 'optimizer',
  optimizer_state_precision: 'optimizer',
  seed: 'optimizer',
  run_type: 'run-type',
  pack: 'pack',
  test_every: 'cadences',
  retention_every: 'cadences',
  save_every: 'save-cadence',
  stop_conditions: 'stop-conditions',
  adapter: 'adapter',
};

export function refusalField(refusal: DecodedRefusal): FormField {
  // A check that could not RUN is not a field the operator got wrong. It goes on the form,
  // where it reads as "there may be more", rather than marking a field that may be correct.
  if (refusal.code.startsWith('check_skipped:')) return 'form';
  if (refusal.code.startsWith('validation_failed:')) {
    const name = refusal.code.slice('validation_failed:'.length);
    return VALIDATION_FIELDS[name] ?? 'form';
  }
  return fieldFor(refusal);
}

/** Group refusals by the field they belong against, so each field renders its own. */
export function byField(
  refusals: readonly DecodedRefusal[],
): ReadonlyMap<FormField, readonly DecodedRefusal[]> {
  const out = new Map<FormField, DecodedRefusal[]>();
  for (const refusal of refusals) {
    const field = refusalField(refusal);
    const bucket = out.get(field) ?? [];
    bucket.push(refusal);
    out.set(field, bucket);
  }
  return out;
}

export interface RefusalBannerProps {
  /** A **list**. See the module docstring. */
  readonly refusals: readonly DecodedRefusal[];
  /** Render only the refusals for this field, or all of them when omitted. */
  readonly field?: FormField;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function RefusalBanner({
  refusals,
  field,
  className = '',
  ...rest
}: RefusalBannerProps): ReactNode {
  const shown =
    field === undefined ? refusals : refusals.filter((refusal) => refusalField(refusal) === field);
  if (shown.length === 0) return null;

  return (
    <div className={`stack-y-2 ${className}`.trim()} {...rest}>
      {shown.map((refusal, index) => (
        <Callout
          key={`${refusal.code}-${index}`}
          tone="error"
          data-testid={`refusal-${refusal.code}`}
          data-refusal-field={refusalField(refusal)}
        >
          <div className="flex flex-wrap items-baseline gap-2">
            {/* The code, never a paraphrase of it. */}
            <MonoId value={refusal.code} />
            <span>{refusal.message}</span>
          </div>
          <DetailList details={refusal.details} />
        </Callout>
      ))}
    </div>
  );
}
