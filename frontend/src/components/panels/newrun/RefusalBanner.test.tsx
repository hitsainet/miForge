/**
 * Every refusal on its own field (FR-009.35), and **M21**.
 *
 * **M21** clears the form on a 422, or renders the refusal as a page-level error. This
 * component renders *beside* the fields and clears nothing - *a form that clears itself on a
 * refusal makes the operator re-enter nine things to learn the tenth was wrong.*
 *
 * The prop is a **list**, because F6 returns every failure together: *"a validator that stops
 * at the first failure makes an operator resubmit once per mistake."* A component with a
 * single-refusal prop cannot express that response.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  REFUSAL_FIELDS,
  RefusalBanner,
  byField,
  decodeRefusals,
  fromOutcome,
  refusalField,
} from './RefusalBanner';
import { decodeErrorEnvelope, transportFailure } from '@/api/errors';

describe('[decodeRefusals]', () => {
  it('should expand validation_failed into one refusal per offending field', () => {
    // This build's POST /api/v1/runs is closed to unknown keys, so a complete R-02.53
    // definition comes back as `validation_failed` with FastAPI's own field list. Four wrong
    // things must render as four messages on four fields, not one banner.
    const failure = decodeErrorEnvelope(422, {
      error: {
        code: 'validation_failed',
        message: 'The request body or parameters did not validate.',
        details: {
          fields: [
            {
              loc: ['body', 'run_type'],
              msg: 'Extra inputs are not permitted',
              type: 'extra_forbidden',
            },
            {
              loc: ['body', 'pack'],
              msg: 'Extra inputs are not permitted',
              type: 'extra_forbidden',
            },
            // A NESTED location, which is the shape F6's envelope actually produces: the
            // last non-numeric segment decides the field, so this lands on the optimizer
            // block rather than on `body`.
            {
              loc: ['body', 'run_configuration', 'learning_rate'],
              msg: 'Field required',
              type: 'missing',
            },
            {
              // An INDEXED location. `0` is skipped as numeric, so the field is `condition`.
              loc: ['body', 'stop_conditions', 0, 'condition'],
              msg: 'Extra inputs are not permitted',
              type: 'extra_forbidden',
            },
          ],
        },
      },
    });
    const refusals = decodeRefusals(failure);
    expect(refusals).toHaveLength(4);
    expect(refusals.map((row) => row.code)).toEqual([
      'validation_failed:run_type',
      'validation_failed:pack',
      'validation_failed:learning_rate',
      'validation_failed:condition',
    ]);
    // Each lands on a field of its own, except the one whose `loc` names a member the field
    // map does not know - that goes on the form rather than marking a field that may be fine.
    expect(refusals.map(refusalField)).toEqual(['run-type', 'pack', 'optimizer', 'form']);
  });

  it('should pass a named refusal through unchanged', () => {
    const failure = decodeErrorEnvelope(400, {
      error: {
        code: 'run_has_no_stop_condition',
        message: 'Name at least one.',
        details: { available: ['step_count'] },
      },
    });
    const refusals = decodeRefusals(failure);
    expect(refusals).toHaveLength(1);
    expect(refusals[0]?.details['available']).toEqual(['step_count']);
  });

  it('should produce nothing for a transport failure, which is not a refusal', () => {
    // A dead backend is not a refusal of the definition, and marking a field for it would
    // tell the operator their answer was wrong.
    expect(decodeRefusals(transportFailure('fetch failed'))).toEqual([]);
  });
});

describe('[fromOutcome]', () => {
  const outcome = {
    ok: false,
    may_be_more_to_fix: true,
    steps_checked: ['pack', 'loop'],
    failures: [
      {
        step: 'pack',
        code: 'pack_incomplete',
        message: 'Missing slots.',
        details: { missing_slots: ['test_set'] },
      },
      {
        step: 'loop',
        code: 'run_has_no_stop_condition',
        message: 'Name one.',
        details: { available: ['step_count'] },
      },
    ],
    skipped: [
      {
        step: 'budget',
        status: 'skipped',
        waited_on: ['model_source'],
        message: "'budget' could not be checked because model_source did not resolve.",
      },
    ],
    resolved: {},
  };

  it('should read every failure out of the validators 200 body', () => {
    // `POST /runs/validate` answers 200 either way - asking whether a definition is valid is
    // a successful question - so the failures come from the BODY, not from an envelope.
    const refusals = fromOutcome(outcome);
    expect(refusals.map((row) => row.code)).toEqual([
      'pack_incomplete',
      'run_has_no_stop_condition',
      'check_skipped:budget',
    ]);
  });

  it('should carry the validators own step, which is what makes a code attributable', () => {
    expect(fromOutcome(outcome)[0]?.details['step']).toBe('pack');
  });

  it('should carry a SKIPPED check rather than dropping it', () => {
    // A step the validator could not run means the failure list may be short. A form that
    // showed a tidy list of two while three were waiting would teach an operator to trust a
    // number that is not a count.
    const skipped = fromOutcome(outcome).find((row) => row.code.startsWith('check_skipped:'));
    expect(skipped?.details['waited_on']).toEqual(['model_source']);
    expect(skipped?.message).toContain('did not resolve');
  });

  it('should put a skipped check on the FORM, not on a guessed field', () => {
    // It is not a thing the operator did wrong, and marking a field would send them to
    // change one that may be correct.
    expect(refusalField({ code: 'check_skipped:budget', message: '', details: {} })).toBe('form');
  });

  it('should read nothing from an ok body and nothing from a missing one', () => {
    expect(fromOutcome({ ok: true, failures: [], skipped: [] })).toEqual([]);
    expect(fromOutcome(null)).toEqual([]);
    expect(fromOutcome(undefined)).toEqual([]);
  });

  it('should name an unnamed failure rather than rendering a blank code', () => {
    expect(fromOutcome({ failures: [{ message: 'something' }] })[0]?.code).toBe('unnamed_failure');
  });
});

describe('[decodeRefusals, run_definition_invalid]', () => {
  it('should expand F6s own envelope into one refusal per failure', () => {
    // F6 refuses `run_definition_invalid` with `details.failures` as a list of named
    // refusals, "so a form can put each refusal beside the field it belongs to."
    const refusals = decodeRefusals({
      kind: 'refusal',
      status: 400,
      code: 'run_definition_invalid',
      message: 'This run definition has 2 problem(s).',
      details: {
        failures: [
          { step: 'pack', code: 'pack_incomplete', message: 'x', details: {} },
          {
            step: 'budget',
            code: 'budget_exceeded',
            message: 'y',
            details: { estimate_bytes: 1, free_bytes: 0 },
          },
        ],
        may_be_more_to_fix: false,
      },
    });
    expect(refusals.map((row) => row.code)).toEqual(['pack_incomplete', 'budget_exceeded']);
    expect(new Set(refusals.map(refusalField))).toEqual(new Set(['pack', 'budget']));
  });

  it('should fall back to the envelope itself when the failure list is empty', () => {
    // A `run_definition_invalid` with no `failures` should not vanish from the screen.
    const refusals = decodeRefusals({
      kind: 'refusal',
      status: 400,
      code: 'run_definition_invalid',
      message: 'something went wrong',
      details: {},
    });
    expect(refusals).toHaveLength(1);
    expect(refusals[0]?.code).toBe('run_definition_invalid');
  });
});

describe('[refusalField]', () => {
  it('should map every code in the field map to a field', () => {
    for (const [code, field] of Object.entries(REFUSAL_FIELDS)) {
      expect(refusalField({ code, message: '', details: {} }), code).toBe(field);
    }
  });

  it('should put cadence_invalid on the cadence its own details name', () => {
    expect(
      refusalField({ code: 'cadence_invalid', message: '', details: { field: 'save_every' } }),
    ).toBe('save-cadence');
    expect(
      refusalField({ code: 'cadence_invalid', message: '', details: { field: 'test_every' } }),
    ).toBe('cadences');
  });

  it('should put midataworks_not_configured on the pack field, naming the SERVICE', () => {
    // F6's E-22: the service, not the pack. An unconfigured sibling is not a fault in the
    // pack, and marking the pack would send the operator to change the wrong thing.
    expect(REFUSAL_FIELDS['midataworks_not_configured']).toBe('pack');
  });

  it('should put budget_input_missing on the model source rather than on the preview', () => {
    // A prefix rule would have put it on the budget panel, where the operator can do nothing
    // about it. The missing input is the model's.
    expect(REFUSAL_FIELDS['budget_input_missing']).toBe('model-source');
  });

  it('should put an unrecognised code on the form rather than guessing a field', () => {
    // Marking a guessed field would mark a field that may be correct.
    expect(refusalField({ code: 'something_new', message: '', details: {} })).toBe('form');
  });
});

describe('[RefusalBanner]', () => {
  const refusals = [
    {
      code: 'run_has_no_stop_condition',
      message: 'Name at least one stop condition.',
      details: { available: ['step_count', 'loss_plateau'] },
    },
    {
      code: 'pack_incomplete',
      message: 'That pack is not usable yet.',
      details: { missing_slots: ['test_set', 'retention_set'] },
    },
    {
      code: 'budget_exceeded',
      message: 'This run does not fit.',
      details: {
        estimate_bytes: 25_000_000_000,
        free_bytes: 21_300_000_000,
        terms: ['weights', 'optimizer'],
      },
    },
    {
      code: 'tokenizer_required',
      message: 'That directory has no tokenizer.',
      details: { path: '/models/x' },
    },
  ];

  it('should render ALL of them, not the first', () => {
    // AC-009.26: a definition wrong in four places returns four refusals and all four render.
    render(<RefusalBanner refusals={refusals} />);
    for (const refusal of refusals) {
      expect(screen.getByTestId(`refusal-${refusal.code}`)).toBeInTheDocument();
    }
  });

  it('should attach each refusal to the field its own details name', () => {
    render(<RefusalBanner refusals={refusals} />);
    expect(
      screen.getByTestId('refusal-run_has_no_stop_condition').getAttribute('data-refusal-field'),
    ).toBe('stop-conditions');
    expect(screen.getByTestId('refusal-pack_incomplete').getAttribute('data-refusal-field')).toBe(
      'pack',
    );
    expect(screen.getByTestId('refusal-budget_exceeded').getAttribute('data-refusal-field')).toBe(
      'budget',
    );
    expect(
      screen.getByTestId('refusal-tokenizer_required').getAttribute('data-refusal-field'),
    ).toBe('model-source');
  });

  it('should render the details rather than summarising them', () => {
    // `budget_exceeded` without both figures is a refusal an operator cannot act on;
    // `pack_incomplete` without the slot names is the generic failure AC10 forbids.
    render(<RefusalBanner refusals={refusals} />);
    const budget = screen.getByTestId('refusal-budget_exceeded');
    expect(budget).toHaveTextContent('estimate_bytes');
    expect(budget).toHaveTextContent('free_bytes');
    const pack = screen.getByTestId('refusal-pack_incomplete');
    expect(pack).toHaveTextContent('test_set');
    expect(pack).toHaveTextContent('retention_set');
    const stop = screen.getByTestId('refusal-run_has_no_stop_condition');
    expect(stop).toHaveTextContent('step_count');
    expect(stop).toHaveTextContent('loss_plateau');
  });

  it('should render an unrecognised code as the code itself, with no generic branch', () => {
    render(
      <RefusalBanner
        refusals={[
          { code: 'weights_are_haunted', message: 'Try again on a Tuesday.', details: {} },
        ]}
      />,
    );
    const callout = screen.getByTestId('refusal-weights_are_haunted');
    expect(callout).toHaveTextContent('weights_are_haunted');
    expect(callout).toHaveTextContent('Try again on a Tuesday.');
  });

  it('should filter to one field when asked, so each field shows only its own', () => {
    render(<RefusalBanner refusals={refusals} field="pack" />);
    expect(screen.getByTestId('refusal-pack_incomplete')).toBeInTheDocument();
    expect(screen.queryByTestId('refusal-budget_exceeded')).toBeNull();
  });

  it('should render nothing at all when there is nothing to say', () => {
    const { container } = render(<RefusalBanner refusals={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe('[byField]', () => {
  it('should group several refusals for one field together', () => {
    const grouped = byField([
      { code: 'stop_condition_unknown', message: '', details: {} },
      { code: 'run_has_no_stop_condition', message: '', details: {} },
      { code: 'pack_incomplete', message: '', details: {} },
    ]);
    expect(grouped.get('stop-conditions')).toHaveLength(2);
    expect(grouped.get('pack')).toHaveLength(1);
  });
});
