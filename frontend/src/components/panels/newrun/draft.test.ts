/**
 * The submitted body (FR-009.33, .34, .39), and **M19**.
 *
 * **M19** defaults a field the operator left empty - a batch size, a cadence, a stop condition.
 * The assertion that catches it is `buildBody` carrying **only what was entered**: ADR-018
 * says a default on a submission form is a value the operator never chose, and the run log
 * will record it as theirs, which is worse than a refusal because it is silent and permanent.
 */
import { describe, expect, it } from 'vitest';
import { DOCUMENTED_STOP_CONDITIONS, EMPTY_DRAFT, STOP_VALUE_KEYS, buildBody } from './draft';

describe('[EMPTY_DRAFT]', () => {
  it('should carry no value at all, for any field', () => {
    // Not a batch size, not a cadence, not a stop condition, not a rank, not even the
    // acquisition mode the backend marks `default: true`.
    for (const [key, value] of Object.entries(EMPTY_DRAFT)) {
      if (key === 'stopConditions') {
        expect(value, key).toEqual([]);
        continue;
      }
      expect(value, `${key} carries an initial value`).toBe('');
    }
  });
});

describe('[buildBody]', () => {
  it('should submit an empty body for an empty form', () => {
    // M19 in its purest form: the operator entered nothing, so nothing is submitted - and
    // the backend refuses naming every required part.
    expect(buildBody(EMPTY_DRAFT)).toEqual({});
  });

  it('should carry only what was entered, inside F6s own envelope', () => {
    const body = buildBody({ ...EMPTY_DRAFT, batchSize: '8', runType: 'change' });
    // `run_configuration` is F1's RunConfig and carries ONLY the batch size; the four other
    // members the operator left empty are absent from it, not zero and not defaulted.
    expect(body).toEqual({ run_type: 'change', run_configuration: { batch_size: 8 } });
    const config = body['run_configuration'] as Record<string, unknown>;
    for (const key of [
      'learning_rate',
      'warmup_steps',
      'gradient_accumulation',
      'optimizer_state_precision',
      'seed',
    ]) {
      expect(key in config, key).toBe(false);
    }
    // And the envelope's own parts the operator did not touch are omitted ENTIRELY rather
    // than sent as `{}` - an empty `model_source` is a model source that names nothing, and
    // F1 would have to refuse it as though one had been given.
    for (const key of [
      'model_source',
      'signal_source',
      'cadences',
      'pack',
      'stop_conditions',
      'adapter',
    ]) {
      expect(key in body, key).toBe(false);
    }
  });

  it('should supply neither cadence, so F6s declared defaults stay F6s', () => {
    // F6's `Cadences` declares `test_every` and `retention_every` with defaults, on the
    // stated ground that "a cadence may have a declared default, because it changes only
    // WHEN a recorded action happens and the record says when it happened" - while anything
    // deciding WHETHER a run ends has none. So an unfilled cadence is F6's default recorded
    // as F6's, not this form's guess recorded as the operator's.
    expect('cadences' in buildBody(EMPTY_DRAFT)).toBe(false);
  });

  it('should name the retention floor an OVERRIDE, since the pack carries the canonical one', () => {
    // F6's field is `retention_floor_override`. Whether a run may override the pack's floor
    // is F7's OQ-8 and is not decided here; the field is offered and the backend decides.
    const body = buildBody({ ...EMPTY_DRAFT, retentionFloor: '0.72' });
    expect(body).toEqual({ retention_floor_override: 0.72 });
    expect('retention_floor' in body).toBe(false);
  });

  it('should put a stop conditions value under its own conditions key, never a guessed one', () => {
    // F6 declares no shape for `config` precisely so no module invents a `window` or a
    // `threshold` with a value nobody chose. This maps the one number the field collects.
    expect(
      buildBody({ ...EMPTY_DRAFT, stopConditions: [{ condition: 'step_count', value: '800' }] })[
        'stop_conditions'
      ],
    ).toEqual([{ condition: 'step_count', config: { steps: 800 } }]);
    expect(
      buildBody({ ...EMPTY_DRAFT, stopConditions: [{ condition: 'loss_plateau', value: '50' }] })[
        'stop_conditions'
      ],
    ).toEqual([{ condition: 'loss_plateau', config: { window: 50 } }]);
    // A condition this table does not know still carries the number, under `value`, so the
    // backend refuses it BY NAME rather than seeing nothing at all.
    expect(
      buildBody({ ...EMPTY_DRAFT, stopConditions: [{ condition: 'brand_new', value: '3' }] })[
        'stop_conditions'
      ],
    ).toEqual([{ condition: 'brand_new', config: { value: 3 } }]);
  });

  it('should submit neither a pack nor parts when neither was chosen, and let F6 refuse', () => {
    // F6's model refuses "both or neither" - a run whose material came half from a pack and
    // half from a loose list is a run whose inputs no pack version names. The form does not
    // pre-empt that check.
    const body = buildBody(EMPTY_DRAFT);
    expect('pack' in body).toBe(false);
    expect('parts' in body).toBe(false);
  });

  it('should submit no stop conditions when none were added', () => {
    // R-02.53 requires at least one, and the form does not enforce that: the refusal is
    // `run_has_no_stop_condition` from the backend, listing the available set.
    expect('stop_conditions' in buildBody(EMPTY_DRAFT)).toBe(false);
  });

  it('should carry a stop condition with no value as a condition with no value', () => {
    const body = buildBody({
      ...EMPTY_DRAFT,
      stopConditions: [{ condition: 'step_count', value: '' }],
    });
    expect(body['stop_conditions']).toEqual([{ condition: 'step_count' }]);
  });

  it('should drop a stop-condition row whose condition was never chosen', () => {
    const body = buildBody({
      ...EMPTY_DRAFT,
      stopConditions: [{ condition: '', value: '400' }],
    });
    expect('stop_conditions' in body).toBe(false);
  });

  it('should omit the adapter block entirely when its fields are empty', () => {
    // So a change of run type cannot smuggle a rank into a `change` run.
    expect('adapter' in buildBody(EMPTY_DRAFT)).toBe(false);
    const withRank = buildBody({ ...EMPTY_DRAFT, adapterRank: '8' });
    expect(withRank['adapter']).toEqual({ rank: 8 });
  });

  it('should split target modules on commas and omit the key when there are none', () => {
    expect(
      (
        buildBody({ ...EMPTY_DRAFT, adapterTargetModules: 'q_proj, v_proj' })['adapter'] as Record<
          string,
          unknown
        >
      )['target_modules'],
    ).toEqual(['q_proj', 'v_proj']);
    expect('adapter' in buildBody({ ...EMPTY_DRAFT, adapterTargetModules: '  ,  ' })).toBe(false);
  });

  it('should pin a pack with its version, since a pack without one is not a pin', () => {
    const body = buildBody({ ...EMPTY_DRAFT, packId: 'md-notary', packVersion: '2' });
    expect(body['pack']).toEqual({ pack_id: 'md-notary', version: '2' });
  });

  it('should submit a zero the operator actually typed, since zero is a choice', () => {
    // The distinction the whole module turns on: an EMPTY field is absent; a field the
    // operator typed `0` into carries 0, because they chose it.
    const config = buildBody({ ...EMPTY_DRAFT, warmupSteps: '0' })['run_configuration'] as Record<
      string,
      unknown
    >;
    expect(config['warmup_steps']).toBe(0);
  });

  it('should drop an unparseable number rather than coercing it to something', () => {
    // Coercing `eight` to 8, or to 0, would submit a value the operator did not choose.
    expect('run_configuration' in buildBody({ ...EMPTY_DRAFT, batchSize: 'eight' })).toBe(false);
  });

  it('should trim whitespace rather than submitting a blank string as a value', () => {
    expect('model_source' in buildBody({ ...EMPTY_DRAFT, path: '   ' })).toBe(false);
    expect(
      (
        buildBody({ ...EMPTY_DRAFT, path: ' /models/x ' })['model_source'] as Record<
          string,
          unknown
        >
      )['path'],
    ).toBe('/models/x');
  });
});

describe('[STOP_VALUE_KEYS]', () => {
  it('should name a key for each of the four conditions that take a parameter', () => {
    // `operator_stop` is deliberately absent: it takes no parameter, because an operator
    // stop is an action rather than a threshold.
    expect(Object.keys(STOP_VALUE_KEYS).sort()).toEqual([
      'loss_plateau',
      'retention_floor',
      'step_count',
      'test_threshold',
    ]);
  });
});

describe('[DOCUMENTED_STOP_CONDITIONS]', () => {
  it('should hold R-02.25s five, in F6s precedence order', () => {
    expect([...DOCUMENTED_STOP_CONDITIONS]).toEqual([
      'operator_stop',
      'retention_floor',
      'test_threshold',
      'loss_plateau',
      'step_count',
    ]);
  });
});
