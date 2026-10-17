/**
 * The form end to end (FR-009.33-.39): **M19 and M21 at the screen level.**
 *
 * The two assertions that carry the feature:
 *
 * - **the submitted body carries only what was entered** (M19). Nothing is defaulted, not even
 *   the acquisition mode the registry marks `default: true`;
 * - **a 422 keeps the form** (M21). The operator's other answers survive and the named fields
 *   are marked - *a form that clears itself on a refusal makes the operator re-enter nine
 *   things to learn the tenth was wrong.*
 *
 * The screen's reads (`run-types`, `packs`, `selectors`, `models/sources`, `signal-sources`)
 * are stubbed at the **HTTP boundary** rather than by importing another feature's service
 * (FPRD 12.5, ADR-001). The real backend's 422 envelope shape is reproduced exactly:
 * `{error: {code: "validation_failed", details: {fields: [...]}}}`, which is what
 * `api/app.py`'s own validation handler returns.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { NewRunForm, startLabel } from './NewRunForm';
import { NavigationProvider } from '@/contexts/NavigationContext';
import type { ApiResult } from '@/api/client';
import type { ValidationOutcome } from '@/api/runs';

const EMPTY_LISTINGS: Readonly<Record<string, unknown>> = {
  '/api/v1/run-types': {
    run_types: [{ name: 'change', stored_as: 'change', changes: 'everything' }],
    count: 1,
  },
  '/api/v1/packs': { packs: [] },
  '/api/v1/selectors': [],
  '/api/v1/models/sources': {
    modes: [
      {
        mode: 'load',
        available: true,
        default: true,
        reason: null,
        message: null,
        capabilities_unavailable: [],
        sources: ['local_dir'],
      },
    ],
    default: 'load',
  },
  '/api/v1/signal-sources': { sources: [], count: 0, available: 0 },
};

let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    for (const [path, body] of Object.entries(EMPTY_LISTINGS)) {
      if (url.startsWith(path)) {
        return Promise.resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        );
      }
    }
    return Promise.resolve(
      new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } }),
    );
  }) as typeof globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

type Accepted = ApiResult<{ readonly run_id: string }>;

function mount(
  submit: (body: unknown) => Promise<Accepted>,
  check?: (body: unknown) => Promise<ApiResult<ValidationOutcome>>,
) {
  return render(
    <NavigationProvider initialScreen="new-run">
      <NewRunForm
        submit={submit as never}
        {...(check === undefined ? {} : { check: check as never })}
      />
    </NavigationProvider>,
  );
}

/** `POST /api/v1/runs/define` answers 202 with the accepted run's summary. */
const accepted: Accepted = { ok: true, value: { run_id: 'run-new' } };

/** `POST /api/v1/runs/validate` answers **200 either way**. This is the ok body. */
const validOutcome: ApiResult<ValidationOutcome> = {
  ok: true,
  value: {
    ok: true,
    may_be_more_to_fix: false,
    steps_checked: ['pack', 'loop', 'budget'],
    failures: [],
    skipped: [],
    resolved: {
      pack: { pack_id: 'md-notary', version: '2' },
      budget: { estimate_bytes: 11_600_000_000, free_bytes: 21_300_000_000 },
    },
    footprint: { projected_bytes: 480_000_000_000, free_bytes: 900_000_000_000 },
  },
};

describe('[NewRunForm]', () => {
  it('should submit an EMPTY body for an untouched form', async () => {
    // M19, at the screen level. Nothing is defaulted - not the mode the registry marks
    // `default: true`, not the optimizer precision, not a cadence, not a stop condition.
    const submit = vi.fn(() => Promise.resolve(accepted));
    mount(submit);
    await userEvent.click(screen.getByTestId('start-run'));
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    // The payload AND the call count (ADR-016 #1: "was called" passes against wrong arguments).
    expect(submit).toHaveBeenCalledWith({});
  });

  it('should submit exactly and only the fields the operator filled', async () => {
    const submit = vi.fn(() => Promise.resolve(accepted));
    mount(submit);
    await userEvent.type(screen.getByTestId('optimizer-batchSize'), '8');
    await userEvent.type(screen.getByTestId('cadence-save'), '100');
    await userEvent.click(screen.getByTestId('start-run'));
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    // F6's envelope: the batch size inside `run_configuration`, the save cadence inside
    // `cadences`, and nothing else - no `model_source`, no `pack`, no stop condition.
    expect(submit).toHaveBeenCalledWith({
      run_configuration: { batch_size: 8 },
      cadences: { save_every: 100 },
    });
  });

  it('should show what will be submitted, so a default would be visible', async () => {
    mount(() => Promise.resolve(accepted));
    await userEvent.type(screen.getByTestId('optimizer-batchSize'), '8');
    await waitFor(() =>
      expect(screen.getByTestId('submitted-body')).toHaveTextContent('"batch_size": 8'),
    );
    expect(screen.getByTestId('submitted-body')).toHaveTextContent('run_configuration');
    expect(screen.getByTestId('submitted-body').textContent ?? '').not.toContain('learning_rate');
  });

  it('should keep every answer on a 422 and mark the named fields', async () => {
    // M21. The refusals land beside their fields; nothing is cleared.
    const submit = vi.fn((): Promise<Accepted> =>
      Promise.resolve({
        ok: false,
        failure: {
          kind: 'refusal',
          status: 422,
          code: 'validation_failed',
          message: 'The request body or parameters did not validate.',
          details: {
            fields: [
              { loc: ['body', 'learning_rate'], msg: 'Field required', type: 'missing' },
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
              {
                loc: ['body', 'stop_conditions'],
                msg: 'Extra inputs are not permitted',
                type: 'extra_forbidden',
              },
            ],
          },
        },
      }),
    );
    mount(submit);
    await userEvent.type(screen.getByTestId('optimizer-batchSize'), '8');
    await userEvent.type(screen.getByTestId('run-id-field'), 'md-notary-1');
    await userEvent.click(screen.getByTestId('start-run'));

    // Four refusals, all four rendered (AC-009.26).
    await waitFor(() =>
      expect(screen.getByTestId('refusal-validation_failed:learning_rate')).toBeInTheDocument(),
    );
    for (const field of ['run_type', 'pack', 'stop_conditions']) {
      expect(screen.getByTestId(`refusal-validation_failed:${field}`)).toBeInTheDocument();
    }
    // And every answer survives.
    expect((screen.getByTestId('optimizer-batchSize') as HTMLInputElement).value).toBe('8');
    expect((screen.getByTestId('run-id-field') as HTMLInputElement).value).toBe('md-notary-1');
  });

  it('should attach each 422 field to the form region it belongs to, not to one banner', async () => {
    const submit = () =>
      Promise.resolve<Accepted>({
        ok: false,
        failure: {
          kind: 'refusal',
          status: 422,
          code: 'validation_failed',
          message: 'did not validate',
          details: {
            fields: [{ loc: ['body', 'learning_rate'], msg: 'Field required', type: 'missing' }],
          },
        },
      });
    mount(submit);
    await userEvent.click(screen.getByTestId('start-run'));
    await waitFor(() =>
      expect(
        screen
          .getByTestId('refusal-validation_failed:learning_rate')
          .getAttribute('data-refusal-field'),
      ).toBe('optimizer'),
    );
  });

  it('should read the stop-condition set from a refusal when one names it', async () => {
    // No endpoint publishes the set; F6's refusal carries `details.available`, so the field
    // takes the refusal's own list rather than keeping its documented one.
    const submit = () =>
      Promise.resolve<Accepted>({
        ok: false,
        failure: {
          kind: 'refusal',
          status: 400,
          code: 'run_has_no_stop_condition',
          message: 'Name at least one.',
          details: { available: ['only_this_one'] },
        },
      });
    mount(submit);
    await userEvent.click(screen.getByTestId('start-run'));
    await waitFor(() =>
      expect(screen.getByTestId('refusal-run_has_no_stop_condition')).toBeInTheDocument(),
    );
    await userEvent.click(screen.getByRole('button', { name: /Add a stop condition/ }));
    expect(screen.getByRole('option', { name: 'only_this_one' })).toBeInTheDocument();
  });

  it('should keep Start run enabled even when the budget estimate refuses', async () => {
    // **M20 at the screen level**, which is the level that matters: disabling submit on this
    // screen's reading of the budget moves R-02.53's refusal into the browser, where it can
    // disagree with the gate. The authoritative refusal is the create path's.
    const original = globalThis.fetch;
    globalThis.fetch = ((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.startsWith('/api/v1/budget/estimate')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              estimate_bytes: 40_000_000_000,
              free_bytes: 21_300_000_000,
              verdict: 'exceeds',
              terms: [],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      for (const [path, body] of Object.entries(EMPTY_LISTINGS)) {
        if (url.startsWith(path)) {
          return Promise.resolve(
            new Response(JSON.stringify(body), {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }),
          );
        }
      }
      return Promise.resolve(
        new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } }),
      );
    }) as typeof globalThis.fetch;
    const submit = vi.fn(() => Promise.resolve(accepted));
    try {
      mount(submit);
      // Enough for the estimate to be requestable: a run type and a batch configuration.
      await waitFor(() => expect(screen.getByTestId('run-type-select')).toBeInTheDocument());
      await userEvent.selectOptions(screen.getByTestId('run-type-select'), 'change');
      await userEvent.type(screen.getByTestId('optimizer-learningRate'), '0.00001');
      await userEvent.type(screen.getByTestId('optimizer-batchSize'), '4096');
      await userEvent.click(screen.getByRole('button', { name: /Estimate the footprint/ }));
      await waitFor(() =>
        expect(screen.getByTestId('budget-verdict')).toHaveTextContent('exceeds'),
      );

      // The preview refused and the button is still live, and still submits.
      expect(screen.getByTestId('start-run')).toBeEnabled();
      await userEvent.click(screen.getByTestId('start-run'));
      await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    } finally {
      globalThis.fetch = original;
    }
  });

  it('should read Start run, and never Attach and start, outside attach mode', () => {
    // FR-009.38. "Attach and start" is the mockup's pre-reframe label for section 5.10's
    // mode; on the default path it would describe an attach that does not happen.
    mount(() => Promise.resolve(accepted));
    expect(screen.getByTestId('start-run')).toHaveTextContent('Start run');
    expect(document.body.textContent ?? '').not.toContain('Attach and start');
  });

  it('should read Attach and start in attach mode and only there (F11 16.4)', async () => {
    // PPRD 7: the label is "section 5.10's screen only". Driven through the real mode select
    // with the live listing offering `attach`, so the label follows the operator's choice.
    const listing = {
      modes: [
        {
          mode: 'load',
          available: true,
          default: true,
          reason: null,
          message: null,
          capabilities_unavailable: [],
          sources: ['local_dir'],
        },
        {
          mode: 'attach',
          available: true,
          default: false,
          reason: null,
          message: null,
          capabilities_unavailable: [],
          sources: [],
        },
      ],
      default: 'load',
    };
    const inner = globalThis.fetch;
    globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.startsWith('/api/v1/models/sources')) {
        return Promise.resolve(
          new Response(JSON.stringify(listing), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        );
      }
      return inner(input, init);
    }) as typeof globalThis.fetch;
    mount(() => Promise.resolve(accepted));
    const select = await screen.findByTestId('mode-select');
    await userEvent.selectOptions(select, 'attach');
    expect(screen.getByTestId('start-run')).toHaveTextContent('Attach and start');
    await userEvent.selectOptions(select, 'load');
    expect(screen.getByTestId('start-run')).toHaveTextContent('Start run');
    expect(screen.getByTestId('start-run')).not.toHaveTextContent('Attach');
  });

  it('should keep the label rule in one function', () => {
    expect(startLabel('attach')).toBe('Attach and start');
    for (const mode of ['', 'load', 'create', 'Attach', 'attach ']) {
      expect(startLabel(mode)).toBe('Start run');
    }
  });

  it('should show the adapter block only for an adapter run', async () => {
    mount(() => Promise.resolve(accepted));
    expect(screen.queryByTestId('adapter-block')).toBeNull();
  });

  it('should show the validators resolved summary, and say a green check enables nothing', async () => {
    // AC-009.27. `POST /api/v1/runs/validate` answers 200 either way, and its `resolved`
    // block carries the pinned versions, the budget beside free VRAM and the footprint
    // beside free disk - F6's figures, not F8's asked separately.
    const check = vi.fn(() => Promise.resolve(validOutcome));
    mount(() => Promise.resolve(accepted), check);
    await userEvent.click(screen.getByTestId('check-definition'));
    await waitFor(() => expect(screen.getByTestId('resolved-summary')).toBeInTheDocument());
    const summary = screen.getByTestId('resolved-summary');
    expect(summary).toHaveTextContent('11.6');
    expect(summary).toHaveTextContent('21.3');
    expect(summary).toHaveTextContent('enables nothing');
    expect(check).toHaveBeenCalledTimes(1);
    // And it enabled nothing: the button was already live and still is.
    expect(screen.getByTestId('start-run')).toBeEnabled();
  });

  it('should read a NON-ok validate body from the body, not as a transport failure', async () => {
    // The validator answers 200 with `ok: false` and the complete failure list. Treating
    // that as a broken request would make a perfectly good answer look like one.
    const check = vi.fn((): Promise<ApiResult<ValidationOutcome>> =>
      Promise.resolve({
        ok: true,
        value: {
          ok: false,
          may_be_more_to_fix: true,
          steps_checked: ['pack'],
          failures: [
            {
              step: 'pack',
              code: 'pack_incomplete',
              message: 'That pack is missing its test set.',
              details: { missing_slots: ['test_set'] },
            },
            {
              step: 'loop',
              code: 'run_has_no_stop_condition',
              message: 'Name at least one.',
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
        },
      }),
    );
    mount(() => Promise.resolve(accepted), check);
    await userEvent.click(screen.getByTestId('check-definition'));

    // Both failures, on their own fields…
    await waitFor(() => expect(screen.getByTestId('refusal-pack_incomplete')).toBeInTheDocument());
    expect(screen.getByTestId('refusal-pack_incomplete').getAttribute('data-refusal-field')).toBe(
      'pack',
    );
    expect(
      screen.getByTestId('refusal-run_has_no_stop_condition').getAttribute('data-refusal-field'),
    ).toBe('stop-conditions');
    // …the skipped check named rather than dropped - "there may be more to fix"…
    expect(screen.getByTestId('refusal-check_skipped:budget')).toBeInTheDocument();
    expect(screen.getByTestId('resolved-summary')).toHaveTextContent('There may be more to fix');
    // …and nothing disabled by any of it.
    expect(screen.getByTestId('start-run')).toBeEnabled();
  });

  it('should render every failure of a run_definition_invalid refusal, on its own field', async () => {
    // AC-009.26 against F6's OWN envelope: `details.failures` is a list of named refusals,
    // "so a form can put each refusal beside the field it belongs to."
    const submit = vi.fn((): Promise<Accepted> =>
      Promise.resolve({
        ok: false,
        failure: {
          kind: 'refusal',
          status: 400,
          code: 'run_definition_invalid',
          message: 'This run definition has 4 problem(s).',
          details: {
            failures: [
              {
                step: 'loop',
                code: 'run_has_no_stop_condition',
                message: 'Name one.',
                details: { available: ['step_count'] },
              },
              {
                step: 'pack',
                code: 'pack_incomplete',
                message: 'Missing slots.',
                details: { missing_slots: ['test_set', 'retention_set'] },
              },
              {
                step: 'budget',
                code: 'budget_exceeded',
                message: 'Does not fit.',
                details: { estimate_bytes: 4e10, free_bytes: 2e10 },
              },
              {
                step: 'acquisition',
                code: 'tokenizer_required',
                message: 'No tokenizer.',
                details: { path: '/models/x' },
              },
            ],
            may_be_more_to_fix: false,
          },
        },
      }),
    );
    mount(submit);
    await userEvent.type(screen.getByTestId('optimizer-batchSize'), '8');
    await userEvent.click(screen.getByTestId('start-run'));

    for (const code of [
      'run_has_no_stop_condition',
      'pack_incomplete',
      'budget_exceeded',
      'tokenizer_required',
    ]) {
      await waitFor(() => expect(screen.getByTestId(`refusal-${code}`)).toBeInTheDocument());
    }
    // Four fields, not one banner.
    const fields = [
      'run_has_no_stop_condition',
      'pack_incomplete',
      'budget_exceeded',
      'tokenizer_required',
    ].map((code) => screen.getByTestId(`refusal-${code}`).getAttribute('data-refusal-field'));
    expect(new Set(fields).size).toBe(4);
    // And every answer survives (M21).
    expect((screen.getByTestId('optimizer-batchSize') as HTMLInputElement).value).toBe('8');
  });

  it('should clear a green check when a submit is then refused', async () => {
    // A green summary sitting beside four refusals would read as "this was fine a moment
    // ago", which is the stale-answer problem a cached permission creates.
    const check = () => Promise.resolve(validOutcome);
    const submit = () =>
      Promise.resolve<Accepted>({
        ok: false,
        failure: {
          kind: 'refusal',
          status: 400,
          code: 'run_definition_invalid',
          message: 'one problem',
          details: {
            failures: [{ step: 'pack', code: 'pack_incomplete', message: 'x', details: {} }],
          },
        },
      });
    mount(submit, check);
    await userEvent.click(screen.getByTestId('check-definition'));
    await waitFor(() => expect(screen.getByTestId('resolved-summary')).toBeInTheDocument());
    await userEvent.click(screen.getByTestId('start-run'));
    await waitFor(() => expect(screen.queryByTestId('resolved-summary')).toBeNull());
  });

  it('should say on the screen that it posts to define, not to the acquisition route', async () => {
    // `POST /api/v1/runs` is F1's acquisition half and R-02.53 forbids that shape for a
    // definition; `/define` is the create path.
    mount(() => Promise.resolve(accepted));
    // The path is rendered through MonoId, so the text is in its own element.
    expect(screen.getByText('POST /api/v1/runs/define')).toBeInTheDocument();
    expect(screen.getByText('validates nothing and defaults nothing')).toBeInTheDocument();
  });
});
