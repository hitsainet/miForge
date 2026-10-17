/**
 * The budget preview (FR-009.36), and **M20 — run before the preview was wired in.**
 *
 * **M20** disables the submit button on the preview's verdict. It is the most plausible
 * improvement on this screen and the most damaging: it moves R-02.53's refusal into the
 * browser, where this screen's reading of the budget can disagree with the gate's. The control
 * is structural - this component has **no prop** that could reach a button, returns no boolean
 * and takes no `onValid` - and these tests assert that absence directly, so the mutation
 * requires changing the component's interface.
 */
import { describe, expect, it } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BudgetPreview } from './BudgetPreview';

describe('[BudgetPreview]', () => {
  it('should expose no way to gate anything', () => {
    // M20's structural guard. Grepping the component is the honest check here: a prop that
    // does not exist cannot be wired to a disabled attribute.
    const source = BudgetPreview.toString();
    expect(source).not.toContain('onValid');
    expect(source).not.toContain('setDisabled');
    expect(source).not.toContain('canSubmit');
    // Its only prop is the request.
    expect(BudgetPreview.length).toBe(1);
  });

  it('should ask for nothing until there is enough to estimate from', () => {
    // An estimate derived from nothing is a preview of nothing, which is worse than a refusal.
    render(<BudgetPreview request={null} />);
    expect(screen.getByTestId('budget-preview')).toHaveTextContent(
      'Enter a run type and a batch configuration',
    );
    expect(screen.getByRole('button', { name: /Estimate/ })).toBeDisabled();
  });

  it('should say in words that it is a preview and not a permission', () => {
    render(<BudgetPreview request={null} />);
    expect(screen.getByTestId('budget-preview')).toHaveTextContent('not a permission');
  });

  it('should render the estimate, the breakdown by term and the free figure with their scales', () => {
    return (async () => {
      const request = {
        run_type: 'change',
        config: { learning_rate: 1e-5, warmup_steps: 0, batch_size: 8 },
      };
      render(<BudgetPreview request={request} />);
      // The real client is not reachable here, so the estimate comes from a stubbed fetch.
      const original = globalThis.fetch;
      globalThis.fetch = (() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              estimate_bytes: 11_600_000_000,
              free_bytes: 21_300_000_000,
              verdict: 'fits',
              gpu_name: 'RTX 4090',
              terms: [
                { name: 'weights', bytes: 2_400_000_000 },
                { name: 'optimizer', bytes: 4_800_000_000 },
              ],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )) as typeof globalThis.fetch;
      try {
        await userEvent.click(screen.getByRole('button', { name: /Estimate/ }));
        await waitFor(() => expect(screen.getByTestId('budget-estimate')).toBeInTheDocument());
        expect(screen.getByTestId('budget-estimate')).toHaveTextContent('11.6');
        expect(screen.getByTestId('budget-estimate')).toHaveTextContent('2 terms');
        expect(screen.getByTestId('budget-free')).toHaveTextContent('21.3');
        expect(screen.getByTestId('budget-free')).toHaveTextContent('RTX 4090');
        expect(screen.getByTestId('budget-terms')).toHaveTextContent('weights');
        expect(screen.getByTestId('budget-terms')).toHaveTextContent('optimizer');
        expect(screen.getByTestId('budget-verdict')).toHaveTextContent('fits');
      } finally {
        globalThis.fetch = original;
      }
    })();
  });

  it('should leave every control outside itself untouched when the verdict refuses', () => {
    // **M20's real control.** The earlier form of this test only grepped the component's
    // source for `onValid`, which an interface-only edit sails past - a mutation that changed
    // bytes without changing behaviour, which is the no-op-looks-like-a-survivor family this
    // project keeps paying for. So the assertion is now about BEHAVIOUR: with a refusing
    // verdict rendered, the only disabled control in the tree is this component's own
    // Estimate button, and it is disabled because a request is in flight - never because the
    // budget said no.
    return (async () => {
      const request = {
        run_type: 'change',
        config: { learning_rate: 1e-5, warmup_steps: 0, batch_size: 4096 },
      };
      const { container } = render(
        <>
          <BudgetPreview request={request} />
          {/* A stand-in for the form's primary action, outside this component's control. */}
          <button type="button" data-testid="outside-submit">
            Start run
          </button>
        </>,
      );
      const original = globalThis.fetch;
      globalThis.fetch = (() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              estimate_bytes: 40_000_000_000,
              free_bytes: 21_300_000_000,
              verdict: 'exceeds',
              terms: [{ name: 'weights', bytes: 40_000_000_000 }],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )) as typeof globalThis.fetch;
      try {
        await userEvent.click(screen.getByRole('button', { name: /Estimate/ }));
        await waitFor(() =>
          expect(screen.getByTestId('budget-verdict')).toHaveTextContent('exceeds'),
        );
        expect(screen.getByTestId('outside-submit')).toBeEnabled();
        const disabled = [...container.querySelectorAll('button[disabled]')].map(
          (node) => node.textContent ?? '',
        );
        expect(disabled).toEqual([]);
      } finally {
        globalThis.fetch = original;
      }
    })();
  });

  it('should report an unreadable free figure as absent and the verdict as unknown, never as fits', () => {
    // "I could not read the limit, so it fits" is the pass-through failure wearing the
    // estimator's name, and it would be the most misleading number on the screen.
    return (async () => {
      const request = {
        run_type: 'change',
        config: { learning_rate: 1e-5, warmup_steps: 0, batch_size: 8 },
      };
      render(<BudgetPreview request={request} />);
      const original = globalThis.fetch;
      globalThis.fetch = (() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              estimate_bytes: 11_600_000_000,
              free_bytes: null,
              verdict: 'unknown',
              terms: [],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )) as typeof globalThis.fetch;
      try {
        await userEvent.click(screen.getByRole('button', { name: /Estimate/ }));
        await waitFor(() => expect(screen.getByTestId('budget-no-free')).toBeInTheDocument());
        expect(screen.getByTestId('budget-no-free')).toHaveTextContent('no limit was read');
        expect(screen.getByTestId('budget-verdict')).toHaveTextContent('unknown');
      } finally {
        globalThis.fetch = original;
      }
    })();
  });
});
