/**
 * The run type from the **live registry** (FR-009.33), and **M22**.
 *
 * **M22** hard-codes the three values instead of reading `GET /api/v1/run-types`. The assertion
 * that catches it is that the options are **the response's** - not that they are any particular
 * three, which is the only form of the assertion that still holds when a fourth type is
 * registered. A hard-coded list is a second registry: a new type would not appear, and a
 * de-registered one would still be offered.
 */
import { describe, expect, it } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { RunTypeField } from './RunTypeField';
import type { RunTypeListing } from '@/api/runTypes';
import type { ApiResult } from '@/api/client';

const registry =
  (names: readonly string[]): (() => Promise<ApiResult<RunTypeListing>>) =>
  () =>
    Promise.resolve({
      ok: true,
      value: {
        run_types: names.map((name) => ({ name, stored_as: name, changes: `${name} things` })),
        count: names.length,
      },
    });

describe('[RunTypeField]', () => {
  it('should offer exactly what the registry returned', () => {
    // M22: the registry's answer, whatever it is.
    return (async () => {
      render(
        <RunTypeField
          value=""
          onChange={() => {}}
          load={registry(['adapter', 'change', 'create'])}
        />,
      );
      await waitFor(() => expect(screen.getByTestId('run-type-select')).toBeInTheDocument());
      const options = [...screen.getByTestId('run-type-select').querySelectorAll('option')]
        .map((option) => option.getAttribute('value'))
        .filter((value) => value !== '');
      expect(options).toEqual(['adapter', 'change', 'create']);
    })();
  });

  it('should show a FOURTH registered type with no change to this component', () => {
    // The assertion a hard-coded list cannot satisfy. If M22 is applied, this goes red.
    return (async () => {
      render(
        <RunTypeField
          value=""
          onChange={() => {}}
          load={registry(['adapter', 'change', 'create', 'distil'])}
        />,
      );
      await waitFor(() => expect(screen.getByTestId('run-type-select')).toBeInTheDocument());
      expect(screen.getByRole('option', { name: 'distil' })).toBeInTheDocument();
    })();
  });

  it('should hold no literal list of run types in its own source', () => {
    const source = RunTypeField.toString();
    expect(source).not.toContain("'change'");
    expect(source).not.toContain("'create'");
    expect(source).not.toContain("'adapter'");
  });

  it('should preselect nothing, because a preselected value is a default', () => {
    // FR-009.39 / M19: a non-nullable field with an initial value IS a default, and the run
    // log would record it as the operator's choice.
    return (async () => {
      render(<RunTypeField value="" onChange={() => {}} load={registry(['change'])} />);
      await waitFor(() => expect(screen.getByTestId('run-type-select')).toBeInTheDocument());
      expect((screen.getByTestId('run-type-select') as HTMLSelectElement).value).toBe('');
    })();
  });

  it('should render the registrys own sentence per type, writing no copy of its own', () => {
    return (async () => {
      render(<RunTypeField value="" onChange={() => {}} load={registry(['change'])} />);
      await waitFor(() => expect(screen.getByTestId('run-type-changes')).toBeInTheDocument());
      expect(screen.getByTestId('run-type-changes')).toHaveTextContent('change things');
    })();
  });

  it('should say there is nothing to choose when the registry is empty, not offer a guess', () => {
    return (async () => {
      render(<RunTypeField value="" onChange={() => {}} load={registry([])} />);
      await waitFor(() => expect(screen.getByText(/nothing to choose/)).toBeInTheDocument());
    })();
  });

  it('should render the refusal when the registry cannot be read', () => {
    return (async () => {
      render(
        <RunTypeField
          value=""
          onChange={() => {}}
          load={() =>
            Promise.resolve({
              ok: false,
              failure: { kind: 'transport', status: 0, cause: 'fetch failed' },
            })
          }
        />,
      );
      await waitFor(() =>
        expect(screen.getByText(/run-type registry is unavailable/)).toBeInTheDocument(),
      );
    })();
  });
});
