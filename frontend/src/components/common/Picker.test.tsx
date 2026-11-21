/**
 * The shared picker (task 14.0, D-14a; AC-009.45).
 *
 * What it must do for every field that uses it: list, filter as typed, be operable by keyboard
 * alone, show the label and **submit the value** (M41: submitting the label is the mutation),
 * list an unavailable option without letting it be chosen, preselect nothing, and render
 * loading, empty and failed as states rather than as an empty list.
 */
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Picker, matches, ranked, type PickerListing, type PickerOption } from './Picker';
import { choose, openPicker, pickedValue, pickerInput, pickerValues } from './pickerTestKit';

const OPTIONS: readonly PickerOption[] = [
  { value: 'pk-notary@3', label: 'notary @ 3', detail: 'valid · 7 of 7 slots' },
  { value: 'pk-tax@1', label: 'tax @ 1', detail: 'incomplete', keywords: 'revenue' },
  { value: 'pk-gone@2', label: 'gone @ 2', disabledReason: 'its service is not configured' },
  { value: 'pk-lease@5', label: 'lease @ 5' },
];

const READY: PickerListing = { kind: 'ready', options: OPTIONS };

function Harness({
  listing = READY,
  initial = '',
  onChange,
  freeText,
}: {
  readonly listing?: PickerListing;
  readonly initial?: string;
  readonly onChange?: (value: string) => void;
  readonly freeText?: { readonly reason: string };
}) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <label htmlFor="pick">Pack</label>
      <Picker
        id="pick"
        value={value}
        onChange={(next) => {
          onChange?.(next);
          setValue(next);
        }}
        listing={listing}
        placeholder="Choose a pack version"
        empty="No pack is registered."
        surface="the pack registry"
        {...(freeText === undefined ? {} : { freeText })}
        data-testid="pick"
      />
    </>
  );
}

const key = (name: string) =>
  act(async () => {
    fireEvent.keyDown(pickerInput('pick'), { key: name });
  });

describe('[Picker]', () => {
  it('should list every option, including an unavailable one, in the order given', async () => {
    render(<Harness />);
    expect(await pickerValues('pick')).toEqual([
      'pk-notary@3',
      'pk-tax@1',
      'pk-gone@2',
      'pk-lease@5',
    ]);
  });

  it('should submit the VALUE and show the LABEL (M41)', async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await choose('pick', 'pk-tax@1');
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith('pk-tax@1');
    expect(pickerInput('pick').value).toBe('tax @ 1');
    expect(pickedValue('pick')).toBe('pk-tax@1');
    // The chosen option's detail line is shown beside it.
    expect(screen.getByTestId('pick-status')).toHaveTextContent('incomplete');
  });

  it('should preselect nothing', () => {
    render(<Harness />);
    expect(pickerInput('pick').value).toBe('');
    expect(pickedValue('pick')).toBe('');
    expect(pickerInput('pick').placeholder).toBe('Choose a pack version');
  });

  it('should list an unavailable option with its reason and not let it be chosen', async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    const listbox = await openPicker('pick');
    const gone = [...listbox.querySelectorAll('[role="option"]')].find(
      (row) => row.getAttribute('data-value') === 'pk-gone@2',
    ) as HTMLElement;
    expect(gone.getAttribute('aria-disabled')).toBe('true');
    expect(gone).toHaveTextContent('Unavailable: its service is not configured');
    await act(async () => {
      fireEvent.mouseDown(gone);
    });
    expect(onChange).not.toHaveBeenCalled();
  });

  it('should be operable by keyboard alone: down, up, home, end, enter, escape', async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    pickerInput('pick').focus();
    await key('ArrowDown'); // opens, on the first enabled option
    expect(pickerInput('pick').getAttribute('aria-expanded')).toBe('true');
    const activeValue = () => {
      const id = pickerInput('pick').getAttribute('aria-activedescendant');
      return id === null ? null : document.getElementById(id)?.getAttribute('data-value');
    };
    expect(activeValue()).toBe('pk-notary@3');
    await key('ArrowDown');
    expect(activeValue()).toBe('pk-tax@1');
    await key('ArrowDown'); // skips the disabled row
    expect(activeValue()).toBe('pk-lease@5');
    await key('ArrowDown'); // wraps
    expect(activeValue()).toBe('pk-notary@3');
    await key('ArrowUp');
    expect(activeValue()).toBe('pk-lease@5');
    await key('Home');
    expect(activeValue()).toBe('pk-notary@3');
    await key('End');
    expect(activeValue()).toBe('pk-lease@5');
    await key('Escape');
    expect(pickerInput('pick').getAttribute('aria-expanded')).toBe('false');
    expect(onChange).not.toHaveBeenCalled();

    await key('ArrowDown');
    await key('ArrowDown');
    await key('Enter');
    expect(onChange).toHaveBeenCalledWith('pk-tax@1');
    expect(pickerInput('pick').getAttribute('aria-expanded')).toBe('false');
  });

  it('should filter as the operator types, over label, value, detail and keywords', async () => {
    render(<Harness />);
    await act(async () => {
      fireEvent.change(pickerInput('pick'), { target: { value: 'revenue' } });
    });
    const listbox = screen.getByTestId('pick-listbox');
    expect(
      [...listbox.querySelectorAll('[role="option"]')].map((row) => row.getAttribute('data-value')),
    ).toEqual(['pk-tax@1']);
    // Typing is searching, not choosing: nothing was submitted.
    expect(pickedValue('pick')).toBe('');
    await key('ArrowDown');
    await key('Enter');
    expect(pickedValue('pick')).toBe('pk-tax@1');
  });

  it('should say when nothing matches what was typed', async () => {
    render(<Harness />);
    await act(async () => {
      fireEvent.change(pickerInput('pick'), { target: { value: 'zzz' } });
    });
    expect(screen.getByTestId('pick-listbox')).toHaveTextContent('Nothing in the list matches');
  });

  it('should restore the chosen label when Escape abandons a search', async () => {
    render(<Harness initial="pk-lease@5" />);
    expect(pickerInput('pick').value).toBe('lease @ 5');
    await act(async () => {
      fireEvent.change(pickerInput('pick'), { target: { value: 'not' } });
    });
    await key('Escape');
    expect(pickerInput('pick').value).toBe('lease @ 5');
    expect(pickedValue('pick')).toBe('pk-lease@5');
  });

  it('should show a restored value the list no longer carries, rather than clearing it', () => {
    render(<Harness initial="pk-deleted@9" />);
    expect(pickerInput('pick').value).toBe('pk-deleted@9');
    expect(screen.getByTestId('pick-status')).toHaveTextContent('not in the current list');
  });

  it('should render loading as loading - disabled, and saying what it is reading', () => {
    render(<Harness listing={{ kind: 'loading' }} />);
    expect(pickerInput('pick')).toBeDisabled();
    expect(screen.getByTestId('pick-status')).toHaveTextContent('Reading the pack registry');
  });

  it('should render an empty list as empty, saying what to do', () => {
    render(<Harness listing={{ kind: 'ready', options: [] }} />);
    expect(pickerInput('pick')).toBeDisabled();
    expect(screen.getByTestId('pick-status')).toHaveTextContent('No pack is registered.');
  });

  it('should render a failure by naming what could not be read, never as an empty list', () => {
    render(
      <Harness
        listing={{
          kind: 'failed',
          failure: { kind: 'transport', status: 0, cause: 'connection refused' },
        }}
      />,
    );
    expect(pickerInput('pick')).toBeDisabled();
    expect(document.body.textContent).toMatch(/pack registry is unavailable/);
    expect(document.body.textContent).not.toContain('No pack is registered.');
  });

  it('should carry the ARIA combobox contract', async () => {
    render(<Harness />);
    const input = pickerInput('pick');
    expect(input.getAttribute('role')).toBe('combobox');
    expect(screen.getByLabelText('Pack')).toBe(input);
    expect(input.getAttribute('aria-expanded')).toBe('false');
    const listbox = await openPicker('pick');
    expect(listbox.getAttribute('role')).toBe('listbox');
    expect(input.getAttribute('aria-controls')).toBe(listbox.id);
  });

  describe('with a typed escape hatch', () => {
    const reason = 'the walk is bounded, so a deeper directory is not listed';

    it('should submit what is typed, say it was typed, and show the reason', async () => {
      const onChange = vi.fn();
      render(<Harness onChange={onChange} freeText={{ reason }} />);
      await act(async () => {
        fireEvent.change(pickerInput('pick'), { target: { value: '/data/elsewhere' } });
      });
      expect(onChange).toHaveBeenLastCalledWith('/data/elsewhere');
      expect(pickedValue('pick')).toBe('/data/elsewhere');
      expect(pickerInput('pick').getAttribute('data-picker')).toBe('free');
      expect(pickerInput('pick').getAttribute('data-free-text-reason')).toBe(reason);
      expect(screen.getByTestId('pick-free-text')).toHaveTextContent(reason);
    });

    it('should still choose from the list, submitting the value', async () => {
      const onChange = vi.fn();
      render(<Harness onChange={onChange} freeText={{ reason }} />);
      await choose('pick', 'pk-lease@5');
      expect(onChange).toHaveBeenLastCalledWith('pk-lease@5');
    });

    it('should stay usable when the list could not be read', () => {
      render(
        <Harness
          freeText={{ reason }}
          listing={{ kind: 'failed', failure: { kind: 'transport', status: 0, cause: 'x' } }}
        />,
      );
      expect(pickerInput('pick')).not.toBeDisabled();
    });
  });
});

describe('[Picker] reads as a dropdown (operator, 2026-10-31)', () => {
  it('should carry a down chevron on the closed field, turned up while open', async () => {
    render(<Harness />);
    const field = pickerInput('pick').parentElement as HTMLElement;
    const chevron = field.querySelector('[data-testid="dropdown-chevron"]');
    expect(chevron).not.toBeNull();
    expect(chevron).not.toHaveClass('rotate-180');
    await openPicker('pick');
    expect(field.querySelector('[data-testid="dropdown-chevron"]')).toHaveClass('rotate-180');
  });

  it('should fill and tick the chosen row, and only that row', async () => {
    render(<Harness initial="pk-tax@1" />);
    const list = await openPicker('pick');
    const rows = [...list.querySelectorAll('[role="option"]')];
    const chosen = rows.filter((row) => row.getAttribute('aria-selected') === 'true');
    expect(chosen).toHaveLength(1);
    expect(chosen[0]).toHaveAttribute('data-value', 'pk-tax@1');
    expect(chosen[0]).toHaveClass('bg-text/15');
    expect(chosen[0]?.querySelector('svg')).not.toBeNull();
    for (const row of rows.filter((r) => r !== chosen[0])) {
      expect(row).not.toHaveClass('bg-text/15');
    }
  });

  it('should sit on the card surface, a step off the page, with room for the chevron', () => {
    render(<Harness />);
    expect(pickerInput('pick')).toHaveClass('bg-card', 'pr-9', 'cursor-pointer');
    expect(pickerInput('pick')).not.toHaveClass('bg-panel');
  });
});

describe('[matches]', () => {
  it('should require every typed word, in any order and any case', () => {
    const option: PickerOption = { value: 'GPU-247a', label: 'RTX 3090', detail: '20.0 GiB free' };
    expect(matches(option, 'rtx free')).toBe(true);
    expect(matches(option, '247A')).toBe(true);
    expect(matches(option, 'rtx 3080')).toBe(false);
  });
});

describe('[ranked]', () => {
  it('should put a label match above a row that only mentions the word in its detail', () => {
    const options: readonly PickerOption[] = [
      { value: 'adapter', label: 'adapter', detail: 'leaves the base byte-unchanged' },
      { value: 'change', label: 'change', detail: 'changes every parameter' },
      { value: 'create', label: 'create', detail: 'instantiated from a config' },
    ];
    expect(ranked(options, 'change').map((option) => option.value)).toEqual(['change', 'adapter']);
    expect(ranked(options, '').map((option) => option.value)).toEqual([
      'adapter',
      'change',
      'create',
    ]);
  });
});
