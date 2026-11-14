/**
 * Driving a `Picker` from a test the way an operator does: open it, read what it lists,
 * choose a row (task 14.0).
 *
 * Kept beside the component rather than in each spec, so every test chooses through the same
 * events the browser sends - a `mousedown` on the row, because a `click` lands after the
 * input's blur has closed the list - and a spec cannot pass by setting the input's value,
 * which for a strict picker is the search text, not the choice.
 */
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';

/** The combobox input a `Picker` renders under `data-testid={testId}`. */
export function pickerInput(testId: string): HTMLInputElement {
  return screen.getByTestId(testId) as HTMLInputElement;
}

/** Open the list (as a click does) and return its listbox. */
export async function openPicker(testId: string): Promise<HTMLElement> {
  const input = await waitFor(() => {
    const node = pickerInput(testId);
    if (node.disabled) throw new Error(`${testId} is not usable yet`);
    return node;
  });
  if (input.getAttribute('aria-expanded') !== 'true') {
    await act(async () => {
      fireEvent.click(input);
    });
  }
  return screen.getByTestId(`${testId}-listbox`);
}

/** Every listed value, in order, including disabled rows. Closes the list afterwards. */
export async function pickerValues(testId: string): Promise<string[]> {
  const listbox = await openPicker(testId);
  const values = within(listbox)
    .queryAllByRole('option')
    .map((row) => row.getAttribute('data-value') ?? '');
  await act(async () => {
    fireEvent.keyDown(pickerInput(testId), { key: 'Escape' });
  });
  return values;
}

/** Choose the row whose submitted value is `value`. */
export async function choose(testId: string, value: string): Promise<void> {
  const listbox = await openPicker(testId);
  const row = within(listbox)
    .queryAllByRole('option')
    .find((option) => option.getAttribute('data-value') === value);
  if (row === undefined) throw new Error(`${testId} lists no option with value ${value}`);
  await act(async () => {
    fireEvent.mouseDown(row);
  });
}

/** The value a picker would submit - not its label, which is what the input shows. */
export function pickedValue(testId: string): string {
  return pickerInput(testId).getAttribute('data-value') ?? '';
}
