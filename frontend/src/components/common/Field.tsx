/**
 * Form grammar: a labelled field, an input and a select (foundation task 7.3; ADR-012).
 *
 * > Inputs and buttons 8 px radius. Hints under fields explain a constraint or a
 * > consequence in one sentence.
 *
 * **What this fixes from the mockup.** Its `Field` renders the label as a `div` and the
 * control as a sibling, with no `htmlFor`, no `id` and no `aria-describedby` - so no
 * label is associated with any control and no hint is announced. FR-009.7 requires every
 * screen and every action to be operable by keyboard, and an unlabelled select is
 * operable but unidentifiable. `useId` generates the pair here so a call site cannot
 * forget it.
 *
 * **No field has a default value** - that rule is F9's FR-009.39 and ADR-018's, and it is
 * about the *form*, not about this component. What this component does is make the
 * absence renderable: `placeholder` says what a value would look like, and nothing is
 * pre-selected.
 */

import { useId, type ReactNode, type SelectHTMLAttributes, type InputHTMLAttributes } from 'react';

export interface FieldProps {
  readonly label: string;
  /** One sentence: a constraint or a consequence. Announced via `aria-describedby`. */
  readonly hint?: string;
  /** Rendered in `red` beneath the hint, and announced. For a backend refusal's message. */
  readonly error?: string;
  /** Receives the generated ids to spread onto the control. */
  readonly children: (ids: {
    readonly id: string;
    readonly describedBy: string | undefined;
  }) => ReactNode;
  readonly className?: string;
}

export function Field({ label, hint, error, children, className = '' }: FieldProps): ReactNode {
  const base = useId();
  const id = `${base}-control`;
  const hintId = hint === undefined ? undefined : `${base}-hint`;
  const errorId = error === undefined ? undefined : `${base}-error`;
  const describedBy =
    [hintId, errorId].filter((value) => value !== undefined).join(' ') || undefined;

  return (
    <div className={`mb-4 ${className}`.trim()}>
      <label htmlFor={id} className="mb-1.5 block text-body font-medium">
        {label}
      </label>
      {children({ id, describedBy })}
      {hint === undefined ? null : (
        <div id={hintId} className="mt-1 text-hint text-dim">
          {hint}
        </div>
      )}
      {error === undefined ? null : (
        <div id={errorId} role="alert" className="mt-1 text-hint text-red">
          {error}
        </div>
      )}
    </div>
  );
}

/** Shared control chrome. `line` border, `panel` background, 8 px radius. */
export const CONTROL_CLASSES =
  'w-full rounded-control border border-line bg-panel px-3 py-2.5 text-body text-text ' +
  'placeholder:text-dim';

export function Input(props: InputHTMLAttributes<HTMLInputElement>): ReactNode {
  const { className = '', ...rest } = props;
  return <input className={`${CONTROL_CLASSES} ${className}`.trim()} {...rest} />;
}

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  readonly options: readonly { readonly value: string; readonly label: string }[];
  /**
   * The copy for "nothing chosen". Required, because a select with no placeholder option
   * silently selects its first entry, and the operator then owns a choice they never
   * made - the defect F9's FR-009.39 is about, one layer down.
   */
  readonly placeholder: string;
}

export function Select({
  options,
  placeholder,
  className = '',
  value,
  ...rest
}: SelectProps): ReactNode {
  // `defaultValue` only when the caller is NOT controlling the value. Setting both makes
  // React warn and ignore one of them, and the one it ignores is the controlled value - so a
  // field reading "choose a run type" while its state held `adapter` was the actual symptom.
  // Found by RunTypeField's test, which is the first controlled Select in the codebase.
  const controlled = value !== undefined;
  return (
    <select
      className={`${CONTROL_CLASSES} ${className}`.trim()}
      {...(controlled ? { value } : { defaultValue: '' })}
      {...rest}
    >
      <option value="" disabled>
        {placeholder}
      </option>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}
