/**
 * "Show archived" (operator decision 2026-10-20). Archived runs and parts are hidden from the
 * default lists and pickers; this one checkbox brings them back. Shared so the Runs screen,
 * the part index and the slot pickers say it the same way.
 */
import type { ReactNode } from 'react';

export interface ShowArchivedToggleProps {
  readonly checked: boolean;
  readonly onChange: (next: boolean) => void;
  /** What is archived - "runs", "parts". */
  readonly noun: string;
  readonly 'data-testid'?: string;
}

export function ShowArchivedToggle({
  checked,
  onChange,
  noun,
  ...rest
}: ShowArchivedToggleProps): ReactNode {
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-hint text-dim">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        data-testid={rest['data-testid']}
      />
      {`Show archived ${noun}`}
    </label>
  );
}
