/**
 * The run type, **from the live registry** (FR-009.33; M22).
 *
 * `GET /api/v1/run-types` answers from `training.runtypes.known()`, which F8 built to be a
 * reachability target: *removing a `register(...)` line must shorten this body and turn the
 * suite red.* So this field reads it.
 *
 * **Never a literal `['change', 'create', 'adapter']`.** A hard-coded list is a second
 * registry: a fourth registered type would not appear, and a type whose registration was
 * removed would still be offered. That is mutation **M22**, and the test asserts the options
 * are the response's rather than any particular three values - which is the only form of the
 * assertion that still holds when a fourth type lands.
 *
 * *Worth knowing while reading `models/run.py`:* that module still exports a `RUN_TYPES`
 * frozenset of `{create, full, adapter, merge}`, which **disagrees with the live registry**
 * (`adapter, change, create`) - `full` was renamed `change` and `merge` is an operation, not a
 * run. Nothing imports it. Reported; this field reads the registry, not the constant.
 */
import type { ReactNode } from 'react';
import { Field } from '@/components/common/Field';
import { Picker, type PickerOption } from '@/components/common/Picker';
import { useListing } from '@/hooks/useListing';
import { fetchRunTypes, type RunTypeListing } from '@/api/runTypes';

export interface RunTypeFieldProps {
  readonly value: string;
  readonly onChange: (value: string) => void;
  /** Injected in tests. A field that can only be exercised against a live server is untested. */
  readonly load?: typeof fetchRunTypes;
}

/** The registry's rows as options. The registry's own sentence is the detail. */
export function runTypeOptions(listing: RunTypeListing): readonly PickerOption[] {
  return listing.run_types.map((entry) => ({
    value: entry.name,
    label: entry.name,
    detail: `changes ${entry.changes}`,
  }));
}

export function RunTypeField({
  value,
  onChange,
  load = fetchRunTypes,
}: RunTypeFieldProps): ReactNode {
  const { value: registry, listing } = useListing(load, runTypeOptions);

  return (
    <Field label="Run type" hint="What the run trains. Each type below says what it changes.">
      {({ id, describedBy }) => (
        <>
          <Picker
            id={id}
            describedBy={describedBy}
            /* No preselection. A non-nullable field with an initial value IS a default
               (FR-009.39, M19), and the registry marks none anyway. */
            value={value}
            onChange={onChange}
            listing={listing}
            placeholder="Choose a run type"
            empty="No run types are available."
            surface="the run-type list"
            data-testid="run-type-select"
          />
          {registry === null ? null : (
            <ul className="mt-2 stack-y-0.5 text-hint text-dim" data-testid="run-type-changes">
              {registry.run_types.map((entry) => (
                <li key={entry.name}>
                  {/* The registry's own sentence per type. This screen writes no copy for
                      them, so a renamed or re-described type needs no edit here. */}
                  <span className="font-mono text-ident">{entry.name}</span> changes {entry.changes}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </Field>
  );
}
