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
import { useEffect, useState, type ReactNode } from 'react';
import { Field, Select } from '@/components/common/Field';
import { AbsentValue } from '@/components/common/AbsentValue';
import { FailureNotice } from '@/components/common/FailureNotice';
import { fetchRunTypes, type RunTypeEntry } from '@/api/runTypes';
import type { ApiFailure } from '@/api/errors';

export interface RunTypeFieldProps {
  readonly value: string;
  readonly onChange: (value: string) => void;
  /** Injected in tests. A field that can only be exercised against a live server is untested. */
  readonly load?: typeof fetchRunTypes;
}

export function RunTypeField({
  value,
  onChange,
  load = fetchRunTypes,
}: RunTypeFieldProps): ReactNode {
  const [entries, setEntries] = useState<readonly RunTypeEntry[] | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  useEffect(() => {
    let cancelled = false;
    void load().then((result) => {
      if (cancelled) return;
      if (result.ok) setEntries(result.value.run_types);
      else setFailure(result.failure);
    });
    return () => {
      cancelled = true;
    };
  }, [load]);

  if (failure !== null) {
    return <FailureNotice failure={failure} surface="the run-type registry" />;
  }

  return (
    <Field
      label="Run type"
      hint="From the live registry. A type registered in the backend appears here with no change to this screen."
    >
      {({ id, describedBy }) =>
        entries === null ? (
          <div className="text-hint text-dim">Reading the registry…</div>
        ) : entries.length === 0 ? (
          <AbsentValue
            reason="missing_required_key"
            variant="block"
            detail="the registry returned no run types, so there is nothing to choose"
          />
        ) : (
          <>
            <Select
              id={id}
              aria-describedby={describedBy}
              /* No preselection. A non-nullable field with an initial value IS a default
                 (FR-009.39, M19), and the registry marks none anyway. */
              value={value}
              onChange={(event) => onChange(event.target.value)}
              placeholder="Choose a run type"
              options={entries.map((entry) => ({ value: entry.name, label: entry.name }))}
              data-testid="run-type-select"
            />
            <ul className="mt-2 space-y-0.5 text-hint text-dim" data-testid="run-type-changes">
              {entries.map((entry) => (
                <li key={entry.name}>
                  {/* The registry's own sentence per type. This screen writes no copy for
                      them, so a renamed or re-described type needs no edit here. */}
                  <span className="font-mono text-ident">{entry.name}</span> changes {entry.changes}
                </li>
              ))}
            </ul>
          </>
        )
      }
    </Field>
  );
}
