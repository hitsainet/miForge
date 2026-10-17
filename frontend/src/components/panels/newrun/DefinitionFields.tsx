/**
 * The remaining fields of R-02.53's definition (FR-009.33, .39).
 *
 * **Why several components in one file.** F9's FTASKS lists these as separate modules
 * (`ModelSourceField.tsx`, `OptimizerFields.tsx`, …). They are grouped here because each is a
 * handful of labelled inputs with no logic, every one shares the same two rules, and the two
 * components that *do* carry a rule a mutation targets - `RunTypeField` (M22) and `PackField`
 * (F7's OQ-9) - are in their own files with their own tests, as is `BudgetPreview` (M20) and
 * `RefusalBanner` (M21). A file per text input would spread one rule across seven modules,
 * which is how a rule comes to be followed in six of them. Recorded as a deviation.
 *
 * **The two rules, which are the same rule from both sides:**
 *
 * 1. **No field carries an initial value.** `ModelSourceField` does not preselect the mode
 *    the registry marks `default: true`; `OptimizerFields` does not preselect `8bit`; no
 *    cadence starts at a number. A non-nullable field with an initial value *is* a default,
 *    and ADR-018 says the run log would record it as the operator's choice. **M19.**
 * 2. **No field validates.** No required marker that blocks, no range check, no "at least one
 *    stop condition". The refusals are the backend's copy.
 *
 * An unavailable option is **shown, named and disabled with its reason** - never hidden and
 * never swapped. *An option missing from a response is indistinguishable from one that is
 * broken* (ADR-013), and the same is true of one missing from a form.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { Field, Input, Select } from '@/components/common/Field';
import { Button } from '@/components/common/Button';
import { Card } from '@/components/common/Card';
import { AbsentValue } from '@/components/common/AbsentValue';
import { FailureNotice } from '@/components/common/FailureNotice';
import { fetchModelSources, type ModelSourceMode } from '@/api/modelSources';
import { fetchSignalSources, type SignalSourceRow } from '@/api/signalSources';
import type { ApiFailure } from '@/api/errors';
import { DOCUMENTED_STOP_CONDITIONS, type StopConditionEntry } from './draft';

/* ------------------------------------------------------------------------------------ *
 * Model source - R-02.46's three ways
 * ------------------------------------------------------------------------------------ */

export interface ModelSourceFieldProps {
  readonly mode: string;
  readonly path: string;
  readonly repo: string;
  readonly revision: string;
  readonly configPath: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
  readonly load?: typeof fetchModelSources;
}

export function ModelSourceField({
  mode,
  path,
  repo,
  revision,
  configPath,
  onChange,
  load = fetchModelSources,
}: ModelSourceFieldProps): ReactNode {
  const [modes, setModes] = useState<readonly ModelSourceMode[] | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  useEffect(() => {
    let cancelled = false;
    void load().then((result) => {
      if (cancelled) return;
      if (result.ok) setModes(result.value.modes);
      else setFailure(result.failure);
    });
    return () => {
      cancelled = true;
    };
  }, [load]);

  if (failure !== null) {
    return <FailureNotice failure={failure} surface="the acquisition modes" />;
  }

  return (
    <>
      <Field
        label="Model source"
        hint="Load a directory, load a repository and revision, or create a model from an architecture config. From the live registry, which lists every mode and names each unavailable one."
      >
        {({ id, describedBy }) =>
          modes === null ? (
            <div className="text-hint text-dim">Reading the acquisition modes…</div>
          ) : (
            <>
              <Select
                id={id}
                aria-describedby={describedBy}
                value={mode}
                onChange={(event) => onChange({ mode: event.target.value })}
                placeholder="Choose how the model comes to be in this process"
                options={modes.map((row) => ({
                  value: row.mode,
                  label: row.available ? row.mode : `${row.mode} — unavailable`,
                }))}
                data-testid="mode-select"
              />
              <ul className="mt-2 space-y-0.5 text-hint text-dim" data-testid="mode-reasons">
                {modes
                  .filter((row) => !row.available)
                  .map((row) => (
                    <li key={row.mode} className="text-amber">
                      <span className="font-mono text-ident">{row.mode}</span> is unavailable:{' '}
                      {row.message ?? row.reason ?? 'the response gave no reason'}
                      {row.capabilities_unavailable.length === 0
                        ? null
                        : ` — ${row.capabilities_unavailable.join(', ')} would need it.`}
                    </li>
                  ))}
              </ul>
            </>
          )
        }
      </Field>

      <Field label="Path" hint="For mode load: a checkpoint or model directory on this host.">
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            value={path}
            onChange={(event) => onChange({ path: event.target.value })}
            data-testid="path-input"
          />
        )}
      </Field>

      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-2">
        <Field label="Repository" hint="For mode load from a hub: the repository id.">
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={repo}
              onChange={(event) => onChange({ repo: event.target.value })}
              data-testid="repo-input"
            />
          )}
        </Field>
        <Field label="Revision" hint="The exact revision. A moving reference is not a pin.">
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={revision}
              onChange={(event) => onChange({ revision: event.target.value })}
              data-testid="revision-input"
            />
          )}
        </Field>
      </div>

      <Field
        label="Architecture config path"
        hint="For mode create: the architecture config a model is instantiated from, randomly initialised."
      >
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            value={configPath}
            onChange={(event) => onChange({ configPath: event.target.value })}
            data-testid="config-path-input"
          />
        )}
      </Field>
    </>
  );
}

/* ------------------------------------------------------------------------------------ *
 * Signal source - R-02.4-.7
 * ------------------------------------------------------------------------------------ */

export interface SignalSourceFieldProps {
  readonly value: string;
  readonly onChange: (value: string) => void;
  readonly load?: typeof fetchSignalSources;
}

export function SignalSourceField({
  value,
  onChange,
  load = fetchSignalSources,
}: SignalSourceFieldProps): ReactNode {
  const [rows, setRows] = useState<readonly SignalSourceRow[] | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  useEffect(() => {
    let cancelled = false;
    void load().then((result) => {
      if (cancelled) return;
      if (result.ok) setRows(result.value.sources);
      else setFailure(result.failure);
    });
    return () => {
      cancelled = true;
    };
  }, [load]);

  if (failure !== null) {
    return <FailureNotice failure={failure} surface="the signal sources" />;
  }

  const selected = (rows ?? []).find((row) => row.name === value) ?? null;

  return (
    <Field
      label="Signal source"
      hint="Where the run's training signal comes from. From the live registry; availability is recomputed per call, so configuring a provider is visible without a restart."
    >
      {({ id, describedBy }) =>
        rows === null ? (
          <div className="text-hint text-dim">Reading the signal sources…</div>
        ) : (
          <>
            <Select
              id={id}
              aria-describedby={describedBy}
              value={value}
              onChange={(event) => onChange(event.target.value)}
              placeholder="Choose a signal source"
              options={rows.map((row) => ({
                value: row.name,
                label: row.available ? row.name : `${row.name} — unavailable`,
              }))}
              data-testid="signal-source-select"
            />
            <ul className="mt-2 space-y-0.5 text-hint text-dim" data-testid="signal-source-notes">
              {rows.map((row) => (
                <li key={row.name} className={row.available ? '' : 'text-amber'}>
                  <span className="font-mono text-ident">{row.name}</span>: {row.summary}
                  {/* Named by its service, never hidden and never substituted (ADR-010). */}
                  {row.available ? null : ` — unavailable: ${row.unavailable_reason}`}
                </li>
              ))}
            </ul>
            {selected === null ? null : (
              <div className="mt-2 text-hint text-dim" data-testid="signal-source-requires">
                This source needs: {selected.requires.part_kind ?? 'no part kind'}
                {selected.requires.selector ? ', a selector configuration' : ''}
                {selected.requires.grader ? ', a grader configuration' : ''}
                {selected.requires.generation_provider ? ', a generation provider' : ''}. Its
                objective is {selected.objective ?? 'not recorded'}.
              </div>
            )}
          </>
        )
      }
    </Field>
  );
}

/* ------------------------------------------------------------------------------------ *
 * Optimizer and batch configuration - R-02.13
 * ------------------------------------------------------------------------------------ */

export interface OptimizerFieldsProps {
  readonly learningRate: string;
  readonly warmupSteps: string;
  readonly batchSize: string;
  readonly gradientAccumulation: string;
  readonly optimizerStatePrecision: string;
  readonly seed: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
}

export function OptimizerFields(props: OptimizerFieldsProps): ReactNode {
  const { onChange } = props;
  const rows: readonly {
    readonly key: keyof OptimizerFieldsProps;
    readonly label: string;
    readonly hint: string;
  }[] = [
    {
      key: 'learningRate',
      label: 'Learning rate',
      hint: 'R-02.13. No initial value: a learning rate the operator did not choose would be recorded in the run log as theirs.',
    },
    { key: 'warmupSteps', label: 'Warmup steps', hint: 'In steps.' },
    { key: 'batchSize', label: 'Batch size', hint: 'In examples per step.' },
    {
      key: 'gradientAccumulation',
      label: 'Gradient accumulation',
      hint: 'In micro-batches per step. The backend defaults this when absent; this form does not.',
    },
    {
      key: 'seed',
      label: 'Seed',
      hint: 'Recorded so a create run can state what it started from.',
    },
  ];

  return (
    <>
      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-2">
        {rows.map((row) => (
          <Field key={row.key} label={row.label} hint={row.hint}>
            {({ id, describedBy }) => (
              <Input
                id={id}
                aria-describedby={describedBy}
                inputMode="decimal"
                value={props[row.key] as string}
                onChange={(event) => onChange({ [row.key]: event.target.value })}
                data-testid={`optimizer-${row.key}`}
              />
            )}
          </Field>
        ))}
      </div>
      <Field
        label="Optimizer state precision"
        hint="8-bit optimizer states halve the state footprint. The backend defaults this when absent; this form does not preselect it."
      >
        {({ id, describedBy }) => (
          <Select
            id={id}
            aria-describedby={describedBy}
            value={props.optimizerStatePrecision}
            onChange={(event) => onChange({ optimizerStatePrecision: event.target.value })}
            placeholder="Choose a precision"
            options={[
              { value: '8bit', label: '8bit' },
              { value: '32bit', label: '32bit' },
            ]}
            data-testid="optimizer-precision"
          />
        )}
      </Field>
    </>
  );
}

/* ------------------------------------------------------------------------------------ *
 * Cadences, the floor, and the save cadence
 * ------------------------------------------------------------------------------------ */

export interface CadenceFieldsProps {
  readonly testEvery: string;
  readonly retentionEvery: string;
  readonly saveEvery: string;
  readonly retentionFloor: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
  /** The pack's floor, when a pack is chosen. Shown and **said to be the pack's**. */
  readonly packFloor?: number | null;
}

export function CadenceFields({
  testEvery,
  retentionEvery,
  saveEvery,
  retentionFloor,
  onChange,
  packFloor,
}: CadenceFieldsProps): ReactNode {
  return (
    <>
      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-3">
        <Field
          label="Test every"
          hint="In steps. A coarser cadence means fewer points on the curve."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="numeric"
              value={testEvery}
              onChange={(event) => onChange({ testEvery: event.target.value })}
              data-testid="cadence-test"
            />
          )}
        </Field>
        <Field
          label="Retention every"
          hint="In steps. With a cadence of N, up to N−1 steps can pass between a breach happening and being seen."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="numeric"
              value={retentionEvery}
              onChange={(event) => onChange({ retentionEvery: event.target.value })}
              data-testid="cadence-retention"
            />
          )}
        </Field>
        <Field
          label="Save every"
          hint="In steps. This is also the blast radius of an interruption: a crash costs up to one save interval."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="numeric"
              value={saveEvery}
              onChange={(event) => onChange({ saveEvery: event.target.value })}
              data-testid="cadence-save"
            />
          )}
        </Field>
      </div>

      <Field
        label="Retention floor"
        hint={
          packFloor === undefined || packFloor === null
            ? 'The pack carries the canonical floor. Whether a run may override it is F7’s open question and is not decided here.'
            : `The pack’s floor is ${packFloor.toFixed(3)}. It is the pack’s value, not this form’s. Whether a run may override it is F7’s open question and is not decided here.`
        }
      >
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            inputMode="decimal"
            value={retentionFloor}
            /* Deliberately NOT prefilled with the pack's floor. Showing the pack's value is
               information; putting it in the input is this form choosing an override the
               operator did not ask for - and F7's OQ-8 has not said an override is allowed. */
            onChange={(event) => onChange({ retentionFloor: event.target.value })}
            data-testid="retention-floor"
          />
        )}
      </Field>
    </>
  );
}

/* ------------------------------------------------------------------------------------ *
 * Stop conditions - at least one, and the form does not enforce that
 * ------------------------------------------------------------------------------------ */

export interface StopConditionsFieldProps {
  readonly entries: readonly StopConditionEntry[];
  readonly onChange: (entries: readonly StopConditionEntry[]) => void;
  /** `details.available` from a `run_has_no_stop_condition` refusal, when one has arrived. */
  readonly available?: readonly string[];
}

export function StopConditionsField({
  entries,
  onChange,
  available,
}: StopConditionsFieldProps): ReactNode {
  const options = available ?? DOCUMENTED_STOP_CONDITIONS;
  const fromRefusal = available !== undefined;

  return (
    <Card className="mb-4" data-testid="stop-conditions">
      <div className="mb-1 text-heading">Stop conditions</div>
      <div className="mb-2 text-hint text-dim">
        R-02.53 requires <strong>at least one</strong>. This form does not enforce that: a
        definition with none is submitted and refused{' '}
        <code className="font-mono">run_has_no_stop_condition</code> listing the available set,
        which is the only behaviour that keeps the refusal the backend&apos;s.
        {fromRefusal ? (
          <> The set below is the one that refusal named.</>
        ) : (
          <>
            {' '}
            The set below is F6&apos;s documented precedence order —{' '}
            <em>no endpoint publishes it</em>, unlike run types, so it is named as documentation
            rather than presented as a live registry.
          </>
        )}
      </div>

      {entries.length === 0 ? (
        <AbsentValue
          reason="missing_required_key"
          variant="block"
          detail="no stop condition has been added"
        />
      ) : (
        <ul className="space-y-2">
          {entries.map((entry, index) => (
            <li key={index} className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_1fr_auto]">
              <Select
                value={entry.condition}
                onChange={(event) =>
                  onChange(
                    entries.map((row, at) =>
                      at === index ? { ...row, condition: event.target.value } : row,
                    ),
                  )
                }
                placeholder="Choose a condition"
                options={options.map((name) => ({ value: name, label: name }))}
                data-testid={`stop-condition-${index}`}
              />
              <Input
                value={entry.value}
                placeholder="threshold, count or window"
                onChange={(event) =>
                  onChange(
                    entries.map((row, at) =>
                      at === index ? { ...row, value: event.target.value } : row,
                    ),
                  )
                }
                data-testid={`stop-value-${index}`}
              />
              <Button onClick={() => onChange(entries.filter((_, at) => at !== index))}>
                Remove this condition
              </Button>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-3">
        <Button onClick={() => onChange([...entries, { condition: '', value: '' }])}>
          Add a stop condition
        </Button>
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------------------------ *
 * The adapter block - only for an adapter run
 * ------------------------------------------------------------------------------------ */

export interface AdapterBlockFieldsProps {
  readonly rank: string;
  readonly alpha: string;
  readonly targetModules: string;
  readonly dropout: string;
  readonly onChange: (patch: Readonly<Record<string, string>>) => void;
}

export function AdapterBlockFields({
  rank,
  alpha,
  targetModules,
  dropout,
  onChange,
}: AdapterBlockFieldsProps): ReactNode {
  return (
    <Card className="mb-4" data-testid="adapter-block">
      <div className="mb-1 text-heading">Adapter</div>
      <div className="mb-2 text-hint text-dim">
        Bounded by <code className="font-mono text-ident">adapter-manifest-v1.json</code>&apos;s{' '}
        <code className="font-mono text-ident">$defs.AdapterConfig</code>, whose required set is
        rank, alpha and target modules. Scaling is alpha over rank and is not an operator choice: an
        unrecognised rule must be refused rather than guessed, because guessing the scale changes
        the served model silently.
      </div>
      <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-2">
        <Field label="Rank" hint="The adapter's inner dimension.">
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="numeric"
              value={rank}
              onChange={(event) => onChange({ adapterRank: event.target.value })}
              data-testid="adapter-rank"
            />
          )}
        </Field>
        <Field label="Alpha" hint="Scaling is alpha over rank.">
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              inputMode="decimal"
              value={alpha}
              onChange={(event) => onChange({ adapterAlpha: event.target.value })}
              data-testid="adapter-alpha"
            />
          )}
        </Field>
      </div>
      <Field label="Target modules" hint="Comma-separated. PEFT's own module names.">
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            value={targetModules}
            onChange={(event) => onChange({ adapterTargetModules: event.target.value })}
            data-testid="adapter-target-modules"
          />
        )}
      </Field>
      <Field label="Dropout" hint="Optional in the schema. Left empty means absent, not zero.">
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            inputMode="decimal"
            value={dropout}
            onChange={(event) => onChange({ adapterDropout: event.target.value })}
            data-testid="adapter-dropout"
          />
        )}
      </Field>
    </Card>
  );
}
