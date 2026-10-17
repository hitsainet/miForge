/**
 * R-02.53's screen. **It collects, submits one body, and renders the refusal** (FR-009.33-.39).
 *
 * R-02.53 is **F6's requirement**; this screen is F9's. F6 defines the run definition,
 * validates it and refuses naming what failed; F9 renders the definition, submits it, and
 * renders the refusal. **This form defines no validation and no default** - every value on it
 * is something the operator chose or something another feature reported.
 *
 * ## One body, one POST, nothing staged
 *
 * `buildBody` produces the whole definition and `submitRun` sends it to `POST /api/v1/runs` -
 * F1's route, extended by F6. Nothing is staged, chunked, pre-committed or persisted:
 * R-02.53's guarantee is that *the whole definition* is validated before anything starts, and
 * a form submitting in pieces breaks exactly that. A persisted draft is F7's requirement
 * (R-02.39), not this one.
 *
 * ## The seam, stated on the screen as well as here
 *
 * **This build's `POST /api/v1/runs` accepts F1's `RunSubmit` and is closed to unknown keys**
 * (`extra="forbid"`), carrying the model source and the optimizer configuration and **none**
 * of R-02.53's remaining parts - no `run_type`, no pack, no signal source, no cadences, no
 * stop conditions, no `save_every`, no adapter block. So a complete definition is refused
 * `validation_failed` naming each unexpected field until F6 extends the route (its
 * FR-006.56). That refusal is **rendered against the fields it names**, which is the same
 * behaviour F6's collected refusals will get. The alternative - sending only the subset the
 * route accepts - would discard answers the operator gave, which is the one thing this form
 * may never do.
 *
 * **`POST /api/v1/runs/validate` is not mounted either**, so check-before-commit is reported
 * unavailable by name rather than simulated. A green check would enable nothing anyway: F6
 * re-runs the same validator on the create path and a part can stop resolving in between.
 *
 * ## The copy
 *
 * The primary button reads **"Start run"** - and **"Attach and start" when, and only when,
 * the acquisition mode is `attach`** (F11 task 16.4; PPRD §7 marks the label *"§5.10's
 * screen only"*). It was the mockup's pre-reframe label for every run, and F1's review
 * recorded it as stale for the default path: starting a `load` or `create` run attaches to
 * nothing, so the label would describe an action that does not happen. In `attach` mode it
 * is exactly what the button does. `startLabel` is the one place the rule lives.
 */
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Callout } from '@/components/common/Callout';
import { Button } from '@/components/common/Button';
import { Field, Input } from '@/components/common/Field';
import { MonoId } from '@/components/common/MonoId';
import { FailureNotice } from '@/components/common/FailureNotice';
import { useNavigation } from '@/contexts/NavigationContext';
import { defineRun, validateRun, type ValidationOutcome } from '@/api/runs';
import type { ApiFailure } from '@/api/errors';
import type { BudgetRequest } from '@/api/budget';
import { EMPTY_DRAFT, buildBody, type RunDefinitionDraft } from './draft';

/** The primary button's label: §5.10's wording for the attach mode and for nothing else. */
export function startLabel(mode: string): string {
  return mode === 'attach' ? 'Attach and start' : 'Start run';
}
import { RunTypeField } from './RunTypeField';
import { PackField } from './PackField';
import {
  AdapterBlockFields,
  CadenceFields,
  ModelSourceField,
  OptimizerFields,
  SignalSourceField,
  StopConditionsField,
} from './DefinitionFields';
import { BudgetPreview } from './BudgetPreview';
import {
  RefusalBanner,
  byField,
  decodeRefusals,
  fromOutcome,
  type DecodedRefusal,
} from './RefusalBanner';
import { ResolvedSummary } from './ResolvedSummary';
import { SelectorAvailability } from './SelectorAvailability';

export interface NewRunFormProps {
  /** Injected in tests. The real one is `defineRun` - `POST /api/v1/runs/define`. */
  readonly submit?: typeof defineRun;
  /** Injected in tests. The real one is `validateRun` - `POST /api/v1/runs/validate`. */
  readonly check?: typeof validateRun;
}

export function NewRunForm({
  submit = defineRun,
  check = validateRun,
}: NewRunFormProps = {}): ReactNode {
  const navigation = useNavigation();
  const [draft, setDraft] = useState<RunDefinitionDraft>(EMPTY_DRAFT);
  const [refusals, setRefusals] = useState<readonly DecodedRefusal[]>([]);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [checkFailure, setCheckFailure] = useState<ApiFailure | null>(null);
  const [outcome, setOutcome] = useState<ValidationOutcome | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const patch = useCallback((values: Readonly<Record<string, string>>) => {
    setDraft((current) => ({ ...current, ...values }));
  }, []);

  const body = useMemo(() => buildBody(draft), [draft]);

  /**
   * The budget request, or `null`.
   *
   * `null` while the operator has entered neither a run type nor a batch configuration -
   * because an estimate derived from nothing is a preview of nothing, and F8's own request
   * model is closed to unknown keys for the same reason.
   */
  const budgetRequest = useMemo<BudgetRequest | null>(() => {
    const learningRate = Number(draft.learningRate);
    const warmupSteps = Number(draft.warmupSteps);
    const batchSize = Number(draft.batchSize);
    if (draft.runType.length === 0) return null;
    if (!Number.isFinite(learningRate) || !Number.isFinite(batchSize)) return null;
    if (draft.learningRate.length === 0 || draft.batchSize.length === 0) return null;
    return {
      run_type: draft.runType,
      config: {
        learning_rate: learningRate,
        warmup_steps:
          Number.isFinite(warmupSteps) && draft.warmupSteps.length > 0 ? warmupSteps : 0,
        batch_size: batchSize,
      },
    };
  }, [draft.runType, draft.learningRate, draft.warmupSteps, draft.batchSize]);

  const onSubmit = useCallback(async () => {
    setSubmitting(true);
    const result = await submit(body);
    setSubmitting(false);
    if (result.ok) {
      setRefusals([]);
      setFailure(null);
      setOutcome(null);
      navigation.go('run', { runId: result.value.run_id });
      return;
    }
    // The resolved summary is cleared on a refusal: a green check sitting beside four
    // refusals would read as "this was fine a moment ago", which is exactly the stale-answer
    // problem a cached permission creates.
    setOutcome(null);
    // A 422 keeps the form. Nothing below is cleared and the draft is untouched (M21).
    const decoded = decodeRefusals(result.failure);
    setRefusals(decoded);
    setFailure(decoded.length === 0 ? result.failure : null);
  }, [body, submit, navigation]);

  /**
   * Check before committing, through **F6's own validator** (its FR-006.62).
   *
   * It answers **200 either way** - `ok: false` with the complete failure list is a
   * successful answer to a successful question - so a non-ok body is read from the body and
   * rendered beside the fields, not treated as a transport failure.
   *
   * **A green check enables nothing.** `/define` re-runs the same validator, and a part can
   * stop resolving or free VRAM change in between, so nothing here is cached as permission
   * and nothing on this form is disabled on the strength of it.
   */
  const onCheck = useCallback(async () => {
    const result = await check(body);
    if (!result.ok) {
      setCheckFailure(result.failure);
      setOutcome(null);
      return;
    }
    setCheckFailure(null);
    setOutcome(result.value);
    setRefusals(fromOutcome(result.value as unknown as Readonly<Record<string, unknown>>));
  }, [body, check]);

  const grouped = byField(refusals);
  /** `details.available`, when a `run_has_no_stop_condition` refusal named the set. */
  const availableStopConditions = useMemo(() => {
    const refusal = refusals.find((row) => row.code === 'run_has_no_stop_condition');
    const available = refusal?.details['available'];
    return Array.isArray(available) ? available.map((row) => String(row)) : undefined;
  }, [refusals]);

  return (
    <>
      <PageHead
        title="New run"
        subtitle="One definition: the model, the pack, the signal, the configuration, the cadences and at least one stop condition"
      />

      <Callout tone="info" title="What this form does and does not do" className="mb-4">
        It collects the definition and submits it whole to{' '}
        <MonoId value="POST /api/v1/runs/define" />, which is R-02.53&apos;s create path — not{' '}
        <MonoId value="POST /api/v1/runs" />, which is the acquisition half and which R-02.53
        forbids for a run definition. It <strong>validates nothing and defaults nothing</strong>: a
        field left empty is submitted absent and refused by its owner, naming what failed.{' '}
        <strong>Check before committing</strong> runs the same validator the create path runs,
        read-only — and a green check enables nothing, because that validator runs again at submit
        and a part can stop resolving in between.
      </Callout>

      <RefusalBanner
        refusals={grouped.get('form') ?? []}
        className="mb-4"
        data-testid="form-refusals"
      />
      {failure === null ? null : (
        <FailureNotice failure={failure} surface="starting this run" className="mb-4" />
      )}

      <Card className="mb-4">
        <ModelSourceField
          mode={draft.mode}
          path={draft.path}
          repo={draft.repo}
          revision={draft.revision}
          configPath={draft.configPath}
          onChange={patch}
        />
        <RefusalBanner refusals={grouped.get('model-source') ?? []} field="model-source" />
      </Card>

      <Card className="mb-4">
        <RunTypeField value={draft.runType} onChange={(runType) => patch({ runType })} />
        <RefusalBanner refusals={grouped.get('run-type') ?? []} field="run-type" />
      </Card>

      <Card className="mb-4">
        <PackField
          packId={draft.packId}
          version={draft.packVersion}
          onChange={(packId, packVersion) => patch({ packId, packVersion })}
        />
        <RefusalBanner refusals={grouped.get('pack') ?? []} field="pack" />
      </Card>

      <Card className="mb-4">
        <SignalSourceField
          value={draft.signalSource}
          onChange={(signalSource) => patch({ signalSource })}
        />
        <RefusalBanner refusals={grouped.get('signal-source') ?? []} field="signal-source" />
      </Card>

      <SelectorAvailability />

      <Card className="mb-4">
        <OptimizerFields
          learningRate={draft.learningRate}
          warmupSteps={draft.warmupSteps}
          batchSize={draft.batchSize}
          gradientAccumulation={draft.gradientAccumulation}
          optimizerStatePrecision={draft.optimizerStatePrecision}
          seed={draft.seed}
          onChange={patch}
        />
        <RefusalBanner refusals={grouped.get('optimizer') ?? []} field="optimizer" />
      </Card>

      <Card className="mb-4">
        <CadenceFields
          testEvery={draft.testEvery}
          retentionEvery={draft.retentionEvery}
          saveEvery={draft.saveEvery}
          retentionFloor={draft.retentionFloor}
          onChange={patch}
        />
        <RefusalBanner refusals={grouped.get('cadences') ?? []} field="cadences" />
        <RefusalBanner refusals={grouped.get('save-cadence') ?? []} field="save-cadence" />
        <RefusalBanner refusals={grouped.get('retention-floor') ?? []} field="retention-floor" />
      </Card>

      <StopConditionsField
        entries={draft.stopConditions}
        onChange={(stopConditions) => setDraft((current) => ({ ...current, stopConditions }))}
        {...(availableStopConditions === undefined ? {} : { available: availableStopConditions })}
      />
      <RefusalBanner
        refusals={grouped.get('stop-conditions') ?? []}
        field="stop-conditions"
        className="mb-4"
      />

      {/* The adapter block appears only for an adapter run. It is not hidden-but-submitted:
          `buildBody` omits it when its fields are empty, so a change of run type does not
          smuggle a rank into a `change` run. */}
      {draft.runType === 'adapter' ? (
        <>
          <AdapterBlockFields
            rank={draft.adapterRank}
            alpha={draft.adapterAlpha}
            targetModules={draft.adapterTargetModules}
            dropout={draft.adapterDropout}
            onChange={patch}
          />
          <RefusalBanner refusals={grouped.get('adapter') ?? []} field="adapter" className="mb-4" />
        </>
      ) : null}

      <BudgetPreview request={budgetRequest} />
      <RefusalBanner refusals={grouped.get('budget') ?? []} field="budget" className="mb-4" />

      <Card className="mb-4">
        <Field
          label="Run identifier"
          hint="Optional. Left empty, the backend assigns one. A branch is a new run and its new run_id is what makes it one."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={draft.runId}
              onChange={(event) => patch({ runId: event.target.value })}
              data-testid="run-id-field"
            />
          )}
        </Field>
        <Field
          label="GPU request"
          hint="Left empty, the backend resolves a card by UUID at submit from live nvidia-smi. Never an index and never a bus address."
        >
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={draft.gpuRequest}
              onChange={(event) => patch({ gpuRequest: event.target.value })}
              data-testid="gpu-request"
            />
          )}
        </Field>
      </Card>

      <Card className="mb-4" data-testid="submitted-body">
        <div className="mb-1 text-heading">What will be submitted</div>
        <div className="mb-2 text-hint text-dim">
          Exactly the values entered above. A field left empty is absent from this body, not zero
          and not a default — the run log records what the operator chose.
        </div>
        <MonoId value={JSON.stringify(body, null, 2)} variant="block" />
      </Card>

      {checkFailure === null ? null : (
        <FailureNotice
          failure={checkFailure}
          surface="checking this definition before committing"
          className="mb-4"
        />
      )}

      <ResolvedSummary outcome={outcome} />

      <div className="flex flex-wrap items-center gap-2">
        {/* "Start run", or "Attach and start" in attach mode only - see the module docstring. */}
        <Button
          variant="primary"
          onClick={() => void onSubmit()}
          disabled={submitting}
          data-testid="start-run"
        >
          {submitting ? 'Starting…' : startLabel(draft.mode)}
        </Button>
        <Button onClick={() => void onCheck()} data-testid="check-definition">
          Check before committing
        </Button>
        <Button onClick={() => setDraft(EMPTY_DRAFT)}>Clear every field</Button>
      </div>
    </>
  );
}
