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
 * F1's route, extended by F6. Nothing is staged, chunked or pre-committed on the server:
 * R-02.53's guarantee is that *the whole definition* is validated before anything starts, and
 * a form submitting in pieces breaks exactly that.
 *
 * **The draft is kept in this browser until it is submitted** (task 13.0, operator request
 * 2026-10-17): `usePersistentState` restores it after a reload, a successful submit and
 * "Clear every field" remove it, and a restored draft is announced so it is never read as a
 * default. Server answers - refusals, the resolved summary - are not kept: they are asked
 * again, because a stale answer replayed after a reload is a cached permission (9A.5b).
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
import { DRAFT_VERSION, EMPTY_DRAFT, buildBody, decodeDraft } from './draft';
import { usePersistentState } from '@/hooks/usePersistentState';
import { DRAFT_KEYS } from '@/stores/draftStorage';
import { DraftRestoredNote } from '@/components/common/DraftRestoredNote';

/** The primary button's label: §5.10's wording for the attach mode and for nothing else. */
export function startLabel(mode: string): string {
  return mode === 'attach' ? 'Attach and start' : 'Start run';
}
import { RunTypeField } from './RunTypeField';
import { PackField } from './PackField';
import {
  AdapterBlockFields,
  CadenceFields,
  GpuRequestField,
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
  const persisted = usePersistentState(DRAFT_KEYS.newRun, EMPTY_DRAFT, {
    version: DRAFT_VERSION,
    decode: decodeDraft,
  });
  const draft = persisted.value;
  const setDraft = persisted.set;
  const clearDraft = persisted.clear;
  const [refusals, setRefusals] = useState<readonly DecodedRefusal[]>([]);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [checkFailure, setCheckFailure] = useState<ApiFailure | null>(null);
  const [outcome, setOutcome] = useState<ValidationOutcome | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const patch = useCallback(
    (values: Readonly<Record<string, string>>) => {
      setDraft((current) => ({ ...current, ...values }));
    },
    [setDraft],
  );

  const body = useMemo(() => buildBody(draft), [draft]);

  /**
   * The budget request, or `null`.
   *
   * `null` while the operator has entered neither a run type nor a batch configuration -
   * because an estimate derived from nothing is a preview of nothing, and the estimate's
   * request model is closed to unknown keys for the same reason.
   *
   * **The model source goes as the body carries it** (`body.model_source`), and the server
   * reads the model's size from it. Until 2026-10-20 this request carried no model source and
   * no size, so every estimate was refused `budget_spec_incomplete`. The optimizer settings,
   * the adapter block and the GPU request come from the same body, so the estimate is of the
   * run that would be submitted.
   */
  const budgetRequest = useMemo<BudgetRequest | null>(() => {
    const learningRate = Number(draft.learningRate);
    const warmupSteps = Number(draft.warmupSteps);
    const batchSize = Number(draft.batchSize);
    if (draft.runType.length === 0) return null;
    if (!Number.isFinite(learningRate) || !Number.isFinite(batchSize)) return null;
    if (draft.learningRate.length === 0 || draft.batchSize.length === 0) return null;
    const entered = (body['run_configuration'] ?? {}) as Readonly<Record<string, unknown>>;
    const modelSource = body['model_source'] as Readonly<Record<string, unknown>> | undefined;
    const adapter = body['adapter'] as Readonly<Record<string, unknown>> | undefined;
    const gpuRequest = body['gpu_request'] as string | undefined;
    return {
      run_type: draft.runType,
      config: {
        ...entered,
        learning_rate: learningRate,
        // A preview needs a number here; an empty warm-up is estimated as none. Warm-up does
        // not change memory, and nothing from this preview is submitted.
        warmup_steps:
          Number.isFinite(warmupSteps) && draft.warmupSteps.length > 0 ? warmupSteps : 0,
        batch_size: batchSize,
      },
      ...(modelSource === undefined ? {} : { model_source: modelSource }),
      ...(adapter === undefined ? {} : { adapter }),
      ...(gpuRequest === undefined ? {} : { gpu_request: gpuRequest }),
    };
  }, [body, draft.runType, draft.learningRate, draft.warmupSteps, draft.batchSize]);

  const onSubmit = useCallback(async () => {
    setSubmitting(true);
    const result = await submit(body);
    setSubmitting(false);
    if (result.ok) {
      setRefusals([]);
      setFailure(null);
      setOutcome(null);
      // Submitted: the draft has become a run and must not come back on the next visit.
      clearDraft();
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
  }, [body, submit, navigation, clearDraft]);

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
      // A 422 from the check names fields, exactly as one from the submit does, so it is
      // rendered beside those fields rather than as one block of raw JSON at the bottom.
      setOutcome(null);
      const decoded = decodeRefusals(result.failure);
      setRefusals(decoded);
      setCheckFailure(decoded.length === 0 ? result.failure : null);
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
        subtitle="Define a training run: the model, the pack, the training signal, the settings, how often to test and save, and at least one stop condition."
      />

      <Callout tone="info" title="How this form works" className="mb-4">
        The whole definition is sent in one request when you start the run. Nothing is filled in for
        you: an empty field is left out, and the server names anything that is missing or wrong.{' '}
        <strong>Check before committing</strong> runs the same checks without starting anything. A
        passing check is not a reservation: the checks run again when you start, and a part or the
        free GPU memory can change in between.
      </Callout>

      <DraftRestoredNote
        restoredAt={persisted.restoredAt}
        onDiscard={clearDraft}
        className="mb-4"
      />

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
          tokenizerPath={draft.tokenizerPath}
          tokenizerRepo={draft.tokenizerRepo}
          tokenizerRevision={draft.tokenizerRevision}
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
        <Field label="Run identifier" hint="Optional. Left empty, miForge assigns one.">
          {({ id, describedBy }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              value={draft.runId}
              onChange={(event) => patch({ runId: event.target.value })}
              entry="new-name"
              data-testid="run-id-field"
            />
          )}
        </Field>
        <GpuRequestField
          value={draft.gpuRequest}
          onChange={(gpuRequest) => patch({ gpuRequest })}
        />
      </Card>

      <Card className="mb-4" data-testid="submitted-body">
        <div className="mb-1 text-heading">What will be submitted</div>
        <div className="mb-2 text-hint text-dim">
          Exactly the values entered above. An empty field is left out rather than sent as zero or a
          default, so the run log records only what you chose.
        </div>
        <MonoId value={JSON.stringify(body, null, 2)} variant="block" />
      </Card>

      {checkFailure === null ? null : (
        <FailureNotice failure={checkFailure} surface="the check" className="mb-4" />
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
        <Button onClick={clearDraft} data-testid="clear-draft">
          Clear every field
        </Button>
      </div>
    </>
  );
}
