/**
 * The Onboard a domain screen (F7 task 12.6; FR-007.17, FR-007.18, FR-007.30).
 *
 * Three-column grid: the step list, the current step's panel, and the draft's state. Subtitle
 * `step N of 7 · progress is saved as you go`, **taken from the server's own `subtitle`** so
 * the claim and the save cannot disagree - the screen says progress is saved, and a claimed
 * save must have happened.
 *
 * **The server is the truth and there is no optimistic anything** (F7 task 12.2): the draft is
 * re-read after every step commit, because `step_states`, the subtitle and the validity are
 * the backend's answers. A draft is persisted immediately, so a browser closed on step 1 loses
 * nothing - and going back to step 3 does **not** discard steps 4 to 6.
 *
 * `Back` / `Save and continue`, and every `Primary` takes an `onClick` - the mockup's did not,
 * so every primary action on it was inert (handoff section 6).
 *
 * ## A reload (task 13.0)
 *
 * The draft id and the step are **in the URL** (`#/packs/onboard?draftId=…&step=…`), so a
 * reload re-reads the same draft. *Until 2026-10-17 the id was kept nowhere, and every reload
 * called `POST /pack-drafts` again - one orphan server draft per refresh - and opened it at the
 * server's step rather than the one on screen.* A `step` outside 1..`step_count`, or not a
 * number, falls back to the server's `current_step`.
 *
 * The **uncommitted** patch of the step on screen (`pending`) is kept in this browser by
 * `usePersistentState`, keyed by draft id, and removed when a step commit or the pack save
 * succeeds. Committed steps are the server's and are not duplicated here.
 */
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { HelpLink } from '@/components/help/HelpLink';
import { HELP_FOR_ONBOARD_STEP } from '@/config/help';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Callout } from '@/components/common/Callout';
import { MonoId } from '@/components/common/MonoId';
import { ValidityPill } from '@/components/common/ValidityPill';
import { FailureNotice } from '@/components/common/FailureNotice';
import { useNavigation } from '@/contexts/NavigationContext';
import {
  commitDraft,
  commitStep,
  createDraft,
  dryRunDraft,
  fetchDraft,
  fetchDraftValidity,
  sampleSlot,
  scoreSample,
} from '@/api/packDrafts';
import type { ApiFailure } from '@/api/errors';
import type {
  DraftRecord,
  DryRunResult,
  PackValidity,
  ScoredSample,
  SlotKind,
  SlotSample,
} from '@/types/packs';
import { StepList, STEP_TITLES } from './StepList';
import { STEP_PANELS, type StepPanelProps } from './OnboardSteps';
import { stepBody } from './stepBody';
import { usePersistentState } from '@/hooks/usePersistentState';
import { DRAFT_KEYS, asRecord } from '@/stores/draftStorage';
import { DraftRestoredNote } from '@/components/common/DraftRestoredNote';
import { formatEt } from '@/components/common/time';

type Pending = Readonly<Record<string, unknown>>;
const NO_PENDING: Pending = {};
export const ONBOARD_PENDING_VERSION = 1;

/** A stored step patch: a plain object of scalar answers, or nothing. */
export function decodePending(data: unknown): Pending | null {
  const record = asRecord(data);
  if (record === null) return null;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    if (value === null || ['string', 'number', 'boolean'].includes(typeof value)) out[key] = value;
  }
  return out;
}

/** The URL's step, when it is a step this draft has. */
function stepFrom(raw: string | undefined, stepCount: number): number | null {
  if (raw === undefined || !/^\d+$/.test(raw)) return null;
  const step = Number(raw);
  return step >= 1 && step <= stepCount ? step : null;
}

export function OnboardPage(): ReactNode {
  const navigation = useNavigation();
  const draftIdFromNav = navigation.params['draftId'] ?? null;
  const stepFromNav = navigation.params['step'];
  const stepFromNavRef = useRef(stepFromNav);
  stepFromNavRef.current = stepFromNav;
  /** The draft already on screen, so writing its id into the URL does not read it twice. */
  const loadedId = useRef<string | null>(null);

  const [draft, setDraft] = useState<DraftRecord | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [sample, setSample] = useState<SlotSample | ScoredSample | null>(null);
  const [dryRun, setDryRun] = useState<DryRunResult | null>(null);
  const [validity, setValidity] = useState<PackValidity | null>(null);
  const persistedPending = usePersistentState<Pending>(
    draft === null ? null : DRAFT_KEYS.onboard(draft.draft_id),
    NO_PENDING,
    { version: ONBOARD_PENDING_VERSION, decode: decodePending },
  );
  const pending = persistedPending.value;
  const setPending = persistedPending.set;
  const clearPending = persistedPending.clear;
  const [step, setStep] = useState<number | null>(null);

  const take = useCallback((next: DraftRecord) => {
    loadedId.current = next.draft_id;
    setDraft(next);
    setFailure(null);
    // The URL's step on a reload, else the server's: the flow decides what follows a commit.
    setStep(
      (current) =>
        current ?? stepFrom(stepFromNavRef.current, next.step_count) ?? next.current_step,
    );
  }, []);

  useEffect(() => {
    // The draft on screen is already this one - or the nav was clicked while a draft is open,
    // which re-selects this screen and must not start a second draft (the effect below puts
    // the open draft's id back in the URL).
    if (
      loadedId.current !== null &&
      (draftIdFromNav === null || draftIdFromNav === loadedId.current)
    )
      return;
    let cancelled = false;
    const start = async (): Promise<void> => {
      const result =
        draftIdFromNav === null ? await createDraft({}) : await fetchDraft(draftIdFromNav);
      if (cancelled) return;
      if (result.ok) take(result.value);
      else setFailure(result.failure);
    };
    void start();
    return () => {
      cancelled = true;
    };
  }, [draftIdFromNav, take]);

  // Draft id and step -> URL, replacing the entry: a step is not a page to go back to.
  const go = navigation.go;
  useEffect(() => {
    if (draft === null || step === null) return;
    if (draftIdFromNav === draft.draft_id && stepFromNav === String(step)) return;
    go('onboard', { draftId: draft.draft_id, step: String(step) }, { replace: true });
  }, [draft, step, draftIdFromNav, stepFromNav, go]);

  const reread = useCallback(
    async (draftId: string) => {
      const result = await fetchDraft(draftId);
      if (result.ok) take(result.value);
      else setFailure(result.failure);
    },
    [take],
  );

  const saveAndContinue = useCallback(async () => {
    if (draft === null || step === null) return;
    // The step's answers as `StepRequest` reads them: a chosen part resolved into its slot
    // document, the floor inside the retention slot (task 14.0; see `stepBody.ts`).
    const built = await stepBody(draft, pending);
    if (!built.ok) {
      setFailure(built.failure);
      return;
    }
    const result = await commitStep(draft.draft_id, step, built.body as never);
    if (!result.ok) {
      setFailure(result.failure);
      return;
    }
    clearPending();
    // Re-read rather than trust the response: one shape, one source of truth.
    await reread(draft.draft_id);
    setStep(result.value.current_step);
    setSample(null);
  }, [draft, step, pending, reread, clearPending]);

  const onSample = useCallback(
    async (slot: SlotKind) => {
      if (draft === null) return;
      const result = await sampleSlot(draft.draft_id, slot);
      if (result.ok) setSample(result.value);
      else setFailure(result.failure);
    },
    [draft],
  );

  const onScore = useCallback(
    async (slot: SlotKind, floor: number | null) => {
      if (draft === null) return;
      const config = (draft.slots[slot] ?? {}) as Readonly<Record<string, unknown>>;
      const result = await scoreSample(draft.draft_id, slot, {
        config,
        ...(floor === null ? {} : { floor }),
      });
      if (result.ok) setSample(result.value);
      else setFailure(result.failure);
    },
    [draft],
  );

  const onDryRun = useCallback(async () => {
    if (draft === null) return;
    const config = (draft.slots['selector_config'] ?? {}) as Readonly<Record<string, unknown>>;
    const result = await dryRunDraft(draft.draft_id, { config });
    if (result.ok) setDryRun(result.value);
    else setFailure(result.failure);
  }, [draft]);

  const review = useCallback(async () => {
    if (draft === null) return;
    const result = await fetchDraftValidity(draft.draft_id);
    if (result.ok) setValidity(result.value);
    else setFailure(result.failure);
  }, [draft]);

  const save = useCallback(async () => {
    if (draft === null) return;
    const result = await commitDraft(draft.draft_id);
    if (result.ok) {
      clearPending();
      navigation.go('packs');
      return;
    }
    // The draft is intact on the other side of this refusal; the screen stays on step 7.
    setFailure(result.failure);
  }, [draft, navigation, clearPending]);

  if (draft === null) {
    return (
      <>
        <PageHead title="Onboard a domain" subtitle="Seven steps · progress is saved as you go" />
        {failure === null ? (
          <Card data-testid="onboard-loading">Starting a draft…</Card>
        ) : (
          <FailureNotice failure={failure} surface="the authoring flow" />
        )}
      </>
    );
  }

  const current = step ?? draft.current_step;
  const Panel = STEP_PANELS[current];
  const panelProps: StepPanelProps = {
    draft,
    pending,
    onPatch: (patch) => setPending((value) => ({ ...value, ...patch })),
    sample,
    dryRun,
    onSample: (slot) => void onSample(slot),
    onDryRun: () => void onDryRun(),
    onScore: (slot, floor) => void onScore(slot, floor),
  };

  return (
    <>
      <PageHead
        title="Onboard a domain"
        /* The server's own subtitle, so the claim "progress is saved" is the backend's. */
        subtitle={
          draft.subtitle ??
          `step ${draft.current_step} of ${draft.step_count} · progress is saved as you go`
        }
        actions={<MonoId value={draft.draft_id} truncateTo={6} className="text-dim" />}
      />

      <Callout tone="info" title="A draft is not a pack" className="mb-4">
        It never appears in the registry and it is resumable or deletable at any time. Later steps
        are not discarded when you go back to an earlier one.
      </Callout>

      <DraftRestoredNote
        restoredAt={persistedPending.restoredAt}
        onDiscard={clearPending}
        className="mb-4"
      />

      {failure === null ? null : (
        <FailureNotice failure={failure} surface="this step" className="mb-4" />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[14rem_1fr_16rem]">
        <Card>
          <StepList draft={draft} onGo={setStep} />
        </Card>

        <Card data-testid="onboard-panel">
          <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
            <div className="text-heading">
              {current}. {STEP_TITLES[current] ?? 'This step'}
            </div>
            {/* ENH-004 D4: the step's own Help topic, mapped by step (steps 4-6 have their own). */}
            {HELP_FOR_ONBOARD_STEP[current] === undefined ? null : (
              <HelpLink topic={HELP_FOR_ONBOARD_STEP[current]} label="Help with this step" />
            )}
          </div>
          {Panel === undefined ? (
            <div className="text-hint text-amber">
              There is no step {current}; the flow has {draft.step_count}, numbered 1 to{' '}
              {draft.step_count}.
            </div>
          ) : (
            <Panel {...panelProps} />
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            <Button onClick={() => setStep(Math.max(1, current - 1))}>Back</Button>
            <Button variant="primary" onClick={() => void saveAndContinue()}>
              Save and continue
            </Button>
            {current === draft.step_count ? (
              <>
                <Button onClick={() => void review()}>Check what is missing</Button>
                <Button variant="primary" onClick={() => void save()} data-testid="commit-draft">
                  Save this pack
                </Button>
              </>
            ) : null}
          </div>
        </Card>

        <Card data-testid="onboard-state">
          <div className="mb-2 text-heading">This draft</div>
          <dl className="stack-y-1 text-hint">
            <div>
              <dt className="inline text-dim">name: </dt>
              <dd className="inline">{draft.name ?? 'not named yet'}</dd>
            </div>
            <div>
              <dt className="inline text-dim">version: </dt>
              <dd className="inline">{draft.version}</dd>
            </div>
            <div>
              <dt className="inline text-dim">domain: </dt>
              <dd className="inline">{draft.domain ?? 'no domain'}</dd>
            </div>
            <div>
              <dt className="inline text-dim">updated: </dt>
              <dd className="inline" title={draft.updated_at}>
                {formatEt(draft.updated_at)}
              </dd>
            </div>
          </dl>
          {validity === null ? null : (
            <div className="mt-3" data-testid="onboard-validity">
              <ValidityPill validity={validity} />
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
