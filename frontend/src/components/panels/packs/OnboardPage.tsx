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
 */
import { useCallback, useEffect, useState, type ReactNode } from 'react';
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

export function OnboardPage(): ReactNode {
  const navigation = useNavigation();
  const draftIdFromNav = navigation.params['draftId'] ?? null;

  const [draft, setDraft] = useState<DraftRecord | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [sample, setSample] = useState<SlotSample | ScoredSample | null>(null);
  const [dryRun, setDryRun] = useState<DryRunResult | null>(null);
  const [validity, setValidity] = useState<PackValidity | null>(null);
  const [pending, setPending] = useState<Readonly<Record<string, unknown>>>({});
  const [step, setStep] = useState<number | null>(null);

  const take = useCallback((next: DraftRecord) => {
    setDraft(next);
    setFailure(null);
    // The server's step, not a local cursor: the flow decides what follows a commit.
    setStep((current) => current ?? next.current_step);
  }, []);

  useEffect(() => {
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
    const result = await commitStep(draft.draft_id, step, pending as never);
    if (!result.ok) {
      setFailure(result.failure);
      return;
    }
    setPending({});
    // Re-read rather than trust the response: one shape, one source of truth.
    await reread(draft.draft_id);
    setStep(result.value.current_step);
    setSample(null);
  }, [draft, step, pending, reread]);

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
      navigation.go('packs');
      return;
    }
    // The draft is intact on the other side of this refusal; the screen stays on step 7.
    setFailure(result.failure);
  }, [draft, navigation]);

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

      {failure === null ? null : (
        <FailureNotice failure={failure} surface="this step" className="mb-4" />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[14rem_1fr_16rem]">
        <Card>
          <StepList draft={draft} onGo={setStep} />
        </Card>

        <Card data-testid="onboard-panel">
          <div className="mb-3 text-heading">
            {current}. {STEP_TITLES[current] ?? 'This step'}
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
          <dl className="space-y-1 text-hint">
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
              <dd className="inline">{draft.updated_at}</dd>
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
