/**
 * F7's seven-step authoring flow (F7 task 12.6, 12.7; FR-007.17-.20).
 *
 * **The server is the truth and there is no optimistic validity** (F7 task 12.2): every step
 * commit re-reads the draft, because the draft's `step_states`, its `subtitle` and its
 * validity are the backend's answers and a client that computed them would be a second one.
 * A draft is persisted immediately, so a browser closed on step 1 loses nothing - and the
 * subtitle claims progress is saved, so a claimed save must have happened.
 */
import { apiDelete, apiGet, apiPost, apiPut, type ApiResult } from './client';
import { V1 } from './v1';
import type {
  DraftRecord,
  DryRunResult,
  PackListingRow,
  PackValidity,
  ScoredSample,
  SlotSample,
} from '@/types/packs';

export const DRAFTS_PATH = `${V1}/pack-drafts`;

const draftPath = (draftId: string): string => `${DRAFTS_PATH}/${encodeURIComponent(draftId)}`;

export const createDraft = (body: {
  readonly name?: string | null;
  readonly version?: string;
  readonly domain?: string | null;
}): Promise<ApiResult<DraftRecord>> => apiPost<DraftRecord>(DRAFTS_PATH, body);

export const fetchDrafts = (): Promise<ApiResult<{ readonly drafts: readonly DraftRecord[] }>> =>
  apiGet<{ readonly drafts: readonly DraftRecord[] }>(DRAFTS_PATH);

export const fetchDraft = (draftId: string): Promise<ApiResult<DraftRecord>> =>
  apiGet<DraftRecord>(draftPath(draftId));

export const deleteDraft = (draftId: string): Promise<ApiResult<unknown>> =>
  apiDelete<unknown>(draftPath(draftId));

/**
 * One step's answers.
 *
 * `domain_given` is separate from `domain` because **null is meaningful**: it asserts the
 * pack is not domain-scoped, which a general pretraining pack genuinely is not. An absent
 * answer and "no domain" are different facts, and the pack schema's nullable-but-required
 * `domain` exists to keep them apart.
 */
export interface StepBody {
  readonly name?: string | null;
  readonly version?: string | null;
  readonly domain?: string | null;
  readonly domain_given?: boolean;
  readonly slots?: Readonly<Record<string, unknown>>;
  readonly skip?: boolean;
}

export const commitStep = (
  draftId: string,
  number: number,
  body: StepBody,
): Promise<ApiResult<DraftRecord>> =>
  apiPut<DraftRecord>(`${draftPath(draftId)}/steps/${number}`, body);

/** A seeded, stratified sample through F4's own sampler - the sample a run would score. */
export const sampleSlot = (draftId: string, slot: string): Promise<ApiResult<SlotSample>> =>
  apiPost<SlotSample>(`${draftPath(draftId)}/samples/${encodeURIComponent(slot)}`);

export interface ScoreSampleBody {
  readonly config: Readonly<Record<string, unknown>>;
  readonly catalog?: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  readonly form?: string;
  readonly provider?: string | null;
  readonly n?: number | null;
  /** The **proposed** floor at step 6: the panel says whether it is already breached. */
  readonly floor?: number | null;
}

export const scoreSample = (
  draftId: string,
  slot: string,
  body: ScoreSampleBody,
): Promise<ApiResult<ScoredSample>> =>
  apiPost<ScoredSample>(`${draftPath(draftId)}/samples/${encodeURIComponent(slot)}/score`, body);

export interface DryRunBody {
  readonly config: Readonly<Record<string, unknown>>;
  readonly catalog?: readonly Readonly<Record<string, unknown>>[];
  readonly provider?: string | null;
  readonly n?: number | null;
}

/**
 * Answer a sample of the pack's own prompts and show a verdict with a reason (AC11).
 *
 * Returns **three counts and their denominator together**, inside F3's `SelectionHealth`
 * record passed through whole. A human-review configuration comes back all `undecided` with
 * `asynchronous_note` naming why, and the step still saves.
 */
export const dryRunDraft = (draftId: string, body: DryRunBody): Promise<ApiResult<DryRunResult>> =>
  apiPost<DryRunResult>(`${draftPath(draftId)}/dry-run`, body);

/** Step 7's refusal path. On a refusal **the draft is left intact.** */
export const commitDraft = (draftId: string): Promise<ApiResult<PackListingRow>> =>
  apiPost<PackListingRow>(`${draftPath(draftId)}/commit`);

/** Step 7: a **200 describing invalidity**, naming every missing slot rather than the first. */
export const fetchDraftValidity = (draftId: string): Promise<ApiResult<PackValidity>> =>
  apiGet<PackValidity>(`${draftPath(draftId)}/validity`);
