/**
 * The seven step panels - **one component per step, not one switching on a number**
 * (F7 task 12.6).
 *
 * The rule is the second half of that sentence, and it is why there is no `renderStep(n)` with
 * a `switch` in it: a step's panel is a different *thing* at each number - a text field, a
 * part picker, a dry-run, a scored sample against a proposed floor - and one component
 * branching seven ways is where "step 6 shows step 5's panel" comes from. Seven named
 * components live in one file because each is small and they share the same props; the
 * prohibition is on the switch, not on the file. Recorded as a deviation from a file-per-
 * component reading of the task.
 *
 * Every panel takes the whole draft and reports a patch. **None of them validates**: the
 * backend refuses a slot a step does not fill rather than ignoring it, and a client-side
 * required-field check would be a second answer.
 *
 * **Each slot a step fills has a picker** over the indexed parts of that kind (task 14.0).
 * *Until 2026-10-18 the panels only displayed what the server held: no control could fill a
 * slot, so a pack could not be authored from this screen at all.* The slots a step fills are
 * read from `DRAFT_STEPS` (the backend's `STEPS`), not listed per panel.
 */
import { useCallback, type ReactNode } from 'react';
import { Field, Input } from '@/components/common/Field';
import { Picker, type PickerOption } from '@/components/common/Picker';
import { useListing } from '@/hooks/useListing';
import { PART_SOURCE_CONTRACT, PART_SOURCE_EXTERNAL, fetchParts, type PartRow } from '@/api/packs';
import { ShowArchivedToggle } from '@/components/common/ShowArchivedToggle';
import { usePersistentFlag } from '@/hooks/usePersistentFlag';
import { DRAFT_KEYS } from '@/stores/draftStorage';
import { FLOOR_KEY, slotKey } from './stepBody';
import { Button } from '@/components/common/Button';
import { AbsentValue } from '@/components/common/AbsentValue';
import { MonoId } from '@/components/common/MonoId';
import { DRAFT_STEPS, SLOT_LABELS, type DraftRecord, type SlotKind } from '@/types/packs';
import { TryPanel } from './TryPanel';
import { DryRunPanel } from './DryRunPanel';
import type { DryRunResult, ScoredSample, SlotSample } from '@/types/packs';

export interface StepPanelProps {
  readonly draft: DraftRecord;
  /**
   * The step's uncommitted answers. **An input shows these over the server's draft** - until
   * 2026-10-17 the inputs were bound to `draft` alone while `onPatch` wrote here, so every
   * keystroke was sent but reverted on screen (task 13.5).
   */
  readonly pending: Readonly<Record<string, unknown>>;
  readonly onPatch: (patch: Readonly<Record<string, unknown>>) => void;
  readonly sample: SlotSample | ScoredSample | null;
  readonly dryRun: DryRunResult | null;
  readonly onSample: (slot: SlotKind) => void;
  readonly onDryRun: () => void;
  readonly onScore: (slot: SlotKind, floor: number | null) => void;
}

/** A slot's current document, rendered as what it is: a reference, or an honest absence. */
function SlotState({
  draft,
  slot,
}: {
  readonly draft: DraftRecord;
  readonly slot: SlotKind;
}): ReactNode {
  const value = draft.slots[slot];
  if (value === null || value === undefined) {
    return (
      <AbsentValue
        reason="missing_required_key"
        variant="block"
        detail={`${SLOT_LABELS[slot]} is empty — the pack document states all seven slots even when empty`}
      />
    );
  }
  return <MonoId value={JSON.stringify(value)} variant="block" />;
}

/**
 * The parts of one kind as options. A miDataworks version with **no contract reference** is
 * listed and disabled with the reason: F7 refuses to pin it (`part_unpinnable`), and hiding it
 * would make an imported part look missing.
 */
export function partOptions(listing: {
  readonly items: readonly PartRow[];
}): readonly PickerOption[] {
  return listing.items.map((row) => {
    const unpinnable = row.source === PART_SOURCE_CONTRACT && row.contract_reference === null;
    const archived = (row.archived_at ?? null) !== null;
    const name =
      row.rendered ??
      (row.dataset === null
        ? row.id
        : `${row.dataset}${row.version_number === null ? '' : ` v${row.version_number}`}`);
    return {
      value: row.id,
      label: name,
      detail: [
        archived ? 'archived' : null,
        row.source === PART_SOURCE_EXTERNAL ? 'external part' : 'from miDataworks',
        row.row_count === null ? 'row count not recorded' : `${row.row_count} rows`,
        row.caveats.length === 0 ? null : `${row.caveats.length} caveat(s)`,
        row.licence_classes.length === 0 ? null : row.licence_classes.join(', '),
      ]
        .filter((part): part is string => part !== null)
        .join(' · '),
      keywords: archived ? `${row.id} archived` : row.id,
      ...(unpinnable
        ? {
            disabledReason:
              'this version carries no contract reference, so it cannot pin a slot (part_unpinnable)',
          }
        : {}),
    };
  });
}

/** One slot's picker. The choice is a part id under `slot:<kind>`; Save resolves it. */
export function SlotPicker({
  slot,
  pending,
  onPatch,
  includeArchived = false,
  load = fetchParts,
}: {
  readonly slot: SlotKind;
  readonly pending: Readonly<Record<string, unknown>>;
  readonly onPatch: (patch: Readonly<Record<string, unknown>>) => void;
  /** List archived parts too (operator decision 2026-10-20). Hidden by default. */
  readonly includeArchived?: boolean;
  readonly load?: typeof fetchParts;
}): ReactNode {
  // The slot's own name filters the index: the contract spelling IS the slot's (see PartRow).
  const read = useCallback(
    () => load({ partKind: slot, includeArchived }),
    [load, slot, includeArchived],
  );
  const { listing } = useListing(read, partOptions);
  const key = slotKey(slot);
  const value = typeof pending[key] === 'string' ? (pending[key] as string) : '';
  return (
    <Field
      label={`${SLOT_LABELS[slot]} — choose a part`}
      hint="From the part index. Leave it unchosen to keep what the slot holds; Save resolves the choice into a pinned reference."
    >
      {({ id, describedBy }) => (
        <Picker
          id={id}
          describedBy={describedBy}
          value={value}
          onChange={(partId) => onPatch({ [key]: partId })}
          listing={listing}
          placeholder={`Choose the ${SLOT_LABELS[slot].toLowerCase()}`}
          empty={
            includeArchived
              ? `No ${SLOT_LABELS[slot].toLowerCase()} part is indexed. Import one (POST /api/v1/parts/import) or publish one from miDataworks.`
              : `No ${SLOT_LABELS[slot].toLowerCase()} part is listed. Import one (POST /api/v1/parts/import), publish one from miDataworks, or tick "Show archived parts".`
          }
          surface="the part index"
          data-testid={`slot-picker-${slot}`}
        />
      )}
    </Field>
  );
}

/** A picker for every slot step `number` fills, from `DRAFT_STEPS`. */
function SlotPickers({
  number,
  pending,
  onPatch,
}: {
  readonly number: number;
  readonly pending: Readonly<Record<string, unknown>>;
  readonly onPatch: (patch: Readonly<Record<string, unknown>>) => void;
}): ReactNode {
  const [showArchived, setShowArchived] = usePersistentFlag(DRAFT_KEYS.slotPickersArchived);
  const spec = DRAFT_STEPS.find((step) => step.number === number);
  if (spec === undefined || spec.slots.length === 0) return null;
  return (
    <div className="mb-3">
      {spec.slots.map((slot: SlotKind) => (
        <SlotPicker
          key={slot}
          slot={slot}
          pending={pending}
          onPatch={onPatch}
          includeArchived={showArchived}
        />
      ))}
      <ShowArchivedToggle
        checked={showArchived}
        onChange={setShowArchived}
        noun="parts"
        data-testid={`slot-show-archived-${number}`}
      />
    </div>
  );
}

/** Step 1 — the domain sentence. `null` is meaningful and is offered explicitly. */
/** The uncommitted answer when there is one (`null` reads as empty), else the server's. */
function shown(
  pending: Readonly<Record<string, unknown>>,
  key: string,
  committed: string | null | undefined,
): string {
  if (key in pending) {
    const value = pending[key];
    return value === null || value === undefined ? '' : String(value);
  }
  return committed ?? '';
}

export function DescribeStep({ draft, pending, onPatch }: StepPanelProps): ReactNode {
  return (
    <>
      <Field
        label="Pack name"
        hint="What this domain is called. Used in every part's display form."
      >
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            value={shown(pending, 'name', draft.name)}
            onChange={(event) => onPatch({ name: event.target.value })}
            entry="new-name"
            data-testid="draft-name"
          />
        )}
      </Field>
      <Field
        label="Domain"
        hint="One sentence. Leaving it empty and saying so asserts the pack is not domain-scoped, which a general pretraining pack genuinely is not — an absent answer and “no domain” are different facts."
      >
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            value={shown(pending, 'domain', draft.domain)}
            onChange={(event) => onPatch({ domain: event.target.value, domain_given: true })}
            entry="prose"
            data-testid="draft-domain"
          />
        )}
      </Field>
      <Button
        onClick={() => onPatch({ domain: null, domain_given: true })}
        data-testid="assert-no-domain"
      >
        This pack is not domain-scoped
      </Button>
    </>
  );
}

/** Step 2 — the prompt set, with a seeded sample of it. */
export function PromptsStep(props: StepPanelProps): ReactNode {
  return (
    <>
      <SlotPickers number={2} pending={props.pending} onPatch={props.onPatch} />
      <SlotState draft={props.draft} slot="prompt_set" />
      <div className="my-3">
        <Button onClick={() => props.onSample('prompt_set')}>Try the prompt set</Button>
      </div>
      <TryPanel sample={props.sample} />
    </>
  );
}

/** Step 3 — the corpus and the preference pairs. Skippable, and marked so. */
export function AnswersStep(props: StepPanelProps): ReactNode {
  return (
    <>
      <div className="mb-2 text-hint text-dim">
        Optional. A run can take its signal from the prompts alone; a corpus or preference pairs
        give it more to learn from.
      </div>
      <SlotPickers number={3} pending={props.pending} onPatch={props.onPatch} />
      <SlotState draft={props.draft} slot="corpus" />
      <div className="my-3 flex flex-wrap gap-2">
        <Button onClick={() => props.onSample('corpus')}>Try the corpus</Button>
        <Button onClick={() => props.onSample('preference_pairs')}>Try the preference pairs</Button>
      </div>
      <TryPanel sample={props.sample} />
    </>
  );
}

/** Step 4 — judgment, with the dry-run. The selector is **selected, not authored**. */
export function JudgmentStep(props: StepPanelProps): ReactNode {
  return (
    <>
      <div className="mb-2 text-hint text-dim">
        A selector and grader configuration is <strong>selected</strong>, not authored here — those
        are versioned in miDataworks and cited by hash. What this step does is answer a sample of
        this pack&apos;s own prompts and show the verdicts with their reasons.
      </div>
      <SlotPickers number={4} pending={props.pending} onPatch={props.onPatch} />
      <SlotState draft={props.draft} slot="selector_config" />
      <div className="my-3">
        <Button onClick={props.onDryRun} data-testid="run-dry-run">
          Try it on this pack&apos;s prompts
        </Button>
      </div>
      <DryRunPanel result={props.dryRun} />
    </>
  );
}

/** Step 5 — the test set, scored. No floor is in play here. */
export function ImprovementStep(props: StepPanelProps): ReactNode {
  return (
    <>
      <SlotPickers number={5} pending={props.pending} onPatch={props.onPatch} />
      <SlotState draft={props.draft} slot="test_set" />
      <div className="my-3">
        <Button onClick={() => props.onScore('test_set', null)}>Score the test set now</Button>
      </div>
      <TryPanel sample={props.sample} />
    </>
  );
}

/** Step 6 — the retention set and its floor, with the one thing an author cannot guess. */
export function RetentionStep(props: StepPanelProps): ReactNode {
  const floorValue = shown(props.pending, FLOOR_KEY, readFloor(props.draft));
  return (
    <>
      <SlotPickers number={6} pending={props.pending} onPatch={props.onPatch} />
      <SlotState draft={props.draft} slot="retention_set" />
      <Field
        label="Retention floor"
        hint="The score this set must not fall below. The panel says whether the floor is already breached — which is the one thing an author cannot guess."
      >
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            inputMode="decimal"
            entry="number"
            value={floorValue}
            onChange={(event) => props.onPatch({ [FLOOR_KEY]: event.target.value })}
            data-testid="draft-floor"
          />
        )}
      </Field>
      <div className="my-3">
        <Button
          onClick={() =>
            props.onScore('retention_set', floorValue.length === 0 ? null : Number(floorValue))
          }
        >
          Score the retention set against this floor
        </Button>
      </div>
      <TryPanel sample={props.sample} />
    </>
  );
}

function readFloor(draft: DraftRecord): string {
  const slot = draft.slots['retention_set'];
  if (slot === null || typeof slot !== 'object') return '';
  const floor = (slot as Record<string, unknown>)['floor'];
  return typeof floor === 'number' ? String(floor) : '';
}

/** Step 7 — review. It renders validity and **does not decide it**. */
export function ReviewStep({ draft }: StepPanelProps): ReactNode {
  return (
    <>
      <div className="mb-2 text-hint text-dim">
        What is still missing, by name. Saving refuses with every missing slot listed — not the
        first — and <strong>leaves the draft intact</strong>: two hours of someone&apos;s work is on
        the other side of that call.
      </div>
      <ul className="stack-y-1 text-hint">
        {Object.entries(draft.slots).map(([slot, value]) => (
          <li key={slot} className="flex flex-wrap items-baseline gap-2">
            <span className="min-w-44 text-dim">{SLOT_LABELS[slot as SlotKind] ?? slot}</span>
            {value === null || value === undefined ? (
              <AbsentValue reason="missing_required_key" detail="empty" />
            ) : (
              <MonoId value={JSON.stringify(value)} />
            )}
          </li>
        ))}
      </ul>
    </>
  );
}

/** Step number to panel. A **table**, so no component switches on a number. */
export const STEP_PANELS: Readonly<Record<number, (props: StepPanelProps) => ReactNode>> = {
  1: DescribeStep,
  2: PromptsStep,
  3: AnswersStep,
  4: JudgmentStep,
  5: ImprovementStep,
  6: RetentionStep,
  7: ReviewStep,
};
