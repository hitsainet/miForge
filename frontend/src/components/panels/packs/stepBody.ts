/**
 * One authoring step's request body, from what the operator entered (F9 task 14.0).
 *
 * ## Two defects this replaced
 *
 * 1. **A slot could not be filled from the screen.** Steps 2-6 displayed the slot the server
 *    held and offered no control to change it. Each slot now has a `Picker` over the parts
 *    of its kind; the choice is kept as `slot:<kind>` → part id (a scalar, so it survives a
 *    reload through `usePersistentState`), and on Save it is resolved through
 *    `GET /api/v1/pack-part-refs/{id}` into the **slot document the backend wrote**.
 * 2. **The retention floor was refused.** Step 6 sent a top-level `retention_floor`, which
 *    `StepRequest` (`extra="forbid"`) does not have - so every step-6 save with a floor was
 *    a `validation_failed`. The floor belongs to the retention set's slot document
 *    (`PackSlot.floor`), and travels there.
 *
 * **Nothing is validated here.** A floor typed with no retention set chosen or held is sent
 * as a slot naming neither source, and the backend refuses it by name
 * (`pack_document_invalid`); a floor that is not a number is sent as typed. The refusals are
 * the backend's copy.
 */
import type { ApiResult } from '@/api/client';
import type { ApiFailure } from '@/api/errors';
import { fetchPinnableReference, type PinnableReference, type SlotDocument } from '@/api/packs';
import type { DraftRecord, SlotKind } from '@/types/packs';

export const SLOT_KEY_PREFIX = 'slot:';

/** The pending key a slot's chosen part id is kept under. */
export const slotKey = (slot: SlotKind): string => `${SLOT_KEY_PREFIX}${slot}`;

/** The pending key the step-6 floor is kept under. Not a request field. */
export const FLOOR_KEY = 'retention_floor';

export type StepBody = Readonly<Record<string, unknown>>;

export async function stepBody(
  draft: DraftRecord,
  pending: Readonly<Record<string, unknown>>,
  resolve: (partId: string) => Promise<ApiResult<PinnableReference>> = fetchPinnableReference,
): Promise<
  | { readonly ok: true; readonly body: StepBody }
  | { readonly ok: false; readonly failure: ApiFailure }
> {
  const body: Record<string, unknown> = {};
  const slots: Record<string, SlotDocument | Readonly<Record<string, unknown>>> = {};

  for (const [key, value] of Object.entries(pending)) {
    if (key === FLOOR_KEY) continue;
    if (key.startsWith(SLOT_KEY_PREFIX)) {
      if (typeof value !== 'string' || value.length === 0) continue;
      // The part id is resolved, not submitted: a slot holds a REFERENCE, and the id is
      // this installation's index key, which no exported pack would carry.
      const resolved = await resolve(value);
      if (!resolved.ok) return { ok: false, failure: resolved.failure };
      slots[key.slice(SLOT_KEY_PREFIX.length)] = resolved.value.slot_document;
      continue;
    }
    body[key] = value;
  }

  if (FLOOR_KEY in pending) {
    const raw = pending[FLOOR_KEY];
    const text = raw === null || raw === undefined ? '' : String(raw).trim();
    const parsed = Number(text);
    const floor = text.length === 0 ? null : Number.isFinite(parsed) ? parsed : text;
    const base = slots['retention_set'] ??
      draft.slots['retention_set'] ?? { contract_reference: null, external: null, floor: null };
    slots['retention_set'] = { ...(base as Readonly<Record<string, unknown>>), floor };
  }

  if (Object.keys(slots).length > 0) body['slots'] = slots;
  return { ok: true, body };
}
