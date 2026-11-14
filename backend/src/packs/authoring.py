"""The seven-step authoring flow  (task 9.x; FR-007.17, FR-007.18, FR-007.22, R-02.39).

R-02.39's order, which the mockup already renders as seven steps:

=== ============================= ========================================== =========
#   Step                          Slots it fills                             Skippable
=== ============================= ========================================== =========
1   Describe the domain           none - the name and the domain sentence     no
2   What must it answer?          ``prompt_set``                              no
3   Do you have good answers?     ``corpus``, ``preference_pairs``            **yes**
4   How do we tell good from bad? ``selector_config``, ``grader_config``      **yes**
5   How do we know it improved?   ``test_set``                                no
6   What must it not forget?      ``retention_set`` **and its floor**         no
7   Review and save               none - it renders ``validate()``'s result   no
=== ============================= ========================================== =========

**The table is explicit rather than a pile of ``if``s**, so the skippable steps are a
property of the flow that a test can read, and so step 7 cannot drift out of sync with the
required-slot set.

## The draft is this feature's primary user's work, and it is treated that way

The authoring flow's user is the **domain expert** - *"who has material but does not run
infrastructure"* - and this is the only screen in the product whose primary user is not the
operator. Two consequences implemented here rather than described:

* **Persisted after every step** (FR-007.18), synchronously and atomically. The subtitle says
  *"progress is saved as you go"*, and a save that is claimed must have happened. Server-side,
  never only in browser state, so the draft survives closing the browser **and** a backend
  restart.
* **A failed commit leaves the draft intact** and returns ``pack_incomplete`` with every
  missing slot. Someone two hours into the flow must not lose their work to a validation
  failure, and re-entering a done step must not discard the later ones.

**Step 4 selects a configuration; it does not compose one** (FR-007.22). Rule terms, judge
prompt templates and reference sets are authored and versioned in miDataworks and cited by
hash; no endpoint in this feature accepts one as authored content. An operator with no
miDataworks supplies a configuration as an **external part**, which is the standalone path
and the only authoring-shaped escape hatch this feature has.

**A draft is not a pack.** It never appears in the registry listing, and it is resumable or
deletable at any time (E13).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from core.errors import refuse
from packs import registry, store
from packs.definition import (
    FilledSlot,
    PackVersionDefinition,
    slot_document,
    slot_from_document,
)
from packs.references import PART_REF_ADAPTER, PartRef
from packs.resolver import resolve_slots
from packs.slots import SLOT_ORDER, SlotKind
from packs.validation import PackValidity, validate

STATE_TODO = "todo"
STATE_CURRENT = "current"
STATE_DONE = "done"


@dataclass(frozen=True, slots=True)
class StepSpec:
    """One step of the flow."""

    number: int
    name: str
    slots: tuple[SlotKind, ...]
    skippable: bool
    #: What the panel tries, or ``None`` for a step with nothing to sample (1 and 7).
    panel: str | None


STEPS: tuple[StepSpec, ...] = (
    StepSpec(1, "describe", (), False, None),
    StepSpec(2, "prompts", (SlotKind.PROMPT_SET,), False, "sample"),
    StepSpec(3, "answers", (SlotKind.CORPUS, SlotKind.PREFERENCE_PAIRS), True, "sample"),
    StepSpec(4, "judgment", (SlotKind.SELECTOR_CONFIG, SlotKind.GRADER_CONFIG), True, "dry_run"),
    StepSpec(5, "improvement", (SlotKind.TEST_SET,), False, "scored_sample"),
    StepSpec(6, "retention", (SlotKind.RETENTION_SET,), False, "scored_sample"),
    StepSpec(7, "review", (), False, None),
)

STEP_COUNT = len(STEPS)


def step(number: int) -> StepSpec:
    """One step by number, refusing an unknown one rather than clamping.

    Raises:
        Refusal: ``pack_draft_step_unknown``. Clamping to the nearest step would silently
            commit the author's answers to a different step from the one they were looking
            at, which is the worst available failure for a seven-step form.
    """
    for spec in STEPS:
        if spec.number == number:
            return spec
    raise refuse(
        "pack_draft_step_unknown",
        f"There is no step {number}; the flow has {STEP_COUNT}, numbered 1 to {STEP_COUNT}.",
        step=number,
        steps=STEP_COUNT,
    )


@dataclass(frozen=True, slots=True)
class Draft:
    """An unfinished pack version. The authoring flow's whole state."""

    draft_id: str
    name: str | None
    version: str
    #: Nullable **and meant**: null asserts the pack is not domain-scoped, which a general
    #: pretraining pack genuinely is not. The flow therefore records whether step 1 was
    #: completed separately, in ``step_states`` - an absent answer and "no domain" are
    #: different facts and the schema's nullable-but-required ``domain`` exists to keep them
    #: apart.
    domain: str | None
    current_step: int
    step_states: dict[int, str]
    slots: dict[SlotKind, FilledSlot | None]
    created_at: datetime
    updated_at: datetime
    #: The last dry-run, **for display continuity only**, flagged advisory. It is never an
    #: input to validity: a pack is valid or not for structural reasons, and a selector that
    #: accepted everything is advice to the author rather than a defect in the composition.
    last_dry_run: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        """The draft as the API and the store carry it (one shape, so a restart loses nothing)."""
        return {
            "draft_id": self.draft_id,
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "current_step": self.current_step,
            "step_count": STEP_COUNT,
            "step_states": {
                str(number): state for number, state in sorted(self.step_states.items())
            },
            "slots": {slot.value: slot_document(self.slots.get(slot)) for slot in SLOT_ORDER},
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "last_dry_run": self.last_dry_run,
        }

    def subtitle(self) -> str:
        """The screen's subtitle, from the mockup: *"step 4 of 7 · progress is saved as you go"*."""
        return f"step {self.current_step} of {STEP_COUNT} · progress is saved as you go"


def create(*, name: str | None = None, version: str = "1", domain: str | None = None) -> Draft:
    """Start a draft. Persisted immediately, so a browser closed on step 1 loses nothing."""
    now = datetime.now(tz=UTC)
    draft = Draft(
        draft_id=uuid.uuid4().hex,
        name=name,
        version=version,
        domain=domain,
        current_step=1,
        step_states={spec.number: STATE_TODO for spec in STEPS} | {1: STATE_CURRENT},
        slots=dict.fromkeys(SLOT_ORDER),
        created_at=now,
        updated_at=now,
    )
    _persist(draft)
    return draft


def get(draft_id: str) -> Draft:
    """Read a draft.

    Raises:
        Refusal: ``pack_draft_not_found``. Named rather than returning ``None``: every
            caller of this would have to check, and one of them would forget.
    """
    payload = store.read_draft(draft_id)
    if payload is None:
        raise refuse(
            "pack_draft_not_found",
            f"No pack draft {draft_id!r} is in the pack store. A draft is deletable at any "
            f"time, so it may have been deleted; start a new one.",
            draft_id=draft_id,
        )
    return _from_payload(payload)


def commit_step(
    draft_id: str,
    number: int,
    *,
    name: str | None = None,
    domain: str | None = None,
    domain_given: bool = False,
    version: str | None = None,
    slots: Mapping[str, Any] | None = None,
    skip: bool = False,
) -> Draft:
    """Record one step's answers and persist. **Does not discard later steps.**

    Args:
        draft_id: The draft.
        number: Which step's answers these are.
        name: The pack's slug, at step 1.
        domain: The domain sentence, at step 1. May be ``None`` *meaningfully*.
        domain_given: Whether ``domain`` was supplied at all - so "not domain-scoped" (null)
            is distinguishable from "the field was not in the request". The schema's
            nullable-but-required ``domain`` makes that distinction load-bearing, and a
            single ``None`` parameter cannot carry it.
        version: The version to commit as, if not ``1``.
        slots: ``{slot kind: slot document | null}`` for the slots this step fills. A slot
            named here that the step does not fill is **refused**, not ignored: an author
            who believes step 3 set the test set has a different pack from the one they
            think they have.
        skip: Mark a skippable step skipped. Refused for a step that is not skippable.

    Returns:
        The persisted draft, advanced to the next step.

    Raises:
        Refusal: ``pack_draft_step_unknown``; ``pack_draft_step_not_skippable``;
            ``pack_draft_slot_not_in_step``.
        PackRefusal: ``pack_document_invalid`` for a slot document that is not one.
    """
    spec = step(number)
    draft = get(draft_id)

    if skip and not spec.skippable:
        raise refuse(
            "pack_draft_step_not_skippable",
            f"Step {number} ({spec.name}) fills {[slot.value for slot in spec.slots]}, and "
            f"those slots are required for a pack to be valid, so the step cannot "
            f"be skipped. Steps 3 and 4 can.",
            step=number,
            slots=[slot.value for slot in spec.slots],
        )

    new_slots = dict(draft.slots)
    for key, document in (slots or {}).items():
        slot = _slot_named(key)
        if slot not in spec.slots:
            raise refuse(
                "pack_draft_slot_not_in_step",
                f"Step {number} ({spec.name}) fills "
                f"{[item.value for item in spec.slots]} and this request set "
                f"{slot.value!r}. Refusing rather than accepting it quietly: a slot set "
                f"from the wrong step is a pack whose author believes a different step "
                f"filled it.",
                step=number,
                slot=slot.value,
                step_slots=[item.value for item in spec.slots],
            )
        new_slots[slot] = _filled_from(slot, document)
    if skip:
        for slot in spec.slots:
            new_slots[slot] = None

    states = dict(draft.step_states)
    states[number] = STATE_DONE
    # The NEXT step becomes current only if it has not already been done. Re-entering step 2
    # to correct a prompt set must not reset step 6 to `todo` and must not drag the author
    # back through the flow - FR-007.18's "without discarding later steps", implemented as
    # one condition rather than asserted in a comment.
    following = min(number + 1, STEP_COUNT)
    if states.get(following) != STATE_DONE:
        states[following] = STATE_CURRENT

    updated = Draft(
        draft_id=draft.draft_id,
        name=name if name is not None else draft.name,
        version=version if version is not None else draft.version,
        domain=domain if domain_given else draft.domain,
        current_step=following,
        step_states=states,
        slots=new_slots,
        created_at=draft.created_at,
        updated_at=datetime.now(tz=UTC),
        last_dry_run=draft.last_dry_run,
    )
    _persist(updated)
    return updated


def record_dry_run(draft_id: str, result: Mapping[str, Any]) -> Draft:
    """Cache the last dry-run on the draft, **for display only**.

    Flagged advisory by the payload it stores, and never read by :func:`packs.validation.validate`
    - a test asserts validity is unchanged by any dry-run outcome. The cache exists so that
    re-opening step 4 shows what was tried rather than an empty panel.
    """
    draft = get(draft_id)
    updated = Draft(
        draft_id=draft.draft_id,
        name=draft.name,
        version=draft.version,
        domain=draft.domain,
        current_step=draft.current_step,
        step_states=draft.step_states,
        slots=draft.slots,
        created_at=draft.created_at,
        updated_at=datetime.now(tz=UTC),
        last_dry_run={**dict(result), "advisory": True},
    )
    _persist(updated)
    return updated


def review(session: Any, draft_id: str) -> tuple[Draft, PackValidity]:
    """Step 7: **render** ``validate()``'s result. It does not re-implement validity.

    Returns the draft and the verdict, so the screen can list every missing slot by name
    (AC10) beside the step list.
    """
    draft = get(draft_id)
    definition = definition_of(draft)
    validity = validate(definition, resolve_slots(session, definition))
    return draft, validity


def commit(session: Any, draft_id: str) -> PackVersionDefinition:
    """Turn a draft into a committed pack version, or refuse **leaving the draft intact**.

    Raises:
        PackRefusal: ``pack_incomplete`` naming every missing slot, ``retention_floor_missing``,
            ``pack_part_unresolved``, or ``midataworks_not_configured`` - whichever applies,
            in :meth:`packs.validation.PackValidity.raise_if_invalid`'s order.
        Refusal: ``pack_draft_unnamed`` when step 1 was never completed, which is the one
            thing a pack cannot be committed without and which the schema's slug pattern
            would otherwise report as a regex failure.
    """
    draft = get(draft_id)
    if not draft.name:
        raise refuse(
            "pack_draft_unnamed",
            "This draft has no name, so there is nothing to commit it as. Step 1 names the "
            "domain; the name is the pack's identity and appears in the display rendering "
            "of every part addressed through it.",
            draft_id=draft_id,
        )
    definition = definition_of(draft)
    validity = validate(definition, resolve_slots(session, definition))
    # Refuse BEFORE writing anything, and do not touch the draft. Two hours of someone's
    # work is on the other side of this call.
    validity.raise_if_invalid()

    committed = registry.create(session, definition)
    store.delete_draft(draft_id)
    return committed


def listing() -> list[dict[str, Any]]:
    """Every draft. **Drafts are not packs** and never appear in the registry listing (E13)."""
    return [dict(payload) for payload in store.iter_drafts()]


def delete(draft_id: str) -> None:
    """Delete a draft. Always permitted: nothing cites a draft (E13)."""
    store.delete_draft(draft_id)


# ----------------------------------------------------------------------------- internals


def definition_of(draft: Draft) -> PackVersionDefinition:
    """The definition a draft would commit as.

    Built even when incomplete, because :func:`packs.validation.validate` is what names the
    missing slots and it needs a definition to read. ``name`` falls back to a placeholder the
    schema's pattern accepts so that an unnamed draft's refusal comes from
    :func:`commit`'s own check rather than from a regex.
    """
    return PackVersionDefinition(
        name=draft.name or "unnamed-draft",
        version=draft.version,
        domain=draft.domain,
        slots=dict(draft.slots),
        created_at=draft.created_at,
    )


def _slot_named(key: str) -> SlotKind:
    try:
        return SlotKind(key)
    except ValueError:
        raise refuse(
            "pack_draft_slot_unknown",
            f"{key!r} is not one of the seven slots "
            f"{[slot.value for slot in SLOT_ORDER]}. The slot set is the pack definition "
            f"contract's and a eighth slot is a schema change, not a request.",
            slot=key,
            slots=[slot.value for slot in SLOT_ORDER],
        ) from None


def _filled_from(slot: SlotKind, document: Any) -> FilledSlot | None:
    """One slot from the request, through the **same** reader the document uses.

    Deliberately not a second parser: a slot filled by the authoring flow and the same slot
    read back from an exported pack must be the same object, and two readers would agree
    until one of them changed.
    """
    return slot_from_document(slot, document)


def _persist(draft: Draft) -> None:
    store.write_draft(draft.draft_id, draft.payload())


def _from_payload(payload: Mapping[str, Any]) -> Draft:
    slots: dict[SlotKind, FilledSlot | None] = {}
    raw = payload.get("slots") or {}
    for slot in SLOT_ORDER:
        slots[slot] = _filled_from(slot, raw.get(slot.value))
    return Draft(
        draft_id=str(payload["draft_id"]),
        name=payload.get("name"),
        version=str(payload.get("version") or "1"),
        domain=payload.get("domain"),
        current_step=int(payload.get("current_step") or 1),
        step_states={
            int(number): str(state) for number, state in (payload.get("step_states") or {}).items()
        },
        slots=slots,
        created_at=_parse(str(payload["created_at"])),
        updated_at=_parse(str(payload["updated_at"])),
        last_dry_run=payload.get("last_dry_run"),
    )


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def refs_in(draft: Draft) -> Sequence[PartRef]:
    """Every reference the draft holds. For a UI that renders them before a commit."""
    return [filled.ref for filled in draft.slots.values() if filled is not None]


def ref_from_document(document: Mapping[str, Any]) -> PartRef:
    """Validate one reference document into a typed reference, discriminated on ``source``.

    Exposed so a route can refuse a malformed reference at the edge with the discriminator's
    own message rather than constructing a slot first.
    """
    return PART_REF_ADAPTER.validate_python(dict(document))
