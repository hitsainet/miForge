"""The authoring endpoints: drafts, steps, panels and the dry-run  (FR-007.17 - FR-007.22).

## There is no queue on this surface, and that absence is a requirement

**miForge keeps no review queue or review screen of its own** (R-02.36; the in-run queue is
miDataworks', BRD-03 R-03.41). So this module defines **no** queue route, no inbox, no
pending-decision store, no reviewer identity and no route that records a decision coming back
later. ``test_pack_no_review_queue.py`` asserts that absence against the **mounted** app's
route table rather than by reading this file.

The dry-run returns **within the request**. If it ever needs a worker, shrink the sample; do
not add a queue - *"a panel full of ``undecided`` is where the queue is most tempting and
where it is most clearly forbidden."*

## Step 4 selects a configuration; it does not compose one

No route here accepts rule terms, a judge prompt template or a reference set as **authored
content** (FR-007.22). A configuration body arrives to be *run* - validated against
``selector-config-v1.json`` by F3 and cited by hash - and an operator with no miDataworks
supplies one as an external part. A write endpoint for a configuration would be a second
authoring surface and a second hash, which R-02.10 forbids.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.agent_origin import requires_approval_when_agent
from api.deps import session_dependency
from packs.service import pack_service
from packs.slots import SlotKind
from services import approval_gates

router = APIRouter(prefix="/pack-drafts", tags=["packs"])

SessionDep = Annotated[Session, Depends(session_dependency)]


class CreateDraftRequest(BaseModel):
    """``POST /api/v1/pack-drafts``. Everything optional: step 1 is where the naming happens."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    version: str = "1"
    domain: str | None = None


class StepRequest(BaseModel):
    """One step's answers (FR-007.17).

    ``extra="forbid"`` so a misspelled field refuses rather than being silently dropped - an
    author who believes they set the test set and did not has a different pack from the one
    they think they have.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    version: str | None = None
    #: The domain sentence. **Null is meaningful** - it asserts the pack is not
    #: domain-scoped - so ``domain_given`` says whether the field was supplied at all. An
    #: absent key is an omission and a null value is a fact; the pack schema's
    #: nullable-but-required ``domain`` is the same distinction.
    domain: str | None = None
    domain_given: bool = False
    #: ``{slot kind: slot document | null}``, for the slots **this** step fills. A slot the
    #: step does not fill is refused, not ignored.
    slots: dict[str, Any] = Field(default_factory=dict)
    skip: bool = False


class DryRunRequest(BaseModel):
    """``POST /api/v1/pack-drafts/{id}/dry-run`` (FR-007.20, AC11)."""

    model_config = ConfigDict(extra="forbid")

    #: A ``selector-config-v1.json`` body, **selected rather than authored** (FR-007.22).
    config: dict[str, Any]
    #: The other configured selectors by name, for a composite's ``children`` - which the
    #: contract defines as *names of other configured selectors*, so a composite cannot be
    #: resolved without them.
    catalog: list[dict[str, Any]] = Field(default_factory=list)
    provider: str | None = None
    n: int | None = Field(default=None, ge=1)


class ScoredSampleRequest(BaseModel):
    """``POST /api/v1/pack-drafts/{id}/samples/{slot}/score`` - steps 5 and 6's panels."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    catalog: dict[str, dict[str, Any]] = Field(default_factory=dict)
    form: str = "selector"
    provider: str | None = None
    n: int | None = Field(default=None, ge=1)
    #: The **proposed** floor, at step 6: the panel says whether it is already breached,
    #: which is the one thing an author cannot guess (FR-007.19).
    floor: float | None = Field(default=None, ge=0.0, le=1.0)


@router.post("", status_code=201)
def create_draft(request: CreateDraftRequest) -> dict[str, Any]:
    """Start a draft. **Persisted immediately** (FR-007.18)."""
    draft = pack_service.create_draft(
        name=request.name, version=request.version, domain=request.domain
    )
    return draft.payload()


@router.get("")
def list_drafts() -> dict[str, Any]:
    """Every draft. **Drafts are not packs** and never appear in the registry listing (E13)."""
    return {"drafts": pack_service.drafts()}


@router.get("/{draft_id}")
def get_draft(draft_id: str) -> dict[str, Any]:
    """Resume a draft: the same step, with the same answers (FAC-8).

    Served from the pack store on disk, so a draft survives a backend restart - the subtitle
    claims progress is saved, and a claimed save must have happened.
    """
    draft = pack_service.draft(draft_id)
    return {**draft.payload(), "subtitle": draft.subtitle()}


@router.put("/{draft_id}/steps/{number}")
def commit_step(draft_id: str, number: int, request: StepRequest) -> dict[str, Any]:
    """Record one step and advance. **Later steps are not discarded** (FR-007.18)."""
    draft = pack_service.commit_step(
        draft_id,
        number,
        name=request.name,
        version=request.version,
        domain=request.domain,
        domain_given=request.domain_given,
        slots=request.slots,
        skip=request.skip,
    )
    return {**draft.payload(), "subtitle": draft.subtitle()}


@router.post("/{draft_id}/samples/{slot}")
def sample_slot(draft_id: str, slot: SlotKind, session: SessionDep) -> dict[str, Any]:
    """A seeded, stratified sample of one slot (FR-007.19).

    Seeded so re-opening the step shows the same rows - a sample that changed on every view
    would read as the data changing - and stratified through F4's own sampler, so the panel
    shows the sample a run would score rather than a head slice.
    """
    return pack_service.sample(session, draft_id, slot).payload()


@router.post("/{draft_id}/samples/{slot}/score")
def score_sample(
    draft_id: str, slot: SlotKind, request: ScoredSampleRequest, session: SessionDep
) -> dict[str, Any]:
    """A **scored** sample (step 5) or the current score against a proposed floor (step 6).

    The same path a run's evaluation takes, so the number an author sees here is the number
    the run will compare against the floor.
    """
    return pack_service.scored_sample(
        session,
        draft_id,
        slot,
        expectation_config=request.config,
        catalog=request.catalog,
        form=request.form,
        n=request.n,
        provider_name=request.provider,
        floor=request.floor,
    ).payload()


@router.post("/{draft_id}/dry-run")
def dry_run(draft_id: str, request: DryRunRequest, session: SessionDep) -> dict[str, Any]:
    """Answer a sample of the pack's **own** prompts and show a verdict with a reason (AC11).

    Returns **three counts and their denominator together**, inside F3's ``SelectionHealth``
    record passed through whole: ``accept`` / ``reject`` / ``undecided`` over ``n``. There is
    no separate ``accepted``/``sample_n`` pair, because a second place holding the counts is
    a second answer - and the UI cannot print a bare percentage when the denominator arrives
    beside the numerator (ADR-012).

    A human-review configuration comes back **all ``undecided`` with a note naming why**, and
    the step still saves: it answers from a queue this feature does not have.
    """
    return pack_service.dry_run(
        session,
        draft_id,
        config=request.config,
        catalog=request.catalog,
        provider_name=request.provider,
        n=request.n,
    ).payload()


@router.post("/{draft_id}/commit", status_code=201)
def commit_draft(draft_id: str, session: SessionDep) -> dict[str, Any]:
    """Commit the draft as a pack version, or refuse **leaving the draft intact**.

    Step 7 renders :func:`packs.validation.validate`'s result; this is where the refusal
    fires. ``pack_incomplete`` names every missing slot, and the draft is untouched - two
    hours of someone's work is on the other side of this call.
    """
    definition = pack_service.commit_draft(session, draft_id)
    return pack_service.get(session, definition.name, definition.version)


@router.get("/{draft_id}/validity")
def review_draft(draft_id: str, session: SessionDep) -> dict[str, Any]:
    """Step 7: what is still missing, by name (AC10). A 200 describing invalidity.

    **The path is ``/validity``, not ``/review``, and that is deliberate.** Step 7's screen
    is *"Review and save"* in the mockup's copy, but in this repository the word *review*
    names miDataworks' human-review queue - the thing FR-007.21 forbids this feature from
    having - and F10 ships a guard asserting that the only route mentioning it is its own
    read-only one. A pack-authoring route with ``review`` in its path would have made that
    guard red, and weakening another feature's guard to fit a URL would be the wrong fix:
    the guard is right, and this endpoint does not review anything - it reports validity.
    """
    draft, validity = pack_service.review_draft(session, draft_id)
    return {"draft": draft.payload(), "validity": validity.payload()}


@router.delete("/{draft_id}", status_code=204)
@requires_approval_when_agent(
    "delete",
    validate=approval_gates.validate_draft_delete,
    summary=approval_gates.summarise_draft_delete,
)
def delete_draft(draft_id: str) -> Response:
    """Delete a draft. Always permitted: nothing cites a draft (E13)."""
    pack_service.delete_draft(draft_id)
    return Response(status_code=204)
