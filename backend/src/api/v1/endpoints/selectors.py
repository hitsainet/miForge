"""The selector endpoints  (FR-003.12; R-02.9, R-02.10, ADR-013).

Two **read** paths, and no write path.

* ``GET /api/v1/selectors`` - the live registry's contents, each type with what it requires
  and whether that requirement is **currently satisfied**, naming the unsatisfied service.
* ``POST /api/v1/selectors/evaluate`` - evaluate a set of completions against a
  configuration, returning verdicts in input order plus the health record. **This is the one
  call F7's dry-run (R-02.39) makes.** F7 owns the sample, the flow and the screen; F3 owns
  the verdicts and their reasons.

## There is no write endpoint, and that is a requirement rather than an omission

F3 authors no configuration - that is miDataworks' (BRD-03 R-03.48). An endpoint accepting
a configuration for storage would create a **second authoring surface and a second hash**,
which R-02.10 forbids: the hash recorded is the handoff contract's, not one miForge invents.

## Refusals go through the envelope, never a bare ``HTTPException``

``core.errors.Refusal`` is installed as an application-level handler (see
:mod:`api.app`), so a refusal raised three calls down arrives at the client in one shape. A
bare ``HTTPException`` here would bypass it and produce a second error shape on the same
API - and a client that has to recognise two shapes is a client that will recognise one.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from services.selectors import ensure_registered, registry
from services.selectors.resolve import resolve
from services.selectors.service import evaluate
from services.selectors.settings import get_selector_settings
from services.selectors.types import CapabilityView, EvaluationContext

router = APIRouter(prefix="/selectors", tags=["selectors"])

# The import side effect is triggered explicitly rather than relied upon. An import side
# effect nobody triggers is not a side effect, and the symptom would be an empty listing
# on a cold process.
ensure_registered()


class CompletionItem(BaseModel):
    """One completion to evaluate, with whatever the configuration needs beside it."""

    model_config = ConfigDict(extra="forbid")

    completion: str
    #: The prompt it answers. A judge reads it; a rule does not. Defaulting it to empty
    #: rather than refusing, because four of the five types genuinely do not need it - but
    #: a judge configuration with empty prompts is judging an answer to nothing, which the
    #: dry-run's own output makes visible.
    prompt: str = ""
    #: The reference answer for a reference-match configuration (R-02.16's item carries it,
    #: not the selector's configuration - there is no reference field in the contract).
    reference: str | None = None


class EvaluateRequest(BaseModel):
    """``POST /api/v1/selectors/evaluate``. ``extra="forbid"`` so a misspelling refuses.

    A silently-ignored field in a *selector* request is worse than elsewhere: the caller
    believes a configuration is in force, and the verdicts come back from a different one.
    """

    model_config = ConfigDict(extra="forbid")

    #: A ``selector-config-v1.json`` body. Validated against the vendored schema, and
    #: refused with the failing JSON pointer.
    config: dict[str, Any]
    #: The other configured selectors by name, for a composite's ``children`` - which the
    #: contract defines as *"names of other configured selectors"*, so a composite cannot
    #: be resolved without them.
    catalog: list[dict[str, Any]] = Field(default_factory=list)
    items: list[CompletionItem]
    #: Where a locally supplied configuration came from (R-02.40's honesty rule). A local
    #: hash is never recorded as a contract reference, and this says what it *is*.
    source: str = "inline"


class VerdictOut(BaseModel):
    """One verdict as the wire carries it, with its provenance."""

    model_config = ConfigDict(extra="forbid")

    value: str
    selector: str
    #: The stable machine code a UI maps on - a mapping keyed on prose breaks on a wording
    #: change.
    reason_code: str
    #: Operator-facing, naming the thing that fired.
    reason: str | None
    score: float | None
    judge: dict[str, Any] | None = None
    caveats: list[str] = Field(default_factory=list)
    match_mode: str | None = None
    match_metric: str | None = None
    review_item_id: str | None = None


class EvaluateResponse(BaseModel):
    """Verdicts in **input order**, the health record, and the configuration's identity."""

    model_config = ConfigDict(extra="forbid")

    config_sha256: str
    config_kind: str
    hash_provenance: str
    verdicts: list[VerdictOut]
    health: dict[str, Any]
    hashes: dict[str, Any]


@router.get("", response_model=list[CapabilityView])
def list_selectors() -> list[CapabilityView]:
    """The live registry's contents, with **live** availability (FR-003.12).

    Availability is computed from settings **per call**, not cached at import, so an
    operator who configures miDataworks sees the human-review type become available without
    a restart of this layer's logic.

    An unavailable type is **listed, naming the service it needs** - never omitted and never
    a mute disabled control. An option missing from a response is indistinguishable from one
    that is broken (ADR-013).
    """
    ensure_registered()
    return registry.list_capabilities(get_selector_settings())


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_completions(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate a set of completions. **The call F7's dry-run makes.**

    Synchronous, with the count capped from configuration: a dry-run is five prompts, not
    five thousand, and an unbounded synchronous endpoint over a judge is a request that
    outlives its client.
    """
    ensure_registered()
    catalog = {str(body["name"]): body for body in request.catalog if "name" in body}
    resolved = resolve(request.config, catalog=catalog, source=request.source)

    context = EvaluationContext(
        batch_id="dry-run",
        items=tuple(
            {"reference": item.reference} if item.reference is not None else {}
            for item in request.items
        ),
    )
    result = evaluate(
        [item.completion for item in request.items],
        resolved,
        context,
        prompts=[item.prompt for item in request.items],
    )

    return EvaluateResponse(
        config_sha256=resolved.config_sha256,
        config_kind=resolved.kind,
        hash_provenance=resolved.hash.provenance,
        verdicts=[
            VerdictOut(
                value=record.verdict.value.value,
                selector=record.verdict.selector,
                reason_code=record.reason_code,
                reason=record.verdict.reason,
                score=record.verdict.score,
                judge=record.judge.model_dump(mode="json") if record.judge else None,
                caveats=list(record.caveats),
                match_mode=record.match_mode,
                match_metric=record.match_metric,
                review_item_id=record.review_item_id,
            )
            for record in result.records
        ],
        health=result.health.model_dump(mode="json"),
        hashes=result.hashes.model_dump(mode="json"),
    )
