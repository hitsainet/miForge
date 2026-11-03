"""The review-queue port  (FR-003.7; R-02.36, BRD-03 R-03.41, ADR-010).

**miForge keeps no review queue and no review screen.** The queue is miDataworks' - that is
R-02.36 and BRD-03 R-03.41 - and rebuilding it here is the error this project has already
made once. What F3 declares is the narrow interface its ``human_review`` plugin calls;
**F10 implements it** at ADR-010's single outbound call site in ``clients/``.

So this module contains no ``httpx``, no URL and no header. A static test asserts that is
still true, because "F10 is not ready yet so I will just call it directly" is how a port
stops being a port and the seam stops being findable.

## The two activities that are not the same activity

This port serves **in-run review** (R-02.36): a person adjudicating generated candidates
while a run is stepping. It is **not** the **authoring dry-run** of R-02.39, where the
operator checks a selector's verdicts on a sample before any run exists - that is
**miForge's**, uses **no queue**, and belongs to F7. Different call sites, different
records, different tests.

## The mapping is F10's and is consumed, not re-derived

miDataworks' decisions are not two-valued. :data:`schemas.selector.DECISION_TO_VERDICT` is
the single statement of how one becomes a verdict value - ``accept``/``reject`` pass
through, ``override`` and ``skip`` become ``undecided``, and an unrecognised kind is a named
refusal. ``human_review.py`` calls :func:`schemas.selector.map_decision` and interprets no
decision vocabulary of its own: two mappings of one decision set is how ``flag`` ends up
meaning accept in one of them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class ReviewCandidate(BaseModel):
    """One completion sent to a person.

    ``external_id`` is **derived deterministically** from the batch and the completion
    (see :func:`services.selectors.plugins.human_review.derive_external_id`) because
    miDataworks returns an existing item for a repeated ``external_id`` - so a retried post
    is idempotent and nobody is asked to read the same candidate twice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    external_id: str
    prompt: str
    completion: str
    #: Where the candidate came from, for the person reading it. Never a filesystem tour
    #: and never a credential (ADR-020).
    provenance: dict[str, Any] = {}


class ReviewDecision(BaseModel):
    """What came back. ``decision`` is miDataworks' vocabulary, mapped by the contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    external_id: str
    #: Stable identifier of the decision, recorded on the batch record per R-02.36's last
    #: sentence.
    decision_id: str
    #: ``accept`` | ``reject`` | ``override`` | ``skip``, per miDataworks' shipped routes.
    #: An unrecognised kind is refused by :func:`schemas.selector.map_decision`, not
    #: defaulted.
    decision: str
    #: Free text from the queue's ``label_set`` (2-64 free strings - verified). Rides
    #: **beside** the verdict value, never as a fourth enum member.
    label: str | None = None
    reason: str | None = None


class ReviewItem(BaseModel):
    """The queue item a candidate became. Its id is what an ``undecided`` verdict carries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    external_id: str
    item_id: str


@runtime_checkable
class ReviewQueuePort(Protocol):
    """Three methods, the shape miDataworks already serves.

    Narrow by design. Every method raises a named ``Refusal`` rather than returning a
    sentinel - ``midataworks_not_configured`` when the service is absent, which is
    **normal** (ADR-010), and a transport refusal when it is configured but unreachable.
    **Never a default verdict**: a fabricated accept trains on an answer nobody gave.
    """

    def available(self) -> bool:
        """Whether miDataworks is configured **right now**.

        Separate from construction so the type can be *listed* as unavailable by name
        rather than being absent from the list - an option missing from a response is
        indistinguishable from one that is broken (ADR-013).
        """
        ...

    def ensure_queue(self, *, name: str, question: str, label_set: list[str]) -> str:
        """Create or find the queue, returning its id. ``label_set`` needs >= 2 values."""
        ...

    def post_candidates(self, queue_id: str, candidates: list[ReviewCandidate]) -> list[ReviewItem]:
        """Post at most 500 candidates. Idempotent per ``external_id``."""
        ...

    def read_decisions(
        self, queue_id: str, *, external_ids: list[str] | None = None
    ) -> list[ReviewDecision]:
        """Read decisions back. A candidate with no decision is simply absent."""
        ...
