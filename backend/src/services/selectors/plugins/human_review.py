"""In-run human review  (FR-003.7; R-02.9, R-02.36, R-02.50, BRD-03 R-03.41).

**This is the type that forces the third verdict value.** It is necessarily asynchronous:
at the moment it returns, the answer is ``undecided``, which is neither a soft accept nor a
soft reject. The contract's :class:`schemas.selector.Selector` Protocol has **no async
variant on purpose** - a second shape is a second contract - so this plugin returns
``undecided`` immediately, having enqueued the candidate, and **the caller reconciles the
decision later**.

## There is no path from a deadline to an accept

``timeout_seconds`` in the configuration bounds how long a *run* waits, and the contract
spells out the consequence: *"a timeout does not admit the candidate - it leaves it
undecided and excluded."* So this module never blocks and never converts the passage of
time into a verdict. Defaulting silence to accept would turn an unattended queue into an
unselected run; defaulting it to reject would silently discard every candidate a person
simply had not reached yet.

## The queue is miDataworks' and is reached through F10's one call site

miForge keeps **no review queue and no review screen** (R-02.36, BRD-03 R-03.41). This
module contains no ``httpx``, no URL and no header: it calls
:class:`services.selectors.review_port.ReviewQueuePort`, which **F10** implements at
ADR-010's single outbound call site. A static test asserts there is no second one.

## THE BINDING, AND THE GAP IT CLOSES  (ADR-016 #1)

:func:`bind_review_queue` was called in **eleven test files and nowhere in
``backend/src/``** until 2026-10-16, and
:class:`clients.midataworks_review.MiDataworksReviewQueue` was constructed only in its own
test. So :func:`current_review_queue` was **always** ``None``,
:meth:`HumanReviewSelector.prepare` **always** refused ``midataworks_not_configured`` -
*naming an absent service that was present*, with miDataworks configured and answering -
and ``GET /api/v1/selectors`` listed the type unavailable for the same reason. Dead with
it: ``verdicts_for_candidates``, :meth:`HumanReviewSelector.item_ids`,
:attr:`HumanReviewSelector.timeout_seconds` and ``MiDataworksClient.get_queue``. Every one
of those tests passed by binding its own stub - the sibling's sixteen unregistered MCP
tools, in this package.

**The binding is made on F10's side**, by :func:`clients.midataworks_review.wire`, and
**not here**: ``test_selector_guards.TestNoImportOfTheForbiddenPackages`` forbids every
module under ``services/selectors/`` from importing ``clients``, and its reason is this
exact temptation - *"depending on F10's concrete client instead of the port is how the port
stops being a port."* The same fence closes ``services/scoring/`` (F4 has no outbound call
site) and ``training/`` (AC20). All three were observed red against earlier sitings of the
binding, which is why it ended up where ADR-010 says the sibling lives.

## The queue is per run; this slot is a process global

The thing bound is :class:`clients.midataworks_review.RunScopedReviewQueue` rather than a
queue, because the two halves have different scopes: the queue id lands on **one run's**
binding row and the decisions are read back through it, while the API process answering
``GET /api/v1/selectors`` has no run id at all. So the bound port answers
:meth:`~clients.midataworks_review.RunScopedReviewQueue.available` with no run and resolves
the run itself from :func:`current_review_run`; **the call sites in this module are
unchanged.** A queue operation attempted outside a run refuses
``review_queue_run_unknown``, which is a different fact from an absent miDataworks and is
spelled differently.

**It is in-run review, not R-02.39's authoring dry-run.** The dry-run is miForge's, uses no
queue, and belongs to F7. Conflating the two is an error this project has already made once.

## The decision vocabulary is not interpreted here

:func:`schemas.selector.map_decision` is the single statement of how a human decision
becomes a verdict value - ``accept``/``reject`` pass through, ``override`` and ``skip``
become ``undecided`` carrying the label, and an unrecognised kind is a **named refusal**.
Two mappings of one decision set is how ``flag`` ends up meaning accept in one of them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from core.errors import not_configured, refuse
from schemas.selector import SelectorKind, Verdict, VerdictValue, map_decision
from services.selectors.resolve import ResolvedSelector
from services.selectors.review_port import ReviewCandidate, ReviewDecision, ReviewQueuePort
from services.selectors.settings import get_selector_settings

#: The label set every miForge-created queue carries. At least two values, which
#: miDataworks requires - verified against that repo's model
#: (``Field(min_length=2, max_length=64)``).
DEFAULT_LABEL_SET = ["accept", "reject"]

DEFAULT_QUESTION = "Should this completion be admitted into training?"


def derive_external_id(batch_id: str, index: int, completion: str) -> str:
    """A deterministic candidate id, so a retried post is idempotent.

    miDataworks returns the **existing** item for a repeated ``external_id`` - verified - so
    a network retry cannot create a duplicate and nobody is asked to read the same
    candidate twice. Derived from the batch, the position **and** the completion digest:
    position alone would collide across a re-assembled batch, and the digest alone would
    merge two identical completions, which are two candidates.
    """
    digest = hashlib.sha256(completion.encode("utf-8")).hexdigest()[:16]
    return f"{batch_id}:{index}:{digest}"


# --------------------------------------------------------------------------------------
# The port binding. One slot, like the provider resolver - F10 installs its client here.
# --------------------------------------------------------------------------------------

_port: ReviewQueuePort | None = None


def bind_review_queue(port: ReviewQueuePort | None) -> None:
    """Install F10's miDataworks client. ``None`` unbinds it.

    The unbind path exists and is used: a test that configured the queue and could not
    configure it *away* would make the standalone case (AC20) unassertable.
    """
    global _port
    _port = port


def current_review_queue() -> ReviewQueuePort | None:
    """The bound port, or ``None`` when miDataworks is not wired - which is **normal**.

    **No lazy default here.** Falling back to F10's ``wire`` on a miss would make
    :func:`bind_review_queue` ``(None)`` a no-op, and the unbind path *"exists and is
    used"*: eleven test files use it to express the unconfigured world (AC20), and a getter
    that quietly re-bound would make that case unassertable. It would also need a
    ``clients`` import, which the layer guard forbids.
    """
    return _port


#: The run whose review queue the bound port speaks for, or ``None`` outside a run.
#:
#: A :class:`~contextvars.ContextVar` rather than a plain global, for two reasons that both
#: bite. :meth:`~contextvars.ContextVar.reset` restores the *previous* value rather than
#: clearing to ``None``, so the scope nests and one run cannot leak into the next; and a
#: ContextVar does not propagate into a thread nobody told about it, which is the behaviour
#: wanted here - a worker is ``--pool=solo -c 1`` (ADR-006), and a run id visible from an
#: unrelated thread would be a run id some other run's code could post candidates under.
#:
#: **It lives here and not in** :mod:`clients.midataworks_review` **because
#: :func:`training.run_driver.drive_recorded_run` is what opens it**, and ``training/**``
#: may not import ``clients.*`` - ``test_F1_has_no_outbound_sibling_call_site``.
_run_id: ContextVar[str | None] = ContextVar("miforge_review_run_id", default=None)


@contextmanager
def review_run(run_id: str) -> Iterator[str]:
    """Scope the bound review-queue port to one run, for the duration of the block.

    Entered once per run by :func:`training.run_driver.drive_recorded_run`, around
    everything that can resolve a selector - which includes F2's source pre-flight and not
    only the step loop, because ``preflight`` resolves the configuration and
    :meth:`HumanReviewSelector.prepare` runs there.

    Args:
        run_id: The run. Refused empty rather than scoped to a falsy value: a scope
            indistinguishable from no scope is worse than no scope, because the refusal it
            suppresses is the one naming what is missing.

    Yields:
        The scoped run id, so a caller can assert what it entered.

    Raises:
        Refusal: ``review_queue_run_unknown`` for an empty run id.
    """
    if not run_id:
        raise refuse(
            "review_queue_run_unknown",
            "A review-queue scope was opened with no run id. The queue id lands on that "
            "run's binding row, so an unnamed run has nowhere to record it and no way to "
            "read decisions back. Pass the run id.",
        )
    token = _run_id.set(run_id)
    try:
        yield run_id
    finally:
        _run_id.reset(token)


def current_review_run() -> str | None:
    """The run the bound port speaks for, or ``None`` outside a run. ``None`` is normal."""
    return _run_id.get()


class HumanReviewSelector:
    """Satisfies :class:`schemas.selector.Selector`. Returns ``undecided`` and enqueues."""

    kind: SelectorKind = "human_review"

    def __init__(self, resolved: ResolvedSelector) -> None:
        self._resolved = resolved
        #: ``external_id`` -> decision, filled by :meth:`prefetch` for a whole batch.
        self._decisions: dict[str, ReviewDecision] = {}
        self._items: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self._resolved.name

    @property
    def queue(self) -> str:
        return str(self._resolved.body["queue"])

    @property
    def timeout_seconds(self) -> int | None:
        value = self._resolved.body.get("timeout_seconds")
        return None if value is None else int(value)

    def prepare(self, resolved: ResolvedSelector, *, run_target: Any = None) -> None:
        """Refuse **by name** when miDataworks is absent (E-9, R-02.50, ADR-010)."""
        del run_target
        port = self._require_port()
        resolved.bindings["review_queue"] = port

    def _require_port(self) -> ReviewQueuePort:
        port = current_review_queue()
        if port is None or not port.available():
            # ADR-010's spelling, and the ONLY thing that happens: no local queue is
            # invented, no verdict is defaulted, and the other four types are unaffected.
            raise not_configured("midataworks", "in-run human review (the human_review selector)")
        return port

    # ---------------------------------------------------------------- batch path
    def prefetch(
        self,
        prompts: list[str],
        completions: list[str],
        *,
        batch_id: str,
        start_index: int = 0,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Post a whole batch and read back whatever has already been decided.

        ``is_cancelled`` is checked before each chunk is posted, so a cancelled run does not
        keep putting candidates in front of a person (suite tracker 4.28).

        Batched at **at most 500 candidates per post** - miDataworks' cap, verified - so a
        large batch is chunked rather than refused or silently truncated.

        Called by the evaluation service before the per-completion loop, and by
        :meth:`evaluate` for a single candidate nothing has posted yet. ``start_index``
        exists for that second caller: the ``external_id`` is derived from the candidate's
        position in its **batch**, so a one-candidate post must say which position it is or
        it would mint a different id from the batch post and the person would be asked
        twice.

        The Protocol stays one method wide; this is an extra entry point the service uses
        when a plugin offers it, not a second contract.
        """
        port = self._require_port()
        queue_id = port.ensure_queue(
            name=self.queue, question=DEFAULT_QUESTION, label_set=list(DEFAULT_LABEL_SET)
        )
        settings = get_selector_settings()
        limit = settings.selector_max_text_chars

        candidates: list[ReviewCandidate] = []
        for offset, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
            index = start_index + offset
            for field, text in (("prompt", prompt), ("completion", completion)):
                if len(text) > limit:
                    raise refuse(
                        "selector_text_too_long",
                        f"A review candidate's {field} is {len(text)} characters, over "
                        f"miDataworks' {limit}-character bound. Refusing rather than "
                        f"truncating: a person asked to judge a truncated answer is "
                        f"judging a different answer.",
                        field=field,
                        length=len(text),
                        limit=limit,
                    )
            candidates.append(
                ReviewCandidate(
                    external_id=derive_external_id(batch_id, index, completion),
                    prompt=prompt,
                    completion=completion,
                    provenance={"selector": self.name, "batch_id": batch_id},
                )
            )

        chunk = settings.review_batch_max
        for start in range(0, len(candidates), chunk):
            if is_cancelled is not None and is_cancelled():
                return
            for item in port.post_candidates(queue_id, candidates[start : start + chunk]):
                self._items[item.external_id] = item.item_id

        external_ids = [candidate.external_id for candidate in candidates]
        # One read, not a poll. The contract says this type returns undecided immediately;
        # a decision made since a previous round is picked up because the repost is
        # idempotent, so the second pass over the same batch sees it.
        for decision in port.read_decisions(queue_id, external_ids=external_ids):
            self._decisions[decision.external_id] = decision

    def evaluate(self, *, prompt: str, completion: str, context: dict[str, Any]) -> Verdict:
        """The mapped decision when there is one, ``undecided`` otherwise.

        **It enqueues the candidate if nothing has**, which is the contract's own wording:
        *"``human_review`` returns ``undecided`` immediately, **having enqueued the
        candidate**."* :meth:`prefetch` is an optimisation for a whole batch, not the only
        route - and it cannot be the only route, because a composite child is reached one
        completion at a time after its cheaper siblings have had their say.

        *Found by a test:* without this, a candidate inside a multi-child composite came
        back ``undecided`` with the reason *"queued for review"* **while nothing had been
        posted.** A verdict that misstates what happened is worse than a refusal, because
        the operator would wait for a decision on a candidate no person could ever see.

        ``undecided`` carries the queue item id in its ``label`` so the caller can
        reconcile it later. It carries **no score** - the contract refuses that combination
        at the type, because a score on a pending candidate is how a consumer reconstructs
        the forbidden path from undecided to admitted.
        """
        index = int(context.get("index", 0))
        batch_id = str(context.get("batch_id", "batch"))
        external_id = derive_external_id(batch_id, index, completion)

        if external_id not in self._items:
            self.prefetch([prompt], [completion], batch_id=batch_id, start_index=index)

        decision = self._decisions.get(external_id)
        if decision is not None:
            return map_decision(decision.decision, label=decision.label, selector=self.name)

        item_id = self._items.get(external_id)
        return Verdict(
            value=VerdictValue.UNDECIDED,
            selector=self.name,
            label=item_id or external_id,
            reason=(
                f"waiting on review item {item_id or external_id}"
                if item_id
                else f"queued for review as {external_id}"
            ),
        )

    def decision_ids(self) -> dict[str, str]:
        """``external_id -> decision_id`` for the batch record (R-02.36's last sentence)."""
        return {
            external_id: decision.decision_id for external_id, decision in self._decisions.items()
        }

    def item_ids(self) -> dict[str, str]:
        """``external_id -> queue item id``, for an ``undecided`` verdict's reconciliation."""
        return dict(self._items)


def factory(resolved: ResolvedSelector) -> HumanReviewSelector:
    return HumanReviewSelector(resolved)
