"""F10's implementation of F3's review-queue port  (F10 task 7.1; FR-010.21-23).

F3 declares :class:`services.selectors.review_port.ReviewQueuePort` and its own module
docstring says *"**F10 implements it** at ADR-010's single outbound call site in
``clients/``"*. This is that implementation, and the division it rests on is the one both
features agreed before either started:

**F3 owns the Protocol, the registry, the ``Verdict`` type and the ``human_review``
plugin. F10 owns everything behind the port.** Nothing above the interface is F10's, and
nothing below it is F3's - which is what stops two modules owning one selector.

So this class holds no verdict logic: mapping a decision to a verdict is
:mod:`services.review.verdicts`, and the durable run state - the queue id, the external-id
map, the watermark, the counts - is :mod:`services.review.binding`. What is here is the
narrowing of four port methods onto the one outbound call site.

**``available()`` is a question, not a construction.** F3 lists the ``human_review``
selector as *unavailable by name* when miDataworks is unconfigured rather than omitting it,
because an option missing from a response is indistinguishable from one that is broken
(ADR-013). So this method must answer without opening a client, and it does.

**The queue is created by miForge, not by the operator.** ``origin_app`` is set from the
*creating* caller's agent header and the poster's app must match it - verified in their
``external_service.post_candidates`` (``ORIGIN_APP_REQUIRED``, ``ORIGIN_APP_MISMATCH``) and
in the ``CheckConstraint("kind <> 'external' OR origin_app IS NOT NULL")`` on their model.
An operator-created queue is therefore one **miForge cannot post to**, which settles OQ-2's
first half on a mechanism rather than a preference.

**OQ-2's second half - one queue per run, or one per domain - is answered "per run" here**,
and the data model does not foreclose the other: the binding row is keyed by ``run_id``, so
a later decision to share a queue across a domain's runs becomes a different lookup, not a
migration. Recorded as a decision rather than left implicit.

## THE PORT, AND THE GAP IT CLOSES  (ADR-016 #1)

:class:`MiDataworksReviewQueue` was constructed in **one test file** and nowhere in
``backend/src/``, and ``services.selectors.plugins.human_review.bind_review_queue`` was
called in eleven test files and nowhere in ``backend/src/``. So
``current_review_queue()`` was **always** ``None`` and
``HumanReviewSelector.prepare`` **always** refused ``midataworks_not_configured`` -
naming an absent service that was present, with miDataworks configured, reachable and
answering. ``GET /api/v1/selectors`` listed the type unavailable for the same reason.
Dead with it: ``verdicts_for_candidates``, ``HumanReviewSelector.item_ids``,
``.timeout_seconds`` and ``MiDataworksClient.get_queue``. Every one of those tests passed
by constructing the class directly - the sibling's sixteen unregistered MCP tools, in this
package.

**A single global binding cannot be the fix, and that is the whole design problem.** This
class needs a ``run_id`` - the queue id lands on *that run's* binding row, and losing it
loses the ability to read decisions back - while the port slot is a process global and the
**API process answering availability has no run id at all.** So the slot holds
:class:`RunScopedReviewQueue`: availability answered without a run, the three queue
operations answered within one.
:func:`services.selectors.plugins.human_review.review_run` is the scope, entered once per
run by :func:`training.run_driver.drive_recorded_run`, and this class is handed a callable
that reads it - **never an import of it**, because the scope has to live on the selector
side for ``training/**`` to be able to open it without importing ``clients.*``.

**Where the binding is made, and what is still owed.** :func:`wire` at the foot of this
module, as an import side effect - and *nothing in ``backend/src/`` imports this module*,
which was the original bug and is the one half that cannot be closed from inside this
change's permitted file set. :func:`wire`'s own docstring carries the owed line verbatim
and names the four fences that rule out every other home for it.

**Why a scope rather than a factory-shaped port.** F3's
:class:`services.selectors.review_port.ReviewQueuePort` is a four-method Protocol that
``human_review.py`` and ``services.selectors.availability`` both call directly, and
``services.selectors.resolve.resolve`` - which is what invokes ``prepare`` - carries no
``run_id`` and never has. A factory-shaped port would therefore have had to change the
Protocol, both of its callers and the resolver's signature to thread a run id that only
one of five selector types wants. The scope changes none of them: the port keeps its
shape, the plugin keeps its call sites, and the run id travels with the run rather than
with the request.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from clients import availability
from clients.midataworks import MiDataworksClient
from core.errors import refuse
from services.review import binding as binding_service

if TYPE_CHECKING:  # pragma: no cover - typing only
    # `httpx` is imported for a TYPE ANNOTATION ONLY, under TYPE_CHECKING, and
    # `test_midataworks_one_call_site.py` excludes typing-only imports for a stated reason:
    # an import that does not exist at runtime cannot issue a request, so it is not a second
    # outbound call site. The transport this module accepts is passed straight through to
    # `MiDataworksClient`, which is the one site.
    import httpx
    from sqlalchemy.orm import Session

log = logging.getLogger("miforge.clients.midataworks_review")

#: The action string every refusal from this seam names, in the operator's words.
ACTION = "running with the human-review selector"
CAPABILITY = "human_review_selector"

SessionScope = Callable[[], AbstractContextManager["Session"]]


class MiDataworksReviewQueue:
    """The review queue, as F3's port sees it.

    Args:
        run_id: The run this queue belongs to. Required, because the queue id must land on
            **that run's** binding row: losing it loses the ability to read decisions back,
            and a request-scoped value is exactly how it gets lost.
        session_scope: How to open a transaction. Defaults to
            :func:`db.session.session_scope`, imported lazily so that importing this module
            needs no database - the no-sibling-import guard, the AST tests and the schema
            sync test all import across the tree.
        transport: An ``httpx`` transport, for tests. Production passes ``None``.
    """

    def __init__(
        self,
        *,
        run_id: str,
        session_scope: SessionScope | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.run_id = run_id
        self._transport = transport
        self._scope = session_scope

    # ------------------------------------------------------------------ plumbing

    def _session(self) -> AbstractContextManager[Session]:
        if self._scope is not None:
            return self._scope()
        from db.session import session_scope

        return session_scope()

    def _client(self) -> MiDataworksClient:
        """A client, or the ``midataworks_not_configured`` refusal from its constructor."""
        return MiDataworksClient(action=ACTION, capability=CAPABILITY, transport=self._transport)

    # ------------------------------------------------------------------ the port

    def available(self) -> bool:
        """Whether miDataworks is configured **right now**. Never raises."""
        return availability.is_configured()

    def ensure_queue(self, *, name: str, question: str, label_set: list[str]) -> str:
        """Create or find this run's external queue, returning its id.

        Idempotent: the binding row is read first, and an existing queue is returned
        without a round trip. Creating a second queue for a run would orphan the first
        **and its decisions**, which are the only copy - miForge stores no decision content
        as authority.

        The queue id is written in the same transaction as the lookup, so a crash between
        creating the queue at miDataworks and recording it here is the one case that can
        leave an orphan queue at miDataworks. That is the safe direction: an unreferenced
        queue is visible and harmless, while a recorded id for a queue that does not exist
        would make every read-back refuse.

        Raises:
            Refusal: ``midataworks_not_configured`` when unset - **before** any row is
                written (FR-010.26's rule applied here as well as at submission);
                ``review_label_set_too_small`` when the label set cannot pose a question;
                the client's transport refusals otherwise.
        """
        if len(label_set) < 2:
            raise refuse(
                "review_label_set_too_small",
                f"A review question needs at least two labels to be answerable and this "
                f"one has {len(label_set)}. miDataworks requires 2-64; refusing here so "
                f"the run says why rather than surfacing their 422 on the first post.",
                label_set=list(label_set),
            )

        with self._session() as session:
            existing = binding_service.find_binding(session, self.run_id)
            if existing is not None:
                return existing.queue_id

        with self._client() as api:
            created = api.create_external_queue(
                question=question,
                label_set=label_set,
                # Their `external_ref` is a free-form object. The run id is what makes a
                # queue findable from miForge's side and vice versa; naming the app as
                # well means an operator looking at the queue list in miDataworks can see
                # which installation created it.
                external_ref={"app": "miforge", "run_id": self.run_id, "name": name},
            )
        queue_id = str(created.get("id") or "")
        if not queue_id:
            raise refuse(
                "review_queue_create_failed",
                "miDataworks accepted the queue creation but returned no id, so there is "
                "nothing to post to. Refusing rather than proceeding with an empty "
                "identifier.",
                run_id=self.run_id,
                response_keys=sorted(created),
            )

        with self._session() as session:
            binding_service.record_binding(
                session,
                run_id=self.run_id,
                queue_id=queue_id,
                question=question,
                label_set=label_set,
            )
        log.info(
            "review_queue_created",
            extra={
                "code": "review_queue_created",
                "run_id": self.run_id,
                "queue_id": queue_id,
                "origin_app": availability.agent_identity(),
            },
        )
        return queue_id

    def post_candidates(self, queue_id: str, candidates: list[Any]) -> list[Any]:
        """Post candidates, splitting to miDataworks' limits **before** sending.

        Accepts F3's ``ReviewCandidate`` objects - anything carrying ``external_id``,
        ``prompt``, ``completion`` and ``provenance``. The ids are F3's to derive and are
        used as given: deriving them again here would be a second derivation, and two
        derivations of an idempotency key is how a re-post creates duplicates.

        Returns:
            F3's ``ReviewItem`` records, one per candidate, including the ones miDataworks
            already had (``created=False`` on its side) - because "we posted 500 and 500
            already existed" and "we posted 500 new ones" are different facts about a
            resumed run, and a caller that saw only new items would re-post forever.
        """
        from services.selectors.review_port import ReviewItem

        payloads = [
            binding_service.candidate_payload(
                str(candidate.external_id),
                binding_service.Candidate(
                    prompt=str(candidate.prompt),
                    completion=str(candidate.completion),
                    provenance=dict(getattr(candidate, "provenance", {}) or {}),
                ),
            )
            for candidate in candidates
        ]
        batches = binding_service.split_batches(payloads)

        items: list[ReviewItem] = []
        accepted: list[tuple[str, str]] = []
        step = _step_of(candidates)
        with self._client() as api:
            for batch in batches:
                for acknowledged in api.post_candidates(queue_id, batch):
                    accepted.append((acknowledged.external_id, acknowledged.item_id))
                    items.append(
                        ReviewItem(
                            external_id=acknowledged.external_id, item_id=acknowledged.item_id
                        )
                    )

        if accepted:
            with self._session() as session:
                bound = binding_service.require_binding(session, self.run_id)
                binding_service.record_posted(session, bound, step=step, accepted=accepted)
        return items

    def read_decisions(self, queue_id: str, *, external_ids: list[str] | None = None) -> list[Any]:
        """Read decisions back by watermark **and** by explicit pending ids - in **two** calls.

        Two, not one, and this is the subtlest thing in the feature. **miDataworks ANDs the
        two filters**: its ``read_back`` applies
        ``WHERE external_id IN (...) AND created_at > since`` when both are given (verified
        in ``services/review/external_service.py``). So a single request carrying both is
        equivalent to the watermark alone for the pending set, and it **loses exactly the
        case the second filter exists to catch**: a decision on a still-pending candidate
        whose ``created_at`` precedes the cursor - which happens whenever miForge posts a
        candidate late, or a reviewer answers an old item after newer ones.

        This feature's own FTID said *"``since=read_back_cursor`` catches the common case;
        ``external_ids=[...]`` for the still-pending set catches a decision whose
        ``created_at`` precedes a candidate miForge posted late"* without noticing that one
        request cannot do both. Found by a test, not by re-reading.

        So:

        1. ``since=cursor`` - everything new, cheaply;
        2. ``external_ids=<still pending>`` with **no** ``since`` - the pending set whatever
           its age.

        The union is recorded. One extra request per poll is the whole cost, against a lost
        decision being a candidate that stays pending forever while a human believes they
        answered it.

        The cursor is advanced **after** the decisions are committed, in a second
        transaction. An advanced cursor over an uncommitted write loses a decision
        permanently.

        Returns:
            F3's ``ReviewDecision`` records for everything newly known.
        """
        from services.selectors.review_port import ReviewDecision

        with self._session() as session:
            bound = binding_service.require_binding(session, self.run_id)
            since = bound.read_back_cursor
            ids = (
                list(external_ids)
                if external_ids
                else binding_service.pending_external_ids(session, bound)
            )

        with self._client() as api:
            # De-duplicated by decision id, keeping the first - the two reads overlap by
            # design and a decision counted twice would double the `decided` count.
            seen: dict[str, Any] = {}
            for record in api.read_decisions(queue_id, since=since):
                seen[record.decision_id] = record
            if ids:
                for record in api.read_decisions(queue_id, external_ids=ids):
                    seen.setdefault(record.decision_id, record)
            decisions = list(seen.values())

        with self._session() as session:
            bound = binding_service.require_binding(session, self.run_id)
            read_back = binding_service.record_decisions(session, bound, decisions)

        # A separate transaction, deliberately: the cursor may only move once the rows it
        # covers are durable.
        with self._session() as session:
            bound = binding_service.require_binding(session, self.run_id)
            binding_service.advance_cursor(session, bound, read_back.watermark)

        log.info(
            "review_read_back",
            extra={
                "code": "review_read_back",
                "run_id": self.run_id,
                "queue_id": queue_id,
                "read": len(decisions),
                **read_back.counts.as_dict(),
            },
        )
        return [
            ReviewDecision(
                external_id=decision.external_id,
                decision_id=decision.decision_id,
                decision=decision.decision,
                label=decision.override_label,
                reason=decision.reason or None,
            )
            for decision in decisions
            if decision.external_id in read_back.decision_ids
        ]


def _step_of(candidates: Sequence[Any]) -> int:
    """The step these candidates belong to, read from their external ids.

    ``miforge:<run_id>:<step>:<digest>`` is the shape :func:`
    services.review.binding.derive_external_id` produces. Parsed rather than passed as an
    argument because F3's port signature carries no step - and **refusing to guess** would
    mean refusing a legitimate post, so an unparseable id records step 0 and the id itself
    remains the authority. The step column is for an operator's diagnosis, not for
    correctness: nothing joins on it.
    """
    for candidate in candidates:
        parts = str(getattr(candidate, "external_id", "")).split(":")
        if len(parts) >= 4 and parts[0] == "miforge":
            try:
                return int(parts[2])
            except ValueError:
                continue
    return 0


# ======================================================================================
# THE PROCESS-WIDE PORT
# ======================================================================================


class RunScopedReviewQueue:
    """F3's :class:`~services.selectors.review_port.ReviewQueuePort` for a whole process.

    One object lives in the port slot for the life of the process. It answers
    :meth:`available` with no run and no round trip - which is what ``GET
    /api/v1/selectors`` needs, since the API process has no run id and ADR-013 requires
    the ``human_review`` type to be *listed unavailable by name* rather than omitted - and
    routes the three queue operations to the :class:`MiDataworksReviewQueue` for the run
    currently in scope.

    **With miDataworks unset this still refuses ``midataworks_not_configured``**, from
    :meth:`MiDataworksReviewQueue.ensure_queue`'s own client construction, and
    :meth:`available` reports ``False``. That is ADR-010's absent-and-normal and is the
    standalone path (AC20): the bug this class fixes is not that miForge refused, it is
    that refusing was the **only** thing it could do.
    """

    def __init__(
        self,
        *,
        run_id: Callable[[], str | None],
        session_scope: SessionScope | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Args:
            run_id: How to read the run currently in scope - normally
                :func:`services.selectors.plugins.human_review.current_review_run`.
                **Injected rather than imported, and required.** Injected because
                ``training/**`` may not import ``clients.*`` (the AC20 fence
                ``test_F1_has_no_outbound_sibling_call_site`` enforces it), so the scope
                has to live on the selector side and this module must not reach back for
                it. Required because a default would make a port that silently speaks for
                no run.
            session_scope: Passed through to each per-run queue. ``None`` lets it use
                :func:`db.session.session_scope`.
            transport: An ``httpx`` transport, for tests. Production passes ``None``.
        """
        self._run_id = run_id
        self._scope = session_scope
        self._transport = transport

    # ------------------------------------------------------------------ the port

    def available(self) -> bool:
        """Whether miDataworks is configured **right now**. No run, no client, never raises.

        Reads :func:`clients.availability.is_configured` - the same function
        :meth:`MiDataworksReviewQueue.available` reads - so the listing and the per-run
        port cannot give two answers about one state.
        """
        return availability.is_configured()

    def ensure_queue(self, *, name: str, question: str, label_set: list[str]) -> str:
        return self._for_run().ensure_queue(name=name, question=question, label_set=label_set)

    def post_candidates(self, queue_id: str, candidates: list[Any]) -> list[Any]:
        return self._for_run().post_candidates(queue_id, candidates)

    def read_decisions(self, queue_id: str, *, external_ids: list[str] | None = None) -> list[Any]:
        return self._for_run().read_decisions(queue_id, external_ids=external_ids)

    # ------------------------------------------------------------------ routing

    def for_run(self, run_id: str) -> MiDataworksReviewQueue:
        """The per-run queue for ``run_id``. Public so a test can assert what it routes to."""
        return MiDataworksReviewQueue(
            run_id=run_id, session_scope=self._scope, transport=self._transport
        )

    def _for_run(self) -> MiDataworksReviewQueue:
        """The queue for the run in scope, or refuse **by name**.

        Constructed per call rather than cached. A cache keyed by run id would be a second
        answer to "which run is this", and the constructor opens nothing - all of this
        class's state is the binding row.

        Raises:
            Refusal: ``review_queue_run_unknown`` when no run is in scope. Reached by the
                authoring path - ``POST /api/v1/selectors/evaluate`` with a
                ``human_review`` configuration - which has no run and therefore no binding
                row to put a queue id on. Refusing by name is the honest answer there, and
                a different one from ``midataworks_not_configured``: R-02.39's dry-run
                is miForge's own and needs no queue at all.
        """
        run_id = self._run_id()
        if not run_id:
            raise refuse(
                "review_queue_run_unknown",
                "In-run human review needs a run: the queue id is recorded on that run's "
                "binding row and the decisions are read back through it, so there is "
                "nowhere to put a queue created outside one. Run the selector inside a "
                "run, or use the authoring dry-run, which uses no queue.",
                capability=CAPABILITY,
                action=ACTION,
            )
        return self.for_run(run_id)


# ======================================================================================
# THE BINDING  (ADR-016 #1)
# ======================================================================================


def wire() -> None:
    """Bind this port into F3's review slot. **The line, and the point of the module.**

    Idempotent, and it does not clobber: a port already in the slot is left alone, so a
    test that bound a stub keeps it, and production - where nothing else ever binds - gets
    this one.

    ## Why the binding is here and not on F3's side

    Four structural fences forbid every other candidate, and each is right:

    * ``services/selectors/**`` - ``test_selector_guards`` forbids a ``clients`` import,
      for this exact temptation: *"depending on F10's concrete client instead of the port
      is how the port stops being a port"*;
    * ``services/scoring/*.py`` - ``test_scoring_static_guards.TestF4NeverCallsMiDataworks``
      and ``test_F4_no_scoring_module_reaches_a_sibling_or_an_http_client``: F4 reads rows
      and resolves providers and has no outbound call site;
    * ``training/**`` - ``test_F1_has_no_outbound_sibling_call_site``, which is AC20's
      fence;
    * and nothing may construct a second client, which is why the thing bound is this
      class rather than anything new.

    All four were observed red against earlier sitings. ADR-010 puts the sibling in
    ``clients/``, so the binding of the sibling's implementation into the port belongs
    here, called by whoever imports this module.

    ## WHAT IS STILL OWED, AND IT IS ONE LINE

    **Nothing in ``backend/src/`` imports this module**, which was the original defect and
    is the half that cannot be closed from here: the import has to come from a package that
    both reaches every process and is permitted to import ``clients``. That is F10's own
    :mod:`services.review` - it already imports ``clients.midataworks`` and
    ``clients.availability``, and ``api/v1/endpoints/runs_review.py`` imports it, so the API
    process has it. The worker half is wired: ``services/scoring/wiring.py`` imports
    ``services.review``, and ``services.scoring.tasks`` is in ``workers.celery_app``'s
    ``TASK_MODULES``.

    So the owed change is one line at the foot of ``backend/src/services/review/__init__.py``::

        # F10's review client binds itself into F3's review-queue port on import. Here
        # because `services/selectors/`, `services/scoring/` and `training/` are all
        # fenced against importing `clients` (ADR-016 #1).
        from clients import midataworks_review as _review_port_wiring  # noqa: F401

    ``test_review_queue_wiring.TestTheImportTriggerIsOwed`` is the tripwire: it is a
    **strict xfail**, so it turns red the moment that line lands and the marker has to come
    off. Until then ``GET /api/v1/selectors`` still reports ``human_review`` unavailable
    with miDataworks configured.
    """
    from services.selectors.plugins import human_review

    if human_review.current_review_queue() is not None:
        return
    # THE LINE. Delete it and `test_review_queue_wiring.py` must go red: the `human_review`
    # selector refuses `midataworks_not_configured` whatever the operator configured, while
    # all eleven of its own test files still pass because each binds its own stub.
    human_review.bind_review_queue(RunScopedReviewQueue(run_id=human_review.current_review_run))
    log.debug(
        "miDataworks review queue bound into the selector review port",
        extra={"code": "review_queue_port_bound"},
    )


# The import side effect. Importing this module is what installs the port; see `wire`'s
# "what is still owed" for the one production import that is not yet in place.
wire()
