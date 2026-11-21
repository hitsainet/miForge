"""The run's pointer into miDataworks' review queue  (F10 tasks 7.3-7.10; FR-010.21-23, .28).

Everything durable about in-run review lives here: the queue id, the candidates' external
ids, the decisions as read, the watermark and the counts. **All of it on rows**, because
every worker is ``--pool=solo -c 1`` (ADR-006) - ``revoke(terminate=True)`` has no pool
child to signal and a busy solo worker never reads the control queue - so anything that
must reach a running job goes through the database.

## Four decisions that are easy to get wrong, and why each is the way it is

**``ensure_queue`` is idempotent and reads the binding row first.** Creating a second queue
for a run would orphan the first *and its decisions*, which are the only copy: miForge does
not store decision content as authority.

**The queue id is stored in the same transaction that records the run's intent.** Losing it
loses the ability to read decisions back - the same class of mistake as holding an
attachment id in a request-scoped variable.

**``external_id`` is deterministic**, hashed over ``(run_id, step, candidate_index,
prompt_sha256)`` through the one canonical-JSON function (ADR-005). Determinism is what
makes a re-post after a crash idempotent: miDataworks returns the existing item for a
repeated external id, and nobody is asked to read the same candidate twice.

**Read back by watermark *and* by explicit pending ids.** ``since`` catches the common case
cheaply; the explicit ids catch a decision whose ``created_at`` precedes a candidate miForge
posted late. Doing only one of the two loses decisions, and a lost decision is a candidate
that stays pending forever while a human believes they answered it. The cursor advances
**only after the read's results are committed** - an advanced cursor over an uncommitted
write loses a decision permanently.

## Counts go on the row as well as the socket

Progress reported *only* over the socket is invisible to the janitor - that is how miStudio
reaped a live 5.8-hour job as "worker lost" (ADR-007). And every number names its scale and
sample (ADR-012): *"128 pending of 500 posted"*, never *"128"*.

## Everything read back is untrusted data

``reason``, the reviewer's identity, an override label and the queue's question are **data**
(FR-010.29). Stored as data, rendered as text, never interpreted as instructions, never used
to construct a filesystem path, a shell argument or a prompt instruction block.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from clients.midataworks import CANDIDATE_BATCH_MAX, CANDIDATE_MAX_BYTES_DEFAULT
from core.canonical import canonical_bytes
from core.errors import refuse
from schemas.selector import Verdict
from services.parts.tables import DECISION_NONE, REVIEW_BINDINGS, REVIEW_CANDIDATE_REFS
from services.review.verdicts import Anomaly, verdict_for, verdict_for_absent

log = logging.getLogger("miforge.review.binding")

#: The capability key the guard is called with for every path in this module.
CAPABILITY = "human_review_selector"
ACTION_QUEUE = "running with the human-review selector"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One completion on its way to a person.

    ``provenance`` is for the reader - which part the prompt came from, which model wrote
    the completion. **Never a filesystem tour and never a credential** (ADR-020).
    """

    prompt: str
    completion: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    model_output: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ReviewCounts:
    """Posted, decided and pending - each naming its sample.

    ``pending`` is a ``COUNT`` over the ``(binding_id, decision)`` index rather than a set
    difference, because a candidate's undecided condition is a **stored value**
    (``decision = 'none'``) and not an absence. That is the whole of FR-010.24 expressed in
    the data model: an absence invites a default, a value does not.
    """

    posted: int
    decided: int
    pending: int

    def as_dict(self) -> dict[str, int]:
        return {"posted": self.posted, "decided": self.decided, "pending": self.pending}

    def describe(self) -> str:
        """``"128 pending of 500 posted"`` - ADR-012: every number names its scale."""
        return f"{self.pending} pending of {self.posted} posted, {self.decided} decided"


def derive_external_id(
    *, run_id: str, step: int, candidate_index: int, prompt: str, completion: str
) -> str:
    """A deterministic external id for one candidate.

    Hashed through the **one** canonical-JSON function (ADR-005) over the run, the step, the
    candidate's position and a digest of its text. Two processes computing this for the same
    candidate get the same answer, which is what makes a re-post after a crash idempotent.

    The completion is in the material as well as the prompt: the same prompt generated twice
    at one step is two different candidates, and an id that collided would silently drop one
    of them by returning the other's queue item.

    Capped well under miDataworks' ``external_id`` limit of 256 characters, with a prefix
    that makes a stray id recognisable in their UI as miForge's.
    """
    digest = hashlib.sha256(
        canonical_bytes(
            {
                "run_id": run_id,
                "step": step,
                "candidate_index": candidate_index,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
            }
        )
    ).hexdigest()
    return f"miforge:{run_id}:{step}:{digest[:32]}"


def candidate_payload(external_id: str, candidate: Candidate) -> dict[str, Any]:
    """The body miDataworks' ``Candidate`` model expects, key for key.

    Verified against their ``schemas/review.py``: ``external_id`` (1-256), ``prompt`` and
    ``completion`` (each at most 65,536), optional ``model_output``, and ``provenance``
    defaulting to an empty object. ``extra="forbid"`` on their side, so an extra key here
    would be a 422 rather than an ignored field - which is why this builder exists instead
    of each call site assembling a dict.
    """
    payload: dict[str, Any] = {
        "external_id": external_id,
        "prompt": candidate.prompt,
        "completion": candidate.completion,
        "provenance": dict(candidate.provenance),
    }
    if candidate.model_output is not None:
        payload["model_output"] = dict(candidate.model_output)
    return payload


def payload_bytes(payload: Mapping[str, Any]) -> int:
    """How large miDataworks will consider this candidate.

    Measured **the way the receiver measures it** - the serialized candidate, through a
    canonical serialiser, exactly as their ``post_candidates`` does
    (``len(canonical_json(c.model_dump(mode="json")))``) - and not the completion string's
    length. Measuring the string would pass a candidate whose provenance object is the
    thing over the limit.
    """
    return len(canonical_bytes(payload))


def split_batches(
    payloads: Sequence[Mapping[str, Any]],
    *,
    batch_max: int = CANDIDATE_BATCH_MAX,
    byte_ceiling: int = CANDIDATE_MAX_BYTES_DEFAULT,
) -> list[list[Mapping[str, Any]]]:
    """Split candidates to fit miDataworks' limits **before** sending.

    Never send-and-handle-413: a rejected batch costs the whole payload's upload for an
    answer miForge already had.

    Args:
        payloads: Candidate bodies from :func:`candidate_payload`.
        batch_max: At most this many per request (their ``CANDIDATE_BATCH_MAX`` is 500).
        byte_ceiling: Per-candidate ceiling (their configured
            ``review_candidate_max_bytes``, default 65,536).

    Returns:
        Batches, each within both limits. An empty input gives no batches - zero requests,
        not one empty request.

    Raises:
        Refusal: ``review_candidate_too_large`` for a **single** candidate over the
            ceiling. Refused rather than split, because a candidate cannot be halved: a
            person must read one prompt and one completion together, and sending half of
            each would ask them to judge something that was never generated.
    """
    oversized = [p for p in payloads if payload_bytes(p) > byte_ceiling]
    if oversized:
        first = oversized[0]
        raise refuse(
            "review_candidate_too_large",
            f"Candidate {first.get('external_id')!r} serialises to "
            f"{payload_bytes(first)} bytes and miDataworks' per-candidate ceiling is "
            f"{byte_ceiling}. A candidate cannot be split - a person reads one prompt and "
            f"one completion together - so this one is refused rather than truncated. "
            f"Shorten the generation or raise miDataworks' "
            f"REVIEW_CANDIDATE_MAX_BYTES.",
            external_id=first.get("external_id"),
            bytes=payload_bytes(first),
            ceiling=byte_ceiling,
            oversized_count=len(oversized),
        )
    return [list(payloads[i : i + batch_max]) for i in range(0, len(payloads), batch_max)]


# ====================================================================== the binding row


@dataclass(frozen=True, slots=True)
class Binding:
    """One run's pointer to its queue. **Not a queue** - see :mod:`services.review`."""

    id: str
    run_id: str
    queue_id: str
    question: str
    label_set: tuple[str, ...]
    read_back_cursor: datetime | None


def _binding_id(run_id: str) -> str:
    return "rvb-" + hashlib.sha256(canonical_bytes({"run_id": run_id}).strip()).hexdigest()[:24]


def find_binding(session: Session, run_id: str) -> Binding | None:
    """The run's binding row, or ``None``."""
    row = session.execute(
        sa.select(REVIEW_BINDINGS).where(REVIEW_BINDINGS.c.run_id == run_id)
    ).one_or_none()
    if row is None:
        return None
    mapping = dict(row._mapping)
    return Binding(
        id=str(mapping["id"]),
        run_id=str(mapping["run_id"]),
        queue_id=str(mapping["queue_id"]),
        question=str(mapping["question"]),
        label_set=tuple(str(v) for v in (mapping["label_set"] or [])),
        read_back_cursor=_aware(mapping["read_back_cursor"]),
    )


def require_binding(session: Session, run_id: str) -> Binding:
    """The run's binding, or refuse by name.

    Raises:
        Refusal: ``review_binding_missing``. Named, because the two reasons it can happen -
            the run does not use human review, or its queue was never created - have
            different fixes, and an empty result would read as "no candidates".
    """
    binding = find_binding(session, run_id)
    if binding is None:
        raise refuse(
            "review_binding_missing",
            f"Run {run_id!r} has no review binding, so there is no miDataworks queue to "
            f"read. Either this run's selector is not human review, or its queue was "
            f"never created.",
            run_id=run_id,
        )
    return binding


def record_binding(
    session: Session,
    *,
    run_id: str,
    queue_id: str,
    question: str,
    label_set: Sequence[str],
) -> Binding:
    """Store the queue id for a run, idempotently.

    **Called in the same transaction that created the queue.** If the row already exists
    with a different queue id the call refuses rather than overwriting: the old queue holds
    decisions, and pointing the run at a new one would make them unreachable while the UI
    showed a healthy, empty queue.

    Raises:
        Refusal: ``review_queue_already_bound`` when the run already points elsewhere.
    """
    existing = find_binding(session, run_id)
    if existing is not None:
        if existing.queue_id != queue_id:
            raise refuse(
                "review_queue_already_bound",
                f"Run {run_id!r} is already bound to miDataworks queue "
                f"{existing.queue_id!r} and cannot be repointed at {queue_id!r}. The "
                f"first queue holds this run's decisions; repointing would orphan them.",
                run_id=run_id,
                bound_to=existing.queue_id,
                requested=queue_id,
            )
        return existing

    now = datetime.now(tz=UTC)
    session.execute(
        sa.insert(REVIEW_BINDINGS).values(
            id=_binding_id(run_id),
            run_id=run_id,
            queue_id=queue_id,
            question=question,
            label_set=list(label_set),
            posted=0,
            decided=0,
            pending=0,
            read_back_cursor=None,
            created_at=now,
            updated_at=now,
        )
    )
    session.flush()
    log.info(
        "review_queue_bound",
        extra={"code": "review_queue_bound", "run_id": run_id, "queue_id": queue_id},
    )
    binding = find_binding(session, run_id)
    assert binding is not None  # just inserted
    return binding


# ====================================================================== candidates


def record_posted(
    session: Session,
    binding: Binding,
    *,
    step: int,
    accepted: Sequence[tuple[str, str]],
) -> int:
    """Record the candidates miDataworks acknowledged. Returns how many were new rows.

    Args:
        accepted: ``(external_id, item_id)`` pairs as miDataworks returned them.

    A repeated ``external_id`` updates its ``item_id`` and leaves the decision alone - a
    re-post after a crash must not reset a decision that arrived in between.
    """
    created = 0
    for external_id, item_id in accepted:
        savepoint = session.begin_nested()
        try:
            session.execute(
                sa.insert(REVIEW_CANDIDATE_REFS).values(
                    binding_id=binding.id,
                    step=step,
                    external_id=external_id,
                    item_id=item_id,
                    decision=DECISION_NONE,
                    created_at=datetime.now(tz=UTC),
                )
            )
            savepoint.commit()
            created += 1
        except sa.exc.IntegrityError:
            savepoint.rollback()
            session.execute(
                sa.update(REVIEW_CANDIDATE_REFS)
                .where(
                    sa.and_(
                        REVIEW_CANDIDATE_REFS.c.binding_id == binding.id,
                        REVIEW_CANDIDATE_REFS.c.external_id == external_id,
                    )
                )
                .values(item_id=item_id)
            )
    session.flush()
    _refresh_counts(session, binding)
    return created


def pending_external_ids(session: Session, binding: Binding, *, limit: int = 500) -> list[str]:
    """The still-undecided candidates, for the explicit half of the read-back.

    Capped at miDataworks' ``external_ids`` query limit (``Query(None, max_length=500)``
    on their route), so a run with thousands pending asks for the oldest 500 rather than
    sending a URL their framework refuses.
    """
    rows = session.execute(
        sa.select(REVIEW_CANDIDATE_REFS.c.external_id)
        .where(
            sa.and_(
                REVIEW_CANDIDATE_REFS.c.binding_id == binding.id,
                REVIEW_CANDIDATE_REFS.c.decision == DECISION_NONE,
            )
        )
        .order_by(REVIEW_CANDIDATE_REFS.c.id)
        .limit(limit)
    ).scalars()
    return [str(value) for value in rows]


@dataclass(frozen=True, slots=True)
class ReadBack:
    """What one read-back changed.

    ``decision_ids`` is keyed by ``external_id`` and is **what R-02.36's closing sentence
    asks for**: the decision identifiers recorded with the batch they admitted. F10 supplies
    the values; the batch record's schema is the contract gate's and is not defined here.
    """

    verdicts: Mapping[str, Verdict]
    decision_ids: Mapping[str, str]
    anomalies: tuple[Anomaly, ...]
    counts: ReviewCounts
    #: The newest ``created_at`` seen, which becomes the cursor **after** the write commits.
    watermark: datetime | None


def record_decisions(
    session: Session,
    binding: Binding,
    decisions: Sequence[Any],
) -> ReadBack:
    """Record decisions read back, mapping each to a verdict.

    Args:
        decisions: :class:`clients.midataworks.ExternalDecision` records, as read.

    Returns:
        A :class:`ReadBack`. **The cursor is not advanced here** - :func:`advance_cursor`
        does that, and the caller must call it only after this write has committed. An
        advanced cursor over an uncommitted write loses a decision permanently, and the
        separation is what makes that ordering expressible.

    A decision for an ``external_id`` miForge does not know is recorded as an anomaly and
    **never admitted** (E-17). It is not an error: two miForge installs can share a queue
    id in a misconfiguration, and the honest outcome is to say so rather than to crash a
    run on someone else's candidate.
    """
    known = {
        str(row[0]): row[1]
        for row in session.execute(
            sa.select(REVIEW_CANDIDATE_REFS.c.external_id, REVIEW_CANDIDATE_REFS.c.decision).where(
                REVIEW_CANDIDATE_REFS.c.binding_id == binding.id
            )
        ).all()
    }

    verdicts: dict[str, Verdict] = {}
    decision_ids: dict[str, str] = {}
    anomalies: list[Anomaly] = []
    watermark: datetime | None = None

    for decision in decisions:
        external_id = str(decision.external_id)
        created_at = _parse_timestamp(decision.created_at)
        if created_at is not None and (watermark is None or created_at > watermark):
            watermark = created_at

        if external_id not in known:
            anomalies.append(
                Anomaly(
                    code="decision_for_unknown_candidate",
                    external_id=external_id,
                    detail=(
                        f"miDataworks queue {binding.queue_id} returned decision "
                        f"{decision.decision_id} for a candidate this run never posted"
                    ),
                )
            )
            log.warning(
                "review_anomaly",
                extra={
                    "code": "decision_for_unknown_candidate",
                    "external_id": external_id,
                    "queue_id": binding.queue_id,
                    "run_id": binding.run_id,
                },
            )
            continue

        verdict, anomaly = verdict_for(
            decision.decision,
            label=decision.override_label,
            label_set=binding.label_set,
            external_id=external_id,
        )
        if anomaly is not None:
            anomalies.append(anomaly)
        verdicts[external_id] = verdict
        decision_ids[external_id] = str(decision.decision_id)

        session.execute(
            sa.update(REVIEW_CANDIDATE_REFS)
            .where(
                sa.and_(
                    REVIEW_CANDIDATE_REFS.c.binding_id == binding.id,
                    REVIEW_CANDIDATE_REFS.c.external_id == external_id,
                )
            )
            .values(
                decision_id=str(decision.decision_id),
                decision=decision.decision,
                override_label=decision.override_label,
                # Untrusted text, stored as text.
                reason=decision.reason or None,
                decided_by=decision.decided_by or None,
                decided_by_origin=decision.decided_by_origin or None,
                decided_at=created_at,
            )
        )

    session.flush()
    counts = _refresh_counts(session, binding)
    return ReadBack(
        verdicts=verdicts,
        decision_ids=decision_ids,
        anomalies=tuple(anomalies),
        counts=counts,
        watermark=watermark,
    )


def advance_cursor(session: Session, binding: Binding, watermark: datetime | None) -> None:
    """Move the read-back watermark forward. **Call only after the decisions committed.**

    Never moves backwards: a read that returned nothing new leaves the cursor where it was,
    and two concurrent reads cannot rewind each other.
    """
    if watermark is None:
        return
    session.execute(
        sa.update(REVIEW_BINDINGS)
        .where(
            sa.and_(
                REVIEW_BINDINGS.c.id == binding.id,
                sa.or_(
                    REVIEW_BINDINGS.c.read_back_cursor.is_(None),
                    REVIEW_BINDINGS.c.read_back_cursor < watermark,
                ),
            )
        )
        .values(read_back_cursor=watermark, updated_at=datetime.now(tz=UTC))
    )
    session.flush()


def verdicts_for_candidates(
    session: Session, binding: Binding, external_ids: Sequence[str]
) -> dict[str, Verdict]:
    """The current verdict for each candidate, with **no decision meaning undecided**.

    The function a selector's caller uses to turn "what do we know" into verdicts. A
    candidate with no row and a candidate with ``decision = 'none'`` both come back
    ``undecided`` - the absence is turned into a value here rather than being left for a
    caller to default.
    """
    rows = {
        str(row[0]): (row[1], row[2])
        for row in session.execute(
            sa.select(
                REVIEW_CANDIDATE_REFS.c.external_id,
                REVIEW_CANDIDATE_REFS.c.decision,
                REVIEW_CANDIDATE_REFS.c.override_label,
            ).where(
                sa.and_(
                    REVIEW_CANDIDATE_REFS.c.binding_id == binding.id,
                    REVIEW_CANDIDATE_REFS.c.external_id.in_(list(external_ids)),
                )
            )
        ).all()
    }
    out: dict[str, Verdict] = {}
    for external_id in external_ids:
        found = rows.get(external_id)
        if found is None or found[0] == DECISION_NONE:
            out[external_id] = verdict_for_absent()
            continue
        verdict, _ = verdict_for(
            str(found[0]),
            label=found[1],
            label_set=binding.label_set,
            external_id=external_id,
        )
        out[external_id] = verdict
    return out


def counts_for(session: Session, binding: Binding) -> ReviewCounts:
    """The run's current review counts, read from the rows."""
    posted = int(
        session.execute(
            sa.select(sa.func.count())
            .select_from(REVIEW_CANDIDATE_REFS)
            .where(REVIEW_CANDIDATE_REFS.c.binding_id == binding.id)
        ).scalar_one()
    )
    pending = int(
        session.execute(
            sa.select(sa.func.count())
            .select_from(REVIEW_CANDIDATE_REFS)
            .where(
                sa.and_(
                    REVIEW_CANDIDATE_REFS.c.binding_id == binding.id,
                    REVIEW_CANDIDATE_REFS.c.decision == DECISION_NONE,
                )
            )
        ).scalar_one()
    )
    return ReviewCounts(posted=posted, decided=posted - pending, pending=pending)


def _refresh_counts(session: Session, binding: Binding) -> ReviewCounts:
    """Write the counts onto the binding row.

    **On the row, not only over the socket.** Progress reported only over the socket is
    invisible to the janitor, and that is how a live 5.8-hour job was reaped as "worker
    lost" (ADR-007).
    """
    counts = counts_for(session, binding)
    session.execute(
        sa.update(REVIEW_BINDINGS)
        .where(REVIEW_BINDINGS.c.id == binding.id)
        .values(
            posted=counts.posted,
            decided=counts.decided,
            pending=counts.pending,
            updated_at=datetime.now(tz=UTC),
        )
    )
    session.flush()
    return counts


def review_payload(session: Session, run_id: str) -> dict[str, Any]:
    """What ``GET /api/v1/runs/{run_id}/review`` serves (FR-010.28).

    Carries the queue's identity and the three counts **with their scale named**. The queue
    id is here because an operator who needs to open the queue in miDataworks needs it, and
    because a run whose counts look wrong is diagnosed by looking at the queue it is
    actually bound to.
    """
    binding = require_binding(session, run_id)
    counts = counts_for(session, binding)
    return {
        "run_id": run_id,
        "queue_id": binding.queue_id,
        "question": binding.question,
        "label_set": list(binding.label_set),
        "counts": counts.as_dict(),
        "summary": counts.describe(),
        "read_back_cursor": (
            None if binding.read_back_cursor is None else binding.read_back_cursor.isoformat()
        ),
        # Named so a UI does not have to know which sibling owns the queue - and so the
        # indigo state colour (ADR-012) has something to key on.
        "queue_owner": "miDataworks",
    }


def _aware(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return _parse_timestamp(value)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse miDataworks' ISO-8601 timestamp, refusing to invent one.

    Returns ``None`` rather than ``now`` when it cannot parse: a fabricated timestamp would
    move the watermark past decisions miForge has not seen, and losing a decision silently
    is worse than leaving the cursor where it was.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        log.warning(
            "review_decision_timestamp_unparseable",
            extra={"code": "review_decision_timestamp_unparseable", "value": str(value)[:64]},
        )
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
