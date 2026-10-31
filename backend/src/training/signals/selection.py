"""The one verdict partition  (FR-002.12, FR-002.13, FR-002.14, FR-002.15).

**Three values, four destinations, and no fourth decision site.**

    accept    -> admitted set
    reject    -> rejected count
    undecided -> pending set    (NEVER admitted, NEVER counted as rejected)
                 health record (F3's counts, rate with its n, histogram, signals)

FR-002.13's rule touches four fields, so it is implemented in **one** function: four sites
that each decide what ``undecided`` means is four chances for one of them to call it a
reject, and the symptom is a health record that lies about an unattended queue while the
run looks like it is selecting.

**There is no branch, flag, deadline or shortfall path from pending to admitted.** The
deadline bounds the *wait* and the only thing it can produce is a **count**. Defaulting
silence to accept turns an unattended queue into an unselected run, which is the recursion
R-02.11 exists to prevent and which F10's FR-010.24 forbids from the queue's side.

:func:`schemas.selector.is_admitted` is the **only** route to "admitted" and this module
calls it rather than comparing to a value itself - that is the contract's own rule, and a
static test forbids the comparison.

## F3 decides; F2 partitions and records

F2 writes no rule, no threshold, no judge call, no verdict and no configuration hash. It
calls F3's ``evaluate`` **once** per candidate set and **carries** F3's health record -
never recomputes it. Recomputing any of its numbers would produce a second number that can
disagree with F3's, and the two would be read as one fact.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from schemas.selector import Verdict, VerdictValue, is_admitted
from services.selectors.service import evaluate as f3_evaluate
from services.selectors.types import (
    EvaluationContext,
    EvaluationResult,
    PublishedHashes,
    SelectionHealth,
    VerdictRecord,
)
from training.signals.settings import SignalSettings

#: How often a bounded wait re-asks F3 about the still-pending candidates. Small and fixed:
#: it is a poll interval, not a policy, and the thing it must never do is grow into a
#: timeout that produces an admission.
POLL_INTERVAL_S = 0.5


@dataclass(frozen=True)
class Candidate:
    """One generated candidate, before it is judged."""

    prompt_id: str
    prompt: str
    completion: str
    index: int


@dataclass(frozen=True)
class SelectionOutcome:
    """The partition's result. **Four collections, kept apart by construction.**

    Separate fields rather than one list with a status, so a caller cannot iterate the
    whole thing and treat every element the same way - which is exactly how a pending
    candidate would become an admitted one.
    """

    admitted: tuple[Candidate, ...]
    rejected: tuple[Candidate, ...]
    pending: tuple[Candidate, ...]
    #: F3's full record, carried verbatim. Its ``signals`` are what F6 stops on (R-02.25).
    health: SelectionHealth
    hashes: PublishedHashes
    #: Per-candidate provenance, in input order: the reason code, the judge, the mode and
    #: the review item. Identifiers and codes only - never the reviewer's text (ADR-020).
    records: tuple[VerdictRecord, ...]
    #: Review item identifiers for the pending candidates, and decision identifiers for
    #: the ones a human settled (R-02.36's closing sentence).
    review_ids: tuple[str, ...] = ()
    #: How long the bounded wait actually waited. A measurement, never a licence.
    waited_s: float = 0.0

    @property
    def counts(self) -> Mapping[str, int]:
        return {
            "admitted": len(self.admitted),
            "rejected": len(self.rejected),
            "pending": len(self.pending),
        }


def partition(
    candidates: Sequence[Candidate], verdicts: Sequence[Verdict]
) -> tuple[tuple[Candidate, ...], tuple[Candidate, ...], tuple[Candidate, ...]]:
    """``(admitted, rejected, pending)``. **The only place admission is decided here.**

    One pass, three destinations, and the order of the tests is deliberate:
    :func:`schemas.selector.is_admitted` first, because it is the contract's one gate;
    then ``undecided``, which is **not** a reject; then everything else, which is.

    Writing it the other way round - "reject unless accepted" - would fold ``undecided``
    into ``rejected`` and the health record would then report an unattended review queue as
    a selector that is working.
    """
    if len(candidates) != len(verdicts):
        from core.errors import refuse

        raise refuse(
            "verdict_count_mismatch",
            f"{len(verdicts)} verdicts were returned for {len(candidates)} candidates. "
            f"Refusing rather than zipping the shorter list: a positional join over "
            f"mismatched lists misattributes every verdict after the gap.",
            candidates=len(candidates),
            verdicts=len(verdicts),
        )
    admitted: list[Candidate] = []
    rejected: list[Candidate] = []
    pending: list[Candidate] = []
    for candidate, verdict in zip(candidates, verdicts, strict=True):
        if is_admitted(verdict):
            admitted.append(candidate)
        elif verdict.value is VerdictValue.UNDECIDED:
            pending.append(candidate)
        else:
            rejected.append(candidate)
    return tuple(admitted), tuple(rejected), tuple(pending)


def apply(
    candidates: Sequence[Candidate],
    resolved: Any,
    *,
    settings: SignalSettings,
    run_id: str | None = None,
    batch_id: str = "batch",
    is_cancelled: Any | None = None,
    emit_signal: Any | None = None,
    items: Sequence[Mapping[str, Any]] = (),
    sleep: Any = time.sleep,
    now: Any = time.monotonic,
) -> SelectionOutcome:
    """Judge a candidate set through F3, partition the answer, and wait - bounded.

    Args:
        candidates: The candidates, in the order they were generated.
        resolved: F3's ``ResolvedSelector``. Resolved at pre-flight, so a configuration
            error is refused **once** before the run rather than once per completion.
        settings: Supplies the pending deadline. The deadline bounds the wait and nothing
            else.
        sleep: Injected so the deadline test runs in milliseconds rather than seconds -
            and so it asserts the *behaviour at expiry*, which is the whole point.
        now: Injected for the same reason.

    Returns:
        A :class:`SelectionOutcome`. Pending candidates are **still pending**, whatever the
        deadline did.

    Raises:
        Refusal: F3's own configuration or transport refusals, which fail the **whole**
            evaluation - a transport failure is not a verdict.
    """
    result = _evaluate(
        candidates,
        resolved,
        run_id=run_id,
        batch_id=batch_id,
        is_cancelled=is_cancelled,
        emit_signal=emit_signal,
        items=items,
    )
    admitted, rejected, pending = partition(candidates, list(result.verdicts))
    records = list(result.records)
    waited = 0.0

    # ---- the bounded wait ------------------------------------------------- FR-002.13
    # It re-asks F3 about the candidates that are still pending, because the human-review
    # selector reads decisions back through F10 and a decision may have arrived. What it
    # CANNOT do - and what no branch below admits - is turn the expiry of this loop into an
    # admission. On expiry the pending set is unchanged and the record says how many.
    if pending and settings.pending_deadline_s > 0.0:
        started = now()
        deadline = started + settings.pending_deadline_s
        while pending and now() < deadline:
            if is_cancelled is not None and is_cancelled():
                break
            sleep(min(POLL_INTERVAL_S, max(0.0, deadline - now())))
            recheck = _evaluate(
                pending,
                resolved,
                run_id=run_id,
                batch_id=batch_id,
                is_cancelled=is_cancelled,
                emit_signal=None,
                items=_items_for(pending, items),
            )
            settled_admitted, settled_rejected, still_pending = partition(
                pending, list(recheck.verdicts)
            )
            admitted = (*admitted, *settled_admitted)
            rejected = (*rejected, *settled_rejected)
            records = _merge_records(records, pending, recheck.records)
            pending = still_pending
        # A measurement of the wait that happened, not of the deadline that was allowed.
        waited = max(0.0, now() - started)

    health = _health_after_wait(
        result.health, admitted=admitted, rejected=rejected, pending=pending
    )
    return SelectionOutcome(
        admitted=admitted,
        rejected=rejected,
        pending=pending,
        health=health,
        hashes=result.hashes,
        records=tuple(records),
        review_ids=_review_ids(records),
        waited_s=waited,
    )


def _evaluate(
    candidates: Sequence[Candidate],
    resolved: Any,
    *,
    run_id: str | None,
    batch_id: str,
    is_cancelled: Any | None,
    emit_signal: Any | None,
    items: Sequence[Mapping[str, Any]],
) -> EvaluationResult:
    """One call to F3's one entry point. **F2 evaluates nothing itself.**"""
    context = EvaluationContext(
        run_id=run_id,
        batch_id=batch_id,
        items=tuple(dict(item) for item in items),
        is_cancelled=is_cancelled,
        emit_signal=emit_signal,
    )
    return f3_evaluate(
        [candidate.completion for candidate in candidates],
        resolved,
        context,
        prompts=[candidate.prompt for candidate in candidates],
    )


def _items_for(
    candidates: Sequence[Candidate], items: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """The per-item context for a subset, **re-indexed by the candidate's own index**.

    F3 joins ``items`` positionally, so handing it the original list alongside a shorter
    candidate list would give candidate *k* the context of candidate *k* of the first pass
    - a reference-match verdict against the wrong reference, which still returns a verdict.
    """
    return [dict(items[c.index]) if c.index < len(items) else {} for c in candidates]


def _merge_records(
    records: list[VerdictRecord],
    rechecked: Sequence[Candidate],
    produced: Sequence[VerdictRecord],
) -> list[VerdictRecord]:
    """Replace the records of the re-asked candidates, keeping input order.

    Replaced rather than appended: a candidate has **one** verdict on the record, and two
    would make the verdict list disagree with the counts beside it.
    """
    merged = list(records)
    for candidate, record in zip(rechecked, produced, strict=True):
        if 0 <= candidate.index < len(merged):
            merged[candidate.index] = record
    return merged


def _health_after_wait(
    health: SelectionHealth,
    *,
    admitted: Sequence[Candidate],
    rejected: Sequence[Candidate],
    pending: Sequence[Candidate],
) -> SelectionHealth:
    """F3's health record, with the three counts the wait may have moved.

    **Only the three counts and the rate are touched**, and only when a decision actually
    arrived during the wait - the reason histogram, the grader statistics and the named
    signals stay exactly as F3 computed them. F2 carries F3's judgement; it does not form
    one of its own (FR-002.15).
    """
    if (health.accepted, health.rejected, health.undecided) == (
        len(admitted),
        len(rejected),
        len(pending),
    ):
        return health
    n = health.n or (len(admitted) + len(rejected) + len(pending))
    return health.model_copy(
        update={
            "accepted": len(admitted),
            "rejected": len(rejected),
            "undecided": len(pending),
            # ADR-012: the rate never travels without its n, which is why `n` is carried
            # above rather than recomputed from the numerator.
            "acceptance_rate": (len(admitted) / n) if n else 0.0,
        }
    )


def _review_ids(records: Sequence[VerdictRecord]) -> tuple[str, ...]:
    """Review item and decision identifiers, de-duplicated, in order.

    **Identifiers only.** A reviewer's label, reason and identity are data F10 stores and a
    person reads; none of it belongs in a batch record that leaves this machine in an
    export (ADR-020).
    """
    found: list[str] = []
    for record in records:
        for value in (record.review_item_id, record.review_decision_id):
            if isinstance(value, str) and value and value not in found:
                found.append(value)
    return tuple(found)
