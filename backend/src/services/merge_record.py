"""Writing ``mf_merges`` - the record of an operation  (R-02.45; F8 tasks 11.5, 12.5).

**The table existed and nothing wrote to it.** F8 built ``MergeOutcome``, the migration and
the table with all three constraints, and reported accurately that the writer was F6's -
both of its tasks are marked ``[?]`` naming this feature. By ADR-016 #1 an unwritten table
is a capability that was not shipped: the index was fully implemented, fully constrained and
existed for no caller.

## A merge is an operation, not a run  (F6 FR-006.55)

So it gets this table and no ``mf_runs`` row. The mechanical reason is F6's: a stepless run
row **would be refused at submit for having no stop condition** (FR-006.17), would never
heartbeat and so would be reaped by the janitor, and would be a run whose step series is
empty - which every stop condition and every curve in F6 assumes is impossible. Both sides
agreed this on 2026-10-13 and F8's own documents model it that way.

## A **preflight** refusal is deliberately not recorded

``POST /api/v1/adapters/{name}/merge`` runs AC17's unmerged-base check before enqueueing
anything, and a refusal there writes **no row**. Recording one would make the table
reachable today and would invent a semantic no requirement states: R-02.45's three outcomes
describe *the merge*, not a request for one. A ``refused`` row is written only for a merge
that was admitted, enqueued, and then refused on the card - which is the outcome an operator
most needs to find afterwards, because it is the difference between *"the base was never
saved so I stopped"* and *"nothing happened and I do not know why"*.

The three outcomes and their constraints are F8's and are **cited, not restated**:
``written`` **iff** a result checkpoint name, and ``refusal_code`` only on a ``refused``
row. This module's job is to pass what it was given to a table that already refuses a row
that lies.
"""

from __future__ import annotations

import logging
from typing import Any

import sqlalchemy as sa

from training.runtypes.adapters.merge import MergeOutcome
from training.runtypes.tables import MERGES
from workers.run_record import utcnow

log = logging.getLogger(__name__)


def record(session: Any, outcome: MergeOutcome) -> int:
    """Insert one ``mf_merges`` row from F8's own outcome object. Returns its id.

    The row is built from :class:`training.runtypes.adapters.merge.MergeOutcome` field by
    field rather than from loose arguments, so the writer cannot assemble a row the merge
    did not produce - which is the only way a row about an **irreversible** operation could
    lie once the table's own constraints are in force.

    Raises:
        IntegrityError: from the table's ``written_iff_result`` or
            ``code_only_when_refused`` constraints. **Not caught**: a row that violates
            either is a claim about an irreversible operation that nothing can explain, and
            swallowing the error would store the lie.
    """
    result = session.execute(
        sa.insert(MERGES).values(
            adapter_name=outcome.adapter_name,
            adapter_step=outcome.adapter_step,
            adapter_manifest_sha256=outcome.adapter_manifest_sha256,
            base_checkpoint_name=outcome.base_checkpoint_name,
            result_checkpoint_name=outcome.result_checkpoint_name,
            outcome=outcome.outcome,
            refusal_code=outcome.refusal_code,
            created_at=utcnow(),
        )
    )
    row_id = result.inserted_primary_key[0] if result.inserted_primary_key else 0
    log.info(
        "merge recorded",
        extra={
            "code": "merge_recorded",
            "merge_id": row_id,
            "adapter": outcome.adapter_name,
            "base_checkpoint": outcome.base_checkpoint_name,
            "result_checkpoint": outcome.result_checkpoint_name,
            "outcome": outcome.outcome,
            "refusal_code": outcome.refusal_code,
        },
    )
    return int(row_id)


def refused(
    *,
    adapter_name: str,
    adapter_step: int,
    adapter_manifest_sha256: str,
    base_checkpoint_name: str,
    refusal_code: str,
) -> MergeOutcome:
    """A ``refused`` outcome, for a merge that was **admitted and then refused on the card**.

    Built through F8's own dataclass so the three-outcome vocabulary has one spelling. The
    caller is :mod:`workers.merge_tasks`; a *preflight* refusal does not come here - see the
    module docstring.
    """
    from training.runtypes.adapters.merge import OUTCOME_REFUSED

    return MergeOutcome(
        outcome=OUTCOME_REFUSED,
        adapter_name=adapter_name,
        adapter_step=adapter_step,
        adapter_manifest_sha256=adapter_manifest_sha256,
        base_checkpoint_name=base_checkpoint_name,
        result_checkpoint_name=None,
        refusal_code=refusal_code,
    )


def interrupted(
    *,
    adapter_name: str,
    adapter_step: int,
    adapter_manifest_sha256: str,
    base_checkpoint_name: str,
) -> MergeOutcome:
    """An ``interrupted`` outcome: **admitted, started, and did not finish.**

    The third of R-02.45's three outcomes, and it was the one with **no producer**. The
    vocabulary, the ``outcome_known`` CHECK and this table's own docstring all carried
    ``interrupted`` while ``grep -rn OUTCOME_INTERRUPTED backend/src`` matched only the
    constant's definition and the CHECK built from it - a value the schema admits, the
    documents explain, and nothing could ever write. An outcome no path produces is the
    same unreachable capability as a function nothing calls.

    **Carries no refusal code, and cannot.** ``code_only_when_refused`` permits one only on
    a ``refused`` row, which is the distinction the two words exist to draw: ``refused``
    means the merge was stopped **for a named reason this system decided**, and
    ``interrupted`` means it stopped for a reason that is not one of ours - an operator's
    cancellation, a crash on the card, a worker that died. The operator reading this table
    needs that difference, because only one of the two has somewhere to look next.

    **And it carries no result checkpoint, which the table enforces** - ``written_iff_result``
    in both directions. See :mod:`workers.merge_tasks` for the one case where an interrupt
    must be recorded as ``written`` instead: the artifact is on disk, so the row saying it
    is not would be the lie the constraint exists to prevent.
    """
    from training.runtypes.adapters.merge import OUTCOME_INTERRUPTED

    return MergeOutcome(
        outcome=OUTCOME_INTERRUPTED,
        adapter_name=adapter_name,
        adapter_step=adapter_step,
        adapter_manifest_sha256=adapter_manifest_sha256,
        base_checkpoint_name=base_checkpoint_name,
        result_checkpoint_name=None,
        refusal_code=None,
    )


def listing(
    session: Any, *, adapter_name: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Merge rows, newest first. The two lookups an operator actually performs.

    *"What happened to this adapter"* and *"what has been merged onto this base, and did any
    of it refuse"* - which is why a ``refused`` row exists at all: for the other two
    outcomes the merged checkpoint's own manifest is authoritative (ADR-004), and a refused
    merge writes no checkpoint, so this table is the only record there is.
    """
    query = sa.select(MERGES).order_by(MERGES.c.created_at.desc(), MERGES.c.id.desc())
    if adapter_name is not None:
        query = query.where(MERGES.c.adapter_name == adapter_name)
    rows = session.execute(query.limit(limit)).all()
    return [dict(row._mapping) for row in rows]


__all__ = ["interrupted", "listing", "record", "refused"]
