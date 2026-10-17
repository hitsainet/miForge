"""``operator_stop`` - the quietest line in the feature  (R-02.25; FR-006.24, FR-006.25).

## Why this is not simply ``SELECT stop_requested``

Every worker is ``--pool=solo -c 1`` (ADR-006). ``revoke(terminate=True)`` signals a pool
child and the solo pool has none, so the call returns successfully and nothing happens - the
worst available failure mode, because the operator sees a cancellation that was accepted.
And a busy solo worker never reads the control queue. So the stop is a column on the
``mf_runs`` row (ADR-007: *the run record is the protocol*).

**"Through the database" is necessary and not sufficient.** A worker that holds one
long-lived transaction for the run's duration reads a snapshot taken before the operator
pressed stop, and then sees the flag **never** - for hours, with no error on any surface.
The operator presses stop, the API returns 202, the row is updated, and the run carries on.

So the read happens in a **new, short session** each boundary. Two things have to be
defeated and only one of them is the transaction:

1. the **open transaction's snapshot**, which is why the session is new (or, on the
   fallback path, committed first);
2. SQLAlchemy's **identity map**, which would hand back the already-loaded ``Run`` object
   even after a commit, which is why ``expire_all`` is called on the fallback path.

*"Re-query" is not enough for either of them, which is the whole reason this module exists
rather than one line in the loop.*

**Mutation M-2** deletes the new-session / commit-and-expire before the read. The test for
it must mutate the row **from a second session** (``integration/test_operator_stop_second_
session.py``): a fixture that updates the row through the loop's own session agrees with the
loop by construction and passes against the broken line, which is exactly the fixture
failure ADR-016 warns about. *The control is worthless without that.*

## It reads F1's column and adds no second flag

``mf_runs.cancel_requested_at`` and :func:`workers.cancellation.stop_requested` are F1's and
the foundation's (task 6.3). F6 adds **no second route, no second column and no second
mechanism** - it owns the *condition*, the precedence and the terminal transition; F1 owns
``POST /runs/{run_id}/cancel`` and the primitive. Reported as **AMB-3**, since F1 defines
that route while its own non-goal assigns stop conditions here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.errors import refuse
from training.stop.base import StopContext, StopReason, Verdict
from workers.cancellation import stop_requested, stop_requested_at

NAME = StopReason.OPERATOR_STOP.value


class OperatorStop:
    """R-02.25's fifth condition. Takes no configuration: it is on or it is off."""

    name = NAME

    def evaluate(self, ctx: StopContext, config: Mapping[str, Any]) -> Verdict:
        del config
        requested, asked_at = _fresh_read(ctx)
        return Verdict(
            condition=NAME,
            triggered=requested,
            observed_at_step=ctx.step,
            detail=(
                f"stop requested at {asked_at.isoformat()}"
                if requested and asked_at is not None
                else "no stop requested"
            ),
            values={
                "requested": requested,
                "requested_at": None if asked_at is None else asked_at.isoformat(),
            },
        )


def _fresh_read(ctx: StopContext) -> tuple[bool, Any]:
    """Read the flag in a transaction that **began after** this boundary did.

    Prefers a brand-new session from ``ctx.session_scope``. Falls back to the loop's own
    session with an explicit ``commit()`` then ``expire_all()``, which ends the snapshot and
    empties the identity map - the same two defeats, one object.

    Raises:
        Refusal: ``operator_stop_unreadable`` when there is neither. **Refused rather than
            answered ``False``**: a condition that cannot read its flag and reports "no stop
            requested" is a stop button that does nothing, and a visible refusal is
            recoverable where a silently wrong answer is not (ADR-013).
    """
    if ctx.session_scope is not None:
        # M-2: A NEW SESSION. Not a re-query - a re-query inside the loop's transaction
        # reads the same snapshot, and the identity map would return the same object even
        # after the snapshot moved. Deleting this branch and the commit/expire below is the
        # mutation, and `integration/test_operator_stop_second_session.py` must go red.
        with ctx.session_scope() as fresh:
            return stop_requested(fresh, ctx.run_id), stop_requested_at(fresh, ctx.run_id)

    if ctx.session is not None:
        # M-2, the fallback half. BOTH calls are load-bearing and for different reasons:
        # `commit` ends the long-lived transaction so the next statement opens a new one,
        # and `expire_all` empties the identity map so the next read goes to the database
        # rather than to the object already loaded.
        ctx.session.commit()
        ctx.session.expire_all()
        return stop_requested(ctx.session, ctx.run_id), stop_requested_at(ctx.session, ctx.run_id)

    raise refuse(
        "operator_stop_unreadable",
        "The operator-stop condition has no session to read mf_runs.cancel_requested_at "
        "from, so it cannot say whether a stop was requested. Refusing rather than "
        "reporting 'no stop requested', which would be a stop button that does nothing.",
        run_id=ctx.run_id,
        step=ctx.step,
    )


CONDITION = OperatorStop()

__all__ = ["CONDITION", "NAME", "OperatorStop"]
