"""Reading F4's durable score rows  (FR-006.19, FR-006.20).

Two conditions need numbers F4 produced, and **neither computes one**. This module is the
one read path, so there is one answer to *"what is the most recent recorded score"* and one
answer to *"has a floor been breached"*.

## It reads ``mf_scores`` and makes no comparison of its own

:func:`latest_breach` selects on F4's ``breached`` column. That column is, in F4's own
words, *"a fact on the pass, not something derived later by comparing two numbers"* -
because ``reindex`` rebuilds from the log alone, and ``score < floor`` recomputed later
against a floor that has since been edited gives a different answer. F6 therefore has
**no floor comparison anywhere**, which
``backend/tests/unit/test_stop_no_floor_comparison.py`` asserts structurally: the division
is kept by what this feature's code cannot contain, which is how F4's TID keeps its half.

## Out-of-cadence rows are excluded, and that is a decision

``mf_scores.out_of_cadence`` marks an ad-hoc check requested through F4's own
``POST /runs/{id}/evaluate``. Its column comment is *"an ad-hoc check must never silently
join the step series"* - and **ending a run** on one would be the loudest possible way of
joining it. An operator spot-checking a run must not thereby stop it, so the series the
stop conditions read is the cadenced one. A breach found out of cadence is still published
by F4 and still visible; it does not by itself terminate the run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import sqlalchemy as sa

from services.scoring.tables import KIND_RETENTION, KIND_TEST, SCORES


def _latest(session: Any, run_id: str, kind: str, *, breached_only: bool = False) -> Any:
    """The newest cadenced row of one kind, or ``None``.

    Ordered by the canonical step and then by ``created_at``, because a re-score **appends**
    (F4's unique index is over ``(run_id, step, kind, created_at)``), so the latest row for
    a step is the one a stop condition must read - an earlier superseded number would stop
    or fail to stop the run on a figure F4 has already replaced.
    """
    query = (
        sa.select(SCORES)
        .where(SCORES.c.run_id == run_id)
        .where(SCORES.c.kind == kind)
        .where(SCORES.c.out_of_cadence.is_(False))
    )
    if breached_only:
        query = query.where(SCORES.c.breached.is_(True))
    row = session.execute(
        query.order_by(SCORES.c.step.desc(), SCORES.c.created_at.desc()).limit(1)
    ).one_or_none()
    return None if row is None else dict(row._mapping)


def latest_test_score(session: Any, run_id: str) -> Mapping[str, Any] | None:
    """The most recent cadenced test-set score row, or ``None`` if none has been recorded."""
    row: Mapping[str, Any] | None = _latest(session, run_id, KIND_TEST)
    return row


def latest_breach(session: Any, run_id: str) -> Mapping[str, Any] | None:
    """The most recent cadenced retention row **F4 marked as a breach**, or ``None``.

    ``None`` means *no breach has been recorded*, which is not the same fact as *no
    retention score has been recorded* - the second is a run whose floor cannot be measured
    and is E-11's concern at the evaluation, not this one's.
    """
    row: Mapping[str, Any] | None = _latest(session, run_id, KIND_RETENTION, breached_only=True)
    return row


def sample_phrase(row: Mapping[str, Any]) -> str:
    """``score over n prompts, aggregated by X, from provider P`` (ADR-012).

    *Every number names its scale and sample*, and it applies hardest to the number that
    ended a run: an accept rate and a normalised grader mean are both "0.84" and are not
    the same measurement, so the aggregation method rides with the figure.
    """
    return (
        f"{float(row['score']):.4f} over {int(row['n'])} prompts "
        f"({row['aggregation']}, {row['provider']})"
    )


__all__ = ["latest_breach", "latest_test_score", "sample_phrase"]
