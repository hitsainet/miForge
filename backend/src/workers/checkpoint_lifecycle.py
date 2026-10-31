"""The checkpoint lifecycle sweep: decide, log, and only then delete  (OQ-4; ADR-004, 006).

OQ-4, settled by the operator on 2026-10-16: **keep the last N saves, every Mth save, and
unconditionally every checkpoint that is a parent in the lineage tree. Delete the rest.**

**Why this needed settling at all.** One 10,000-step run at ``save_every=100`` writes
480-960 GB; the checkpoints volume is 800Gi and fills around step 9,000, at which point the
save fails and the run has nothing to show for the hours since the last one. *The question
as OQ-4 was originally written proposed pruning ``mf_steps`` and justified it with the
480-960 GB figure - which is checkpoint bytes. ``mf_steps`` is forty-odd columns, so ten
thousand rows is tens of megabytes, and acting on the question as written would have
destroyed the commit-duration distribution FR-001.26 needs and left the gigabytes
untouched.* This sweep deletes directories.

------------------------------------------------------------------------------------
THE THREE GATES, IN THE ORDER THEY APPLY
------------------------------------------------------------------------------------

1. **Not enabled** (``CHECKPOINT_LIFECYCLE_ENABLED``, default false) - plan, log, delete
   nothing. The plan is still computed and reported, because *"what would this delete"* is
   the question an operator needs answered **before** turning it on, and a sweep that
   declines without saying what it would have done is a feature nobody can evaluate.
2. **Dry run** (``CHECKPOINT_LIFECYCLE_DRY_RUN``, default **true**) - same, and the report
   says ``dry_run``. Shipping disabled *and* dry-run is deliberate; see the settings.
3. **Enabled and not dry-run** - :func:`services.checkpoints.pruning.execute` stages,
   de-indexes and removes, one artifact at a time.

------------------------------------------------------------------------------------
WHAT IS LOGGED, AND WHY THE KEEPS ARE IN IT
------------------------------------------------------------------------------------

Every sweep logs the keep histogram by reason, **and one line per checkpoint kept because
something descends from it**. That subset is there because it is the one an operator
challenges: *"why is step 300 of a finished run still on my volume"* has exactly one good
answer - *"an adapter or a branch names it, and deleting it would truncate that walk"* -
and that answer has to be in the log at the moment the decision was made, not
reconstructible from a tree query weeks later.

A sweep that reported only its deletions would be unauditable in the same way the janitor
would be if it reported only its kills.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from core.errors import Refusal
from core.runtime_settings import effective_settings
from services.checkpoints import pruning
from services.checkpoints.lifecycle import (
    KEEP_LINEAGE_ANCESTOR,
    LifecyclePlan,
    LifecyclePolicy,
)
from services.checkpoints.lifecycle import (
    plan as build_plan,
)
from workers.run_record import RUNS, TERMINAL_STATES

log = logging.getLogger("miforge.checkpoints.lifecycle")

__all__ = ["SweepReport", "active_run_ids", "policy_from_settings", "sweep"]


@dataclass
class SweepReport:
    """One sweep: the decision, what it did, and which gate stopped it."""

    #: ``disabled`` | ``dry_run`` | ``executed`` | ``refused``.
    outcome: str = "disabled"
    kept: int = 0
    to_delete: int = 0
    deleted: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    orphaned: list[tuple[str, str]] = field(default_factory=list)
    kept_by_reason: dict[str, int] = field(default_factory=dict)
    #: Checkpoints kept **only** because something descends from them - the ones N and M
    #: would have taken. The clause's own evidence that it is doing something.
    protected_ancestors: list[str] = field(default_factory=list)
    active_runs: list[str] = field(default_factory=list)
    refusal_code: str | None = None
    policy: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "kept": self.kept,
            "to_delete": self.to_delete,
            "deleted": list(self.deleted),
            "deleted_count": len(self.deleted),
            "failed": [{"name": n, "error": e} for n, e in self.failed],
            "orphaned": [{"name": n, "error": e} for n, e in self.orphaned],
            "kept_by_reason": dict(self.kept_by_reason),
            "protected_ancestors": list(self.protected_ancestors),
            "active_runs": list(self.active_runs),
            "refusal_code": self.refusal_code,
            "policy": dict(self.policy),
        }


def policy_from_settings() -> LifecyclePolicy:
    """N and M from configuration. **Settings, never literals.**"""
    settings = effective_settings()
    return LifecyclePolicy(
        keep_last_n=settings.checkpoint_lifecycle_keep_last_n,
        keep_every_m=settings.checkpoint_lifecycle_keep_every_m,
    )


def active_run_ids(session: Any) -> frozenset[str]:
    """Runs whose ``mf_runs`` row is **not** terminal - every artifact of theirs is immune.

    *This is the "never delete while a run is actively writing" clause, and it is coarse on
    purpose.* The precise question - *is a save in flight right now* - is not answerable
    from outside the writing process: the save publishes with one ``rename`` and a sweep
    that checked a flag would still race it. So the whole of a non-terminal run is
    untouchable, which costs a few directories that could have gone and removes the race
    entirely.

    ``TERMINAL_STATES`` is imported rather than restated - it includes ``cancelled``, and
    the cost of omitting a member here is deleting under a run that is still writing.
    """
    rows = session.execute(
        sa.select(RUNS.c.id).where(RUNS.c.status.notin_(tuple(TERMINAL_STATES)))
    ).all()
    return frozenset(str(row[0]) for row in rows)


def _log_plan(decided: LifecyclePlan, report: SweepReport) -> None:
    """The keeps, by reason, and every ancestor-protected checkpoint by name."""
    for decision in decided.keep:
        if decision.reason != KEEP_LINEAGE_ANCESTOR:
            continue
        log.info(
            "checkpoint kept because something descends from it",
            extra={
                "code": "checkpoint_kept_lineage_ancestor",
                "checkpoint": decision.name,
                "run_id": decision.run_id,
                "step": decision.step,
                # Named so a reader can see whether N or M would also have kept it. When
                # this is the only reason, the clause is what saved the subtree.
                "reasons": list(decision.reasons),
            },
        )
    log.info(
        "checkpoint lifecycle plan",
        extra={
            "code": "checkpoint_lifecycle_plan",
            "outcome": report.outcome,
            "kept": report.kept,
            "to_delete": report.to_delete,
            "kept_by_reason": report.kept_by_reason,
            "protected_ancestors": len(report.protected_ancestors),
            "active_runs": len(report.active_runs),
            "policy": report.policy,
        },
    )


def sweep(session: Any, *, root: Path | None = None) -> SweepReport:
    """One pass. Decide from manifests, log, and delete only through both gates.

    Args:
        session: an open session. :func:`services.checkpoints.pruning.execute` commits per
            artifact when it deletes; nothing is committed on the other paths.
        root: the artifact root, for tests.

    Returns:
        A :class:`SweepReport`. A :class:`core.errors.Refusal` from the collection step -
        an unreadable manifest, an impossible policy - is **caught and reported**, not
        raised: a sweep on a schedule that raises becomes a Celery traceback every sixty
        minutes and the next sweep has the same problem. The report names the code.
    """
    settings = effective_settings()
    report = SweepReport(
        policy={
            "keep_last_n": settings.checkpoint_lifecycle_keep_last_n,
            "keep_every_m": settings.checkpoint_lifecycle_keep_every_m,
        }
    )

    try:
        collected = pruning.collect(root)
        active = active_run_ids(session)
        decided = build_plan(
            collected.artifacts,
            policy_from_settings(),
            latest_names=collected.latest_names,
            active_run_ids=active,
        )
    except Refusal as refusal:
        report.outcome = "refused"
        report.refusal_code = refusal.code
        log.error(
            "checkpoint lifecycle sweep refused; nothing was deleted",
            extra={
                "code": refusal.code,
                # `detail`, NOT `message`: `message` is a reserved LogRecord attribute and
                # `logging` raises `KeyError: Attempt to overwrite 'message'` on it - so the
                # refusal path, which exists to report a problem, would itself raise. The
                # save service records the same trap one field over (`checkpoint`, not
                # `name`). Found by running the test, not by reading the call.
                "detail": refusal.message,
            },
        )
        return report

    report.kept = len(decided.keep)
    report.to_delete = len(decided.delete)
    report.kept_by_reason = decided.kept_by_reason
    report.active_runs = sorted(active)
    report.protected_ancestors = [
        d.name for d in decided.keep if d.reasons == (KEEP_LINEAGE_ANCESTOR,)
    ]

    if not settings.checkpoint_lifecycle_enabled:
        report.outcome = "disabled"
        _log_plan(decided, report)
        log.info(
            "checkpoint lifecycle is not enabled; the plan above was not acted on",
            extra={
                "code": "checkpoint_lifecycle_disabled",
                "would_delete": report.to_delete,
                "setting": "CHECKPOINT_LIFECYCLE_ENABLED",
            },
        )
        return report

    dry_run = settings.checkpoint_lifecycle_dry_run
    report.outcome = "dry_run" if dry_run else "executed"
    _log_plan(decided, report)

    result = pruning.execute(session, decided, root=root, dry_run=dry_run)
    report.deleted = result.deleted
    report.failed = result.failed
    report.orphaned = result.orphaned
    return report
