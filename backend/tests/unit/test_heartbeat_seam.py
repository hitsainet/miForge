"""The jobs package's seam onto ``mf_runs`` and ``mf_leases``  (foundation 6.4a, 6.7).

**This file exists because the seam was wrong once and nothing could see it.**
:mod:`workers.run_record` originally declared its own ``sa.Table`` objects as a defensive
seam, written while foundation 4.2 was still in flight. The test support built *those*
tables, so every test in this package passed. When 4.2 landed, **six of the seven column
names were wrong**: ``run_id`` for ``id``, ``state`` for ``status``, a boolean
``stop_requested`` for a ``cancel_requested_at`` timestamp, ``error_code`` for
``failure_code``, milliseconds for seconds, and a ``heartbeat_writes`` column that does not
exist. The fixture and the code were two views of one mistake - **fields agreeing by
construction**, which ADR-016 names as the usual reason a suite stays green over a real
bug.

The seam now imports 4.2's models, so a rename stops the import and every test in this
package goes red at once. What remains worth asserting is the part an import cannot check:
that the **names this package publishes as the answer to task 6.4a** are the names on the
real table, and that the terminal set it guards against has not lost a member.
"""

from __future__ import annotations

import sqlalchemy as sa

from models.run import RUN_STATUSES, TERMINAL_STATUSES, Run
from workers import run_record
from workers.heartbeat import HEARTBEAT_COLUMN, HEARTBEAT_LONGEST_GAP_COLUMN


class TestTheHeartbeatColumnIsNamedOnce:
    """Task 6.4a asked for the column to be *named*, because F9's finding F-15 recorded
    that **no heartbeat column was named anywhere in ``0xcc/``** and the run screen's
    liveness rule had no input. These assertions are what keep one name."""

    def test_the_heartbeat_column_exists_on_mf_runs(self):
        assert "heartbeat_at" in Run.__table__.c
        assert HEARTBEAT_COLUMN == "heartbeat_at"

    def test_it_is_a_timezone_aware_timestamp(self):
        """A naive column makes the age arithmetic wrong by the server's offset, silently,
        and the value being computed is whether a run is alive."""
        column = Run.__table__.c["heartbeat_at"]
        assert isinstance(column.type, sa.DateTime)
        assert column.type.timezone is True

    def test_it_is_nullable_because_a_run_that_has_never_beaten_is_a_real_state(self):
        """And the janitor treats it as ambiguous rather than reaping on it."""
        assert Run.__table__.c["heartbeat_at"].nullable is True

    def test_the_longest_gap_column_exists_and_is_in_seconds(self):
        """Task 6.4b. **Seconds, not milliseconds**, and the name says so: it exists to
        give ``JANITOR_LAPSE_SECONDS`` a defensible value, so it is in that threshold's own
        unit and directly comparable with ``age_seconds`` on the same response. F6's
        per-boundary ``longest_gap_ms`` is a different measurement at a different
        granularity, in its own run-log block."""
        assert HEARTBEAT_LONGEST_GAP_COLUMN == "longest_heartbeat_gap_seconds"
        assert HEARTBEAT_LONGEST_GAP_COLUMN in Run.__table__.c
        assert HEARTBEAT_LONGEST_GAP_COLUMN.endswith("_seconds")

    def test_the_longest_gap_is_nullable_so_unobserved_is_not_zero(self):
        """A run that has written once has **no gap to report**; a run showing 0 has
        written twice inside a second. Collapsing them makes a brand-new run look like the
        smoothest run in the system."""
        assert Run.__table__.c[HEARTBEAT_LONGEST_GAP_COLUMN].nullable is True


class TestTheLifecycleSetIsImportedNotRestated:
    def test_the_terminal_set_includes_cancelled(self):
        """**The member it would have been easiest to omit.** A terminal set missing
        ``cancelled`` means the heartbeat happily moves a cancelled row, and the run list
        shows a cancelled run as alive."""
        assert "cancelled" in run_record.TERMINAL_STATES
        assert frozenset(TERMINAL_STATUSES) == run_record.TERMINAL_STATES

    def test_the_terminal_set_is_a_subset_of_the_known_statuses(self):
        assert run_record.TERMINAL_STATES <= RUN_STATUSES

    def test_the_running_and_pending_names_match_the_model(self):
        assert run_record.STATE_RUNNING in RUN_STATUSES
        assert run_record.STATE_PENDING in RUN_STATUSES

    def test_a_non_terminal_status_is_not_in_the_terminal_set(self):
        assert run_record.STATE_RUNNING not in run_record.TERMINAL_STATES
        assert run_record.STATE_PENDING not in run_record.TERMINAL_STATES


class TestTheColumnsThisPackageWrites:
    def test_every_aliased_column_belongs_to_its_real_table(self):
        """The aliases are the only translation layer in the package, so each must be a
        plain re-export of a real column rather than a lookalike."""
        for alias in (
            run_record.RUN_ID,
            run_record.RUN_STATUS,
            run_record.HEARTBEAT_AT,
            run_record.HEARTBEAT_LONGEST_GAP_SECONDS,
            run_record.CANCEL_REQUESTED_AT,
            run_record.FAILURE_CODE,
            run_record.FAILURE_MESSAGE,
            run_record.FINISHED_AT,
        ):
            assert alias.table is Run.__table__

    def test_the_run_primary_key_is_id_not_run_id(self):
        """The spelling that was wrong in the original seam, pinned so a reader does not
        reintroduce ``run_id`` as a column. ``run_id`` is the **wire** name and the API
        translates; the column is ``id``."""
        assert run_record.RUN_ID.name == "id"
        assert [c.name for c in Run.__table__.primary_key] == ["id"]

    def test_cancellation_is_a_timestamp_not_a_flag(self):
        """4.2's choice and the better one: a flag answers *"should I stop"*, a timestamp
        also answers *"when was it asked"* - which is what distinguishes a slow shutdown
        from an ignored request."""
        column = Run.__table__.c["cancel_requested_at"]
        assert isinstance(column.type, sa.DateTime)
        assert column.nullable is True

    def test_a_terminal_status_requires_a_finish_time(self):
        """``terminal_iff_finished``. The janitor must set ``finished_at`` when it reaps, or
        the write raises - so the constraint protects this package from itself. Asserted
        here because the first draft of the janitor set the status alone."""
        # The naming convention in db/base.py prefixes check constraints with `ck_<table>_`,
        # so the match is on the suffix rather than on the bare name a reader of
        # models/run.py sees.
        names = {c.name for c in Run.__table__.constraints if c.name}
        assert "ck_mf_runs_terminal_iff_finished" in names, sorted(names)


class TestTheLeaseTable:
    def test_the_holder_and_the_expiry_are_both_not_null(self):
        """ADR-008's two omissions, closed in the schema rather than in a service's good
        intentions: no holder means nothing can say what to ask, and no expiry means a
        crashed run holds the card forever."""
        from models.lease import Lease

        assert Lease.__table__.c["holder_run_id"].nullable is False
        assert Lease.__table__.c["expires_at"].nullable is False

    def test_exclusivity_is_a_partial_unique_index_over_the_unreleased_rows(self):
        """Which is **why** :func:`workers.lease.acquire` reclaims before it inserts: expiry
        is a fact about ``expires_at`` and the index only knows about ``released_at``, so a
        lapsed lease is still unreleased and still blocks."""
        from models.lease import Lease

        indexes = {i.name: i for i in Lease.__table__.indexes}
        unique = indexes["uq_mf_leases_resource_unreleased"]
        assert unique.unique is True
        assert [c.name for c in unique.columns] == ["resource"]
        assert "released_at IS NULL" in str(unique.dialect_options["postgresql"]["where"])
