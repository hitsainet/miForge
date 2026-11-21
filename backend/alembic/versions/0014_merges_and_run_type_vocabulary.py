"""``mf_merges``, and one spelling for the run type that changes weights.

Revision ID: 0014_merges
Revises: 0013_scoring
Create Date: 2026-10-14

Two pieces of F8's schema debt, both of which the feature correctly declined to take:
``backend/alembic/`` and ``backend/src/models/`` are shared state and the Alembic revision
graph is serialised through the coordinator, because two revisions chaining off one parent
give every developer multiple heads and ``alembic upgrade head`` then fails for all of them
on someone else's clean clone.

**1. ``mf_merges``.** A merge produces a checkpoint but **is not a run** and gets no
``mf_runs`` row - F6's FR-006.55 supplies the mechanical reason: a stepless run row would be
refused at submit for having no stop condition, would never heartbeat and so would be reaped
by the janitor, and would be a run whose step series is empty, which every stop condition and
every curve assumes is impossible. So F8 built :class:`MergeOutcome` as the record of an
*operation* and reported, accurately, that *"it is not persisted: ``mf_merges`` does not
exist in the schema."* Here it does.

Per ADR-004 this table is an **index, not the record**. The merged checkpoint's own
``manifest.json`` carries ``merged_adapter`` and is authoritative; this row exists so that a
refused or interrupted merge - which writes no checkpoint and therefore leaves no manifest -
is visible at all. That is the case worth having a table for: a merge is irreversible, and a
merge that refused is the outcome an operator most needs to find afterwards.

``outcome`` is constrained to the three values R-02.45 allows to be recorded, and
``result_checkpoint_name`` is nullable **because two of those three write no checkpoint**.
The pairing is enforced: ``written`` iff a result name is present. Without that constraint a
``refused`` row carrying a checkpoint name, or a ``written`` row carrying none, would both
store cleanly and either is a lie about an irreversible operation.

**2. ``run_type``: ``full`` becomes ``change``.** The CHECK constraint from ``0001`` permitted
``create|full|adapter|merge`` while BRD-02 §4 names the three run types *"direct weight
**change** on an existing model"*, from-scratch creation, and adapter training - and F8's
registry refuses anything but ``'change'``, ``'create'``, ``'adapter'``. F8 bridged the two
with ``DB_RUN_TYPE = {"change": "full", ...}``.

That map is removed with this migration, and the reason is not tidiness. **A translation
table between two names for one concept is a second source of truth**, which is the defect
class this project has paid for repeatedly - the ``step-`` directory prefix, the three signal
vocabularies conflated by one ``sed``, an invented ``skip`` decision asserted as a sibling's.
Each was two spellings of one thing with something in the middle keeping them in step, until
it did not. ``change`` wins over ``full`` on the merits as well: it is the word the
requirement uses and the word a user sees, whereas a row reading ``run_type='full'`` does not
say full *what*.

``merge`` is **left in the permitted set** although F8's reasoning above means nothing should
ever write it. Removing it is a stronger statement than is currently earned - F6 is not built,
so nothing has yet proved no submit path produces one - and a permitted-but-unused value costs
nothing, while removing a value something turns out to need costs a migration under load.
Flagged rather than silently decided.

**Why the ``copy_from``, and why it is not optional.** Changing a named CHECK constraint on
SQLite requires a table rebuild, and the test suite builds its schema with ``alembic upgrade
head`` against SQLite (``tests/support/db.py``), so this path runs on every test session.
SQLAlchemy does not reflect SQLite CHECK constraints, so a ``batch_alter_table`` left to
reflect would rebuild ``mf_runs`` **without** ``ck_mf_runs_terminal_iff_finished`` and
``ck_mf_runs_status_known`` - dropping two invariants silently, while the migration reported
success. The table is therefore declared in full below and passed as ``copy_from``. The
downgrade test and the constraint assertions in ``test_run_type_vocabulary.py`` are what keep
that true rather than this paragraph.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

revision: str = "0014_merges"
down_revision: str | None = "0013_scoring"
branch_labels: str | None = None
depends_on: str | None = None

#: Wrapped in ``op.f()`` at every use. The migration context carries the metadata naming
#: convention ``ck_%(table_name)s_%(constraint_name)s``, which would expand this
#: already-final name into ``ck_mf_runs_ck_mf_runs_run_type_known`` and fail to find the
#: constraint. ``op.f`` marks a name as literal, which is why ``0001`` uses it too.
_RUN_TYPE_CHECK = "ck_mf_runs_run_type_known"

#: JSON on PostgreSQL, TEXT-backed JSON on SQLite - the same spelling ``0001`` uses, so the
#: two dialects do not diverge in how a definition body is stored.
JSON_COLUMN = sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), "postgresql")


def _mf_runs(run_types: str) -> sa.Table:
    """``mf_runs`` as it stands, for ``batch_alter_table(copy_from=...)``.

    Every column through revision ``0012`` and all three CHECK constraints, with the
    ``run_type`` set passed in so the same declaration serves the upgrade and the downgrade.

    This duplicates ``0001`` plus ``0012`` deliberately. A migration must not import
    ``src/models/`` - the models track the *current* head and this file must keep describing
    the table as it is at **this** revision, or a later model change would silently rewrite
    history the next time someone upgrades from scratch.
    """
    meta = sa.MetaData()
    return sa.Table(
        "mf_runs",
        meta,
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("run_type", sa.String(length=16), nullable=False),
        sa.Column("definition_submitted", JSON_COLUMN, nullable=True),
        sa.Column("definition_resolved", JSON_COLUMN, nullable=True),
        sa.Column("definition_sha256", sa.String(length=64), nullable=True),
        sa.Column("pack_id", sa.String(length=64), nullable=True),
        sa.Column("gpu_request", sa.String(length=64), nullable=True),
        sa.Column("gpu_uuid", sa.String(length=64), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("longest_heartbeat_gap_seconds", sa.Integer(), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_step", sa.Integer(), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        # Added by 0012. Present here because copy_from must describe the table as it IS,
        # not as 0001 left it - omitting them would drop both columns on the rebuild.
        sa.Column("save_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("save_request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.CheckConstraint(
            "(status IN ('completed','failed','cancelled')) = (finished_at IS NOT NULL)",
            name="ck_mf_runs_terminal_iff_finished",
        ),
        sa.CheckConstraint(f"run_type IN ({run_types})", name=_RUN_TYPE_CHECK),
        sa.CheckConstraint(
            "status IN ('pending','running','completed','failed','cancelled')",
            name="ck_mf_runs_status_known",
        ),
        sa.ForeignKeyConstraint(
            ["pack_id"],
            ["mf_packs.id"],
            name="fk_mf_runs_pack_id_mf_packs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_mf_runs"),
        # AND THE INDEX. A `copy_from` table that omits an index loses it on the rebuild,
        # exactly as it would lose an unnamed constraint - and this one was lost on the first
        # attempt. The verification printed `indexes kept: []` and that line was read past;
        # `test_db_migration.py` then caught it, which is the test that exists because the
        # migration and the models are otherwise never compared to each other.
        sa.Index("ix_mf_runs_status_created_at", "status", "created_at"),
    )


_OLD = "'create','full','adapter','merge'"
_NEW = "'create','change','adapter','merge'"
#: Both spellings at once, held only between the two rebuilds below.
_BOTH = "'create','full','change','adapter','merge'"


def _retype(copy_from_set: str, to_set: str) -> None:
    """Rebuild ``mf_runs`` with ``run_type``'s CHECK set to ``to_set``.

    ``copy_from_set`` is the constraint the table carries **now**, because ``copy_from`` must
    describe the table as it is or the rebuild writes a different one.
    """
    with op.batch_alter_table(
        "mf_runs", schema=None, copy_from=_mf_runs(copy_from_set)
    ) as batch_op:
        batch_op.drop_constraint(op.f(_RUN_TYPE_CHECK), type_="check")
        batch_op.create_check_constraint(op.f(_RUN_TYPE_CHECK), f"run_type IN ({to_set})")


def _migrate_run_type(*, before: str, after: str, old_set: str, new_set: str) -> None:
    """Rename a permitted ``run_type`` value **with rows present**.

    Three steps, and the order is the whole point:

    1. widen the constraint to permit both spellings,
    2. rewrite the rows,
    3. narrow it to the new set alone.

    **The two-step version of this was wrong and the suite caught it.** Written as "UPDATE,
    then swap the constraint", the UPDATE writes a value the *old* constraint forbids, so it
    raises ``CHECK constraint failed`` the moment a single row exists. It passed anyway,
    because the table was empty - and this file's own comment said the UPDATE was written
    *"because 'there are no rows' is an assumption about every installation, not a fact"*,
    while the code it introduced depended on exactly that assumption. It surfaced only in the
    downgrade, and only because the verification inserted rows before running it; the upgrade
    half was latent and would have failed on the first installation with any history.

    The cost is a second table copy on SQLite and a second validating scan on PostgreSQL.
    That is the right price for a rename that works on a populated table.
    """
    _retype(old_set, _BOTH)
    op.execute(
        sa.text("UPDATE mf_runs SET run_type = :after WHERE run_type = :before").bindparams(
            before=before, after=after
        )
    )
    _retype(_BOTH, new_set)


def upgrade() -> None:
    op.create_table(
        "mf_merges",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # The adapter that was merged, by the identity its own manifest carries.
        sa.Column("adapter_name", sa.String(length=255), nullable=False),
        sa.Column("adapter_step", sa.Integer(), nullable=False),
        # The adapter manifest's hash, so the row names the exact artifact merged and not
        # merely a name that may since have been rewritten.
        sa.Column("adapter_manifest_sha256", sa.String(length=64), nullable=False),
        # The unmerged base. AC17 requires it to survive the merge, so recording which
        # checkpoint it was is what makes the survival checkable afterwards.
        sa.Column("base_checkpoint_name", sa.String(length=255), nullable=False),
        # NULL for 'refused' and 'interrupted': neither writes a checkpoint. The pairing
        # constraint below is what stops that nullability becoming a place to hide a lie.
        sa.Column("result_checkpoint_name", sa.String(length=255), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        # Set from the raised refusal's code. A refusal is raised rather than returned - a
        # caller can ignore a return value - so this column records what was raised.
        sa.Column("refusal_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "outcome IN ('written','refused','interrupted')",
            name=op.f("ck_mf_merges_outcome_known"),
        ),
        # written IFF there is a result. Both directions, because both are lies about an
        # irreversible operation: a 'written' row with no checkpoint name says a merge
        # happened and cannot say to what, and a 'refused' row carrying one says a merge
        # that did not happen produced an artifact.
        sa.CheckConstraint(
            "(outcome = 'written') = (result_checkpoint_name IS NOT NULL)",
            name=op.f("ck_mf_merges_written_iff_result"),
        ),
        # A refusal code belongs to a refusal. Not the converse: 'interrupted' carries no
        # code and neither does 'written'.
        sa.CheckConstraint(
            "refusal_code IS NULL OR outcome = 'refused'",
            name=op.f("ck_mf_merges_code_only_when_refused"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_merges")),
    )
    with op.batch_alter_table("mf_merges", schema=None) as batch_op:
        # The two lookups an operator actually performs: "what happened to this adapter"
        # and "what has been merged onto this base, and did any of it refuse".
        batch_op.create_index(
            "ix_mf_merges_adapter", ["adapter_name", "adapter_step"], unique=False
        )
        batch_op.create_index("ix_mf_merges_base", ["base_checkpoint_name"], unique=False)

    _migrate_run_type(before="full", after="change", old_set=_OLD, new_set=_NEW)


def downgrade() -> None:
    _migrate_run_type(before="change", after="full", old_set=_NEW, new_set=_OLD)

    with op.batch_alter_table("mf_merges", schema=None) as batch_op:
        batch_op.drop_index("ix_mf_merges_base")
        batch_op.drop_index("ix_mf_merges_adapter")
    op.drop_table("mf_merges")
