"""F6's two chain tables: a curriculum and its stages  (R-02.27; ADR-003, ADR-004).

## Core ``Table`` on its own ``MetaData``, not ``db.base.Base``

Foundation task 4.2's ratchet asserts ``set(Base.metadata.tables)`` **equals** the seven
base-schema tables, with the message *"a table only one feature reads belongs in that
feature's own migration."* ``services/parts/tables.py`` (F10), ``services/scoring/tables.py``
(F4) and ``training/runtypes/tables.py`` (F8) are the worked examples; this follows them,
carrying the same :data:`db.base.NAMING_CONVENTION` — unnamed CHECK constraints cannot be
dropped by name, which makes a migration reversible on PostgreSQL and irreversible on the
SQLite the unit tests run against — and the same ``mf_`` prefix.

**These tables precede their writer, and that is said plainly rather than disguised.** F6
shipped R-02.27's *types* and its safety-critical half — ``schemas/chain.py``'s ``AdvanceOn``,
``assert_advanceable``, ``advance_filter`` and ``halt_reason``, with 28 tests and three red
mutation controls across all three layers — and deliberately did **not** ship the
orchestration, reporting it *"blocked on the DDL"*. So what is owed on top of these tables is
``chain_service``, the ``miforge.maintenance.chain_sweep`` sweeper and four routes. Until
those land nothing writes a row here. That is the same shape as ``mf_merges``, recorded the
same way: a table ahead of its writer is DDL, not a shipped capability, and the owner of the
caller is named.

## The three advance reasons that can never advance a curriculum

R-02.27's correctness rests on one rule: **``retention_floor``, ``operator_stop`` and a
failed stage must never advance to the next stage.** A curriculum that hops after general
capability fell below its floor is a curriculum that trains the next stage on a damaged
model, and one that hops after an operator stopped it has overridden a human.

F6 enforces that in three layers of Python (the ``AdvanceOn`` ``Literal``,
``assert_advanceable`` at insert time, and ``advance_filter`` at decision time), each pinned
by its own mutation — M-8, M-8b and M-8c, all red.

**It is deliberately NOT enforced by a database CHECK**, and the reason is portability rather
than laziness. F6's handed-over DDL expressed it as ``advance_on @> '["retention_floor"]'``,
a **PostgreSQL JSONB containment operator** that SQLite cannot evaluate at all — and the
suite's default engine is SQLite. A constraint present on one engine and absent on the other
is precisely the divergence ``tests/support/db.py`` names as its blind spot: *"a
PostgreSQL-only type or a JSONB operator would pass here and fail there."* Three tested code
layers on both engines beat a fourth layer on one, so the storage layer is left out and the
reason is written down. The same judgement was made for ``mf_packs.derived_from_pack_id``.

## Not a store of record  (ADR-004)

A stage's ``definition`` is R-02.53's run definition, and the authoritative copy of what a
run was asked to do is the run log's ``definition`` block. These rows exist so a sweeper can
find the next stage without reading every run log on disk.
"""

from __future__ import annotations

import sqlalchemy as sa

from db.base import NAMING_CONVENTION, JsonColumn

#: F6's own metadata. See the module docstring for why this is not ``Base.metadata``.
metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

#: A chain's lifecycle. ``halted`` is distinct from ``completed``: a curriculum that stopped
#: early because a stage hit its retention floor did not finish, and reporting it as complete
#: would be the silently-wrong answer ADR-013 forbids.
CHAIN_STATUSES: tuple[str, ...] = ("pending", "running", "completed", "halted")
CHAIN_TERMINAL: tuple[str, ...] = ("completed", "halted")

#: The tables ``services.reindex`` may rebuild from disk. **Empty on purpose.** A chain is
#: not reconstructible from an artifact: its stages' definitions are durable in each stage's
#: run log, but a chain that has not started its second stage has written no log for it, so
#: there is nothing on disk to rebuild the plan from. Named rather than omitted, so the
#: absence reads as a decision; see ``services/reindex.py``'s ``RECORD_TABLES``.
CHAIN_INDEX_TABLES: tuple[str, ...] = ()


CHAINS = sa.Table(
    "mf_chains",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("name", sa.String(120), nullable=False),
    sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
    sa.Column("current_stage", sa.Integer, nullable=False, server_default="0"),
    # FR-006.44: the chain holds the GPU lease for its whole duration and a hop does not
    # release and re-acquire. A blind re-acquire could land on a DIFFERENT card mid
    # curriculum, which silently changes the device every later comparison was taken on -
    # and ADR-024's warning applies, since both PCIe links on that node are x4 not x16.
    sa.Column("gpu_uuid", sa.String(64), nullable=True),
    sa.Column("halt_code", sa.String(64), nullable=True),
    sa.Column("halt_detail", sa.Text, nullable=True),
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
    sa.CheckConstraint(
        "status IN ({})".format(", ".join(f"'{value}'" for value in CHAIN_STATUSES)),
        name="chain_status_known",
    ),
    sa.CheckConstraint(
        "(status IN ({})) = (finished_at IS NOT NULL)".format(
            ", ".join(f"'{value}'" for value in CHAIN_TERMINAL)
        ),
        name="chain_terminal_iff_finished",
    ),
    # A halt names WHY. "The chain stopped" and "the chain stopped because general
    # capability fell below the floor" are different sentences and only one is actionable.
    sa.CheckConstraint("status <> 'halted' OR halt_code IS NOT NULL", name="halted_names_its_code"),
)


STAGES = sa.Table(
    "mf_chain_stages",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column(
        "chain_id",
        sa.String(64),
        sa.ForeignKey("mf_chains.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("stage_index", sa.Integer, nullable=False),
    # R-02.53's definition for this stage, validated at CHAIN CREATION (FR-006.39) so a
    # curriculum's fourth stage cannot be discovered invalid three hours in.
    sa.Column("definition", JsonColumn, nullable=False),
    # Which stop reasons advance PAST this stage. The three that never can are enforced in
    # Python, not here - see the module docstring.
    sa.Column("advance_on", JsonColumn, nullable=False),
    # FK to mf_runs BY NAME rather than by object: that table is on `Base.metadata` and this
    # one is not, so the constraint is emitted by the migration and never resolved by a
    # `create_all` over this metadata alone. Same shape as F4's SCORES and F10's
    # REVIEW_BINDINGS.
    sa.Column("run_id", sa.String(64), nullable=True),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("stage_index >= 0", name="stage_index_non_negative"),
    # THE DOUBLE-START GUARD (FR-006.45). There is exactly one advance path - a sweep on the
    # `maintenance` queue - and this makes even a sweeper that runs twice unable to start a
    # stage twice. An IntegrityError here is read as "already started", which is why the
    # guard is a constraint and not a check-then-insert: the latter has a window.
    sa.UniqueConstraint("chain_id", "stage_index", name="uq_mf_chain_stages_chain_stage"),
    # A stage that has a run has started; one that has not, has not.
    sa.CheckConstraint("(run_id IS NOT NULL) = (started_at IS NOT NULL)", name="started_iff_run"),
    sa.Index("ix_mf_chain_stages_chain", "chain_id", "stage_index"),
)

ALL_TABLES: tuple[str, ...] = (CHAINS.name, STAGES.name)
