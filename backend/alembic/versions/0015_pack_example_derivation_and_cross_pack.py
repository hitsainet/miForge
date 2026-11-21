"""Three pack columns: the Example pill, the derivation record, cross-pack addressing.

Revision ID: 0015_pack_cols
Revises: 0014_merges
Create Date: 2026-10-14

**Operator decision, 2026-10-14.** F7 reported three capabilities its own feature documents
promise and the pack **artifact cannot express**, because `pack-definition-v1.json`'s
top-level object and its `PackSlot` are both closed to unknown keys — so none of the three
can ride in `pack.json` either, and each needed either a column or deleting from the feature.
The operator chose the columns.

What each one unblocks, and what its absence cost:

* **``mf_packs.is_example``** — FR-007.16's Example pill. There was no field anywhere, so F7
  shipped **no example packs at all** rather than seed unmarked ones indistinguishable from
  real ones. That was the right call and it cost the product the two worked examples that are
  how a new operator learns what a pack *is*.
* **``mf_packs.derived_from_pack_id``** — FR-007.12's derivation record. Cloning a pack
  worked; recording *what it was cloned from* did not, so a registry full of near-identical
  packs had no way to say which came from which.
* **``mf_pack_parts.addressed_through_pack_id``** — FR-007.25's cross-pack part addressing.
  The renderer already takes the pack as an argument, so **display** worked; **storage** did
  not, which is the half that survives a restart.

## EXPORT BEFORE DOWNGRADING — and an export does not save these three (task 4.5)

**The downgrade below works and it is lossy.** Round-tripped with rows present
(``tests/integration/test_migration_pack_columns.py``): the pack rows survive, both indexes
come back, ``mf_packs``'s CHECK constraint is still there afterwards, and **all three values
are gone** — ``is_example`` returns as ``false`` for every pack, the two id columns as null.

Nothing undoes that, and the reason is the same one that made each column necessary.
``packs.registry.rebuild_index`` rebuilds both tables from ``pack.json`` alone, so losing the
database loses no **pack** — but ``pack-definition-v1.json``'s top-level object and its
``PackSlot`` are **closed to unknown keys**, so none of these three can ride in the artifact,
which is exactly why none of them could be a field instead of a column.
``packs.registry.DB_ONLY_PACK_FIELDS`` and ``DB_ONLY_PART_FIELDS`` name them, and
``rebuild_index`` *carries* them across its own delete rather than re-reading them, because
there is nowhere to re-read them from.

**So capture the three columns before downgrading** — they are three queries::

    SELECT id, is_example, derived_from_pack_id FROM mf_packs WHERE is_example OR derived_from_pack_id IS NOT NULL;
    SELECT pack_id, part_kind, addressed_through_pack_id FROM mf_pack_parts WHERE addressed_through_pack_id IS NOT NULL;

**Task 4.5 asked for the words "export before downgrading — packs are not reconstructible
from disk" and neither half is written here, deliberately.** The second half was true when
the task was written and the gate contract reversed it hours later by giving a pack a file;
F7's own FTASKS records that and says the sentence *"must not be written as specified"*. The
first half is the more dangerous one: **an export would not have saved any of this.** An
export is ``packs.transfer.export_bytes(definition)`` → ``definition.document()``, which is
the closed document — so it carries none of the three columns, and an operator who followed
that instruction would do the work, downgrade, re-import, and find the Example pills and the
derivation history gone with the receipt in hand. A remedy that does not remedy anything is
worse than no instruction, because it is followed.

## Why plain ``add_column`` and not ``batch_alter_table``

All three are additive and nullable (``is_example`` is NOT NULL with a server default, which
is additive against existing rows), so no table rebuild is needed — and a rebuild is actively
dangerous here. Revision ``0014`` learned this the expensive way: SQLAlchemy does not reflect
SQLite CHECK constraints, so a ``batch_alter_table`` left to reflect rebuilds the table
**without** them, silently, while reporting success. ``mf_packs`` carries one CHECK and
``mf_pack_parts`` carries two plus a foreign key. Touching none of them is the safest
available choice, so this migration takes it.

## ``derived_from_pack_id`` carries no database-level foreign key, deliberately

It is a self-reference to ``mf_packs.id``, and SQLite cannot add a constraint to an existing
table — only a rebuild could, which is exactly what the paragraph above rules out. The
alternative orderings are worse than the gap:

* A rebuild to gain the constraint risks dropping the CHECK constraints it was not reflecting.
* Declaring the FK only in the ORM would make PostgreSQL and SQLite disagree about what is
  enforced, which is the divergence ``tests/support/db.py`` already names as its blind spot:
  *"a PostgreSQL-only type or a JSONB operator would pass here and fail there."*

So the reference is enforced where it can be enforced identically on both engines — in the
writer — and the column is **indexed** so the derivation walk is cheap. ADR-004 makes this
table an index over ``pack.json`` anyway: a dangling ``derived_from_pack_id`` is recoverable
by ``reindex``, where a dropped CHECK constraint is not.

**``ON DELETE`` therefore has no database behaviour either**, and that is the honest reading:
deleting a pack whose clone still exists leaves the clone's ``derived_from_pack_id`` pointing
at a row that is gone. The registry renders an unresolvable derivation as **unknown**, not as
absent — null means "not derived", a dangling id means "derived from something no longer
here", and collapsing the two would erase the fact that a derivation happened.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0015_pack_cols"
down_revision: str | None = "0014_merges"
branch_labels: str | None = None
depends_on: str | None = None

_IX_DERIVED = "ix_mf_packs_derived_from"
_IX_ADDRESSED = "ix_mf_pack_parts_addressed_through"


def upgrade() -> None:
    # NOT NULL with a server default, so existing rows get `false` without a backfill. An
    # example pack is a deliberate act of seeding; nothing already in the registry is one,
    # and `false` states that rather than leaving it unknown.
    op.add_column(
        "mf_packs",
        sa.Column(
            "is_example",
            sa.Boolean(),
            nullable=False,
            # `sa.false()` rather than the literal `0`: `server_default=sa.text("0")`
            # renders as `0`, which PostgreSQL REJECTS for a boolean column. That exact bug
            # shipped in this project once and passed every test, because the suite's
            # default engine is SQLite where `0` is a boolean. Caught by compiling the DDL
            # against the PostgreSQL dialect, not by reading it.
            server_default=sa.false(),
        ),
    )

    # Nullable, no FK - see the module docstring. Null means "not derived from anything".
    op.add_column(
        "mf_packs", sa.Column("derived_from_pack_id", sa.String(length=64), nullable=True)
    )

    # Which pack a part was addressed *through*, when a slot cites a part belonging to
    # another pack (FR-007.25). Null is the ordinary case: the part belongs to this pack.
    op.add_column(
        "mf_pack_parts",
        sa.Column("addressed_through_pack_id", sa.String(length=64), nullable=True),
    )

    # Partial on PostgreSQL in both cases: the overwhelming majority of rows are null, and an
    # index over mostly-null rows is mostly wasted. SQLite ignores the dialect keyword and
    # builds a full index, which is correct if less tidy - and it is the suite's default
    # engine, so the difference is stated rather than hidden.
    op.create_index(
        _IX_DERIVED,
        "mf_packs",
        ["derived_from_pack_id"],
        unique=False,
        postgresql_where=sa.text("derived_from_pack_id IS NOT NULL"),
    )
    op.create_index(
        _IX_ADDRESSED,
        "mf_pack_parts",
        ["addressed_through_pack_id"],
        unique=False,
        postgresql_where=sa.text("addressed_through_pack_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_IX_ADDRESSED, table_name="mf_pack_parts")
    op.drop_index(_IX_DERIVED, table_name="mf_packs")
    op.drop_column("mf_pack_parts", "addressed_through_pack_id")
    op.drop_column("mf_packs", "derived_from_pack_id")
    op.drop_column("mf_packs", "is_example")
