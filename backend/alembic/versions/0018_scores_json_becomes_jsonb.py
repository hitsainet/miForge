"""``mf_scores``' three JSON columns become JSONB on PostgreSQL, as the models always said.

Revision ID: 0018_scores_jsonb
Revises: 0017_hotupdate
Create Date: 2026-10-14

## A type divergence SQLite cannot see

``db.base.JsonColumn`` is ``JSON().with_variant(postgresql.JSONB(), "postgresql")``, and
every migration in this tree declares the same variant locally as ``JSON_COLUMN`` -
``0001``, ``0010``, ``0014``, ``0016`` all do. **``0013`` alone wrote plain ``sa.JSON()``**
for ``mf_scores.set_ref``, ``mf_scores.decode`` and ``mf_scores.split_markers``.

On SQLite both spellings render to the same storage, so the unit suite could not see it and
did not. On PostgreSQL the migration created ``json`` while
``services/scoring/tables.py`` declares ``jsonb``, and
``test_db_migration.py::test_the_migration_produces_exactly_the_models_schema`` - the test
that exists **because** the migration and the models are otherwise never compared - failed
in the CI PostgreSQL job with three ``modify_type`` diffs. That is exactly the blind spot
``tests/support/db.py`` names in terms: *"a PostgreSQL-only type or a JSONB operator would
pass here and fail there."*

## Why a new revision rather than correcting ``0013``

The failing test's own message says *"generate a migration for the difference rather than
editing one of the two to match"*, and the reason is not style. Editing ``0013`` would fix a
fresh clone and leave **every database that has already run it** on ``json`` for good, with
no revision that ever moves it - and the dev compose stack is already at head. A migration
fixes both populations; a rewrite fixes only the one nobody has.

## PostgreSQL only, and stated rather than implied

On SQLite there is nothing to do: ``JSON`` and the variant's base are the same type, so a
``batch_alter_table`` rebuild here would be a full table copy to change nothing, and the
rebuild is the operation that silently dropped an index in ``0014``. The guard is on the
dialect, not on a try/except, so a dialect this project does not support fails loudly
instead of being quietly skipped.

``USING <col>::jsonb`` is required: PostgreSQL will not cast ``json`` to ``jsonb``
implicitly. It is lossy in one direction only - ``jsonb`` does not preserve key order or
duplicate keys - and neither matters here, because every value in these columns is written
through ``core.canonical``, which sorts keys and cannot emit a duplicate. **Key *presence*
is preserved**, which is the distinction ``JsonColumn``'s docstring exists to protect (an
absent key and a null one are different facts).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018_scores_jsonb"
down_revision: str | None = "0017_hotupdate"
branch_labels: str | None = None
depends_on: str | None = None

#: ``(column, nullable)`` for the three columns ``0013`` typed without the variant. **All
#: three are on ``mf_scores``** - ``split_markers`` reads like an item-level field and is
#: not: it records how one score's completions were split, beside ``decode`` and
#: ``aggregation``. The first draft of this revision put it on ``mf_score_items`` by
#: inference and PostgreSQL refused with ``UndefinedColumn``, which is the right failure.
_TABLE = "mf_scores"
_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("set_ref", True),
    ("decode", False),
    ("split_markers", True),
)


def _retype(target: sa.types.TypeEngine, using_cast: str) -> None:
    for column, nullable in _COLUMNS:
        op.alter_column(
            _TABLE,
            column,
            type_=target,
            existing_nullable=nullable,
            postgresql_using=f"{column}::{using_cast}",
        )


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        # Nothing to do, and said out loud: see the module docstring. A no-op here is the
        # correct outcome, not a skipped step.
        return
    _retype(postgresql.JSONB(astext_type=sa.Text()), "jsonb")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    # Reversible, and the round trip is safe for the same reason the upgrade is: canonical
    # JSON has no key order or duplicate keys to lose.
    _retype(postgresql.JSON(astext_type=sa.Text()), "json")
