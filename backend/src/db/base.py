"""The SQLAlchemy 2 declarative base and the ``mf_`` prefix  (task 4.1; ADR-003, ADR-004).

**What this layer is, and what it is not.** ADR-004: *the on-disk manifest is
authoritative and PostgreSQL holds a rebuildable index.* So the tables below divide into
two kinds, and the distinction is load-bearing rather than tidy:

* **Index tables** - ``mf_checkpoints``, ``mf_adapters``. Every column is derived from a
  manifest on disk. Dropping them loses nothing but a ``reindex`` (ADR-004), and the
  equivalence test proves that (task 4.4).
* **Record tables** - ``mf_runs``, ``mf_steps``, ``mf_packs``, ``mf_pack_parts``,
  ``mf_leases``. These hold state no artifact carries: a run's status, a lease's expiry.
  They are *not* reindexable and must not be asserted over by task 4.4's equivalence
  test. The pack registry is the sharp case and is called out on :class:`~models.pack.Pack`.

**The naming convention is not cosmetic.** Without it SQLite emits unnamed CHECK and
UNIQUE constraints, and Alembic then cannot drop or alter them by name - so a migration
that is reversible on PostgreSQL is irreversible on the database the suite's unit tests
run against. Naming them here makes both engines behave the same way.
"""

from __future__ import annotations

from sqlalchemy import JSON, MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase

#: ADR-003: table prefix ``mf_``, matching miDataworks' ``dw_``. Asserted by a ratchet
#: test over the metadata rather than left to reviewer attention, because the cost of a
#: missed prefix is a table that collides in a shared database and is found in production.
TABLE_PREFIX = "mf_"

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

#: JSON that is ``JSONB`` on PostgreSQL and plain ``JSON`` elsewhere.
#:
#: **Why a JSON column at all, when this project prefers explicit fields:** a
#: ``ScoreRecord`` has an optional, NOT-required ``provider`` field, so a conforming score
#: can omit the key entirely *or* carry it as null - and those are different facts for the
#: same reason an absent ``domain`` differs from a null one (task 4.2a). Shredding a score
#: into columns would collapse that distinction on the way in; a JSON round trip preserves
#: key presence for free.
JsonColumn = JSON().with_variant(postgresql.JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Declarative base for every miForge table."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
