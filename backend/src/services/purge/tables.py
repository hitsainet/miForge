"""``mf_purges`` - one tombstone per permanent delete  (operator decision 2026-10-20; 0023).

**Append-only.** A purge removes an archived run or part and everything that belonged only to
it; this row is what is left to say it happened: what was removed, by whom, why, and how many
rows, files and bytes went with it. Migration ``0023_purge_tombstones`` installs a trigger on
**both** engines that refuses ``UPDATE`` and ``DELETE`` on the table, so the audit trail
cannot be edited through SQL by this application or any other client of the database.

**Its own metadata, not ``Base.metadata``**: only the purge service writes it and only the
purge routes read it (foundation 4.2's boundary rule). ``tests/integration/test_db_migration.py``
compares it, with every other feature's, against the migrated database.

**No foreign key** to ``mf_runs`` or ``mf_contract_parts``: the row it names is gone by the
time this one is committed - that is the point of it.
"""

from __future__ import annotations

import sqlalchemy as sa

from db.base import NAMING_CONVENTION, JsonColumn

metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

#: The two kinds of item that can be purged.
PURGE_KINDS: tuple[str, ...] = ("run", "part")

PURGES = sa.Table(
    "mf_purges",
    metadata,
    sa.Column("id", sa.String(40), primary_key=True),
    sa.Column("kind", sa.String(8), nullable=False),
    #: The run id or the part id that was purged.
    sa.Column("item_id", sa.String(64), nullable=False),
    #: What the operator saw it called: the run id with its type, a part's reference.
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("purged_by", sa.String(64), nullable=False),
    sa.Column("reason", sa.String(2000), nullable=True),
    #: ``{table: rows deleted}``.
    sa.Column("rows_removed", JsonColumn, nullable=False),
    sa.Column("files_removed", sa.Integer, nullable=False),
    sa.Column("bytes_removed", sa.BigInteger, nullable=False),
    #: Anything else worth keeping: the run's status and type, directories left in place
    #: because miForge does not own them, a file that could not be removed.
    sa.Column("detail", JsonColumn, nullable=False),
    sa.Column("purged_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("kind IN ('run', 'part')", name="kind_known"),
    sa.CheckConstraint("files_removed >= 0 AND bytes_removed >= 0", name="counts_non_negative"),
    sa.Index("ix_mf_purges_purged_at", "purged_at"),
    sa.Index("ix_mf_purges_kind_item_id", "kind", "item_id"),
)
