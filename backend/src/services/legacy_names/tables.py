"""``mf_legacy_name_uses`` and ``mf_legacy_name_write_errors`` (migration 0026, feature 015).

**Its own metadata, not ``Base.metadata``**: only ``services/legacy_names/store.py`` reads or
writes these rows (foundation 4.2's boundary rule), and
``tests/integration/test_db_migration.py`` compares this metadata with the migrated database.
"""

from __future__ import annotations

import sqlalchemy as sa

from db.base import NAMING_CONVENTION

metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

#: The five kinds of the suite's health block (CF-4); miForge writes only ``env``.
KINDS: tuple[str, ...] = ("env", "host", "header", "tool", "category")
STATES: tuple[str, ...] = ("in_use", "also_set", "conflict", "retired_but_set")

USES = sa.Table(
    "mf_legacy_name_uses",
    metadata,
    sa.Column("kind", sa.String(16), primary_key=True),
    sa.Column("old_name", sa.String(64), primary_key=True),
    sa.Column("state", sa.String(16), primary_key=True),
    sa.Column("new_name", sa.String(64), nullable=True),
    sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False),
    sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("kind IN ('env', 'host', 'header', 'tool', 'category')", name="kind_known"),
    sa.CheckConstraint(
        "state IN ('in_use', 'also_set', 'conflict', 'retired_but_set')", name="state_known"
    ),
    sa.Index("ix_mf_legacy_name_uses_last_seen", "last_seen"),
)

WRITE_ERRORS = sa.Table(
    "mf_legacy_name_write_errors",
    metadata,
    sa.Column("reporter", sa.String(128), primary_key=True),
    sa.Column("error", sa.String(64), nullable=False),
    sa.Column("failed_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
)
