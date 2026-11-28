"""WSP paging: the endpoint's paging setting and the engine's paging counters (2026-11-21;
``0xcc/reviews/018_paging_2026-11-21.md``).

Revision ID: 0034_wsp_paging
Revises: 0033_wsp_prompt_evidence
Create Date: 2026-11-21

* ``mf_wsp_endpoints.paging`` - ``on`` (the default, and every existing row's value: an
  ``miwsp_engine`` run on the endpoint pages bulky tool results out of the history and keeps its
  experiment ledger) or ``off``. ``weightspace`` ignores it.
* ``mf_wsp_runs.pages_out``, ``recalls``, ``ledger_rows``, ``peak_prompt_tokens`` - the ``engine``
  block's paging counters, stored by the intake; null for ``weightspace`` and for an engine that
  writes none.

The endpoint column is not a sealed (bound) field, so a PATCH never breaks a key; it joins the
endpoint fingerprint only away from its default (``off``), so every existing fingerprint is
byte-stable across this migration (the F18-D8 rule of ``0029_wsp_engine``).

Additive, column-level CHECKs on the ``ADD COLUMN`` itself (the ``0029_wsp_engine`` pattern):
neither engine rebuilds a table. Nothing is backfilled: the NOT NULL column takes its default.
"""

from __future__ import annotations

from alembic import op

revision: str = "0034_wsp_paging"
down_revision: str | None = "0033_wsp_prompt_evidence"
branch_labels: str | None = None
depends_on: str | None = None

#: ``(table, column, DDL after the column name)``, in upgrade order. The CHECK names are what
#: ``db.base.NAMING_CONVENTION`` renders for the models' ``CheckConstraint`` names.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "mf_wsp_endpoints",
        "paging",
        "VARCHAR(4) DEFAULT 'on' NOT NULL "
        "CONSTRAINT ck_mf_wsp_endpoints_paging_known "
        "CHECK (paging IN ('on', 'off'))",
    ),
    *(
        (
            "mf_wsp_runs",
            name,
            f"INTEGER CONSTRAINT ck_mf_wsp_runs_{name}_non_negative CHECK ({name} >= 0)",
        )
        for name in ("pages_out", "recalls", "ledger_rows", "peak_prompt_tokens")
    ),
)


def upgrade() -> None:
    # Raw DDL, because Alembic's add_column does not render a column-level CHECK. VARCHAR(4) is
    # what ``sa.String(4)`` renders on both engines (test_db_migration compares the types).
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def downgrade() -> None:
    # Loses every endpoint's paging setting (each is on again) and every engine run's paging
    # counters.
    for table, column, _ddl in reversed(COLUMNS):
        op.drop_column(table, column)
