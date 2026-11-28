"""WSP window and output ceiling (feature 019, phase P1; FR-019.2, FR-019.3, FR-019.14; TDD §4.1).

Revision ID: 0036_wsp_window_ceiling
Revises: 0035_wsp_outcome_reason
Create Date: 2026-11-24

* ``mf_wsp_endpoints.output_ceiling_tokens`` - the most output tokens one editor request may ask
  for; null is 32,000 (W3-D6). At least 256.
* ``mf_wsp_endpoints.registration_window_tokens`` and ``registration_window_read_at`` - the
  window the server reported at registration (FR-019.3) and when that read ran.
* ``mf_wsp_runs.window_tokens``, ``window_source``, ``output_ceiling_tokens``,
  ``output_ceiling_lowered`` - the window this run used and where it came from (``stated``,
  ``reported``, ``assumed``), and the ceiling it used after F19-D12 lowered it.

``mf_wsp_runs.served_context_tokens`` keeps its type and changes meaning for NEW rows only
(F19-D28): the window the server reported at this push, for every serving. Nothing is
backfilled: a run from before 019 reads "not recorded".

Additive and nullable, with column-level CHECKs on the ``ADD COLUMN`` itself (the
``0034_wsp_paging`` pattern), so neither engine rebuilds a table. The endpoint ceiling is not a
sealed field and joins the fingerprint only when set, so every existing fingerprint is
byte-stable across this migration.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0036_wsp_window_ceiling"
down_revision: str | None = "0035_wsp_outcome_reason"
branch_labels: str | None = None
depends_on: str | None = None

#: ``(table, column, DDL after the column name)``, in upgrade order. The CHECK names are what
#: ``db.base.NAMING_CONVENTION`` renders for the models' ``CheckConstraint`` names.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "mf_wsp_endpoints",
        "output_ceiling_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_endpoints_output_ceiling_at_least_256 "
        "CHECK (output_ceiling_tokens >= 256)",
    ),
    (
        "mf_wsp_endpoints",
        "registration_window_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_endpoints_registration_window_positive "
        "CHECK (registration_window_tokens > 0)",
    ),
    (
        "mf_wsp_runs",
        "window_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_window_tokens_positive CHECK (window_tokens > 0)",
    ),
    (
        "mf_wsp_runs",
        "window_source",
        "VARCHAR(16) CONSTRAINT ck_mf_wsp_runs_window_source_known "
        "CHECK (window_source IS NULL OR window_source IN ('stated', 'reported', 'assumed'))",
    ),
    (
        "mf_wsp_runs",
        "output_ceiling_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_output_ceiling_positive "
        "CHECK (output_ceiling_tokens > 0)",
    ),
    ("mf_wsp_runs", "output_ceiling_lowered", "BOOLEAN"),
)


#: The one column with no CHECK, added through Alembic so each engine renders its own type for
#: ``DateTime(timezone=True)`` (FTASKS 3.2, confirmed 2026-11-24: a literal
#: ``TIMESTAMP WITH TIME ZONE`` reflects as NUMERIC on SQLite, and ``test_db_migration`` refuses it;
#: SQLite renders DATETIME, PostgreSQL TIMESTAMP WITH TIME ZONE).
READ_AT = ("mf_wsp_endpoints", "registration_window_read_at")


def upgrade() -> None:
    # Raw DDL, because Alembic's add_column does not render a column-level CHECK.
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    op.add_column(READ_AT[0], sa.Column(READ_AT[1], sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    # Loses every endpoint's ceiling (each is 32,000 again) and registration read, and every
    # run's recorded window, source and ceiling.
    op.drop_column(*READ_AT)
    for table, column, _ddl in reversed(COLUMNS):
        op.drop_column(table, column)
