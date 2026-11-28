"""WSP protocol-test pass (feature 019, phase P2; FR-019.11, FR-019.12; TDD §4.2; F19-D25, D27).

Revision ID: 0037_wsp_protocol_pass
Revises: 0036_wsp_window_ceiling
Create Date: 2026-11-25

* ``mf_wsp_endpoints.protocol_test_result`` (``passed`` / ``failed``), ``protocol_test_run_id``,
  ``protocol_test_at`` and the three tested values (``protocol_tested_base_url``, ``_model``,
  ``_action_protocol``): the latest protocol test's record, written by the run's terminal
  transition (``wsp.protocol_tests.record``). Whether a pass is CURRENT is computed on read.
* ``mf_wsp_runs.protocol_test_endpoint_id``: the marker of a protocol-test run, set by the
  protocol-test route only (``runs.create_run`` with ``prepared.protocol_test``).

**Deviation from TDD §4.2, as TID §4.2 records it:** no CHECK ties ``protocol_test_endpoint_id`` to
``endpoint_id`` (a column constraint naming another column; SQLite cannot add a table CHECK in
place). ``create_run`` is its only writer, and a unit test pins the rule.

None of these columns joins the endpoint fingerprint or the sealed fields, so a stored result never
fails a queued run and never breaks a key. Additive and nullable (the ``0036`` pattern): the
CHECKed columns by raw DDL, the timestamp through ``add_column`` so each engine renders its own
type.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0037_wsp_protocol_pass"
down_revision: str | None = "0036_wsp_window_ceiling"
branch_labels: str | None = None
depends_on: str | None = None

#: ``(table, column, DDL after the column name)``, in upgrade order. The CHECK names are what
#: ``db.base.NAMING_CONVENTION`` renders for the models' ``CheckConstraint`` names.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "mf_wsp_endpoints",
        "protocol_test_result",
        "VARCHAR(8) CONSTRAINT ck_mf_wsp_endpoints_protocol_test_result_known "
        "CHECK (protocol_test_result IS NULL OR protocol_test_result IN ('passed', 'failed'))",
    ),
    ("mf_wsp_endpoints", "protocol_test_run_id", "VARCHAR(16)"),
    ("mf_wsp_endpoints", "protocol_tested_base_url", "VARCHAR(512)"),
    ("mf_wsp_endpoints", "protocol_tested_model", "VARCHAR(256)"),
    (
        "mf_wsp_endpoints",
        "protocol_tested_action_protocol",
        "VARCHAR(16) CONSTRAINT ck_mf_wsp_endpoints_protocol_tested_action_protocol_known "
        "CHECK (protocol_tested_action_protocol IS NULL OR "
        "protocol_tested_action_protocol IN ('tool_calls', 'structured'))",
    ),
    ("mf_wsp_runs", "protocol_test_endpoint_id", "VARCHAR(40)"),
)
#: The one timestamp, through Alembic (a literal TIMESTAMP WITH TIME ZONE reflects NUMERIC on
#: SQLite: 019 FTASKS 3.2).
TESTED_AT = ("mf_wsp_endpoints", "protocol_test_at")


def upgrade() -> None:
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    op.add_column(TESTED_AT[0], sa.Column(TESTED_AT[1], sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    # Loses every endpoint's protocol-test record (each reads "not tested") and every run's marker.
    op.drop_column(*TESTED_AT)
    for table, column, _ddl in reversed(COLUMNS):
        op.drop_column(table, column)
