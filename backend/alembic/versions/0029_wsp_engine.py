"""WSP engine: the endpoint's action protocol and window, and the engine's run counters (feature
018; FR-018.8, FR-018.11a, FR-018.12; FTDD §4; FTID §4).

Revision ID: 0029_wsp_engine
Revises: 0028_wsp_provider
Create Date: 2026-11-19

* ``mf_wsp_endpoints.action_protocol`` - ``tool_calls`` (the default, and every existing row's
  value) or ``structured`` (an OpenAI-kind endpoint served by miServer only; the registration and
  the PATCH refuse it elsewhere, ``wsp_endpoint_protocol_unsupported``).
* ``mf_wsp_endpoints.context_window_tokens`` - the operator's window for the editor model, or null
  (the engine then uses the window miServer reports at push, else its 8,192-token default).
* ``mf_wsp_runs.action_protocol``, ``handoffs``, ``refused_turns`` and ``kernel_restarts`` - the
  ``engine`` block of a ``miwsp_engine`` run's ``result.json``, stored by the intake; null for a
  ``weightspace`` run, a replay's parent, or any run that ended before the worker wrote a result.
  ``kernel_restarts`` is the fourth key the engine writes (018-P1 deviation D-4); the FTDD listed
  three columns, and the fourth costs one nullable integer (018-P2 deviation P2-D1).

None of these is in the sealed key's bound fields (``wsp_gateway/sealing.py``), so a PATCH never
breaks a key; the endpoint fields join the endpoint fingerprint only when away from their defaults
(F18-D8), so every existing fingerprint is byte-stable across this migration.

Additive, column-level CHECKs on the ``ADD COLUMN`` itself (the ``0027_run_hub_access`` pattern):
neither engine rebuilds a table, and SQLite (3.37+) tests each CHECK against the existing rows.
Nothing is backfilled: the NOT NULL column takes its default on every existing endpoint row.
"""

from __future__ import annotations

from alembic import op

revision: str = "0029_wsp_engine"
down_revision: str | None = "0028_wsp_provider"
branch_labels: str | None = None
depends_on: str | None = None

#: ``(table, column, DDL after the column name)``, in upgrade order. The CHECK names are what
#: ``db.base.NAMING_CONVENTION`` renders for the models' ``CheckConstraint`` names.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "mf_wsp_endpoints",
        "action_protocol",
        "VARCHAR(16) DEFAULT 'tool_calls' NOT NULL "
        "CONSTRAINT ck_mf_wsp_endpoints_action_protocol_known "
        "CHECK (action_protocol IN ('tool_calls', 'structured'))",
    ),
    (
        "mf_wsp_endpoints",
        "context_window_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_endpoints_context_window_positive "
        "CHECK (context_window_tokens > 0)",
    ),
    (
        "mf_wsp_runs",
        "action_protocol",
        "VARCHAR(16) CONSTRAINT ck_mf_wsp_runs_action_protocol_known "
        "CHECK (action_protocol IN ('tool_calls', 'structured'))",
    ),
    (
        "mf_wsp_runs",
        "handoffs",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_handoffs_non_negative CHECK (handoffs >= 0)",
    ),
    (
        "mf_wsp_runs",
        "refused_turns",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_refused_turns_non_negative CHECK (refused_turns >= 0)",
    ),
    (
        "mf_wsp_runs",
        "kernel_restarts",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_kernel_restarts_non_negative "
        "CHECK (kernel_restarts >= 0)",
    ),
)


def upgrade() -> None:
    # Raw DDL, because Alembic's add_column does not render a column-level CHECK. VARCHAR(16) is
    # what ``sa.String(16)`` renders on both engines (test_db_migration compares the types).
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def downgrade() -> None:
    # Loses every endpoint's protocol and window (each endpoint is tool_calls with no window
    # again) and every engine run's counters. PostgreSQL drops a column's own CHECK with it;
    # SQLite allows DROP COLUMN for a column whose only CHECK is its own.
    for table, column, _ddl in reversed(COLUMNS):
        op.drop_column(table, column)
