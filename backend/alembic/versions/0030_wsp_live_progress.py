"""WSP live progress: the worker's last heartbeat on the run, and what became of each editor call
(live progress, 2026-11-20; the web UI's WSP run pages).

Revision ID: 0030_wsp_live_progress
Revises: 0029_wsp_engine
Create Date: 2026-11-20

* ``mf_wsp_runs.worker_state`` and ``worker_elapsed_s`` - the worker's state and its elapsed whole
  seconds as its ``GET /status`` last reported them, written by the driver on every heartbeat it
  already reads (``last_status_at`` is that heartbeat's time). The worker is untrusted (P8/P9): a
  state outside ``WORKER_STATES`` is stored ``unknown`` and an elapsed time that is not a bounded
  number is stored null, before the row is written (``wsp.dispatch.worker_heartbeat``).
* ``mf_wsp_usage.outcome``, ``http_status``, ``latency_ms``, ``finish_reason``, ``refusal_code`` -
  what became of one editor call, written by the gateway's settle and its new refusal record.
  **Counts and codes only:** no column holds a prompt, an answer or an upstream message, and
  ``finish_reason`` / ``refusal_code`` are short identifiers checked on the way in
  (``wsp.gateway_ledger.call_columns``).

``mf_wsp_usage`` stays append-only: ``ADD COLUMN`` keeps its UPDATE/DELETE triggers on both
engines (no table is rebuilt). Additive, column-level CHECKs on the ``ADD COLUMN`` itself (the
``0027_run_hub_access`` / ``0029_wsp_engine`` pattern); nothing is backfilled, so a row from before
this migration reads null in every new column.
"""

from __future__ import annotations

from alembic import op

revision: str = "0030_wsp_live_progress"
down_revision: str | None = "0029_wsp_engine"
branch_labels: str | None = None
depends_on: str | None = None

#: ``(table, column, DDL after the column name)``, in upgrade order. The CHECK names are what
#: ``db.base.NAMING_CONVENTION`` renders for the models' ``CheckConstraint`` names.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "mf_wsp_runs",
        "worker_state",
        "VARCHAR(16) CONSTRAINT ck_mf_wsp_runs_worker_state_known "
        "CHECK (worker_state IS NULL OR worker_state IN "
        "('idle', 'receiving', 'running', 'finished', 'exiting', 'unknown'))",
    ),
    (
        "mf_wsp_runs",
        "worker_elapsed_s",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_worker_elapsed_non_negative "
        "CHECK (worker_elapsed_s >= 0)",
    ),
    (
        "mf_wsp_usage",
        "outcome",
        "VARCHAR(16) CONSTRAINT ck_mf_wsp_usage_outcome_known "
        "CHECK (outcome IS NULL OR outcome IN ('served', 'upstream_error', 'upstream_failed', "
        "'session_ended', 'refused', 'closed_at_worst'))",
    ),
    (
        "mf_wsp_usage",
        "http_status",
        "INTEGER CONSTRAINT ck_mf_wsp_usage_http_status_range "
        "CHECK (http_status BETWEEN 100 AND 599)",
    ),
    (
        "mf_wsp_usage",
        "latency_ms",
        "INTEGER CONSTRAINT ck_mf_wsp_usage_latency_non_negative CHECK (latency_ms >= 0)",
    ),
    ("mf_wsp_usage", "finish_reason", "VARCHAR(32)"),
    ("mf_wsp_usage", "refusal_code", "VARCHAR(64)"),
)


def upgrade() -> None:
    # Raw DDL, because Alembic's add_column does not render a column-level CHECK.
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def downgrade() -> None:
    # Loses every run's last heartbeat state and every call's outcome, status, latency, finish
    # reason and refusal code (the spend itself stays). PostgreSQL drops a column's own CHECK with
    # it; SQLite allows DROP COLUMN for a column whose only CHECK is its own. On SQLite, dropping a
    # column of mf_wsp_usage keeps its triggers (they name no column).
    for table, column, _ddl in reversed(COLUMNS):
        op.drop_column(table, column)
