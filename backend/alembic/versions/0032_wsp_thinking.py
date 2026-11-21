"""WSP thinking: the endpoint's thinking setting and reasoning allowance, and the engine's
reasoning-turn counter (2026-11-21; ``0xcc/reviews/018_thinking_2026-11-21.md``).

Revision ID: 0032_wsp_thinking
Revises: 0031_wsp_endpoint_register
Create Date: 2026-11-21

* ``mf_wsp_endpoints.thinking`` - ``off`` (the default, and every existing row's value), ``on``
  (an OpenAI-kind endpoint served by miServer only; the registration and the PATCH refuse it
  elsewhere, ``wsp_endpoint_thinking_unsupported``) or ``default`` (the engine sends nothing).
* ``mf_wsp_endpoints.thinking_tokens`` - the reasoning allowance per turn, or null (the engine's
  default, ``miwsp_engine.context.DEFAULT_THINKING_TOKENS``).
* ``mf_wsp_endpoints.output_cap`` - ``on`` (the default: each request names the engine's per-turn
  output cap) or ``off`` (no cap named; the gateway's window-aware default applies).
* ``mf_wsp_runs.reasoning_turns`` - the ``engine`` block's count of replies that carried
  reasoning, stored by the intake; null for ``weightspace`` and for an engine that writes none.
* ``mf_wsp_runs.served_context_tokens`` - the window miServer reported at push (the push's
  ``spec.editor.served_context_tokens``), served to the gateway with the session so an uncapped
  request is sized to the window miServer actually serves when the endpoint states none.
* ``mf_wsp_runs.repeated_turns`` - the ``engine`` block's count of actions the repeat guard did
  not run (the same tool and arguments on the same weights); null as above.

Neither endpoint column is a sealed (bound) field, so a PATCH never breaks a key; each joins the
endpoint fingerprint only away from its default (``off``, null), so every existing fingerprint is
byte-stable across this migration (the F18-D8 rule of ``0029_wsp_engine``).

Additive, column-level CHECKs on the ``ADD COLUMN`` itself (the ``0029_wsp_engine`` pattern):
neither engine rebuilds a table. Nothing is backfilled: the NOT NULL column takes its default.
"""

from __future__ import annotations

from alembic import op

revision: str = "0032_wsp_thinking"
down_revision: str | None = "0031_wsp_endpoint_register"
branch_labels: str | None = None
depends_on: str | None = None

#: ``(table, column, DDL after the column name)``, in upgrade order. The CHECK names are what
#: ``db.base.NAMING_CONVENTION`` renders for the models' ``CheckConstraint`` names.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "mf_wsp_endpoints",
        "thinking",
        "VARCHAR(8) DEFAULT 'off' NOT NULL "
        "CONSTRAINT ck_mf_wsp_endpoints_thinking_known "
        "CHECK (thinking IN ('off', 'on', 'default'))",
    ),
    (
        "mf_wsp_endpoints",
        "thinking_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_endpoints_thinking_tokens_positive "
        "CHECK (thinking_tokens > 0)",
    ),
    (
        "mf_wsp_endpoints",
        "output_cap",
        "VARCHAR(4) DEFAULT 'on' NOT NULL "
        "CONSTRAINT ck_mf_wsp_endpoints_output_cap_known "
        "CHECK (output_cap IN ('on', 'off'))",
    ),
    (
        "mf_wsp_runs",
        "reasoning_turns",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_reasoning_turns_non_negative "
        "CHECK (reasoning_turns >= 0)",
    ),
    (
        "mf_wsp_runs",
        "served_context_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_served_context_tokens_positive "
        "CHECK (served_context_tokens > 0)",
    ),
    (
        "mf_wsp_runs",
        "repeated_turns",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_repeated_turns_non_negative "
        "CHECK (repeated_turns >= 0)",
    ),
)


def upgrade() -> None:
    # Raw DDL, because Alembic's add_column does not render a column-level CHECK. VARCHAR(8) is
    # what ``sa.String(8)`` renders on both engines (test_db_migration compares the types).
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def downgrade() -> None:
    # Loses every endpoint's thinking setting and allowance (each is off again, at the engine's
    # default allowance) and every engine run's reasoning count.
    for table, column, _ddl in reversed(COLUMNS):
        op.drop_column(table, column)
