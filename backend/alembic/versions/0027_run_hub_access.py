"""Whose Hugging Face credential a run's Hub fetches may use (feature 017, FR-017.27).

Revision ID: 0027_run_hub_access
Revises: 0026_legacy_name_uses
Create Date: 2026-11-16

* ``mf_runs.hub_access`` - nullable text. ``anonymous`` when an agent asked for the run (an
  agent's define, continue or branch, directly or as an approved request run for it), or when
  the run continues or branches from an anonymous run, or is a chain stage after an anonymous
  one; the worker then passes ``token=False`` to every Hub call, so the pod's ``HF_TOKEN`` or
  token file is never sent on an agent's behalf (suite CHAIN_PLAN A17, A28). Null means today's
  behaviour: the operator's run, fetched with the process's own credential.

Additive; no rebuild, nothing backfilled (every existing run was the operator's or predates the
rule, and null keeps its behaviour).

**CHECK ``hub_access IN ('anonymous')``** (named ``ck_mf_runs_hub_access_known``; null passes),
as the FTDD specified (R2 review fix OR2-017-4, 2026-11-16, reversing OVN-mcp-forge-5). It is a
COLUMN constraint on the ``ADD COLUMN`` itself, so neither engine rebuilds ``mf_runs``: PostgreSQL
adds it in place, and SQLite (3.37+) accepts a CHECK on ``ADD COLUMN`` and tests it against the
existing rows - unlike a table constraint added later, which is the ``0014``/``0016`` rebuild.
Both engines run the same statement. The reader, ``services.hub_access.hub_token_for_run``, also
fails closed on any value but null, so a value that got past the database still sends nothing.

**016 (WSP) plans ``0027_wsp_provider`` on the same parent.** Whichever of 016 and 017 lands on
``main`` second re-parents its revision onto the other (one ``down_revision`` line).
"""

from __future__ import annotations

from alembic import op

revision: str = "0027_run_hub_access"
down_revision: str | None = "0026_legacy_name_uses"
branch_labels: str | None = None
depends_on: str | None = None


#: The column CHECK's name, as ``db.base.NAMING_CONVENTION`` renders ``hub_access_known``.
_CHECK = "ck_mf_runs_hub_access_known"


def upgrade() -> None:
    # Raw DDL, because Alembic's add_column does not render a column-level CHECK. VARCHAR(16)
    # is what ``sa.String(16)`` renders on both engines (test_db_migration compares the types).
    op.execute(
        "ALTER TABLE mf_runs ADD COLUMN hub_access VARCHAR(16) "
        f"CONSTRAINT {_CHECK} CHECK (hub_access IN ('anonymous'))"
    )


def downgrade() -> None:
    # Losing it makes every anonymous run the operator's again: run only with no agent run
    # pending or running.
    # PostgreSQL drops the column's own CHECK with it; SQLite allows DROP COLUMN for a column
    # whose only CHECK is its own.
    op.drop_column("mf_runs", "hub_access")
