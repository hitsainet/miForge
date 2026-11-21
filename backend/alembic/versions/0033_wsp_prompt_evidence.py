"""WSP prompt evidence: what the gateway sent with each served call (2026-11-21,
``0xcc/reviews/018_uncapped_tokens_2026-11-21.md``).

Revision ID: 0033_wsp_prompt_evidence
Revises: 0032_wsp_thinking
Create Date: 2026-11-21

The gateway sizes a request that names no cap from the upstream's own ``prompt_tokens``
(``wsp_gateway.meter.PromptMeter``). That evidence lived only in the gateway's memory, so a
restarted gateway sized the next call from a fixed 0.75 tokens per byte, and could clamp an
engine's reasoning allowance to 1. Each served call's row now carries what the gateway needs to
recover it from the session the backend serves:

* ``mf_wsp_usage.sent_bytes`` - the bytes of the body the gateway forwarded;
* ``mf_wsp_usage.prompt_messages`` - how many messages it held;
* ``mf_wsp_usage.conversation_sha256`` - the digest of its conversation (``meter.conversation_key``:
  every field but the per-call ones, and the messages). A digest, never text.

All three are null on every earlier row and on any call that was not served. Additive,
column-level CHECKs on the ``ADD COLUMN`` itself (the ``0029_wsp_engine`` pattern).
"""

from __future__ import annotations

from alembic import op

revision: str = "0033_wsp_prompt_evidence"
down_revision: str | None = "0032_wsp_thinking"
branch_labels: str | None = None
depends_on: str | None = None

COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "mf_wsp_usage",
        "sent_bytes",
        "INTEGER CONSTRAINT ck_mf_wsp_usage_sent_bytes_positive CHECK (sent_bytes > 0)",
    ),
    (
        "mf_wsp_usage",
        "prompt_messages",
        "INTEGER CONSTRAINT ck_mf_wsp_usage_prompt_messages_non_negative "
        "CHECK (prompt_messages >= 0)",
    ),
    ("mf_wsp_usage", "conversation_sha256", "VARCHAR(64)"),
)


def upgrade() -> None:
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def downgrade() -> None:
    # Loses the evidence only: a gateway sizes the next call of each live session from its
    # fixed first-call ratio, as before this revision.
    for table, column, _ddl in reversed(COLUMNS):
        op.drop_column(table, column)
