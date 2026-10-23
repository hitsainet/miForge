"""``mf_approvals`` - an agent's request that waits for the operator  (F12; FR-012.9 - .16).

A **record table**, not an index (ADR-004): nothing on disk carries an approval, so
``reindex`` never drops or rebuilds it. **F12's own metadata, not ``Base.metadata``**: the base
schema holds only tables more than one feature reads (foundation 4.2's boundary rule,
``test_models_base_schema.py``), and only F12 reads these two. ``services.reindex`` drops
``INDEX_TABLES`` only, so a table on another metadata is never touched by a reindex.

Lifecycle: ``pending`` -> ``executing`` -> ``executed`` | ``failed``; or ``pending`` ->
``rejected`` | ``expired``. Two CHECKs hold in **both** directions, the rule ``mf_merges``
uses: an ``executed`` row carries its result and only an executed row does; a ``failed`` row
carries its error envelope and only a failed row does. A row that says "executed" with no
result says something happened and cannot say what.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Index, MetaData, String, func
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from db.base import NAMING_CONVENTION, JsonColumn

#: F12's own metadata. ``tests/integration/test_db_migration.py`` compares it, with every
#: other feature's, against the migrated database.
metadata = MetaData(naming_convention=NAMING_CONVENTION)


class F12Base(DeclarativeBase):
    """Declarative base for F12's two tables, on F12's own metadata."""

    metadata = metadata


#: ``None`` persisted as SQL ``NULL``, not as the JSON literal ``null``. The two CHECKs below
#: test ``IS NOT NULL``, and with SQLAlchemy's default (``none_as_null=False``) a Python
#: ``None`` written to a JSON column is stored as ``'null'`` - which IS NOT NULL, so a row
#: could satisfy "executed iff result" with no result at all.
NullableJson = JSON(none_as_null=True).with_variant(
    postgresql.JSONB(none_as_null=True), "postgresql"
)

#: The four approval actions (FR-012.16). The CHECK constraint is rendered from this tuple,
#: and ``api.agent_origin.APPROVAL_ACTIONS`` is pinned equal to it by a test.
APPROVAL_ACTION_NAMES: tuple[str, ...] = ("gpu_start", "hotupdate_attach", "delete", "config_write")

#: Six states. Rendered into the CHECK constraint so the two cannot diverge.
APPROVAL_STATUSES: tuple[str, ...] = (
    "pending",
    "executing",
    "executed",
    "rejected",
    "expired",
    "failed",
)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Approval(F12Base):
    """One stored request and its outcome."""

    __tablename__ = "mf_approvals"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    #: ``module.qualname`` of the gated route function; ``api.agent_origin.EXECUTORS`` key.
    executor_key: Mapped[str] = mapped_column(String(200), nullable=False)
    #: What the operator sees, built at request time from the validated payload.
    summary: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    #: The route's value parameters, canonical JSON. Replayed exactly on approve.
    payload: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    #: SHA-256 of the canonical payload. A mismatch on approve fails the row as tampered.
    request_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    requested_by: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    result: Mapped[Any | None] = mapped_column(NullableJson, nullable=True)
    error: Mapped[dict[str, Any] | None] = mapped_column(NullableJson, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        CheckConstraint(_in("status", APPROVAL_STATUSES), name="status_known"),
        CheckConstraint(_in("action", APPROVAL_ACTION_NAMES), name="action_known"),
        CheckConstraint("(status = 'executed') = (result IS NOT NULL)", name="executed_iff_result"),
        CheckConstraint("(status = 'failed') = (error IS NOT NULL)", name="failed_iff_error"),
        Index("ix_mf_approvals_status_created_at", "status", "created_at"),
    )
