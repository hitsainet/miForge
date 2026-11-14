"""``mf_settings`` - runtime overrides for the allow-listed settings  (F12; FR-012.7, .8; D5).

A **record table** on F12's own metadata (see ``models/approval.py``; ADR-004): an override
is the operator's decision and lives nowhere else. Only keys on ``core.config.RUNTIME_KEYS``
are ever written (the route refuses any other); a row for any other key is ignored by
``core.runtime_settings.refresh`` and logged.

``value`` is nullable because an override may set an optional URL to ``null`` (absent), which
is a different fact from "no override" - that is the absence of a row.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from db.base import JsonColumn
from models.approval import F12Base


class SettingOverride(F12Base):
    """One runtime override."""

    __tablename__ = "mf_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any | None] = mapped_column(JsonColumn, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_by: Mapped[str] = mapped_column(String(64), nullable=False)


#: The two things a change can do, and the two places a value can come from.
SETTING_CHANGE_ACTIONS: tuple[str, ...] = ("set", "reset")
SETTING_SOURCES: tuple[str, ...] = ("env", "override")


class SettingChange(F12Base):
    """One change to a runtime setting: append-only  (013 G3; ENH-003 D12).

    Written in the same transaction as the override it records, by ``PUT`` (``set``) and
    ``DELETE`` (``reset``) on ``/settings/{key}`` - the operator's own call or an agent's
    approved request (``actor`` ``agent:<name>`` with its ``approval_id``). **The table refuses
    UPDATE and DELETE** on both engines (migration ``0024_setting_changes``), so the record of
    a change cannot itself be changed. Values are stored as ``GET /settings`` serves them: no
    secret is runtime-writable, and the served form is the one place the redaction rule lives.
    """

    __tablename__ = "mf_setting_changes"
    __table_args__ = (
        CheckConstraint("action IN ('set', 'reset')", name="action_known"),
        CheckConstraint("source IN ('env', 'override')", name="source_known"),
        CheckConstraint("previous_source IN ('env', 'override')", name="previous_source_known"),
        Index("ix_mf_setting_changes_changed_at", "changed_at"),
        Index("ix_mf_setting_changes_key_changed_at", "key", "changed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(8), nullable=False)
    #: The value in force before the change, and where it came from.
    previous: Mapped[Any | None] = mapped_column(JsonColumn, nullable=True)
    previous_source: Mapped[str] = mapped_column(String(8), nullable=False)
    #: The value in force after it, and where it comes from now.
    value: Mapped[Any | None] = mapped_column(JsonColumn, nullable=True)
    source: Mapped[str] = mapped_column(String(8), nullable=False)
    #: ``operator``, or ``agent:<name>`` for an agent's approved request.
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    approval_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
