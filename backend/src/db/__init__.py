"""Database access: the declarative base, the engine and the session scope (ADR-003)."""

from __future__ import annotations

from db.base import TABLE_PREFIX, Base, JsonColumn
from db.session import (
    get_engine,
    get_session_factory,
    make_engine,
    reset_engine,
    session_scope,
)

__all__ = [
    "TABLE_PREFIX",
    "Base",
    "JsonColumn",
    "get_engine",
    "get_session_factory",
    "make_engine",
    "reset_engine",
    "session_scope",
]
