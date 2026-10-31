"""The settings surface  (F12; FR-012.7, .8; MR-5; D5).

Until this module miForge's configuration was environment-only (ADR-020) and **no route read
or wrote it**, so D2's "configuration control" had nothing to call.

* ``GET /settings`` - every field of ``core.config.Settings``: its effective value, where it
  came from (``env`` | ``override``) and whether it is runtime-writable. The trainer token is
  ``"<redacted>"`` when set, and the password in the database and Redis URLs is replaced.
* ``PUT /settings/{key}`` / ``DELETE /settings/{key}`` - write or clear an override for a key
  on ``RUNTIME_KEYS``; any other key is ``422 setting_not_runtime``. Both are gated
  ``config_write`` for agents. After a write commits the API's own cache is refreshed from the
  table, and the workers pick the change up before their next task (``task_prerun``).
* ``GET /settings/history`` - the append-only change history (013 G3; ENH-003 D12), newest
  first: one row per write, with the value before and after, who made it (``operator`` or
  ``agent:<name>`` with the approval id) and the optional reason.

Each listed setting also carries what it does and when a change takes effect
(``core.settings_catalog``), its environment value, and its last change (013 G4).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.agent_origin import executing_actor, requires_approval_when_agent
from api.deps import session_dependency
from core import runtime_settings, settings_catalog
from core.config import RUNTIME_KEYS, SECRET_FIELDS, Settings, get_settings
from core.errors import refuse
from services import approval_gates

router = APIRouter(prefix="/settings", tags=["settings"])

REDACTED = "<redacted>"
#: URL fields whose password component is never served.
CREDENTIAL_URLS: frozenset[str] = frozenset({"database_url", "redis_url"})
#: The longest reason a change may carry (the history column's width).
REASON_MAX = 2000
#: How many history rows ``GET /settings/history`` serves by default, and at most.
HISTORY_DEFAULT = 50
HISTORY_MAX = 500


class SettingWrite(BaseModel):
    """``PUT /settings/{key}``'s body: the new value, typed by the field it sets."""

    model_config = ConfigDict(extra="forbid")

    value: Any = None
    #: Why, in the operator's words. Optional; recorded in the history (ENH-003 D12).
    reason: str | None = Field(default=None, max_length=REASON_MAX)


def redact_url(url: str) -> str:
    """``scheme://user:secret@host/db`` -> ``scheme://user:<redacted>@host/db``."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return REDACTED
    if parts.password is None:
        return url
    netloc = parts.netloc.rsplit("@", 1)
    user = parts.username or ""
    return urlunsplit(parts._replace(netloc=f"{user}:{REDACTED}@{netloc[-1]}"))


def served_value(key: str, value: Any) -> Any:
    """A setting's value as it may leave this process."""
    if key in SECRET_FIELDS:
        return None if value is None else REDACTED
    if key in CREDENTIAL_URLS and isinstance(value, str):
        return redact_url(value)
    return value


def _iso(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    if moment.tzinfo is None:  # SQLite returns naive datetimes; every stored time is UTC
        moment = moment.replace(tzinfo=UTC)
    return moment.isoformat()


def change_payload(row: Any) -> dict[str, Any]:
    """One history row as the API serves it."""
    return {
        "id": row.id,
        "key": row.key,
        "action": row.action,
        "previous": row.previous,
        "previous_source": row.previous_source,
        "value": row.value,
        "source": row.source,
        "actor": row.actor,
        "approval_id": row.approval_id,
        "reason": row.reason,
        "changed_at": _iso(row.changed_at),
    }


def _payload(session: Session) -> dict[str, Any]:
    from models.setting import SettingChange, SettingOverride

    effective = runtime_settings.effective_settings().model_dump(mode="json")
    environment = get_settings().model_dump(mode="json")
    active = runtime_settings.overrides()
    rows = {row.key: row for row in session.scalars(sa.select(SettingOverride))}
    newest = sa.select(sa.func.max(SettingChange.id)).group_by(SettingChange.key)
    last = {
        row.key: row
        for row in session.scalars(sa.select(SettingChange).where(SettingChange.id.in_(newest)))
    }
    settings: dict[str, Any] = {}
    for key in Settings.model_fields:
        info = settings_catalog.info_for(key)
        override = rows.get(key)
        settings[key] = {
            "value": served_value(key, effective.get(key)),
            "source": "override" if key in active else "env",
            "runtime_writable": key in RUNTIME_KEYS,
            "label": info.label,
            "description": info.description,
            "effect": settings_catalog.effect_payload(key),
            "group": settings_catalog.GROUP_OF[key],
            "env_value": served_value(key, environment.get(key)),
            "updated_at": None if override is None else _iso(override.updated_at),
            "updated_by": None if override is None else override.updated_by,
            "last_change": None if key not in last else change_payload(last[key]),
        }
    return {
        "settings": settings,
        "count": len(settings),
        "runtime_keys": sorted(RUNTIME_KEYS),
        "overrides": sorted(active),
        "groups": settings_catalog.groups_payload(),
    }


@router.get("", summary="Every setting: effective value, source, and whether it is writable")
def read_settings(session: Annotated[Session, Depends(session_dependency)]) -> dict[str, Any]:
    """Secrets are never served (FR-012.7)."""
    return _payload(session)


@router.get("/history", summary="The settings change history, newest first")
def read_history(
    session: Annotated[Session, Depends(session_dependency)],
    key: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=HISTORY_MAX)] = HISTORY_DEFAULT,
) -> dict[str, Any]:
    """Every recorded change, or one setting's (013 G3). Append-only: nothing here is edited."""
    from models.setting import SettingChange

    query = sa.select(SettingChange).order_by(SettingChange.id.desc()).limit(limit)
    if key is not None:
        if key not in Settings.model_fields:
            raise refuse("setting_unknown", f"{key!r} is not a setting.", key=key)
        query = query.where(SettingChange.key == key)
    changes = [change_payload(row) for row in session.scalars(query)]
    return {"changes": changes, "count": len(changes)}


def _writer() -> tuple[str, str | None]:
    """Who is making this change: the approved request's agent and approval, or the operator.

    An agent's write never runs directly - the gate stores it - so the only agent that can
    reach this line is one whose request the operator approved (013 G2).
    """
    actor = executing_actor()
    if actor is None or actor.origin != "agent" or actor.agent is None:
        return "operator", None
    return actor.agent, actor.approval_id


def _record(
    session: Session,
    *,
    key: str,
    action: str,
    previous: Any,
    previous_source: str,
    value: Any,
    source: str,
    reason: str | None,
) -> tuple[str, str | None]:
    """Add the history row in the write's own transaction; return the writer."""
    from models.setting import SettingChange

    actor, approval_id = _writer()
    session.add(
        SettingChange(
            key=key,
            action=action,
            previous=served_value(key, previous),
            previous_source=previous_source,
            value=served_value(key, value),
            source=source,
            actor=actor,
            approval_id=approval_id,
            reason=reason,
            changed_at=datetime.now(UTC),
        )
    )
    return actor, approval_id


def _after_write(session: Session, key: str) -> None:
    session.commit()
    runtime_settings.refresh(session)
    if key == "log_level":
        from core.logging import configure_logging

        configure_logging(runtime_settings.effective_settings().log_level)


@router.put("/{key}", summary="Set a runtime override for an allow-listed setting")
@requires_approval_when_agent(
    "config_write",
    validate=approval_gates.validate_setting_write,
    summary=approval_gates.summarise_setting_write,
)
def write_setting(
    key: str,
    body: SettingWrite,
    session: Annotated[Session, Depends(session_dependency)],
) -> dict[str, Any]:
    """Validated by the field's own type and validators; stored; applied (FR-012.8)."""
    from models.setting import SettingOverride

    value = approval_gates.validate_setting_write({"key": key, "body": body}, session)["value"]
    previous = runtime_settings.effective_settings().model_dump(mode="json")[key]
    previous_source = "override" if key in runtime_settings.overrides() else "env"
    writer, _ = _record(
        session,
        key=key,
        action="set",
        previous=previous,
        previous_source=previous_source,
        value=value,
        source="override",
        reason=body.reason,
    )
    row = session.get(SettingOverride, key)
    if row is None:
        session.add(SettingOverride(key=key, value=value, updated_by=writer))
    else:
        row.value = value
        row.updated_by = writer
        row.updated_at = datetime.now(UTC)
    _after_write(session, key)
    return {
        "key": key,
        "value": served_value(key, value),
        "previous": served_value(key, previous),
        "source": "override",
    }


@router.delete("/{key}", summary="Clear a runtime override; the environment value applies again")
@requires_approval_when_agent(
    "config_write",
    validate=approval_gates.validate_setting_reset,
    summary=approval_gates.summarise_setting_reset,
)
def reset_setting(
    key: str,
    session: Annotated[Session, Depends(session_dependency)],
    reason: Annotated[str | None, Query(max_length=REASON_MAX)] = None,
) -> dict[str, Any]:
    """Remove the override. ``404 setting_not_overridden`` when there is none to remove."""
    from models.setting import SettingOverride

    approval_gates.validate_setting_reset({"key": key}, session)
    row = session.get(SettingOverride, key)
    if row is None:  # pragma: no cover - validate_setting_reset refused already
        raise refuse("setting_not_overridden", f"{key!r} has no override to clear.", key=key)
    env_value = get_settings().model_dump(mode="json")[key]
    _record(
        session,
        key=key,
        action="reset",
        previous=runtime_settings.effective_settings().model_dump(mode="json")[key],
        previous_source="override",
        value=env_value,
        source="env",
        reason=reason,
    )
    session.delete(row)
    _after_write(session, key)
    return {"key": key, "value": served_value(key, env_value), "source": "env"}
