"""Runtime overrides of the allow-listed settings  (F12; FR-012.7, .8; D5).

**One function every reader calls: :func:`effective_settings`.** It is ``get_settings()``
with the in-process override cache laid over it - no I/O, so it is as cheap to call per
request or per job as ``get_settings()`` itself.

**The cache is filled from ``mf_settings`` by :func:`refresh`**, and only at three wiring
points, each of which has a test that fails when it is removed:

* the settings routes, right after a write commits (the API sees its own change at once);
* the API's startup (``api.app.create_asgi_app``), so a restart keeps the overrides;
* a Celery ``task_prerun`` hook (``workers.celery_app``), so every task - a run, a merge,
  the pruning sweep - starts with the overrides as stored, in the worker's own process.

No reader opens a database connection of its own. That is deliberate: a reader inside a
unit test would otherwise try to reach the configured database on every call.

**Values are validated before they are cached**, by building ``Settings`` with the override
applied, so a value the model would refuse (a credential-bearing base URL, an unknown log
level) can never reach a reader through this path either.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from core.config import RUNTIME_KEYS, Settings, get_settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
_overrides: dict[str, Any] = {}
_loaded = False


def effective_settings() -> Settings:
    """The process settings with the runtime overrides applied."""
    base = get_settings()
    with _lock:
        overrides = dict(_overrides)
    if not overrides:
        return base
    return base.model_copy(update=overrides)


def overrides() -> dict[str, Any]:
    """A copy of the overrides currently in force in this process."""
    with _lock:
        return dict(_overrides)


def is_loaded() -> bool:
    """Whether :func:`refresh` has run in this process since the last reset."""
    return _loaded


def validated_value(key: str, value: Any) -> Any:
    """``value`` as ``Settings`` would hold it for ``key``, or ``ValueError``.

    Built through the model's own constructor so every field validator runs - the refusal of
    a base URL carrying credentials included. Returns the JSON-compatible form stored in
    ``mf_settings``.
    """
    if key not in RUNTIME_KEYS:
        raise ValueError(f"{key!r} is not a runtime setting")
    built = Settings.model_validate({**_env_fields(), key: value})
    return built.model_dump(mode="json")[key]


def _env_fields() -> dict[str, Any]:
    """The current process settings as constructor input, so validation sees the same env."""
    return get_settings().model_dump()


def apply(rows: dict[str, Any]) -> None:
    """Replace the cache with ``rows`` (key -> stored value), allow-listed keys only."""
    global _loaded
    accepted: dict[str, Any] = {}
    for key, value in rows.items():
        if key not in RUNTIME_KEYS:
            # Only the route writes this table and it refuses every other key, so a row
            # here was written by hand. Ignored and said so, never applied.
            log.warning(
                "ignoring a settings override for a key that is not runtime-writable",
                extra={"code": "setting_override_ignored", "key": key},
            )
            continue
        try:
            accepted[key] = validated_value(key, value)
        except ValueError:
            log.warning(
                "ignoring a settings override whose value does not validate",
                extra={"code": "setting_override_invalid", "key": key},
            )
    with _lock:
        _overrides.clear()
        _overrides.update(accepted)
        _loaded = True


def refresh(session: Any) -> dict[str, Any]:
    """Read every row of ``mf_settings`` through ``session`` and replace the cache."""
    import sqlalchemy as sa

    from models.setting import SettingOverride

    rows = session.execute(sa.select(SettingOverride.key, SettingOverride.value)).all()
    apply({str(row.key): row.value for row in rows})
    return overrides()


def refresh_from_database() -> bool:
    """:func:`refresh` through a session of its own. ``False`` (logged) if it could not.

    For the two wiring points that have no session in hand: API startup and the Celery
    ``task_prerun`` hook. A failure leaves the previous cache in place and is logged as a
    warning naming the cause - the environment values still apply, which is what an absent
    override means - rather than failing the startup or the task over a settings read.
    """
    try:
        from db.session import session_scope

        with session_scope() as session:
            refresh(session)
    except Exception as exc:
        log.warning(
            "could not read the runtime settings overrides; environment values apply",
            extra={"code": "setting_overrides_unreadable", "detail": type(exc).__name__},
        )
        return False
    return True


def reset() -> None:
    """Empty the cache. For tests, and for a test that simulates a fresh process."""
    global _loaded
    with _lock:
        _overrides.clear()
        _loaded = False
