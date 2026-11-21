"""The environment names miLLM's rename moves, and which old ones are in use (feature 015).

miLLM became miServer (suite ENH-009, BRD-MIFORGE-REBRAND-001 BR-F02). Three settings read a
miServer-branded variable. Until suite phase P7 the old ``MILLM_*`` name was read as a fallback;
**since P7 it is retired**: set alone it configures nothing and is reported ``retired_but_set``
(FR-015.23). The new name is read by presence: set, even blank, it is the value (a blank becomes
unset by ``Settings._empty_string_is_unset``).

**The field names do not change** (suite R1: ``millm_base_url`` is a stored settings key and a
wire field, register entry DR-39). Only the environment variable that fills it moves.

**One authority, written as literals.** ``ENV_NAMES`` is the only place the pairs are written in
this repository; census holds an independent copy (``ENV_RENAMES["miForge"]``) and
``test_env_names_match_export.py`` compares the two. This module therefore reads no file: a
table loaded from the census export would compare the export with itself (CF-3, CF-25 m13).

**What is recorded.** :class:`LegacyEnvSource` records each old name it saw, with its state, in
``Settings._legacy_names``; ``services/legacy_names`` stores that per process and
``GET /api/health`` serves it: before P7 the evidence the suite read to retire the fallback, since
P7 the report of a retired name still set. A *value* is never recorded, logged or raised (ADR-020).

This module imports only the standard library and pydantic-settings: ``core/config.py`` imports
it, so importing ``core.config`` here would be a cycle.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvName:
    """One settings field, the miServer name that fills it, and its old name (``None`` after P7)."""

    field: str
    env: str
    legacy: str | None


ENV_NAMES: tuple[EnvName, ...] = (
    EnvName("millm_base_url", "MISERVER_BASE_URL", None),
    EnvName("millm_trainer_token", "MISERVER_TRAINER_TOKEN", None),
    EnvName("millm_audit_label", "MISERVER_AUDIT_LABEL", None),
)

#: Old name -> new name, while the old name still configures its field (suite P0 to P7). Empty
#: since P7: derived from ``ENV_NAMES``, whose ``legacy`` is ``None`` for every entry.
LEGACY_ENV_ALIASES: Mapping[str, str] = MappingProxyType(
    {e.legacy: e.env for e in ENV_NAMES if e.legacy is not None}
)

#: Old name -> new name once retired (P7 on): set alone it configures NOTHING and is reported
#: ``retired_but_set``. Filled at P7 (2026-11), then kept (the evidence outlives the fallback).
RETIRED_ENV_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "MILLM_BASE_URL": "MISERVER_BASE_URL",
        "MILLM_TRAINER_TOKEN": "MISERVER_TRAINER_TOKEN",
        "MILLM_AUDIT_LABEL": "MISERVER_AUDIT_LABEL",
    }
)

State = Literal["in_use", "also_set", "conflict", "retired_but_set"]
Source = Literal["environment", "env_file"]


@dataclass(frozen=True)
class LegacyUse:
    """One old name seen while building settings. ``source`` is logged, never served or stored."""

    kind: Literal["env"]
    old: str
    new: str
    state: State
    source: Source | None


@dataclass(frozen=True)
class LegacyResolution:
    """Every old name the process's settings were built with, in table order."""

    uses: tuple[LegacyUse, ...] = ()


#: The uses recorded by the source during one ``Settings.__init__``; ``None`` outside one.
_RESOLUTION: ContextVar[list[LegacyUse] | None] = ContextVar("_legacy_resolution", default=None)


def begin_record() -> Any:
    """Start recording for one ``Settings`` construction; the token goes to :func:`end_record`."""
    return _RESOLUTION.set([])


def end_record(token: Any) -> list[LegacyUse]:
    """Stop recording and return what the source recorded."""
    uses = _RESOLUTION.get() or []
    _RESOLUTION.reset(token)
    return list(uses)


def field_of(use: LegacyUse) -> str | None:
    """The settings field a use belongs to (by its new name)."""
    for entry in ENV_NAMES:
        if entry.env == use.new:
            return entry.field
    return None


def _old_names_for(entry: EnvName) -> tuple[tuple[str, bool], ...]:
    """``(old name, retired?)`` for one entry: the live alias and any retired name."""
    names: list[tuple[str, bool]] = []
    if entry.legacy is not None and entry.legacy in LEGACY_ENV_ALIASES:
        names.append((entry.legacy, False))
    for old, new in RETIRED_ENV_NAMES.items():
        if new == entry.env:
            names.append((old, True))
    return tuple(names)


class LegacyEnvSource(PydanticBaseSettingsSource):
    """Fill the three miServer fields by presence, new name first; record old names used.

    Second in ``Settings.settings_customise_sources``: explicit init values beat it, and it beats
    pydantic's own environment and ``.env`` sources, which would otherwise still read
    ``MILLM_BASE_URL`` for the field literally named ``millm_base_url`` - including after P7,
    which is why a retired name is answered with an explicit ``None``.
    """

    def __init__(self, settings_cls: type[BaseSettings], dotenv_settings: Any) -> None:
        super().__init__(settings_cls)
        self._dotenv = dotenv_settings

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        environment = {k.lower(): v for k, v in os.environ.items()}
        dotenv_vars = getattr(self._dotenv, "env_vars", None) or {}
        dotenv = {k.lower(): v for k, v in dotenv_vars.items()}

        def lookup(name: str) -> tuple[bool, str | None, Source | None]:
            key = name.lower()
            if key in environment:
                return True, environment[key], "environment"
            if key in dotenv:
                return True, dotenv[key], "env_file"
            return False, None, None

        out: dict[str, Any] = {}
        uses: list[LegacyUse] = []
        for entry in ENV_NAMES:
            new_present, new_value, _ = lookup(entry.env)
            if new_present:
                out[entry.field] = new_value
            for old, retired in _old_names_for(entry):
                old_present, old_value, source = lookup(old)
                if not old_present:
                    continue
                if retired:
                    uses.append(LegacyUse("env", old, entry.env, "retired_but_set", source))
                    if not new_present:
                        out[entry.field] = None
                elif new_present:
                    uses.append(LegacyUse("env", old, entry.env, "also_set", source))
                    if old_value != new_value:
                        uses.append(LegacyUse("env", old, entry.env, "conflict", source))
                else:
                    out[entry.field] = old_value
                    uses.append(LegacyUse("env", old, entry.env, "in_use", source))
        recording = _RESOLUTION.get()
        if recording is not None:
            recording.extend(uses)
        return out


def env_label(field: str) -> str:
    """The variable an operator should set for ``field``, naming the old one while it works.

    ``"MISERVER_BASE_URL (formerly MILLM_BASE_URL)"`` while an alias works (P0 to P7), then
    ``"MISERVER_BASE_URL"``: a retired name is never offered as a fix.
    The only producer of "set this variable" text (FR-015.7).
    """
    for entry in ENV_NAMES:
        if entry.field == field:
            if entry.legacy is not None and entry.legacy in LEGACY_ENV_ALIASES:
                return f"{entry.env} (formerly {entry.legacy})"
            return entry.env
    raise KeyError(f"{field} is not a miServer-named setting")


def old_names() -> frozenset[str]:
    """Old names that still configure a field (the brand guard's allowance, PD-8)."""
    return frozenset(LEGACY_ENV_ALIASES)


def retired_names() -> frozenset[str]:
    """Old names that are retired (allowed only in this module, PD-8)."""
    return frozenset(RETIRED_ENV_NAMES)


_LOGGED: set[str] = set()


#: The WARNING for each state that an operator must act on: (message, code).
_WARNINGS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "in_use": ("environment variable %s is deprecated; set %s", "legacy_env_name_in_use"),
        # P7 on: the old name configures nothing, so a deployment that still sets it alone has
        # silently lost the value. Health lists it; the log says so where the operator looks.
        "retired_but_set": (
            "environment variable %s is retired and ignored; set %s",
            "legacy_env_name_retired",
        ),
    }
)


def log_in_use(resolution: LegacyResolution) -> int:
    """One WARNING per old name in use or retired-but-set, once per process.

    Returns how many were logged now.
    """
    logged = 0
    for use in resolution.uses:
        warning = _WARNINGS.get(use.state)
        if warning is None or use.old in _LOGGED:
            continue
        _LOGGED.add(use.old)
        message, code = warning
        log.warning(
            message,
            use.old,
            use.new,
            extra={"code": code, "old": use.old, "new": use.new, "source": use.source},
        )
        logged += 1
    return logged


def reset_logged() -> None:
    """Forget which names were logged. For tests."""
    _LOGGED.clear()
