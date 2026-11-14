"""Every "set this variable" message names the miServer variable (FR-015.7, FTID §3.5).

The text comes from ``core.legacy_names.env_label``, the only producer: while an old name still
worked (P0 to P7) it read "MISERVER_BASE_URL (formerly MILLM_BASE_URL)"; since P7 it is the new
name alone, because a retired name configures nothing and must never be offered as the fix.
The builders are the REAL ones (no copy of the message here), and each case asserts the new
name is present and that no old name appears outside an ``env_label`` phrase.

A deprecated name in use, or a retired one still set, is logged once per process, at WARNING,
with its code.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import pytest

from core import legacy_names
from core.errors import Refusal


def _identity_missing_token() -> str:
    from training.hotupdate.identity import headers_for

    with pytest.raises(Refusal) as refused:
        headers_for("heartbeat", token=None, label="x", attachment_id="a")
    return str(refused.value.message)


def _identity_missing_label() -> str:
    from training.hotupdate.identity import headers_for

    with pytest.raises(Refusal) as refused:
        headers_for("heartbeat", token="t", label=None, attachment_id="a")
    return str(refused.value.message)


def _identity_rejected() -> str:
    from training.hotupdate.identity import identity_rejected

    exc = identity_rejected("heartbeat", 401, label=None)
    return str(getattr(exc, "message", exc))


def _not_configured() -> str:
    from training.hotupdate.settings import CAUSE_NOT_CONFIGURED, unavailable_message

    return unavailable_message(CAUSE_NOT_CONFIGURED)


def _identity_incomplete() -> str:
    from training.hotupdate.settings import CAUSE_IDENTITY_INCOMPLETE, unavailable_message

    return unavailable_message(CAUSE_IDENTITY_INCOMPLETE)


def _client_not_configured() -> str:
    from core.config import Settings
    from training.hotupdate.client import WeightsClient
    from training.hotupdate.settings import load

    with pytest.raises(Refusal) as refused:
        WeightsClient(settings=load(Settings(_env_file=None, millm_base_url=None)))
    return str(refused.value.message)


def _credential_refusal() -> str:
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError) as refused:
        Settings(_env_file=None, millm_base_url="http://user:pw@host:8000")
    return str(refused.value)


BUILDERS: dict[str, tuple[Callable[[], str], tuple[str, ...]]] = {
    "identity.missing_token": (_identity_missing_token, ("millm_trainer_token",)),
    "identity.missing_label": (_identity_missing_label, ("millm_audit_label",)),
    "identity.rejected": (_identity_rejected, ("millm_trainer_token", "millm_audit_label")),
    "settings.not_configured": (_not_configured, ("millm_base_url",)),
    "settings.identity_incomplete": (
        _identity_incomplete,
        ("millm_trainer_token", "millm_audit_label"),
    ),
    "client.not_configured": (_client_not_configured, ("millm_base_url",)),
    "config.credential_refusal": (_credential_refusal, ("millm_trainer_token",)),
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_builders_name_new_variable(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MILLM_BASE_URL", "MISERVER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    build, fields = BUILDERS[name]
    message = build()
    for field in fields:
        label = legacy_names.env_label(field)
        assert label in message, f"{name}: expected {label!r} in: {message}"
    stripped = message
    for entry in legacy_names.ENV_NAMES:
        stripped = stripped.replace(legacy_names.env_label(entry.field), "")
    bare = re.findall(r"\bMILLM_[A-Z_]+\b", stripped)
    assert not bare, f"{name}: an old variable is named outside env_label(): {bare}"


@pytest.fixture
def _fresh_log_state() -> object:
    legacy_names.reset_logged()
    yield
    legacy_names.reset_logged()


def _resolution(*uses: tuple[str, str]) -> legacy_names.LegacyResolution:
    return legacy_names.LegacyResolution(
        tuple(legacy_names.LegacyUse("env", old, new, "in_use", "environment") for old, new in uses)
    )


def test_one_warning_per_old_name(
    caplog: pytest.LogCaptureFixture, _fresh_log_state: object
) -> None:
    resolution = _resolution(
        ("MILLM_BASE_URL", "MISERVER_BASE_URL"), ("MILLM_AUDIT_LABEL", "MISERVER_AUDIT_LABEL")
    )
    with caplog.at_level(logging.WARNING, logger="core.legacy_names"):
        assert legacy_names.log_in_use(resolution) == 2
        assert legacy_names.log_in_use(resolution) == 0, "logged twice in one process"
    records = [r for r in caplog.records if r.__dict__.get("code") == "legacy_env_name_in_use"]
    assert [(r.__dict__["old"], r.__dict__["new"]) for r in records] == [
        ("MILLM_BASE_URL", "MISERVER_BASE_URL"),
        ("MILLM_AUDIT_LABEL", "MISERVER_AUDIT_LABEL"),
    ]
    assert all(r.levelno == logging.WARNING for r in records)


def test_no_warning_with_new_names_only(
    caplog: pytest.LogCaptureFixture, _fresh_log_state: object
) -> None:
    resolution = legacy_names.LegacyResolution(
        (legacy_names.LegacyUse("env", "MILLM_BASE_URL", "MISERVER_BASE_URL", "also_set", None),)
    )
    with caplog.at_level(logging.DEBUG, logger="core.legacy_names"):
        assert legacy_names.log_in_use(legacy_names.LegacyResolution()) == 0
        assert legacy_names.log_in_use(resolution) == 0
    assert not [r for r in caplog.records if r.__dict__.get("code") == "legacy_env_name_in_use"]


def test_one_warning_per_retired_name(
    caplog: pytest.LogCaptureFixture, _fresh_log_state: object
) -> None:
    """P7 on: an old name still set is ignored, so the log says so once, by its own code."""
    resolution = legacy_names.LegacyResolution(
        (
            legacy_names.LegacyUse(
                "env", "MILLM_TRAINER_TOKEN", "MISERVER_TRAINER_TOKEN", "retired_but_set", None
            ),
        )
    )
    with caplog.at_level(logging.WARNING, logger="core.legacy_names"):
        assert legacy_names.log_in_use(resolution) == 1
        assert legacy_names.log_in_use(resolution) == 0, "logged twice in one process"
    records = [r for r in caplog.records if r.__dict__.get("code") == "legacy_env_name_retired"]
    assert [(r.__dict__["old"], r.__dict__["new"]) for r in records] == [
        ("MILLM_TRAINER_TOKEN", "MISERVER_TRAINER_TOKEN")
    ]
    assert "retired and ignored; set MISERVER_TRAINER_TOKEN" in records[0].getMessage()
