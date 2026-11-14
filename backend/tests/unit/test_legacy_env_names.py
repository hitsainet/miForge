"""Environment names: ``MISERVER_*`` configures, ``MILLM_*`` is retired (FR-015.5, FR-015.23).

Resolution rules: FTID §7.1.

miLLM became miServer (suite ENH-009). From P0 to P7 an operator could set either name, so the
settings source decides by PRESENCE (a new name that is set, even blank, wins), says which old
names it used, and records nothing for a field the caller passed explicitly. Since P7 the old
names are RETIRED: set alone one configures nothing, and every old name set is reported
``retired_but_set``.

Two matrices: ``test_resolution_after_p7`` on the REAL tables, and ``test_resolution`` on the
pre-P7 tables (``pre_p7``, monkeypatched), because the transition path is still the code the
rollback re-enables (FTDD §11.8: "re-add the pairs to ``ENV_NAMES``").

Every case sets the two names to DIFFERENT values except "both equal": fixtures that agree by
construction hide a precedence bug (FTID §8.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import legacy_names
from core.config import Settings, reset_settings
from core.legacy_names import LegacyEnvSource, LegacyUse

#: (field, new name, old name, new value, old value) - values that differ, per field.
PAIRS = [
    ("millm_base_url", "MISERVER_BASE_URL", "MILLM_BASE_URL", "http://new:8000", "http://old:8000"),
    ("millm_trainer_token", "MISERVER_TRAINER_TOKEN", "MILLM_TRAINER_TOKEN", "tok-new", "tok-old"),
    ("millm_audit_label", "MISERVER_AUDIT_LABEL", "MILLM_AUDIT_LABEL", "label-new", "label-old"),
]

#: Each scenario: which names are set, to what, and what must result while an alias works
#: (the pre-P7 tables). ``new``/``old`` are the value keys used; ``""`` means "present but blank".
SCENARIOS: dict[str, dict[str, Any]] = {
    "old_only": {"set": {"old": "old"}, "value": "old", "states": {"in_use"}},
    "new_only": {"set": {"new": "new"}, "value": "new", "states": set()},
    "both_equal": {"set": {"new": "new", "old": "new"}, "value": "new", "states": {"also_set"}},
    "both_different": {
        "set": {"new": "new", "old": "old"},
        "value": "new",
        "states": {"also_set", "conflict"},
    },
    "new_blank_old_set": {
        "set": {"new": "", "old": "old"},
        "value": None,
        "states": {"also_set", "conflict"},
    },
}


#: The same scenarios on the real tables (P7 on): the old name never supplies a value.
SCENARIOS_AFTER_P7: dict[str, dict[str, Any]] = {
    "old_only": {"set": {"old": "old"}, "value": None, "states": {"retired_but_set"}},
    "new_only": {"set": {"new": "new"}, "value": "new", "states": set()},
    "both_equal": {
        "set": {"new": "new", "old": "new"},
        "value": "new",
        "states": {"retired_but_set"},
    },
    "both_different": {
        "set": {"new": "new", "old": "old"},
        "value": "new",
        "states": {"retired_but_set"},
    },
    "new_blank_old_set": {
        "set": {"new": "", "old": "old"},
        "value": None,
        "states": {"retired_but_set"},
    },
}


@pytest.fixture
def pre_p7(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tables as they were from P0 to P7 (and as a rollback would restore them)."""
    entries = tuple(legacy_names.EnvName(field, new, old) for field, new, old, _, _ in PAIRS)
    monkeypatch.setattr(legacy_names, "ENV_NAMES", entries)
    monkeypatch.setattr(
        legacy_names, "LEGACY_ENV_ALIASES", {e.legacy: e.env for e in entries if e.legacy}
    )
    monkeypatch.setattr(legacy_names, "RETIRED_ENV_NAMES", {})


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """No MILLM_*/MISERVER_* name leaks in; the working directory holds the only ``.env``."""
    import os

    for name in list(os.environ):
        if name.upper().startswith(("MILLM_", "MISERVER_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    reset_settings()
    yield
    reset_settings()


def _set(monkeypatch: pytest.MonkeyPatch, where: str, values: dict[str, str]) -> None:
    if where == "environment":
        for name, value in values.items():
            monkeypatch.setenv(name, value)
    else:
        Path(".env").write_text("".join(f"{k}={v}\n" for k, v in values.items()))


def _states(settings: Settings, old: str) -> set[str]:
    return {use.state for use in settings._legacy_names.uses if use.old == old}


CASES = [
    pytest.param(pair, where, scenario, id=f"{pair[1]}-{where}-{scenario}")
    for pair in PAIRS
    for where in ("environment", "env_file")
    for scenario in SCENARIOS
]


@pytest.mark.parametrize(("pair", "where", "scenario"), CASES)
def test_resolution(
    monkeypatch: pytest.MonkeyPatch,
    pre_p7: None,
    pair: tuple[str, str, str, str, str],
    where: str,
    scenario: str,
) -> None:
    _check_resolution(monkeypatch, pair, where, SCENARIOS[scenario], scenario)


@pytest.mark.parametrize(("pair", "where", "scenario"), CASES)
def test_resolution_after_p7(
    monkeypatch: pytest.MonkeyPatch,
    pair: tuple[str, str, str, str, str],
    where: str,
    scenario: str,
) -> None:
    """The real tables: an old name is reported and ignored, whatever else is set."""
    _check_resolution(monkeypatch, pair, where, SCENARIOS_AFTER_P7[scenario], scenario)


def _check_resolution(
    monkeypatch: pytest.MonkeyPatch,
    pair: tuple[str, str, str, str, str],
    where: str,
    spec: dict[str, Any],
    scenario: str,
) -> None:
    field, new_name, old_name, new_value, old_value = pair
    values = {"new": new_value, "old": old_value, "": ""}
    to_set = {
        (new_name if role == "new" else old_name): values[key] for role, key in spec["set"].items()
    }
    _set(monkeypatch, where, to_set)

    settings = Settings()

    expected = None if spec["value"] is None else values[spec["value"]]
    assert getattr(settings, field) == expected, f"{scenario}: wrong value for {field}"
    assert _states(settings, old_name) == spec["states"]
    for use in settings._legacy_names.uses:
        assert use.new == new_name
        assert use.source == where, f"the source of {old_name} must be recorded as {where}"


@pytest.mark.parametrize(("field", "new_name", "old_name", "new_value", "old_value"), PAIRS)
def test_new_name_in_env_file_beats_old_name_in_environment(
    monkeypatch: pytest.MonkeyPatch,
    pre_p7: None,
    field: str,
    new_name: str,
    old_name: str,
    new_value: str,
    old_value: str,
) -> None:
    """Presence of the new name wins wherever it is (FTID §7.1, the mixed case)."""
    monkeypatch.setenv(old_name, old_value)
    Path(".env").write_text(f"{new_name}={new_value}\n")
    settings = Settings()
    assert getattr(settings, field) == new_value
    uses = {(u.old, u.state, u.source) for u in settings._legacy_names.uses}
    assert uses == {(old_name, "also_set", "environment"), (old_name, "conflict", "environment")}


def test_env_file_none_ignores_dotenv_new_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Settings(_env_file=None)`` reads no ``.env``: neither name there may configure it."""
    Path(".env").write_text("MISERVER_BASE_URL=http://new:8000\nMILLM_AUDIT_LABEL=label-old\n")
    settings = Settings(_env_file=None)
    assert settings.millm_base_url is None
    assert settings.millm_audit_label is None
    assert settings._legacy_names.uses == ()


def test_retired_name_configures_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """After P7 an old name is retired: set alone it configures NOTHING, and says so.

    The field is literally named ``millm_base_url``, so pydantic's own environment source
    would still read ``MILLM_BASE_URL``; the legacy source must override it with ``None``.
    """
    monkeypatch.setattr(
        legacy_names,
        "ENV_NAMES",
        tuple(
            legacy_names.EnvName(e.field, e.env, None) if e.field == "millm_base_url" else e
            for e in legacy_names.ENV_NAMES
        ),
    )
    monkeypatch.setattr(
        legacy_names, "LEGACY_ENV_ALIASES", {"MILLM_TRAINER_TOKEN": "MISERVER_TRAINER_TOKEN"}
    )
    monkeypatch.setattr(legacy_names, "RETIRED_ENV_NAMES", {"MILLM_BASE_URL": "MISERVER_BASE_URL"})
    monkeypatch.setenv("MILLM_BASE_URL", "http://old:8000")
    settings = Settings()
    assert settings.millm_base_url is None, "a retired old name must not configure the field"
    assert _states(settings, "MILLM_BASE_URL") == {"retired_but_set"}


def test_source_is_second_in_customise_sources() -> None:
    """Order: init kwargs, then the legacy source, then pydantic's environment and ``.env``."""
    init, env, dotenv, secrets = object(), object(), object(), object()
    sources = Settings.settings_customise_sources(Settings, init, env, dotenv, secrets)  # type: ignore[arg-type]
    assert sources[0] is init
    assert isinstance(sources[1], LegacyEnvSource)
    assert sources[2:] == (env, dotenv, secrets)


def test_real_settings_with_only_old_name_is_configured(
    monkeypatch: pytest.MonkeyPatch, pre_p7: None
) -> None:
    """The source is wired into the real class: an old name alone is used AND reported, and a
    new name alone (which pydantic's own source would never read) configures the field."""
    monkeypatch.setenv("MILLM_TRAINER_TOKEN", "tok-old")
    monkeypatch.setenv("MISERVER_BASE_URL", "http://new:8000")
    settings = Settings()
    assert settings.millm_trainer_token == "tok-old"
    assert _states(settings, "MILLM_TRAINER_TOKEN") == {"in_use"}
    assert settings.millm_base_url == "http://new:8000"


def test_init_kwargs_drop_the_record(monkeypatch: pytest.MonkeyPatch, pre_p7: None) -> None:
    """An explicit value wins and the record must not claim the old name supplied it (IPD-3)."""
    monkeypatch.setenv("MILLM_BASE_URL", "http://old:8000")
    monkeypatch.setenv("MILLM_AUDIT_LABEL", "label-old")
    for settings in (
        Settings(millm_base_url="http://kwarg:8000"),
        Settings.model_validate({"millm_base_url": "http://kwarg:8000"}),
    ):
        assert settings.millm_base_url == "http://kwarg:8000"
        assert _states(settings, "MILLM_BASE_URL") == set()
        assert _states(settings, "MILLM_AUDIT_LABEL") == {"in_use"}


def test_record_survives_model_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """``effective_settings()`` returns a ``model_copy``; it must carry the same record."""
    monkeypatch.setenv("MILLM_BASE_URL", "http://old:8000")
    settings = Settings()
    copy = settings.model_copy(update={"log_level": "DEBUG"})
    assert copy._legacy_names == settings._legacy_names
    assert copy._legacy_names.uses


def test_the_record_is_never_in_a_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    """A private attribute: nothing that serialises settings can carry it."""
    monkeypatch.setenv("MILLM_BASE_URL", "http://old:8000")
    settings = Settings()
    assert "_legacy_names" not in settings.model_dump()
    assert "legacy" not in settings.model_dump_json()
    assert isinstance(settings._legacy_names.uses[0], LegacyUse)


def test_after_p7_old_name_is_retired_but_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """P7 (FR-015.23): the REAL tables, no monkeypatch. ``MILLM_BASE_URL`` set alone configures
    nothing, is reported ``retired_but_set`` through ``GET /api/health``, and the refusal an
    operator reads names ``MISERVER_BASE_URL`` (and only it).

    Control N20 returns the old value instead of ignoring it: the field is configured and
    ``integrations.millm.configured`` turns true, so this test goes red.
    """
    from api import health as health_module
    from services.legacy_names import store
    from services.legacy_names.store import StoredUses
    from training.hotupdate.settings import CAUSE_NOT_CONFIGURED, unavailable_message

    assert "MILLM_BASE_URL" in legacy_names.RETIRED_ENV_NAMES, "P7 retires the base URL alias"
    assert "MILLM_BASE_URL" not in legacy_names.LEGACY_ENV_ALIASES
    monkeypatch.setenv("MILLM_BASE_URL", "http://old:8000")
    for probe in ("postgres", "redis"):
        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, probe, lambda: {"status": "ok", "detail": None}
        )
    monkeypatch.setattr(store, "record", lambda *a, **k: 0)
    monkeypatch.setattr(store, "read_recent", lambda **_: StoredUses())
    reset_settings()

    settings = Settings()
    assert settings.millm_base_url is None, "a retired old name must configure nothing"
    assert _states(settings, "MILLM_BASE_URL") == {"retired_but_set"}

    body = health_module.health_payload()
    assert body["integrations"]["millm"]["configured"] is False
    assert body["integrations"]["millm"]["code"] == "millm_not_configured"
    assert body["legacy_names"]["retired_but_set"] == [
        {"old": "MILLM_BASE_URL", "new": "MISERVER_BASE_URL"}
    ]
    assert body["legacy_names"]["in_use"] == []

    message = unavailable_message(CAUSE_NOT_CONFIGURED)
    assert "set MISERVER_BASE_URL to" in message, message
    assert "MILLM_" not in message, f"a retired name is still offered as a fix: {message}"
