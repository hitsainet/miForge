"""Unavailable, by name: module absent, URL unset, identity incomplete  (FR-011.47, FR-011.25).

Three distinguishable answers, and **none of them is a fallback to a load.** R-02.48 forbids
the substitution and it is the dangerous kind: attaching to a serving process's resident
tensors and loading the same weights from disk train *different tensors*, and both report
success.

**Why the mode-level ``reason`` is the same in all three cases**, with the cause in the
details: ``GET /api/v1/models/sources`` reports one ``reason`` per acquisition mode, F1's
tests pin the ``attach`` row at ``hot_update_mode_not_available``, and one of them asserts
that no mode row carries a ``*_not_configured`` reason - *"no F1 capability may be
unavailable because a sibling is unconfigured"*. §5.10's mode is unavailable **by
requirement** first and by configuration second. The TDD's ``millm_not_configured`` spelling
lives in the refusal's ``details.cause``, which is the machine-readable half. Reported.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import configure
from training import acquisition
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.hotupdate import settings as settings_module
from training.hotupdate.attach import AttachAcquirer


@pytest.fixture(autouse=True)
def _reset() -> None:
    from core.config import reset_settings

    yield
    reset_settings()


class TestWithNothingConfigured:
    def test_the_mode_is_unavailable_and_names_the_capabilities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure(monkeypatch, MILLM_BASE_URL="", MILLM_TRAINER_TOKEN="", MILLM_AUDIT_LABEL="")
        state = AttachAcquirer().available()
        assert state.available is False
        assert state.reason == settings_module.NOT_AVAILABLE_REASON
        assert state.unavailable_capabilities == settings_module.UNAVAILABLE_CAPABILITIES
        assert "MILLM_BASE_URL" in (state.message or ""), "the message must say what to set"

    def test_preflight_refuses_with_the_cause_in_the_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure(monkeypatch, MILLM_BASE_URL="", MILLM_TRAINER_TOKEN="", MILLM_AUDIT_LABEL="")
        spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH)
        with pytest.raises(Refusal) as caught:
            AttachAcquirer().preflight(spec)
        assert caught.value.code == "acquisition_mode_unavailable"
        assert caught.value.details["cause"] == settings_module.CAUSE_NOT_CONFIGURED
        assert caught.value.details["capabilities_unavailable"]

    def test_acquire_refuses_rather_than_loading_the_same_weights_from_disk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The substitution R-02.48 forbids** - and ``acquire`` makes the refusal itself
        rather than trusting that pre-flight ran."""
        configure(monkeypatch, MILLM_BASE_URL="")
        spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH)
        with pytest.raises(Refusal) as caught:
            AttachAcquirer().acquire(spec)
        assert caught.value.code == "acquisition_mode_unavailable"


class TestWithTheUrlSetAndTheIdentityIncomplete:
    @pytest.mark.parametrize("missing", ["MILLM_TRAINER_TOKEN", "MILLM_AUDIT_LABEL"])
    def test_the_cause_names_which_half_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        """*The audit label is not optional decoration* (R-02.32): a token without one is a
        weights change nobody can attribute."""
        configure(monkeypatch, **{missing: ""})
        state = AttachAcquirer().available()
        assert state.available is False
        assert "MILLM_AUDIT_LABEL" in (state.message or "")
        with pytest.raises(Refusal) as caught:
            AttachAcquirer().preflight(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
        assert caught.value.details["cause"] == settings_module.CAUSE_IDENTITY_INCOMPLETE


class TestAvailabilityTouchesNothing:
    def test_it_makes_no_network_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """F1's ``GET /api/v1/models/sources`` calls this with **no run in hand** and must
        answer instantly. A mode that could only report its absence by failing to acquire
        cannot satisfy "says so before it starts"."""
        import httpx

        configure(monkeypatch)
        sent: list[str] = []

        def _forbid(*args: object, **kwargs: object) -> None:
            sent.append("request")
            raise AssertionError("available() made a network call")

        monkeypatch.setattr(httpx.Client, "request", _forbid)
        AttachAcquirer().available()
        assert sent == []


class TestTheRegistrySurface:
    def test_the_mode_is_listed_even_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """*An option missing from a response is indistinguishable from one that is broken*
        (ADR-013)."""
        configure(monkeypatch, MILLM_BASE_URL="")
        payload = {entry["mode"]: entry for entry in acquisition.modes()}
        assert payload["attach"]["available"] is False
        assert payload["attach"]["reason"] == settings_module.NOT_AVAILABLE_REASON
        assert payload["attach"]["capabilities_unavailable"]

    def test_no_mode_row_reports_a_not_configured_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F1's own rule, re-asserted from this side because F11 is the feature that would
        break it: *no F1 capability may be unavailable because a sibling is unconfigured.*"""
        configure(monkeypatch, MILLM_BASE_URL="")
        for entry in acquisition.modes():
            assert "not_configured" not in (entry["reason"] or "")

    def test_requesting_the_mode_in_an_unconfigured_build_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure(monkeypatch, MILLM_BASE_URL="")
        spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH)
        with pytest.raises(Refusal) as caught:
            acquisition.resolve(spec).preflight(spec)
        assert caught.value.code == "acquisition_mode_unavailable"
        assert caught.value.details["mode"] == "attach"


class TestTheEnvExampleCanActuallyBeUsed:
    """``.env.example`` ships every §5.10 value **present and empty**, and an empty value for
    a numeric field is not a missing value - it is a parse error at startup.

    This is the whole process failing to boot rather than the mode being unconfigured, and
    the file that causes it is the one a developer copies first. Found by setting the
    variable to the empty string and constructing ``Settings``, not by reading either file.
    """

    def test_an_empty_interval_override_means_unset_rather_than_a_parse_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.config import Settings

        monkeypatch.setenv("HOTUPDATE_HEARTBEAT_INTERVAL_SECONDS", "")
        assert Settings().hotupdate_heartbeat_interval_seconds is None

    def test_every_hot_update_variable_in_env_example_parses_when_left_blank(self) -> None:
        """Over the real file, so a value added to it later without a default is caught."""
        import re
        from pathlib import Path

        from core.config import Settings

        example = (Path(__file__).resolve().parents[3] / ".env.example").read_text("utf-8")
        blank = re.findall(r"^((?:HOTUPDATE|MILLM)_[A-Z0-9_]+)=\s*$", example, re.MULTILINE)
        assert blank, "the §5.10 block is gone from .env.example"
        environment = dict.fromkeys(blank, "")
        settings = Settings(_env_file=None, **{})  # type: ignore[call-arg]
        assert settings is not None
        import os

        saved = {key: os.environ.get(key) for key in environment}
        try:
            os.environ.update(environment)
            Settings()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
