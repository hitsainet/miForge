"""Unconfigured is **normal**, and it says so by name  (F10 task 2.6; AC-010.1, AC-010.15).

ADR-010, R-02.50. miForge was the suite's only application with a hard runtime dependency on
a sibling, and the thing it depended on did not exist. These tests are what stop that growing
back one convenience at a time.

**Codes and ``details`` keys are asserted; message text is not.** Messages are copy and will
be improved; codes are contract, and eleven cross-feature refusal codes are spelled
identically across the feature documents precisely because something matches on them.
"""

from __future__ import annotations

import pytest

from clients import availability, capabilities
from core.config import get_settings, reset_settings
from core.errors import Refusal


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    monkeypatch.delenv("MIDATAWORKS_AGENT", raising=False)
    reset_settings()
    yield
    reset_settings()


def configure(monkeypatch, url: str = "http://dw.invalid:8000") -> None:
    monkeypatch.setenv("MIDATAWORKS_BASE_URL", url)
    reset_settings()


class TestUnsetMeansAbsentAndNormal:
    def test_the_default_is_unset(self):
        """**The posture is inverted from the obvious one** and that is the feature."""
        assert get_settings().midataworks_base_url is None
        assert availability.is_configured() is False

    def test_the_guard_refuses_by_name(self):
        with pytest.raises(Refusal) as caught:
            availability.require_midataworks("importing a part over HTTP", "part_import_http")
        assert caught.value.code == "midataworks_not_configured"

    def test_the_refusal_carries_all_four_details_keys(self):
        """A ``details`` dict that varies by call site is one nobody can handle generically."""
        with pytest.raises(Refusal) as caught:
            availability.require_midataworks("listing versions", "available_parts_listing")
        details = caught.value.details
        for key in ("sibling", "setting", "action", "capability"):
            assert key in details, f"{key} missing from {sorted(details)}"
        assert details["setting"] == "MIDATAWORKS_BASE_URL"
        assert details["capability"] == "available_parts_listing"

    def test_it_also_carries_the_shape_core_errors_fixes(self):
        """``core.errors.not_configured`` fixes ``service`` / ``capabilities_unavailable``
        for **every** optional sibling, so a handler written against either shape works."""
        with pytest.raises(Refusal) as caught:
            availability.require_midataworks("listing versions", "available_parts_listing")
        assert caught.value.details["service"] == "midataworks"
        assert caught.value.details["capabilities_unavailable"]

    def test_the_message_says_what_to_do_instead(self):
        """The sentence most people would drop, and the one that makes it useful.

        Asserted as a *property* - the standalone alternative is named - rather than
        word for word, so improving the copy does not fail the suite.
        """
        with pytest.raises(Refusal) as caught:
            availability.require_midataworks("importing a part over HTTP", "part_import_http")
        message = caught.value.message
        assert "MIDATAWORKS_BASE_URL" in message
        assert "your own files" in message, "the standalone next step is missing"

    def test_the_guard_returns_the_url_when_configured(self, monkeypatch):
        """A guard that only raises invites the caller to re-read settings - a second place
        the setting's name appears."""
        configure(monkeypatch, "http://dw.invalid:8000/")
        assert (
            availability.require_midataworks("x", "part_import_http") == "http://dw.invalid:8000"
        ), "a trailing slash would make every path contain //"

    def test_an_empty_env_value_is_unset_not_the_empty_string(self, monkeypatch):
        """``.env.example`` ships the line blank; blank must mean standalone."""
        monkeypatch.setenv("MIDATAWORKS_BASE_URL", "   ")
        reset_settings()
        assert availability.is_configured() is False


class TestTheCapabilityRegistry:
    def test_the_three_capabilities_are_declared(self):
        assert set(capabilities.CAPABILITIES) == {
            "part_import_http",
            "available_parts_listing",
            "human_review_selector",
        }

    def test_an_unknown_capability_refuses_rather_than_describing_nothing(self):
        """A typo'd key would make health report on a capability nothing gates."""
        with pytest.raises(Refusal) as caught:
            capabilities.describe("human_revieww")
        assert caught.value.code == "capability_unknown"

    def test_the_guard_refuses_a_typo_before_building_a_refusal(self):
        """Order matters: the capability is checked *before* the not-configured refusal, so
        a typo cannot hide behind the sibling being absent."""
        with pytest.raises(Refusal) as caught:
            availability.require_midataworks("x", "not_a_capability")
        assert caught.value.code == "capability_unknown"


class TestTheStatusPayload:
    def test_unconfigured_reports_reachable_as_null_not_false(self):
        """**``None`` is not ``False``.** "Not configured" and "configured and down" are
        different states with different fixes, and a payload reporting both as ``false``
        hides a misconfiguration behind a network message."""
        payload = availability.integration_status()
        assert payload["configured"] is False
        assert payload["reachable"] is None
        assert payload["code"] == "midataworks_not_configured"
        assert set(payload["capabilities_unavailable"]) == set(capabilities.CAPABILITIES)

    def test_configured_and_down_reports_reachable_false(self, monkeypatch):
        configure(monkeypatch)
        payload = availability.integration_status(reachable=False)
        assert payload["configured"] is True
        assert payload["reachable"] is False
        assert payload["capabilities_unavailable"] == []

    def test_every_capability_is_described_either_way(self):
        """Built by **looping the registry**, not by an ``if`` per capability - so a
        capability cannot be added without appearing here."""
        for state in (
            availability.integration_status(),
            availability.integration_status(reachable=True),
        ):
            assert set(state["capabilities"]) == set(capabilities.CAPABILITIES)

    def test_the_agent_identity_is_reported(self, monkeypatch):
        """Configuration, not a secret (ADR-020) - and an operator debugging an
        ORIGIN_APP_MISMATCH needs to see what miForge sends."""
        configure(monkeypatch)
        assert availability.integration_status()["agent"] == "agent:miforge"


class TestHealthNamesTheIntegration:
    def test_health_is_ok_with_midataworks_unconfigured(self, monkeypatch):
        """**Absent is normal.** A health endpoint that degrades on an unconfigured optional
        integration is the first step back toward a hard dependency."""
        from api import health as health_module

        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, "postgres", lambda: {"status": "ok", "detail": None}
        )
        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, "redis", lambda: {"status": "ok", "detail": None}
        )
        payload = health_module.health_payload()
        assert payload["status"] == "ok"
        assert payload["integrations"]["midataworks"]["configured"] is False
        assert payload["integrations"]["midataworks"]["code"] == "midataworks_not_configured"
        assert "midataworks" in payload["unconfigured_integrations"]

    def test_health_names_the_capabilities_that_need_it(self, monkeypatch):
        from api import health as health_module

        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, "postgres", lambda: {"status": "ok", "detail": None}
        )
        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, "redis", lambda: {"status": "ok", "detail": None}
        )
        entry = health_module.health_payload()["integrations"]["midataworks"]
        assert entry["capabilities_unavailable"], "an absent sibling must say what is lost"


class TestABogusCapabilityIsRefusedOnBothPaths:
    """Suite tracker 1.6 (found from miDataworks): the key was validated only inside the
    not-configured refusal, so a typo passed silently whenever miDataworks WAS configured -
    surfacing exactly when the service was down and never when it was up."""

    def test_a_bogus_key_is_refused_with_miDataworks_configured(self, monkeypatch):
        configure(monkeypatch)
        with pytest.raises(Refusal) as caught:
            availability.require_midataworks("importing a part", "part_imprt_http")
        assert caught.value.code == "capability_unknown"

    def test_a_bogus_key_is_refused_with_miDataworks_unconfigured(self):
        with pytest.raises(Refusal) as caught:
            availability.require_midataworks("importing a part", "part_imprt_http")
        assert caught.value.code == "capability_unknown"

    def test_a_real_key_still_returns_the_url(self, monkeypatch):
        configure(monkeypatch)
        assert availability.require_midataworks("x", "human_review_selector")
