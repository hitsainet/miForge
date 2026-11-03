"""Reachability: the ``attach`` mode is in the **live** registry  (AC-011.14; **MUT-10**).

*A capability is not shipped until a test FAILS when its wiring is removed.* This file's
whole job is to be the test that fails when the one line in
``training/acquisition/__init__.py`` that installs §5.10 is deleted.

**Nothing here imports ``AttachAcquirer`` to prove it exists.** That is the failure mode the
rule was written for: a sibling shipped **sixteen** fully-implemented, unit-tested,
documented MCP tools that were never registered with its server, and every one of those
tests passed by importing the module directly - so the suite was green and the docs said ✅
while the feature existed for no caller. Two review rounds read that code.

So every assertion is made by **resolving through the registry**, and the payload and the
call count are asserted rather than "something was registered".
"""

from __future__ import annotations

import pytest

from training import acquisition
from training.acquisition.base import Acquirer, AcquisitionMode, AcquisitionSpec


class TestTheModeIsInTheLiveRegistry:
    def test_resolving_the_attach_spec_returns_this_features_acquirer(self) -> None:
        """**Resolved, not imported.** ``resolve`` is what a caller uses, so this is the
        assertion a caller's experience actually depends on."""
        acquirer = acquisition.resolve(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
        assert type(acquirer).__name__ == "AttachAcquirer"
        assert type(acquirer).__module__ == "training.hotupdate.attach"

    def test_it_is_not_the_unavailable_stub_any_more(self) -> None:
        """The stub F1 ships is the *absence* of this feature. With the package present, the
        registry must hold the real acquirer - otherwise the mode is permanently unavailable
        however well configured the deployment is."""
        acquirer = acquisition.resolve(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
        assert type(acquirer).__name__ != "_UnavailableMode"

    def test_it_satisfies_F1s_protocol(self) -> None:
        acquirer = acquisition.resolve(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
        assert isinstance(acquirer, Acquirer)
        assert acquirer.mode is AcquisitionMode.ATTACH

    def test_the_mode_appears_in_the_payload_the_API_serves(self) -> None:
        """``GET /api/v1/models/sources`` renders this; a mode absent from it is a mode no
        operator can choose."""
        payload = {entry["mode"]: entry for entry in acquisition.modes()}
        assert "attach" in payload
        assert payload["attach"]["sources"] == ["attached"]
        # Every key, present on the available and the unavailable shape alike: *a key that
        # appears only on failure makes the success and failure shapes two different types.*
        assert set(payload["attach"]) == {
            "mode",
            "available",
            "default",
            "reason",
            "message",
            "capabilities_unavailable",
            "sources",
        }


class TestTheWiringIsWhatIsAsserted:
    def test_install_registers_the_acquirer_with_the_payload_and_the_call_count(self) -> None:
        """**The payload AND the count.** *"Was called" passes against a call sending the
        wrong arguments*, and a double registration would silently replace the first."""
        registered: list[object] = []
        from training.hotupdate import install

        returned = install(registered.append)
        assert len(registered) == 1, "exactly one registration"
        assert registered[0] is returned
        assert returned.mode is AcquisitionMode.ATTACH

    def test_the_registry_calls_install_rather_than_building_the_acquirer_itself(self) -> None:
        """So that **removing that call** - and nothing else - turns this file red. If the
        registry constructed the class directly, deleting ``install`` would be a different
        and louder failure, and this test would not be the one that caught it."""
        import ast
        import inspect

        source = inspect.getsource(acquisition)
        tree = ast.parse(source)
        installer = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_install_optional_attach_mode"
        )
        calls = [
            node.func.attr
            for node in ast.walk(installer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "install" in calls

    def test_the_installer_is_called_from_the_registrys_own_defaults(self) -> None:
        """The line whose deletion is MUT-10. Asserted structurally **as well as**
        behaviourally above, because the behavioural assertion would also pass if some other
        module happened to install the mode - and then the fence would be broken instead."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(acquisition))
        defaults = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_install_defaults"
        )
        called = {
            node.func.id
            for node in ast.walk(defaults)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_install_optional_attach_mode" in called


class TestTheSettingsAreReadLive:
    def test_availability_follows_a_change_in_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-011.14's second half: *the §5.10 settings are asserted read from the live
        settings object, not from a module constant.* A constant captured at import makes an
        environment change in a test silently ineffective - which is how a standalone test
        ends up asserting against a configuration it did not set."""
        from core.config import reset_settings
        from tests.support.fixtures_hotupdate import configure

        configure(monkeypatch, MILLM_BASE_URL="")
        assert acquisition.availability(AcquisitionMode.ATTACH).available is False
        configure(monkeypatch)
        state = acquisition.availability(AcquisitionMode.ATTACH)
        # Configured now: either available, or unavailable *for the ML runtime* - never for
        # the base URL, which is what this asserts.
        assert state.available or state.reason == "ml_runtime_unavailable"
        reset_settings()
