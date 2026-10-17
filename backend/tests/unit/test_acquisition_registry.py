"""The acquisition mode registry  (R-02.46, R-02.48; FR-001.1, FR-001.2, FR-001.15).

Two things this file exists to pin, both of which are structural rather than behavioural:

1. **There is no ``if mode == "attach"`` anywhere in F1's tree.** F11 adds a *registration*,
   so ADR-022's fence holds by construction. Asserted by scanning the source, because the
   alternative - trusting that nobody adds one - is what the fence is for.
2. **An unavailable mode refuses by name and never falls back to a load.** R-02.48 forbids
   substituting a different behaviour for a missing one, and the substitution available here
   is the dangerous kind: loading the same model from disk trains a *different set of
   tensors* than attaching to a serving process, and reports success either way.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.errors import Refusal
from training import acquisition
from training.acquisition import (
    DEFAULT_MODE,
    AcquisitionMode,
    AcquisitionSpec,
    Availability,
    SourceKind,
)
from training.acquisition.hub import HubAcquirer
from training.acquisition.local_dir import LocalDirAcquirer
from training.acquisition.scratch import ScratchAcquirer

SRC = Path(__file__).resolve().parents[3] / "backend" / "src"


class TestDispatch:
    def test_load_is_the_default_mode(self) -> None:
        """R-02.46 names ``load`` the default, and the spec agrees without being told."""
        assert DEFAULT_MODE is AcquisitionMode.LOAD
        assert AcquisitionSpec(path=Path("/tmp/x")).mode is AcquisitionMode.LOAD

    def test_a_load_with_a_path_resolves_to_the_directory_acquirer(self) -> None:
        resolved = acquisition.resolve(AcquisitionSpec(path=Path("/tmp/x")))
        assert isinstance(resolved, LocalDirAcquirer)

    def test_a_load_with_a_repo_resolves_to_the_hub_acquirer(self) -> None:
        """One mode, two sources (R-02.46 case 1 names both). Dispatching on the source
        beats inventing ``load_local``/``load_hub`` mode names the requirement lacks and the
        acquisition record would then have to carry."""
        resolved = acquisition.resolve(AcquisitionSpec(repo="org/model", revision="main"))
        assert isinstance(resolved, HubAcquirer)

    def test_a_create_resolves_to_the_scratch_acquirer(self) -> None:
        resolved = acquisition.resolve(
            AcquisitionSpec(mode=AcquisitionMode.CREATE, config={"model_type": "llama"})
        )
        assert isinstance(resolved, ScratchAcquirer)

    def test_all_three_modes_are_registered_in_the_requirement_s_order(self) -> None:
        """``attach`` is **present** though unavailable. An absent entry would make an
        unavailable mode indistinguishable from a typo."""
        assert acquisition.registered_modes() == (
            AcquisitionMode.LOAD,
            AcquisitionMode.CREATE,
            AcquisitionMode.ATTACH,
        )


class TestAvailabilityIsSeparateFromAcquisition:
    def test_available_touches_neither_network_nor_gpu(self, monkeypatch) -> None:
        """``GET /models/sources`` calls this with no run in hand and must answer instantly.

        Enforced by making the two expensive things explode: a probe of either would fail
        the test rather than merely slow it down.
        """
        import subprocess

        def explode(*_args, **_kwargs):  # pragma: no cover - the point is that it is not hit
            raise AssertionError("available() must not shell out or probe the network")

        monkeypatch.setattr(subprocess, "run", explode)
        for mode in acquisition.registered_modes():
            acquisition.availability(mode)

    def test_an_unknown_mode_reports_unavailable_rather_than_raising(self) -> None:
        state = acquisition.availability("not-a-mode")  # type: ignore[arg-type]
        assert state.available is False
        assert state.reason == "acquisition_mode_unknown"


class TestAttachIsRegisteredAsUnavailableAndRefusesByName:
    """E-6, AC-001.12."""

    def test_attach_is_reported_unavailable_with_its_reason_and_capabilities(self) -> None:
        state = acquisition.availability(AcquisitionMode.ATTACH)
        assert state.available is False
        assert state.reason == "hot_update_mode_not_available"
        assert state.unavailable_capabilities, (
            "an unavailable mode must name the capabilities it would have provided; "
            "otherwise a reader learns that something failed rather than what they cannot do"
        )
        assert "§5.10" in (state.message or "")

    def test_preflight_refuses_by_name(self) -> None:
        spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH)
        with pytest.raises(Refusal) as caught:
            acquisition.resolve(spec).preflight(spec)
        assert caught.value.code == "acquisition_mode_unavailable"
        assert caught.value.details["mode"] == "attach"
        assert caught.value.details["capabilities_unavailable"]

    def test_acquire_refuses_and_does_NOT_fall_back_to_a_load(self) -> None:
        """**The substitution R-02.48 forbids.** Quietly loading the same weights from disk
        would train a different set of tensors than the operator asked for and report
        success - so ``acquire`` must raise rather than return an ``AcquiredModel``."""
        spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH)
        with pytest.raises(Refusal) as caught:
            acquisition.resolve(spec).acquire(spec)
        assert caught.value.code == "acquisition_mode_unavailable"

    def test_an_attach_spec_carrying_a_model_source_is_refused(self) -> None:
        """A source on an ``attach`` means the caller expects a load - which is the
        confusion the refusal above exists to prevent, arriving at the type instead."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="takes no model source"):
            AcquisitionSpec(mode=AcquisitionMode.ATTACH, path=Path("/tmp/x"))


class TestTheFenceHoldsByConstruction:
    """Scanned with ``ast``, not with ``in text``.

    A substring scan over these modules reports **this feature's own prose as a violation**:
    both ``base.py`` and ``__init__.py`` explain at length that there is no
    ``if mode == "attach"``, and a string checker cannot tell a disclaimer from a claim. The
    project has already paid for that lesson once - a fresh harness reported 18 scope-creep
    failures and all 18 were documents correctly *disclaiming* a foreign requirement. So
    these two tests parse the syntax tree and look at comparisons and imports, which is also
    the stronger assertion: it is about the code rather than about the file's text.
    """

    def test_no_module_under_training_branches_on_the_attach_mode(self) -> None:
        """ADR-022: ``attach`` is a registration, not a branch.

        A single ``if mode == "attach"`` would make the core path aware of §5.10 and would be
        the first thread of the thirteen-row dependency the reframe removed - *"it grew to
        thirteen required rows once without anyone deciding it should."*
        """
        offenders: list[str] = []
        for path in (SRC / "training").rglob("*.py"):
            if "hotupdate" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare) and _compares_with_attach(node):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} compares with attach")
        assert not offenders, "\n".join(offenders)

    def test_nothing_in_the_core_path_imports_the_hot_update_package(self) -> None:
        """The standalone test runs with ``training/hotupdate/`` **absent from the tree**,
        so a core-path import of it would fail the suite - which is the fence working."""
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            if "hotupdate" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [
                        f"{node.module or ''}.{alias.name}" for alias in node.names
                    ]
                else:
                    continue
                if any(name.startswith("training.hotupdate") for name in names):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
        assert not offenders, f"core-path modules import the fenced package: {offenders}"


def _compares_with_attach(node: ast.Compare) -> bool:
    """Whether a comparison tests a value against the ``attach`` mode.

    Both spellings: the string literal and the enum member. Equality **and** identity, since
    ``is AcquisitionMode.ATTACH`` is the form a typed codebase reaches for first.
    """
    operands = [node.left, *node.comparators]
    for operand in operands:
        if isinstance(operand, ast.Constant) and operand.value == "attach":
            return True
        if isinstance(operand, ast.Attribute) and operand.attr == "ATTACH":
            return True
    return False


class TestTheModesPayload:
    """FR-001.15's surface, and why ``Availability`` is not a boolean."""

    def test_every_mode_appears_with_the_same_key_set(self) -> None:
        """A key that appears only on failure makes the success and failure shapes two
        different types for one endpoint."""
        payload = acquisition.modes()
        assert len(payload) == 3
        shapes = {frozenset(entry) for entry in payload}
        assert len(shapes) == 1, f"inconsistent shapes: {shapes}"
        assert shapes.pop() == {
            "mode",
            "available",
            "default",
            "reason",
            "message",
            "capabilities_unavailable",
            "sources",
        }

    def test_the_default_is_marked_so_the_ui_need_not_know_it(self) -> None:
        payload = {entry["mode"]: entry for entry in acquisition.modes()}
        assert payload["load"]["default"] is True
        assert payload["create"]["default"] is False

    def test_load_declares_both_of_its_sources(self) -> None:
        payload = {entry["mode"]: entry for entry in acquisition.modes()}
        assert payload["load"]["sources"] == [SourceKind.LOCAL_DIR.value, SourceKind.HUB.value]
        assert payload["create"]["sources"] == [SourceKind.CONFIG.value]

    def test_the_unavailable_mode_carries_its_reason_in_the_payload(self) -> None:
        payload = {entry["mode"]: entry for entry in acquisition.modes()}
        assert payload["attach"]["available"] is False
        assert payload["attach"]["reason"] == "hot_update_mode_not_available"
        assert payload["attach"]["capabilities_unavailable"]


class TestRegistrationIsReplaceable:
    def test_a_later_registration_replaces_an_unavailable_stub(self) -> None:
        """**This is F11's mechanism**, exercised here so it is known to work before F11
        needs it: ``attach`` becomes usable with no edit to any file in this feature."""

        class FakeAttach:
            mode = AcquisitionMode.ATTACH

            def available(self) -> Availability:
                return Availability(available=True)

            def preflight(self, spec: AcquisitionSpec) -> AcquisitionSpec:
                return spec

            def acquire(self, spec: AcquisitionSpec):  # pragma: no cover - not called here
                raise AssertionError

        original = acquisition._REGISTRY[AcquisitionMode.ATTACH]
        try:
            acquisition.register(FakeAttach())
            assert acquisition.availability(AcquisitionMode.ATTACH).available is True
        finally:
            acquisition._REGISTRY[AcquisitionMode.ATTACH] = original
        assert acquisition.availability(AcquisitionMode.ATTACH).available is False

    def test_unavailable_refuses_to_register_an_available_mode(self) -> None:
        """Mixing the two entry points would let a mode be registered as "unavailable" while
        reporting itself available, and the API would then disagree with the refusal."""
        with pytest.raises(ValueError):
            acquisition.unavailable(AcquisitionMode.ATTACH, Availability(available=True))
