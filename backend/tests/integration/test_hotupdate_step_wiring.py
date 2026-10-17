"""The §5.10 step is reachable from the production loop  (R-02.52; ADR-016 #1).

**Why this file exists.** Until 2026-10-16 ``barrier.hot_update_step`` - the whole of
R-02.52's commit bracket - had **no production caller**. ``training/tasks.py`` stepped every
acquisition through the run type's plain ``run_one``, so an attached ``change`` run called
``optimizer.step()`` on miLLM's resident tensors with **no quiesce and no commit bracket**:
exactly the half-written parameter set served mid-write that BRD-01 R-01.21/R-01.22 and the
2026-09-19 torn-weights work exist to make impossible. Every F11 test drove
``hot_update_step`` directly, so the suite was green; F11 tasks 7.5 and 10.5 were ticked
against those direct calls.

So this file drives the **production driver** (``tasks.drive_run``) with an acquisition
that carries the real :class:`training.hotupdate.barrier.AttachedStepper` over the real
:class:`WeightsClient` and the fake serving surface, and asserts the **call sequence on the
wire, its count, and the step records' payload** - not that a function exists. Removing the
``step_impl`` dispatch in ``tasks.py`` turns it red (the negative control is recorded in
F11's FTASKS 10.5).
"""

# ruff: noqa: F811 - `rig` is a pytest fixture imported by name; each test parameter that
# receives it is a fixture injection, not a redefinition.

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from core.config import Settings
from tests.support.fake_weights_surface import FakeWeightsSurface
from tests.support.fixtures_hotupdate import BASE_URL, LABEL, TOKEN
from training import run_log, tasks
from training.acquisition.base import AcquiredModel
from training.hotupdate import barrier, state
from training.hotupdate import settings as settings_module
from training.hotupdate.client import WeightsClient
from training.hotupdate.timings import StepTimings

from .test_step_loop import RECORD, RUN, drive, rig  # noqa: F401 - `rig` is a fixture

BRACKET = ["quiesce", "commit-begin", "commit-end", "resume"]


@pytest.fixture
def attached(rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """An attached acquisition: the real stepper, the real client, the fake surface.

    The client's settings are built from an explicit ``Settings`` rather than the
    environment, so this fixture cannot disturb the ``rig``'s patched settings object.
    """
    surface = FakeWeightsSurface()
    live = Settings(  # type: ignore[call-arg]
        _env_file=None,
        millm_base_url=BASE_URL,
        millm_trainer_token=TOKEN,
        millm_audit_label=LABEL,
    )
    cfg = settings_module.load(live)
    client = WeightsClient(settings=cfg, transport=surface.transport())
    response = client.call("attach")
    client.remember_attachment(str(response["attachment_id"]))
    stepper = barrier.AttachedStepper(
        client,
        StepTimings(
            watchdog_ms=cfg.commit_watchdog_ms,
            creep_fraction=cfg.commit_creep_fraction,
            run_id=RUN,
        ),
        expected=state.identity_from_attach(response),
    )
    monkeypatch.setattr(
        tasks,
        "acquire_for_run",
        lambda spec: AcquiredModel(rig["model"], rig["tokenizer"], RECORD, stepper=stepper),
    )
    try:
        yield {"surface": surface, "stepper": stepper, "client": client}
    finally:
        client.close()


def _step_blocks(rig: dict[str, Any]) -> list[dict[str, Any]]:
    log = run_log.run_log_path(tasks.run_log_root(rig["log_root"].parent), RUN)
    return list(run_log.read_blocks(log, block="step"))


def _bracket_ops(surface: FakeWeightsSurface) -> list[str]:
    return [op for op in surface.ops() if op in set(BRACKET)]


class TestTheLoopStepsAnAttachedRunInsideTheBracket:
    def test_every_step_is_bracketed_on_the_wire_in_order(self, rig, attached) -> None:
        """The call, the count and the order. Three steps, three full brackets."""
        result = drive(rig, max_steps=3)
        assert result.steps_completed == 3
        assert attached["stepper"].calls == 3, (
            "the loop did not take its steps through the acquisition's stepper - if this is "
            "0, training/tasks.py is stepping the resident tensors with no bracket"
        )
        assert _bracket_ops(attached["surface"]) == BRACKET * 3
        assert rig["optimizer"].steps == 3

    def test_each_step_is_gated_on_a_clean_set_first(self, rig, attached) -> None:
        """``require_clean`` reads status before every step (FR-011.36; F11 7.5)."""
        drive(rig, max_steps=2)
        ops = attached["surface"].ops()
        # status precedes each quiesce: the precondition, not the bracket, comes first.
        quiesces = [i for i, op in enumerate(ops) if op == "quiesce"]
        assert len(quiesces) == 2
        for index in quiesces:
            assert "status" in ops[:index]
        assert attached["surface"].call_count("status") >= 2

    def test_the_step_records_carry_this_steps_timings(self, rig, attached) -> None:
        """R-02.14's two §5.10 fields reach the run log through the real writer.

        Before 2026-10-16 the writer passed neither, so an attached run's records said
        `null` - "no bracket was held" - for steps that held one.
        """
        drive(rig, max_steps=3)
        blocks = _step_blocks(rig)
        assert [block["step"] for block in blocks] == [1, 2, 3]
        for block in blocks:
            run_log.validate_block(block)
            assert isinstance(block["quiesce_hold_ms"], float)
            assert isinstance(block["commit_duration_ms"], float)
            assert block["quiesce_hold_ms"] >= 0.0
            assert block["commit_duration_ms"] >= 0.0
        # One sample per step - the distribution another repository's watchdog is checked
        # against must not double-count or skip.
        assert len(attached["stepper"].timings.commits_ms) == 3

    def test_a_cancelled_step_opens_no_bracket(self, rig, attached) -> None:
        """ADR-007's boundary is outside the bracket: a cancelled step quiesces nothing."""

        class StopRequestedError(Exception):
            pass

        class StopImmediately:
            def check(self) -> None:
                raise StopRequestedError

        with pytest.raises(StopRequestedError):
            attached["stepper"](
                rig["model"], rig["optimizer"], rig["source"], cancel=StopImmediately()
            )
        assert _bracket_ops(attached["surface"]) == []
        assert rig["optimizer"].steps == 0

    def test_a_step_that_held_no_bracket_borrows_no_timing(
        self, rig, attached, monkeypatch
    ) -> None:
        """``StepTimings.last`` returns the last value recorded by ANY step. A step that
        returned without bracketing must be recorded ``None``, never with its
        predecessor's duration - that would be a fabricated sample in the distribution
        miLLM's watchdog is checked against. Defensive today (``run_one`` always calls
        ``optimizer.step``), so it is driven directly."""
        from training.step import StepOutcome

        stepper = attached["stepper"]
        stepper.timings.record_commit(9.0)  # a previous step's samples
        stepper.timings.record_hold(3.0)
        monkeypatch.setattr(
            barrier,
            "hot_update_step",
            lambda *a, **k: StepOutcome(
                loss=1.0, objective="sft", grad_norm=1.0, wall_time_ms=1.0, batch_sha256=None
            ),
        )
        outcome = stepper(rig["model"], rig["optimizer"], rig["source"])
        assert outcome.quiesce_hold_ms is None
        assert outcome.commit_duration_ms is None


class TestAnAdapterRunOpensNoBracket:
    def test_a_step_through_a_wrapper_is_not_bracketed(self, rig, attached, monkeypatch) -> None:
        """AC-011.18: an adapter run writes no resident tensor, so it quiesces nothing.

        Modelled by a run type whose ``prepare`` returns a module that is not the acquired
        one - which is exactly what F8's adapter ``prepare`` does with its ``peft`` wrapper.
        """
        from dataclasses import replace

        from training import runtypes

        real = runtypes.resolve("change")

        class Wrapped:
            def __init__(self, inner: Any) -> None:
                self.inner = inner

            def parameters(self) -> Any:
                return self.inner.parameters()

            def __getattr__(self, name: str) -> Any:
                return getattr(self.inner, name)

        class AdapterLike:
            def __getattr__(self, name: str) -> Any:
                return getattr(real, name)

            def prepare(self, acquired: Any, adapter: Any = None) -> Any:
                plan = real.prepare(acquired, adapter=adapter)
                return replace(plan, module=Wrapped(plan.module))

        monkeypatch.setattr(tasks.runtypes, "resolve", lambda name: AdapterLike())
        result = drive(rig, max_steps=2)
        assert result.steps_completed == 2
        assert attached["stepper"].calls == 0
        assert _bracket_ops(attached["surface"]) == []


class TestTheDefaultPathIsUnchanged:
    def test_no_stepper_means_the_run_types_own_step(self, rig) -> None:
        """``load``/``create`` carry no stepper and their records keep both fields null."""
        drive(rig, max_steps=2)
        for block in _step_blocks(rig):
            assert block["quiesce_hold_ms"] is None
            assert block["commit_duration_ms"] is None


class TestTheAcquirerSuppliesTheStepper:
    """The other half of the wiring: ``AttachAcquirer.acquire`` must put the stepper on the
    acquisition it returns. Driven through the real ``acquire`` against the fake surface,
    with the CUDA-IPC rebuild and the build-identity check stood in (the first needs torch
    and a card; the second compares against this machine's torch); the heartbeat, the
    identity and the saver are the real ones."""

    def test_acquire_returns_an_acquisition_whose_stepper_brackets(self, rig, monkeypatch) -> None:
        from tests.support.fixtures_hotupdate import configure
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec, Availability
        from training.hotupdate import attach as attach_module
        from training.hotupdate import rebuild

        configure(monkeypatch)
        surface = FakeWeightsSurface()
        acquirer = attach_module.AttachAcquirer(transport=surface.transport())
        # The ML-runtime gate, not the behaviour under test (see test_hotupdate_run_state).
        monkeypatch.setattr(type(acquirer), "available", lambda self: Availability(available=True))

        class Base:
            device_uuid = "GPU-test"
            tensors: ClassVar[dict[str, Any]] = {}

        monkeypatch.setattr(rebuild, "build", lambda response, **kw: Base())
        # The fake advertises a fixed producer build; this machine's torch differs. The
        # build-identity refusal has its own tests (test_hotupdate_build_identity.py).
        monkeypatch.setattr(rebuild, "check_build_identity", lambda response: None)
        monkeypatch.setattr(attach_module, "build_module", lambda response, base: rig["model"])
        monkeypatch.setattr(attach_module, "acquisition_record", lambda **kw: RECORD)
        try:
            acquired = acquirer.acquire(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
            stepper = acquired.stepper
            assert isinstance(stepper, barrier.AttachedStepper)
            assert stepper is acquirer.stepper
            assert stepper.client is acquirer.client
            # Its baseline is the saver's (re-read after a save), not a frozen snapshot.
            assert stepper.expected == acquirer.saver.expected
            assert stepper.timings.watchdog_ms == settings_module.load().commit_watchdog_ms

            before = len(_bracket_ops(surface))
            stepper(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
            assert _bracket_ops(surface)[before:] == BRACKET
        finally:
            if acquirer.heartbeat is not None:
                acquirer.heartbeat.stop()


class TestTheRunReleasesWhatItAcquired:
    """``AcquiredModel.release`` is called once on every exit from the loop.

    Until 2026-10-16 nothing in the loop called ``detach``: the heartbeat thread kept the
    attachment alive in the solo worker after the run ended, holding miLLM's single slot
    (and every model load behind it) until the worker restarted.
    """

    def _with_release(self, rig, monkeypatch, release) -> None:
        monkeypatch.setattr(
            tasks,
            "acquire_for_run",
            lambda spec: AcquiredModel(rig["model"], rig["tokenizer"], RECORD, release=release),
        )

    def test_a_finished_run_releases_once(self, rig, monkeypatch) -> None:
        calls: list[str] = []
        self._with_release(rig, monkeypatch, lambda: calls.append("release"))
        drive(rig, max_steps=2)
        assert calls == ["release"]

    def test_a_failed_run_releases_and_keeps_its_own_error(self, rig, monkeypatch) -> None:
        calls: list[str] = []

        def release() -> None:
            calls.append("release")
            raise RuntimeError("detach refused")

        self._with_release(rig, monkeypatch, release)

        class Boom(Exception):  # noqa: N818 - the run's own error, named for the assertion
            pass

        def explode() -> Any:
            raise Boom

        rig["source"].next = explode
        with pytest.raises(Boom):
            drive(rig, max_steps=2)
        assert calls == ["release"]

    def test_a_release_failure_after_a_clean_finish_is_not_swallowed(
        self, rig, monkeypatch
    ) -> None:
        def release() -> None:
            raise RuntimeError("detach refused: set not clean")

        self._with_release(rig, monkeypatch, release)
        with pytest.raises(RuntimeError, match="set not clean"):
            drive(rig, max_steps=1)

    def test_the_attach_acquirer_supplies_detach_as_the_release(self, rig, monkeypatch) -> None:
        """Through the real ``acquire`` and the real loop: the wire ends with ``detach``."""
        from tests.support.fixtures_hotupdate import configure
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec, Availability
        from training.hotupdate import attach as attach_module
        from training.hotupdate import rebuild

        configure(monkeypatch)
        surface = FakeWeightsSurface()
        acquirer = attach_module.AttachAcquirer(transport=surface.transport())
        monkeypatch.setattr(type(acquirer), "available", lambda self: Availability(available=True))

        class Base:
            device_uuid = "GPU-test"
            tensors: ClassVar[dict[str, Any]] = {}

        monkeypatch.setattr(rebuild, "build", lambda response, **kw: Base())
        monkeypatch.setattr(rebuild, "check_build_identity", lambda response: None)
        monkeypatch.setattr(attach_module, "build_module", lambda response, base: rig["model"])
        monkeypatch.setattr(attach_module, "acquisition_record", lambda **kw: RECORD)
        # R-02.2's check is its own subject (test_hotupdate_verify_wiring.py); here a pass.
        monkeypatch.setattr(
            attach_module.AttachAcquirer, "_verifier", lambda self, *a: (lambda: None)
        )
        monkeypatch.setattr(
            tasks,
            "acquire_for_run",
            lambda spec: acquirer.acquire(AcquisitionSpec(mode=AcquisitionMode.ATTACH)),
        )
        try:
            drive(rig, max_steps=2)
            ops = surface.ops()
            assert ops.count("detach") == 1
            assert ops[-1] == "detach"
            assert _bracket_ops(surface) == BRACKET * 2
            assert acquirer.heartbeat is None, "the heartbeat must stop before detach"
        finally:
            if acquirer.heartbeat is not None:
                acquirer.heartbeat.stop()
