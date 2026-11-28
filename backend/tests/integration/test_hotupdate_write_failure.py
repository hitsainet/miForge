"""A write that fails partway WITHHOLDS commit-end; the watchdog tears the set; recovery runs.

Operator decision (2026-10-17). Until then the bracket sent ``commit-end`` on every exit, so
an optimizer update that raised half-way through the resident tensors was reported to miLLM
as ``clean`` and served. Now the commit is left open, miLLM's commit watchdog (BRD-01 R-01.22;
**200 ms** since miLLM ``db44fea``) declares the set ``torn`` - the fake runs that watchdog in
real time here - and R-02.30's recovery takes over: rollback to the last accepted checkpoint,
or stop. Never a late commit-end (that would bless the half-written set), never a retry.

Driven through ``tasks.drive_run`` with the real ``AttachedStepper`` and client.
"""

# ruff: noqa: F811 - `rig` is a pytest fixture imported by name; parameters receive it.

from __future__ import annotations

from typing import Any

import pytest

from core.config import Settings
from core.errors import Refusal
from tests.support.fake_weights_surface import FakeWeightsSurface
from tests.support.fixtures_hotupdate import BASE_URL, LABEL, TOKEN
from training import tasks
from training.acquisition.base import AcquiredModel
from training.hotupdate import barrier, state
from training.hotupdate import settings as settings_module
from training.hotupdate.client import WeightsClient
from training.hotupdate.timings import StepTimings

from .test_step_loop import RECORD, RUN, drive, rig  # noqa: F401 - `rig` is a fixture

WATCHDOG_MS = 200.0  # miLLM's measured default (db44fea)


class Saver:
    def __init__(self, last_saved: tuple[int, str] | None) -> None:
        self.expected = None
        self.last_saved = last_saved


class FailingOptimizer:
    """Writes "some" tensors, then raises - a write that failed partway."""

    def __init__(self, fail_on_call: int = 1) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.param_groups: list[dict[str, Any]] = [{"lr": 1e-5}]

    def step(self) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("CUDA error: an illegal memory access, tensor 41 of 148")

    def zero_grad(self, **_kw: Any) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"calls": self.calls}


def _attached(rig, monkeypatch, *, saved, watchdog_ms: float | None = WATCHDOG_MS):  # type: ignore[no-untyped-def]
    surface = FakeWeightsSurface(commit_watchdog_ms=watchdog_ms)
    cfg = settings_module.load(
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            millm_base_url=BASE_URL,
            millm_trainer_token=TOKEN,
            millm_audit_label=LABEL,
        )
    )
    client = WeightsClient(settings=cfg, transport=surface.transport())
    response = client.call("attach")
    client.remember_attachment(str(response["attachment_id"]))
    stepper = barrier.AttachedStepper(
        client,
        StepTimings(watchdog_ms=WATCHDOG_MS, creep_fraction=0.5, run_id=RUN),
        expected=state.identity_from_attach(response),
        saver=Saver(saved),
    )
    optimizer = FailingOptimizer()
    monkeypatch.setattr(
        tasks,
        "acquire_for_run",
        lambda spec: AcquiredModel(rig["model"], rig["tokenizer"], RECORD, stepper=stepper),
    )
    monkeypatch.setattr(
        tasks.optimizer_module,
        "build",
        lambda module, cfg, state="fresh": (optimizer, _identity(cfg, state)),
    )
    return surface, stepper, client, optimizer


def _identity(cfg, state):  # type: ignore[no-untyped-def]
    from training.optimizer import OptimizerIdentity

    return OptimizerIdentity(name=cfg.optimizer_name(), state=state, config_hash=cfg.config_hash())


class TestAFailedWriteWithholdsCommitEnd:
    def test_rollback_after_the_watchdog_declares_the_set_torn(self, rig, monkeypatch) -> None:
        surface, _stepper, client, optimizer = _attached(
            rig, monkeypatch, saved=(40, "run-x:step-40")
        )
        try:
            with pytest.raises(Refusal) as caught:
                drive(rig, max_steps=3)
        finally:
            client.close()
        details = caught.value.details
        assert caught.value.code == "hotupdate_step_abandoned"
        # No commit-end at all: neither the bracket's nor a late "completion".
        assert surface.call_count("commit-end") == 0
        assert details["watchdog_declared_torn"] is True
        assert details["recovery_path"] == "rollback"
        assert details["resumed_from_step"] == 40
        assert details["retried"] is False and details["discontinuity"] is True
        assert surface.bodies_of("load") == [{"name": "run-x:step-40"}]
        # The rollback load returned the set to clean; the write was attempted ONCE.
        assert surface.weights_state == "clean"
        assert optimizer.calls == 1
        assert surface.served_while_not_clean == []

    def test_the_torn_state_came_from_the_watchdog_in_real_time(self, rig, monkeypatch) -> None:
        """Before the rollback, a status read showed ``torn`` - and only after the watchdog
        period elapsed with the commit open."""
        surface, _stepper, client, _ = _attached(rig, monkeypatch, saved=(40, "run-x:step-40"))
        seen: list[str] = []
        real = surface._op_status

        def watching(body):  # type: ignore[no-untyped-def]
            response = real(body)
            seen.append(response.json()["weights_state"])
            return response

        surface._op_status = watching  # type: ignore[method-assign]
        try:
            with pytest.raises(Refusal):
                drive(rig, max_steps=3)
        finally:
            client.close()
        assert "writing" in seen, "the set was still writing before the watchdog fired"
        assert seen[-1] == "torn", "recovery acted only once miLLM reported torn"

    def test_stop_when_there_is_no_checkpoint_to_roll_back_to(self, rig, monkeypatch) -> None:
        surface, _stepper, client, _ = _attached(rig, monkeypatch, saved=None)
        try:
            with pytest.raises(Refusal) as caught:
                drive(rig, max_steps=3)
        finally:
            client.close()
        details = caught.value.details
        assert details["recovery_path"] == "stop"
        assert details["watchdog_declared_torn"] is True
        assert surface.call_count("commit-end") == 0
        assert surface.call_count("load") == 0
        assert surface.weights_state == "torn", "nothing blessed or replaced the torn set"

    def test_no_recovery_against_a_set_the_watchdog_never_tore(self, rig, monkeypatch) -> None:
        """If miLLM never reports torn, miForge does not act on an unknown state."""
        surface, stepper, client, _ = _attached(
            rig, monkeypatch, saved=(40, "run-x:step-40"), watchdog_ms=None
        )
        stepper.timings.watchdog_ms = 50.0  # bound the wait for the test
        try:
            with pytest.raises(Refusal) as caught:
                drive(rig, max_steps=3)
        finally:
            client.close()
        details = caught.value.details
        assert details["recovery_path"] == "stop"
        assert details["watchdog_declared_torn"] is False
        assert surface.call_count("load") == 0
        assert surface.call_count("commit-end") == 0


class TestEveryOtherExitIsUnchanged:
    def test_a_successful_write_still_commits_and_resumes(self, rig, monkeypatch) -> None:
        surface, _stepper, client, optimizer = _attached(rig, monkeypatch, saved=None)
        optimizer.fail_on_call = 0  # never fails
        try:
            result = drive(rig, max_steps=2)
        finally:
            client.close()
        assert result.steps_completed == 2
        assert surface.call_count("commit-end") == 2
        assert surface.call_count("resume") == 2
        assert surface.weights_state == "clean"
