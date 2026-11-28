"""R-02.30 / R-02.31 recovery, in the production loop  (F11 FR-011.36 - FR-011.45).

Until 2026-10-16 ``training/hotupdate/recovery.py`` decided COMPLETE / ROLLBACK / STOP and
nothing in the run asked it: an interrupted commit raised out of the loop with the serving
set left ``writing`` or ``torn``. These tests drive ``tasks.drive_run`` with the real
``AttachedStepper`` over the real client and the fake serving surface, and assert the wire
sequence, the count of optimizer writes (the step is never re-applied) and the record.
"""

# ruff: noqa: F811 - `rig` is a pytest fixture imported by name; parameters receive it.

from __future__ import annotations

from collections.abc import Iterator
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


class Saver:
    """Stands in for ``save.AttachedSaver``: only ``expected`` and ``last_saved`` are read."""

    def __init__(self, last_saved: tuple[int, str] | None) -> None:
        self.expected = None
        self.last_saved = last_saved


def _attached(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *, saved=None, retries: int = 1
):  # type: ignore[no-untyped-def]
    surface = FakeWeightsSurface()
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
        StepTimings(watchdog_ms=200.0, creep_fraction=0.5, run_id=RUN),
        expected=state.identity_from_attach(response),
        saver=Saver(saved),
        precommit_retries=retries,
    )
    monkeypatch.setattr(
        tasks,
        "acquire_for_run",
        lambda spec: AcquiredModel(rig["model"], rig["tokenizer"], RECORD, stepper=stepper),
    )
    return surface, stepper, client


@pytest.fixture
def closing() -> Iterator[list[Any]]:
    clients: list[Any] = []
    yield clients
    for client in clients:
        client.close()


class TestARefusalBeforeCommitBeginIsRetried:
    def test_one_precommit_refusal_is_retried_and_the_run_completes(
        self, rig, monkeypatch, closing, caplog
    ) -> None:
        """BRD AC7 / F11 14.7's CPU half: nothing was written, so the step is retried - and
        the refusal is NAMED (live 14.7 logged it `weights_refusal_unmapped`)."""
        surface, stepper, client = _attached(rig, monkeypatch)
        closing.append(client)
        surface.refuse_once["commit-begin"] = (409, "NOT_QUIESCED")
        with caplog.at_level("WARNING"):
            result = drive(rig, max_steps=2)
        retries = [
            r for r in caplog.records if getattr(r, "code", None) == "hotupdate_precommit_retry"
        ]
        assert [getattr(r, "upstream", None) for r in retries] == ["weights_not_quiesced"]
        assert result.steps_completed == 2
        assert stepper.precommit_retries_taken == 1
        assert rig["optimizer"].steps == 2, "the refused attempt wrote nothing"
        assert surface.call_count("commit-begin") == 3

    def test_without_retries_the_refusal_ends_the_run(self, rig, monkeypatch, closing) -> None:
        surface, stepper, client = _attached(rig, monkeypatch, retries=0)
        closing.append(client)
        surface.refuse_once["commit-begin"] = (409, "NOT_QUIESCED")
        with pytest.raises(Refusal):
            drive(rig, max_steps=2)
        assert rig["optimizer"].steps == 0
        assert stepper.precommit_retries_taken == 0


class TestAnInterruptedCommitIsNeverRetried:
    def test_complete_a_late_commit_end_then_stop_with_the_record(
        self, rig, monkeypatch, closing
    ) -> None:
        """R-02.30(a): the same process sends a late commit-end and the set is clean."""
        surface, stepper, client = _attached(rig, monkeypatch, saved=(1, "ckpt-step-1"))
        closing.append(client)
        surface.refuse_once["commit-end"] = (503, "WEIGHTS_TORN")
        with pytest.raises(Refusal) as caught:
            drive(rig, max_steps=3)
        refusal = caught.value
        assert refusal.code == "hotupdate_step_abandoned"
        assert refusal.details["recovery_path"] == "complete"
        assert refusal.details["retried"] is False
        assert refusal.details["discontinuity"] is True
        assert refusal.details["step"] == 1
        assert rig["optimizer"].steps == 1, "the interrupted update is applied once, never again"
        assert surface.call_count("commit-end") == 2  # the failed one and the late one
        assert surface.call_count("load") == 0
        assert surface.weights_state == "clean"
        assert len(stepper.recoveries) == 1

    def test_rollback_loads_the_last_checkpoint_when_completion_fails(
        self, rig, monkeypatch, closing
    ) -> None:
        """R-02.30(b): load the last saved checkpoint by name (BRD-01 R-01.25)."""
        surface, _stepper, client = _attached(rig, monkeypatch, saved=(40, "run-x:step-40"))
        closing.append(client)
        surface.refuse_always["commit-end"] = (503, "WEIGHTS_TORN")
        with pytest.raises(Refusal) as caught:
            drive(rig, max_steps=3)
        details = caught.value.details
        assert details["recovery_path"] == "rollback"
        assert details["resumed_from_step"] == 40
        assert "holds_the_computed_update" in details["completion_unavailable_because"]
        assert surface.bodies_of("load") == [{"name": "run-x:step-40"}]
        assert surface.weights_state == "clean"
        assert rig["optimizer"].steps == 1

    def test_stop_when_nothing_can_be_rolled_back_to(self, rig, monkeypatch, closing) -> None:
        surface, _stepper, client = _attached(rig, monkeypatch, saved=None)
        closing.append(client)
        surface.refuse_always["commit-end"] = (503, "WEIGHTS_TORN")
        with pytest.raises(Refusal) as caught:
            drive(rig, max_steps=3)
        details = caught.value.details
        assert details["recovery_path"] == "stop"
        assert details["operator_action_required"]
        assert surface.call_count("load") == 0
        assert rig["optimizer"].steps == 1
