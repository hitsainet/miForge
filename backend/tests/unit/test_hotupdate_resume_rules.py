"""The rules a resume is held to, one at a time  (D52; F11 FTASKS 18.3).

The end-to-end path is ``tests/integration/test_hotupdate_resume.py``; this file pins each
rule on its own so a mutation of one line names the rule it broke.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from core.canonical import write_canonical
from core.config import Settings
from core.errors import Refusal
from models.run import Run
from services.checkpoints import manifest as manifest_module
from services.checkpoints import resume as resume_module
from services.checkpoints.paths import checkpoint_dir
from tests.support.jobs import insert_run, memory_engine, session_for
from training import run_log, run_state
from training.acquisition.base import AcquisitionMode, AcquisitionSpec, ResumeFrom
from training.hotupdate import resume

NAME = "LiquidAI/LFM2-1.2B:run-r:step-3"
HASH = "a" * 64


def _expected(config: dict[str, Any] | None = None) -> resume.Expected:
    return resume.Expected(
        run_id="run-r", step=3, name=NAME, manifest_file_sha256=HASH, config=config
    )


def _check(**overrides: Any) -> resume.Verified:
    kwargs: dict[str, Any] = {
        "model_block": {"model_id": 7, "name": NAME, "manifest_hash": HASH},
        "steps_since_load": 0,
        "config": None,
        "stage": "attach",
    }
    kwargs.update(overrides)
    expected = kwargs.pop("expected", _expected())
    return resume.check_served(expected, **kwargs)


def _fields(caught: pytest.ExceptionInfo[Refusal]) -> list[str]:
    assert caught.value.code == resume.MISMATCH
    return [m["field"] for m in caught.value.details["mismatches"]]


class TestTheServedModelIsTheCheckpoint:
    def test_a_match_on_every_field_passes_and_says_what_it_proved(self) -> None:
        verified = _check()
        assert verified.record() == {
            "run_id": "run-r",
            "step": 3,
            "checkpoint_name": NAME,
            "manifest_file_sha256": HASH,
            "stage": "attach",
            "config_keys_compared": 0,
        }

    def test_the_name_must_match(self) -> None:
        with pytest.raises(Refusal) as caught:
            _check(model_block={"name": "other", "manifest_hash": HASH})
        assert _fields(caught) == ["name"]

    def test_the_manifest_hash_must_match(self) -> None:
        with pytest.raises(Refusal) as caught:
            _check(model_block={"name": NAME, "manifest_hash": "b" * 64})
        assert _fields(caught) == ["manifest_hash"]

    def test_a_null_hash_is_a_statement_not_a_wildcard(self) -> None:
        """R-01.1: null means "not loaded from a checkpoint manifest" - never a match."""
        with pytest.raises(Refusal) as caught:
            _check(model_block={"name": NAME, "manifest_hash": None})
        assert _fields(caught) == ["manifest_hash"]

    @pytest.mark.parametrize("count", [1, 100, None, True, "0"])
    def test_any_commit_since_load_refuses_even_when_name_and_hash_match(self, count) -> None:
        with pytest.raises(Refusal) as caught:
            _check(steps_since_load=count)
        assert _fields(caught) == ["steps_since_load"]

    def test_every_mismatch_is_named_at_once(self) -> None:
        with pytest.raises(Refusal) as caught:
            _check(model_block={}, steps_since_load=4)
        assert _fields(caught) == ["name", "manifest_hash", "steps_since_load"]
        details = caught.value.details
        assert details["operator_action"] == {
            "call": "POST /api/weights/load",
            "body": {"name": NAME},
            "identity": "operator",
        }
        assert details["miforge_can_perform_it"] is False
        assert "before any optimizer state was restored" in caught.value.message


class TestTheConfigComparison:
    SAVED: ClassVar[dict[str, Any]] = {
        "model_type": "llama",
        "hidden_size": 8,
        "transformers_version": "4.57.0",
    }

    def test_keys_both_state_must_agree(self) -> None:
        with pytest.raises(Refusal) as caught:
            _check(
                expected=_expected(self.SAVED),
                config={"model_type": "llama", "hidden_size": 16},
            )
        [mismatch] = caught.value.details["mismatches"]
        assert mismatch["differences"] == [{"key": "hidden_size", "served": 16, "saved": 8}]

    def test_serialisation_metadata_is_not_architecture(self) -> None:
        served = {"model_type": "llama", "hidden_size": 8, "transformers_version": "9.9.9"}
        assert _check(expected=_expected(self.SAVED), config=served).config_keys_compared == 2

    def test_a_key_the_diff_form_omits_is_not_a_disagreement(self) -> None:
        """``config.json`` is save_pretrained's difference-from-defaults form."""
        served = {"model_type": "llama", "hidden_size": 8, "rope_theta": 10000.0}
        _check(expected=_expected(self.SAVED), config=served)

    def test_a_different_model_type_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            _check(expected=_expected(self.SAVED), config={"model_type": "lfm2", "hidden_size": 8})
        [mismatch] = caught.value.details["mismatches"]
        assert mismatch["differences"][0]["key"] == "model_type"


class TestTheSpecField:
    def test_it_names_run_and_step_or_a_path_never_both(self) -> None:
        ResumeFrom(run_id="r", step=3)
        ResumeFrom(path=Path("/x"))
        for bad in ({"run_id": "r"}, {"step": 3}, {}, {"run_id": "r", "step": 3, "path": "/x"}):
            with pytest.raises(ValidationError):
                ResumeFrom(**bad)

    @pytest.mark.parametrize(
        "source",
        [{"mode": "load", "path": "/data/checkpoints/x"}, {"mode": "create", "config": {}}],
    )
    def test_only_an_attach_takes_it(self, source: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="resume_from"):
            AcquisitionSpec(**source, resume_from={"run_id": "r", "step": 3})

    def test_an_attach_does(self) -> None:
        spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH, resume_from={"run_id": "r", "step": 3})
        assert spec.resume_from == ResumeFrom(run_id="r", step=3)


# --------------------------------------------------------------------- the generic half


def _checkpoint(root: Path, run_id: str = "run-r", step: int = 3) -> Path:
    directory = checkpoint_dir(run_id, step, root=root)
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"weights")
    (directory / "config.json").write_text(json.dumps({"model_type": "llama"}))
    doc = manifest_module.build(
        run_id=run_id,
        step=step,
        domain=None,
        signal_source="corpus",
        base_model="LiquidAI/LFM2-1.2B",
        parent=None,
        optimizer={"name": "adamw-32bit", "state": "fresh", "config_hash": None},
        weights_sha256=hashlib.sha256(b"weights").hexdigest(),
    )
    write_canonical(directory / manifest_module.MANIFEST_NAME, doc)
    return directory


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    ours, millm, states = tmp_path / "ours", tmp_path / "millm", tmp_path / "states"
    for directory in (ours, millm, states):
        directory.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            checkpoint_root=ours,
            checkpoint_read_roots=str(millm),
            trainer_state_root=states,
        ),
    )
    return {"ours": ours, "millm": millm, "states": states}


class TestResolvingTheCheckpoint:
    def test_by_run_and_step_through_the_read_roots(self, roots) -> None:
        directory = _checkpoint(roots["millm"])
        target = resume_module.resolve_resume_from(ResumeFrom(run_id="run-r", step=3))
        assert target.directory == directory
        assert target.name == "LiquidAI/LFM2-1.2B:run-r:step-3"
        # The FILE's bytes, which is what miLLM hashes - not the canonical digest.
        assert (
            target.manifest_file_sha256
            == hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        )
        assert target.config == {"model_type": "llama"}

    def test_by_path_when_it_is_that_runs_canonical_directory(self, roots) -> None:
        directory = _checkpoint(roots["millm"])
        target = resume_module.resolve_resume_from(ResumeFrom(path=directory))
        assert (target.run_id, target.step) == ("run-r", 3)

    def test_a_path_outside_every_checkpoint_root_is_refused(self, roots, tmp_path) -> None:
        stray = _checkpoint(tmp_path / "elsewhere")
        with pytest.raises(Refusal) as caught:
            resume_module.resolve_resume_from(ResumeFrom(path=stray))
        assert caught.value.code == "resume_from_outside_roots"

    def test_a_missing_checkpoint_is_refused_by_name(self, roots) -> None:
        with pytest.raises(Refusal) as caught:
            resume_module.resolve_resume_from(ResumeFrom(run_id="run-r", step=9))
        assert caught.value.code == "checkpoint_not_found"


class TestOnlyFromTheLatestArtifact:
    def test_a_later_checkpoint_refuses(self, roots) -> None:
        _checkpoint(roots["millm"], step=3)
        _checkpoint(roots["millm"], step=6)
        with pytest.raises(Refusal) as caught:
            resume_module.refuse_unless_latest("run-r", 3)
        assert caught.value.code == "resume_not_from_latest"
        assert caught.value.details["checkpoints"] == [6]

    def test_a_later_trainer_state_refuses(self, roots) -> None:
        from schemas.trainer_state import REQUIRED_FILES

        _checkpoint(roots["millm"], step=3)
        later = roots["states"] / "run-r" / "step-000006"
        later.mkdir(parents=True)
        for name in REQUIRED_FILES:
            (later / name).write_bytes(b"x")
        with pytest.raises(Refusal) as caught:
            resume_module.refuse_unless_latest("run-r", 3)
        assert caught.value.details["trainer_states"] == [6]

    def test_the_latest_passes(self, roots) -> None:
        _checkpoint(roots["millm"], step=3)
        resume_module.refuse_unless_latest("run-r", 3)


class TestTheIndexKeepsTheLaterRecord:
    def test_a_rewritten_step_appears_once_with_its_later_values(self, tmp_path: Path) -> None:
        path = run_log.run_log_path(tmp_path, "run-r")
        for step, loss in ((1, 5.0), (2, 4.0), (3, 3.0), (2, 2.5), (3, 2.0)):
            run_log.append(
                path,
                run_log.step_block(
                    run_id="run-r",
                    step=step,
                    loss=loss,
                    objective="sft",
                    learning_rate=1e-3,
                    wall_time_ms=1.0,
                ),
            )
        rows = run_log.step_rows(path)
        assert [(r["step"], r["loss"]) for r in rows] == [(1, 5.0), (2, 2.5), (3, 2.0)]


class TestReopeningARun:
    @pytest.fixture
    def session(self):
        with session_for(memory_engine()) as session:
            yield session

    def test_a_failed_run_becomes_pending_and_its_ending_is_returned(self, session) -> None:
        insert_run(session, "run-r", status="running")
        run_state.mark_failed(session, "run-r", code="heartbeat_lapsed", message="lost")
        previous = run_state.reopen_for_continue(session, "run-r")
        assert previous["status"] == "failed"
        assert previous["failure_code"] == "heartbeat_lapsed"
        row = session.execute(sa.select(Run).where(Run.id == "run-r")).scalar_one()
        assert (row.status, row.finished_at, row.failure_code, row.failure_message) == (
            "pending",
            None,
            None,
            None,
        )

    def test_a_completed_run_can_be_continued_too(self, session) -> None:
        insert_run(session, "run-r", status="running")
        run_state.mark_completed(session, "run-r", stop_reason="operator_stop")
        assert run_state.reopen_for_continue(session, "run-r")["stop_reason"] == "operator_stop"

    @pytest.mark.parametrize("status", ["pending", "running"])
    def test_a_live_run_is_not_continued(self, session, status: str) -> None:
        insert_run(session, "run-r", status=status)
        with pytest.raises(Refusal) as caught:
            run_state.reopen_for_continue(session, "run-r")
        assert caught.value.code == "run_not_continuable"
