"""A crashed ATTACHED run continues from the checkpoint miLLM saved for it  (D52; F11 18.3).

**Through the production path, end to end, with no stand-in for any wiring point:**

1. ``POST /api/v1/runs/{id}/continue {"submit": true}`` - the real route, the real
   :func:`services.run_continue.submit_continue`, the real ``run_launch.launch`` and the real
   validator's budget step, which runs the **real** :class:`AttachAcquirer` pre-flight
   (reached through the acquisition registry, not called directly) against the
   conformance-checked fake miLLM surface. The pack gate, the part resolver and the service
   check are stubbed - they are other features' and need part rows this test has no reason
   to build.
2. the worker - :func:`training.run_driver.drive_recorded_run`, reading the run log's NEW
   definition block - into :func:`training.tasks.drive_run`, the real ``acquire_for_run``,
   the real attach + identity check on the attach response, the real bracketed stepper, the
   real optimizer restore through F1's trainer-state reader, and a real ``sft`` step over a
   real torch module.

What is doubled, and why: the CUDA-IPC rebuild (``rebuild.build`` / ``build_module``) - it
needs a card - returns a tiny torch module holding the checkpoint's weights; R-02.2's
verifier is off (its own wiring is pinned by ``test_hotupdate_verify_wiring.py``); the data
cursor is handed to the source, because **the product does not restore the data position**
(FTDD §6.9 records it; the continuity test makes the same choice and says why).

**The assertion that matters is the loss curve**: the resumed steps 4-6 rejoin an
uninterrupted run's curve within ``1e-5``, which is only true if the restored optimizer
state reached the step. With the restore call removed the curve jumps by ~0.4 - asserted
by :class:`TestTheControls`, so the window can never be narrowed to a seam that cannot see
it.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.canonical import sha256_of_file, write_canonical
from core.config import reset_settings
from core.errors import Refusal
from models.run import Run, Step
from services import run_definition as validator
from services.checkpoints import manifest as manifest_module
from services.checkpoints.paths import checkpoint_dir
from tests.support.checkpoints import create_feature_tables
from tests.support.fake_weights_surface import FakeWeightsSurface
from tests.support.fixtures_hotupdate import configure
from tests.support.jobs import memory_engine, session_for
from training import acquisition, run_driver, run_log, run_state, tasks, trainer_state
from training import optimizer as optimizer_module
from training.acquisition.base import AcquisitionMode

from .test_continuity_across_a_resume import (
    CONTINUITY_TOLERANCE,
    DISCRIMINATION_FACTOR,
    _cfg,
    _fixed_init_model,
    _FixedCycleSource,
    _take_steps,
)

RUN = "f11-resume"
SAVE_AT = 3
LAST_LOGGED = 5  # the crashed attempt recorded steps 4 and 5 after its step-3 save
TOTAL = 6
BASE = "LiquidAI/LFM2-1.2B"
# `num_attention_heads` is stated: transformers 5 VALIDATES a config on load (hidden_size
# must divide by the head count, default 32), and `AutoTokenizer` reads `config.json` first.
CONFIG = {"model_type": "llama", "hidden_size": 8, "num_attention_heads": 2, "num_hidden_layers": 1}


def _definition() -> dict[str, Any]:
    cfg = _cfg()
    return {
        "run_id": RUN,
        "model_source": {"mode": "attach"},
        "run_type": "change",
        "run_configuration": {
            "learning_rate": cfg.learning_rate,
            "warmup_steps": cfg.warmup_steps,
            "batch_size": cfg.batch_size,
            "gradient_accumulation": 1,
            "optimizer_state_precision": "32bit",
        },
        "signal_source": {"source": "corpus"},
        "parts": [{"part_id": "part-corpus", "kind": "corpus"}],
        "cadences": {"test_every": 100, "retention_every": 100, "save_every": SAVE_AT},
        "stop_conditions": [{"condition": "step_count", "config": {"total": TOTAL}}],
        "domain": "legal",
        "seed": cfg.seed,
    }


class _PackDouble:
    class _Definition:
        domain = "legal"

    definition = _Definition()

    def payload(self) -> dict[str, Any]:
        return {"pack_id": "legal", "version": "1", "retention_floor": None, "parts": []}

    def part_records(self) -> list[Any]:
        return []


class _Base:
    device_uuid = "GPU-test"
    tensors: ClassVar[dict[str, Any]] = {}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """The state an attached run leaves when it is killed two steps after its step-3 save.

    * miLLM's checkpoint for step 3 under ITS directory, mounted read-only here (D47) -
      real safetensors weights, a manifest built and hashed by F5's production code;
    * miForge's trainer state for step 3, written by F1's real writer from a real AdamW;
    * a run log holding the definition and steps 1..5, the row ``failed``;
    * a fake miLLM whose resident model the OPERATOR LOADED from that checkpoint:
      name = the checkpoint's, manifest hash = sha256 of its manifest bytes, 0 commits.
    """
    states = tmp_path / "trainer-state"
    ours = tmp_path / "checkpoints"
    millm = tmp_path / "millm-checkpoints"
    for directory in (states, ours, millm):
        directory.mkdir()
    configure(
        monkeypatch,
        CHECKPOINT_ROOT=str(ours),
        TRAINER_STATE_ROOT=str(states),
        CHECKPOINT_READ_ROOTS=str(millm),
    )

    # ---- the uninterrupted reference, and the first attempt up to its save --------------
    reference_model = _fixed_init_model()
    reference_opt, _ = optimizer_module.build(reference_model, _cfg())
    reference = _take_steps(reference_model, reference_opt, TOTAL, from_step=0)

    model = _fixed_init_model()
    optimizer, identity = optimizer_module.build(model, _cfg())
    _take_steps(model, optimizer, SAVE_AT, from_step=0)
    trainer_state.write(
        run_id=RUN, step=SAVE_AT, optimizer=optimizer, identity=identity, root=states
    )
    weights_at_save = copy.deepcopy(model.state_dict())

    # ---- miLLM's checkpoint of step 3, as its save writes it ---------------------------
    from safetensors.torch import save_file

    directory = checkpoint_dir(RUN, SAVE_AT, root=millm)
    directory.mkdir(parents=True)
    save_file(
        {k: v.contiguous() for k, v in weights_at_save.items()},
        str(directory / "model.safetensors"),
    )
    (directory / "config.json").write_text(__import__("json").dumps(CONFIG))
    # miLLM's save writes the SERVING tokenizer at the checkpoint's root (R-01.10); the
    # resumed run's tokenizer is read from here by the REAL loader (F11 window 3's defect).
    from tests.support.tiny_hf_model import write_tiny_tokenizer

    write_tiny_tokenizer(directory)
    doc = manifest_module.build(
        run_id=RUN,
        step=SAVE_AT,
        domain="legal",
        signal_source="corpus",
        base_model=BASE,
        parent=None,
        optimizer=identity.as_manifest_block(),
        weights_sha256=sha256_of_file(directory / "model.safetensors"),
    )
    write_canonical(directory / manifest_module.MANIFEST_NAME, doc)
    manifest_hash = hashlib.sha256(
        (directory / manifest_module.MANIFEST_NAME).read_bytes()
    ).hexdigest()

    # ---- the database, the run log, the crashed attempt ---------------------------------
    engine = memory_engine()
    create_feature_tables(engine)
    stubs = {
        "gate": lambda *a, **k: _PackDouble(),
        "resolve_parts": lambda *a, **k: {"source": "parts", "parts": []},
        "services": lambda *a, **k: {"needed": []},
    }
    for name, fn in (
        ("_live_gate", stubs["gate"]),
        ("_live_parts", stubs["resolve_parts"]),
        ("_live_services", stubs["services"]),
    ):
        monkeypatch.setattr(validator, name, fn)
    from services import acquisition as acquisition_service

    monkeypatch.setattr(acquisition_service, "resolve_submit_device", lambda request: None)

    surface = FakeWeightsSurface(
        resident_model=str(doc["name"]),
        manifest_sha256=manifest_hash,
        steps_since_load=0,
    )
    surface.config = dict(CONFIG)
    from training.hotupdate import attach as attach_module
    from training.hotupdate import rebuild

    acquirer = attach_module.AttachAcquirer(transport=surface.transport())
    # THROUGH THE REGISTRY: `acquire_for_run` and the validator's pre-flight both resolve
    # the attach mode by name, so the acquirer under test is the one production reaches.
    monkeypatch.setitem(acquisition._REGISTRY, AcquisitionMode.ATTACH, acquirer)
    resumed_model = _fixed_init_model()
    resumed_model.load_state_dict(weights_at_save)
    monkeypatch.setattr(rebuild, "build", lambda response, **kw: _Base())
    monkeypatch.setattr(rebuild, "check_build_identity", lambda response: None)
    monkeypatch.setattr(attach_module, "build_module", lambda response, base: resumed_model)
    # NOT injected: the real `_tokenizer_loader` runs, with the real `from_pretrained`
    # observed (and called through) so the source it received is asserted. Until 2026-10-18
    # this was replaced by `lambda: None`, which is why F11 window 3 was the first thing ever
    # to run the line that handed `from_pretrained` the served `<base>:<run>:step-N` name.
    import transformers

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    real_from_pretrained = transformers.AutoTokenizer.from_pretrained.__func__
    tokenizer_sources: list[str] = []

    def observed(cls: Any, source: Any, *args: Any, **kwargs: Any) -> Any:
        tokenizer_sources.append(str(source))
        return real_from_pretrained(cls, source, *args, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(observed))
    monkeypatch.setattr(attach_module.AttachAcquirer, "_verifier", lambda self, *a: None)

    enqueued: list[str] = []

    class _Task:
        @staticmethod
        def delay(run_id: str) -> None:
            enqueued.append(run_id)

    monkeypatch.setattr(tasks, "run_step_loop", _Task)

    from schemas.run_definition import RunDefinition
    from services import run_launch

    with session_for(engine) as session:
        # The first attempt: defined (real launch, budget stubbed - the resident model
        # was the base then), started, logged steps 1..5, then reaped.
        run_launch.launch(
            RunDefinition(**_definition()),
            session=session,
            enqueue=lambda run_id: None,
            budget=lambda *a, **k: {"param_count": 10, "gpu_uuid": None, "mode": "attach"},
            **stubs,
        )
        run_state.mark_running(session, RUN)
        log_path = run_log.run_log_path(tasks.run_log_root(), RUN)
        for step in range(1, LAST_LOGGED + 1):
            run_log.append(
                log_path,
                run_log.step_block(
                    run_id=RUN,
                    step=step,
                    loss=9.0 + step,  # the crashed attempt's numbers; superseded after 3
                    objective="sft",
                    learning_rate=0.05,
                    wall_time_ms=1.0,
                ),
            )
        run_state.mark_failed(session, RUN, code="heartbeat_lapsed", message="worker lost")
        session.commit()

    yield {
        "engine": engine,
        "surface": surface,
        "acquirer": acquirer,
        "reference": reference,
        "enqueued": enqueued,
        "states": states,
        "millm": millm,
        "checkpoint": directory,
        "doc": doc,
        "manifest_hash": manifest_hash,
        "log_path": log_path,
        "resumed_model": resumed_model,
        "monkeypatch": monkeypatch,
        "tokenizer_sources": tokenizer_sources,
    }
    reset_settings()


def _client(world: dict[str, Any]) -> TestClient:
    app = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(world["engine"]) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    return TestClient(app, raise_server_exceptions=False)


def _submit(world: dict[str, Any], **body: Any) -> Any:
    return _client(world).post(f"/api/v1/runs/{RUN}/continue", json={"submit": True, **body})


def _mismatch(response: Any) -> dict[str, Any]:
    """The served-model refusal, as the submission reports it.

    A continue is a submission, so it answers in the validator's envelope - every failure
    listed (FR-006.59) - and the attach pre-flight's refusal is one named entry of it.
    """
    error = response.json()["error"]
    assert error["code"] == "run_definition_invalid", error
    [failure] = [f for f in error["details"]["failures"] if f["step"] == "budget"]
    assert failure["code"] == "resume_served_model_mismatch", failure
    return dict(failure["details"])


def _work(world: dict[str, Any]) -> dict[str, Any]:
    """The worker's half: the production driver over the recorded definition."""
    with session_for(world["engine"]) as session:

        @contextmanager
        def scope() -> Iterator[Any]:
            yield session
            session.flush()

        return run_driver.drive_recorded_run(
            RUN,
            session=session,
            session_scope=scope,
            batch_source=_FixedCycleSource(cursor=SAVE_AT),
            objectives=None,
            evaluate=lambda step: None,
        )


def _resumed_losses(world: dict[str, Any]) -> dict[int, float]:
    blocks = list(run_log.read_blocks(world["log_path"], block="step"))
    after_reopen = blocks[LAST_LOGGED:]  # the new attempt's records, in written order
    return {int(b["step"]): float(b["loss"]) for b in after_reopen}


class TestAnAttachedRunContinuesFromItsCheckpoint:
    def test_the_route_submits_and_records_the_discontinuity(self, world) -> None:
        response = _submit(world)
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["submitted"] is True
        assert body["plan"]["step"] == SAVE_AT
        assert body["plan"]["optimizer"]["state"] == "restored"
        assert body["resume"]["superseded_steps"] == {"first": SAVE_AT + 1, "last": LAST_LOGGED}
        assert body["resume"]["previous_attempt"]["status"] == "failed"
        assert body["resume"]["previous_attempt"]["failure_code"] == "heartbeat_lapsed"
        # Payload AND count: one enqueue, for this run.
        assert world["enqueued"] == [RUN]
        # The run log's LAST definition block is what the worker reads: resume_from in the
        # submitted model source, the discontinuity in the resolved block.
        definitions = list(run_log.read_blocks(world["log_path"], block="definition"))
        assert len(definitions) == 2
        last = definitions[-1]
        assert last["submitted"]["model_source"] == {
            "mode": "attach",
            "resume_from": {"run_id": RUN, "step": SAVE_AT},
        }
        assert last["resolved"]["resume"]["resumed_from"]["manifest_file_sha256"] == (
            world["manifest_hash"]
        )
        with session_for(world["engine"]) as session:
            row = session.execute(sa.select(Run).where(Run.id == RUN)).scalar_one()
            assert (row.status, row.finished_at, row.failure_code) == ("pending", None, None)
        # The pre-flight read the SERVED model (status) and attached nothing.
        assert world["surface"].call_count("status") >= 1
        assert world["surface"].call_count("attach") == 0

    def test_the_worker_resumes_at_step_plus_one_and_the_curve_is_continuous(self, world) -> None:
        assert _submit(world).status_code == 202
        result = _work(world)
        assert result["status"] == "completed", result
        losses = _resumed_losses(world)
        assert sorted(losses) == [4, 5, 6], "a resume numbers its steps from step + 1"
        reference = world["reference"]
        gap = max(abs(losses[s] - reference[s - 1]) for s in (4, 5, 6))
        assert gap <= CONTINUITY_TOLERANCE, (
            f"the resumed curve left the uninterrupted one by {gap:.3e}: the optimizer state "
            f"restored from step {SAVE_AT} did not reach the step"
        )
        # The attach response was checked, and the acquisition carried the proof.
        verified = world["acquirer"].resume_verified
        assert verified is not None and verified.stage == "attach"
        assert verified.config_keys_compared >= 2
        # Every resumed step was bracketed on the wire.
        assert world["surface"].call_count("commit-begin") == TOTAL - SAVE_AT
        # The serving tokenizer came from the checkpoint directory being resumed - ONCE, and
        # never from the served `<base>:<run>:step-N` name (F11 window 3, HFValidationError).
        assert world["tokenizer_sources"] == [str(world["checkpoint"])]

    def test_the_index_holds_one_row_per_step_the_later_record_winning(self, world) -> None:
        assert _submit(world).status_code == 202
        _work(world)
        from training.step_record import rebuild_step_index

        with session_for(world["engine"]) as session:
            rebuild_step_index(session, log_root=tasks.run_log_root(), run_id=RUN)
            rows = session.execute(
                sa.select(Step.step, Step.loss).where(Step.run_id == RUN).order_by(Step.step)
            ).all()
        assert [r.step for r in rows] == [1, 2, 3, 4, 5, 6]
        by_step = {r.step: r.loss for r in rows}
        assert by_step[4] != pytest.approx(9.0 + 4), "the superseded record won"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def reopen_log() -> Iterator[_Records]:
    """Records from BOTH modules that could log the reopen, held on the loggers themselves:
    the app's logging setup stops propagation, so ``caplog`` sees nothing here (checked - it
    captured zero records, which would have made the refused case pass vacuously)."""
    handler = _Records()
    loggers = [
        logging.getLogger(name)
        for name in ("services.run_continue", "training.run_state", "api.app")
    ]
    for logger in loggers:
        logger.addHandler(handler)
    yield handler
    for logger in loggers:
        logger.removeHandler(handler)


class TestTheReopenIsLoggedOnlyWhenItHappened:
    """F11 window 5: a REFUSED continue logged ``run_reopened_for_continue`` while its
    transaction rolled the reopen back and the row stayed terminal - a log line announcing a
    transition that never happened. Asserted with the payload and the count."""

    @staticmethod
    def _reopens(handler: _Records) -> list[Any]:
        return [
            r for r in handler.records if getattr(r, "code", None) == "run_reopened_for_continue"
        ]

    def test_a_refused_continue_logs_no_reopen(self, world, reopen_log) -> None:  # type: ignore[no-untyped-def]
        world["surface"].steps_since_load = 2
        _mismatch(_submit(world))
        # The handler is live: it saw the route's own refusal, so the empty list is real.
        codes = [getattr(r, "code", None) for r in reopen_log.records]
        assert "run_definition_invalid" in codes, f"the handler saw nothing useful: {codes}"
        assert self._reopens(reopen_log) == []

    def test_an_accepted_continue_logs_it_once_with_the_previous_attempt(  # type: ignore[no-untyped-def]
        self, world, reopen_log
    ) -> None:
        assert _submit(world).status_code == 202
        [record] = self._reopens(reopen_log)
        assert record.run_id == RUN
        assert record.previous["status"] == "failed"
        assert record.previous["failure_code"] == "heartbeat_lapsed"


class TestItRefusesOntoOtherWeights:
    def _run_row(self, world) -> str:
        with session_for(world["engine"]) as session:
            return str(session.execute(sa.select(Run.status).where(Run.id == RUN)).scalar_one())

    def test_a_resident_set_with_commits_since_load_is_refused_at_submit(self, world) -> None:
        """The live case: killed after its step-3 save, the set is NAMED step 3 and carries
        later commits. Name and hash match; only the counter can tell."""
        world["surface"].steps_since_load = 2
        details = _mismatch(_submit(world))
        assert [m["field"] for m in details["mismatches"]] == ["steps_since_load"]
        assert details["stage"] == "preflight"
        assert details["operator_action"]["body"] == {"name": world["doc"]["name"]}
        assert details["miforge_can_perform_it"] is False
        assert world["enqueued"] == []
        assert self._run_row(world) == "failed", "a refused continue must not reopen the row"
        assert world["surface"].call_count("load") == 0, "miForge does not load it itself"

    def test_a_different_manifest_hash_is_refused(self, world) -> None:
        world["surface"].manifest_sha256 = "f" * 64
        details = _mismatch(_submit(world))
        assert [m["field"] for m in details["mismatches"]] == ["manifest_hash"]
        assert world["enqueued"] == []

    def test_a_different_model_name_is_refused(self, world) -> None:
        world["surface"].resident_model = "LiquidAI/LFM2-1.2B"
        details = _mismatch(_submit(world))
        assert [m["field"] for m in details["mismatches"]] == ["name"]

    def test_a_change_between_submit_and_attach_is_refused_at_attach_before_any_restore(
        self, world
    ) -> None:
        """The attach response is checked too - it is the one whose handles are trained."""
        assert _submit(world).status_code == 202
        surface = world["surface"]
        original = surface._op_attach

        def commits_happened(body: Any) -> Any:
            surface.steps_since_load = 7  # someone trained the resident set meanwhile
            return original(body)

        world["monkeypatch"].setattr(surface, "_op_attach", commits_happened)
        restores: list[Any] = []
        world["monkeypatch"].setattr(
            tasks, "restore_optimizer_state", lambda *a, **k: restores.append(a)
        )
        # The pre-flight (status) must still pass, so only the attach-stage check can fire.
        surface_status = surface._op_status

        def clean_status(body: Any) -> Any:
            saved = surface.steps_since_load
            surface.steps_since_load = 0
            try:
                return surface_status(body)
            finally:
                surface.steps_since_load = saved

        world["monkeypatch"].setattr(surface, "_op_status", clean_status)
        with pytest.raises(Refusal) as caught:
            _work(world)
        assert caught.value.code == "resume_served_model_mismatch"
        assert caught.value.details["stage"] == "attach"
        assert restores == [], "optimizer state was restored onto unverified weights"
        assert surface.call_count("detach") == 1, "the attachment must be released"
        assert surface.call_count("commit-begin") == 0
        assert self._run_row(world) == "failed"

    def test_a_config_that_disagrees_is_refused_at_attach(self, world) -> None:
        assert _submit(world).status_code == 202
        world["surface"].config = {**CONFIG, "hidden_size": 16}
        with pytest.raises(Refusal) as caught:
            _work(world)
        mismatches = caught.value.details["mismatches"]
        assert [m["field"] for m in mismatches] == ["config"]
        assert mismatches[0]["differences"] == [{"key": "hidden_size", "served": 16, "saved": 8}]

    def test_a_corrupted_checkpoint_is_refused_before_anything_is_attached(self, world) -> None:
        weights = world["checkpoint"] / "model.safetensors"
        weights.write_bytes(weights.read_bytes() + b"rot")
        error = _submit(world).json()["error"]
        assert error["code"] == "checkpoint_hash_mismatch"
        assert world["surface"].call_count("attach") == 0
        assert world["enqueued"] == []


class TestTheControls:
    """Each wiring point, removed, turns a test above red - and these pin the ones whose
    removal would otherwise look like a pass."""

    def test_without_the_restore_the_curve_breaks_by_far_more_than_the_tolerance(
        self, world
    ) -> None:
        assert _submit(world).status_code == 202
        world["monkeypatch"].setattr(
            tasks, "restore_optimizer_state", lambda optimizer, identity, **k: identity
        )
        _work(world)
        losses = _resumed_losses(world)
        reference = world["reference"]
        gap = max(abs(losses[s] - reference[s - 1]) for s in (4, 5, 6))
        assert gap > CONTINUITY_TOLERANCE * DISCRIMINATION_FACTOR, (
            f"a resume with no optimizer restore moved the curve by only {gap:.3e}; the "
            f"continuity assertion could not tell a restored optimizer from a fresh one"
        )

    def test_the_loop_refuses_a_resume_whose_acquisition_proved_nothing(self, world) -> None:
        """The loop's own guard: an acquisition that skipped the identity check (or a
        future one that forgets it) cannot get optimizer state restored."""
        assert _submit(world).status_code == 202
        acquirer = world["acquirer"]
        original = acquirer.acquire

        def unproved(spec: Any, **kw: Any) -> Any:
            acquired = original(spec, **kw)
            acquired.resume_verified = None
            return acquired

        world["monkeypatch"].setattr(acquirer, "acquire", unproved)
        restores: list[Any] = []
        world["monkeypatch"].setattr(
            tasks, "restore_optimizer_state", lambda *a, **k: restores.append(a)
        )
        with pytest.raises(Refusal) as caught:
            _work(world)
        assert caught.value.code == "resume_identity_unverified"
        assert restores == []
        assert world["surface"].call_count("detach") == 1


class TestWhatIsNotSubmitted:
    def test_a_live_run_is_not_continued(self, world) -> None:
        with session_for(world["engine"]) as session:
            session.execute(
                sa.update(Run).where(Run.id == RUN).values(status="running", finished_at=None)
            )
            session.commit()
        error = _submit(world).json()["error"]
        assert error["code"] == "run_not_continuable"
        assert world["enqueued"] == []

    def test_a_load_mode_run_is_planned_but_not_submitted(self, world) -> None:
        from schemas.run_definition import RunDefinition
        from services import run_launch

        with session_for(world["engine"]) as session:
            run_launch.launch(
                RunDefinition(
                    **{
                        **_definition(),
                        "run_id": "load-run",
                        "model_source": {"mode": "load", "path": "/data/checkpoints/base"},
                    }
                ),
                session=session,
                enqueue=lambda run_id: None,
                gate=lambda *a, **k: _PackDouble(),
                resolve_parts=lambda *a, **k: {"source": "parts", "parts": []},
                budget=lambda *a, **k: {"param_count": 10, "gpu_uuid": None, "mode": "load"},
                services=lambda *a, **k: {"needed": []},
            )
            run_state.mark_failed(session, "load-run", code="x")
            session.commit()
        response = _client(world).post("/api/v1/runs/load-run/continue", json={"submit": True})
        error = response.json()["error"]
        assert error["code"] == "continue_submit_mode_unsupported"
        assert error["details"]["mode"] == "load"
        with session_for(world["engine"]) as session:
            status = session.execute(sa.select(Run.status).where(Run.id == "load-run")).scalar()
        assert status == "failed"

    def test_without_submit_the_route_still_only_plans(self, world) -> None:
        """F5's answer is unchanged for every existing caller."""
        response = _client(world).post(f"/api/v1/runs/{RUN}/continue", json={})
        assert response.status_code == 200
        assert response.json()["plan"]["step"] == SAVE_AT
        assert world["enqueued"] == []
        assert world["surface"].call_count("status") == 0

    def test_a_caller_block_cannot_replace_a_resolved_key(self, world) -> None:
        from schemas.run_definition import RunDefinition
        from services import run_launch

        with session_for(world["engine"]) as session, pytest.raises(Refusal) as caught:
            run_launch.launch(
                RunDefinition(**{**_definition(), "run_id": "other"}),
                session=session,
                enqueue=lambda run_id: None,
                extra_resolved={"budget": {}},
                gate=lambda *a, **k: _PackDouble(),
                resolve_parts=lambda *a, **k: {"source": "parts", "parts": []},
                budget=lambda *a, **k: {"param_count": 10, "gpu_uuid": None},
                services=lambda *a, **k: {"needed": []},
            )
        assert caught.value.code == "run_definition_resolved_key_taken"


class TestTheRunRecordNamesItsAcquisitionMode:
    """The run view offers Continue on this field - pinned so dropping it is red."""

    def test_an_attach_run_reads_attach(self, world) -> None:
        body = _client(world).get(f"/api/v1/runs/{RUN}").json()
        assert body["acquisition_mode"] == "attach"
        assert body["status"] == "failed"


class TestOnlyTheLatestCheckpointContinues:
    def _later_checkpoint(self, world) -> None:
        later = checkpoint_dir(RUN, TOTAL, root=world["millm"])
        later.mkdir(parents=True)
        (later / manifest_module.MANIFEST_NAME).write_text("{}")

    def test_the_route_refuses_an_earlier_step(self, world) -> None:
        self._later_checkpoint(world)
        error = _submit(world, step=SAVE_AT).json()["error"]
        assert error["code"] == "resume_not_from_latest"
        assert error["details"]["checkpoints"] == [TOTAL]
        assert world["enqueued"] == []

    def test_the_define_path_refuses_it_too(self, world) -> None:
        """A resume can also be DEFINED (model_source.resume_from); the pre-flight folds the
        same rule in when it knows the run, so the define path is not a way around it."""
        from services import acquisition as acquisition_service
        from training.acquisition.base import AcquisitionSpec

        self._later_checkpoint(world)
        spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH, resume_from={"run_id": RUN, "step": 3})
        with pytest.raises(Refusal) as caught:
            acquisition_service.preflight_checkpoint_source(spec, run_id=RUN)
        assert caught.value.code == "resume_not_from_latest"
