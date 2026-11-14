"""A ``create`` run produces a LOADABLE checkpoint  (R-02.41, R-02.21, R-02.51; AC12, AC23).

**The defect this file is the regression for, and it was on the core path.** A ``create``
run could not complete. ``training/acquisition/scratch.py``'s ``_instantiate`` returned
``(module, None)`` for the tokenizer, its docstring said *"the caller supplies it"*, and
**nothing in ``backend/src/`` was that caller** - there was no field on ``AcquisitionSpec``
to supply it through. So the run trained for however long it was configured for and then:

* **at the first save point**, ``services/checkpoints/writer.py``'s
  ``tokenizer.save_pretrained(staging)`` raised ``AttributeError: 'NoneType' object has no
  attribute 'save_pretrained'`` - a traceback, hours in, having produced nothing;
* **at the first scoring pass**, ``providers/model_handle.py`` raised ``TypeError: 'NoneType'
  object is not callable``.

``CreateRunType.prepare`` does raise ``tokenizer_required`` and **has no production
caller**, which is why neither crash was ever a refusal (``training/run_driver.py`` records
that correction).

## Why these cases run on the CPU

Instantiating a 2-layer / 64-hidden LLaMA and saving it needs no card, and AC23's strongest
form - a load in a process whose ``sys.path`` has no miForge on it - needs none either. So
the proof that a created checkpoint is *loadable* is unmarked and runs in CI, where
``test_create_run_gpu.py``'s equivalent **skipped unless ``MIFORGE_TEST_TOKENIZER`` was
set** and therefore proved it on no machine at all. What still needs the card is the
training step over those parameters, which is that file's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.errors import Refusal
from services.checkpoints import service as checkpoint_service
from services.checkpoints.manifest import ManifestFields
from tests.support.tiny_hf_model import LLAMA_TINY, write_tiny_tokenizer, write_tokenizer_json_only
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.acquisition.scratch import ScratchAcquirer

pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Checkpoint and parts roots under ``tmp_path``. Returns the checkpoint root."""
    from core.config import reset_settings

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    reset_settings()
    yield checkpoints
    reset_settings()


@pytest.fixture
def tokenizer_dir(roots: Path) -> Path:
    """A **real** tokenizer, written by the real writer, sized from ``LLAMA_TINY``.

    Sized from the config and not from a 32k vocabulary: ``LLAMA_TINY`` declares
    ``vocab_size: 512``, and a bigger tokenizer hands the model an id past its embedding
    table - ``IndexError: index out of range in self`` - the moment anything generates.
    """
    directory = roots / "given-tokenizer"
    directory.mkdir()
    write_tiny_tokenizer(directory, vocab_size=LLAMA_TINY["vocab_size"])
    return directory


def create_spec(tokenizer: Path, **overrides: Any) -> AcquisitionSpec:
    base: dict[str, Any] = {
        "mode": AcquisitionMode.CREATE,
        "config": dict(LLAMA_TINY),
        "seed": 13,
        "tokenizer_path": tokenizer,
    }
    base.update(overrides)
    return AcquisitionSpec(**base)


def manifest_fields(record: Any, *, run_id: str = "run-create", step: int = 0) -> ManifestFields:
    return ManifestFields(
        run_id=run_id,
        step=step,
        domain=None,
        signal_source="corpus",
        base_model=record.base_model,
        parent=record.parent,
        optimizer={"name": "adamw-8bit", "state": "fresh", "config_hash": "a" * 64},
    )


def save_one(acquired: Any, *, root: Path, run_id: str = "run-create") -> Any:
    from tests.support.checkpoints import create_feature_tables
    from tests.support.jobs import insert_run, memory_engine, session_for

    engine = memory_engine()
    create_feature_tables(engine)
    with session_for(engine) as session:
        insert_run(session, run_id, status="running")
        outcome = checkpoint_service.save(
            acquired.module,
            acquired.tokenizer,
            session=session,
            fields=manifest_fields(acquired.record, run_id=run_id),
            root=root,
        )
        session.commit()
    return outcome


class TestACreateRunCompletesItsFirstSave:
    def test_the_acquisition_CARRIES_the_tokenizer_the_spec_named(
        self, tokenizer_dir: Path
    ) -> None:
        """The whole fix in one assertion: ``acquired.tokenizer`` is not ``None``."""
        acquirer = ScratchAcquirer()
        acquired = acquirer.acquire(acquirer.preflight(create_spec(tokenizer_dir)))
        assert acquired.tokenizer is not None, (
            "this was None for the whole life of scratch.py, which is what killed every "
            "create run at its first save point"
        )
        assert acquired.record.tokenizer_source == str(tokenizer_dir)
        assert acquired.record.tokenizer_vocab_size == LLAMA_TINY["vocab_size"]
        assert acquired.record.parent is None
        assert acquired.record.base_model.startswith("config:")

    def test_the_save_writes_TOKENIZER_FILES_AT_THE_ROOT(
        self, tokenizer_dir: Path, roots: Path
    ) -> None:
        """ADR-015 / R-02.51. Asserted on the published directory rather than on the
        writer's intent, and flat - a ``tokenizer/`` subdirectory is the defect the
        requirement records."""
        acquirer = ScratchAcquirer()
        acquired = acquirer.acquire(acquirer.preflight(create_spec(tokenizer_dir)))
        outcome = save_one(acquired, root=roots)

        names = sorted(p.name for p in outcome.directory.iterdir())
        assert "tokenizer.json" in names
        assert "tokenizer_config.json" in names
        assert "model.safetensors" in names
        assert "config.json" in names
        assert not [p.name for p in outcome.directory.iterdir() if p.is_dir()], (
            "a conforming checkpoint is a FLAT Hugging Face directory; a tokenizer/ "
            "subdirectory cannot be loaded by transformers, vLLM, TGI or miLLM's loader"
        )
        manifest = json.loads((outcome.directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["parent"] is None
        assert manifest["base_model"] == acquired.record.base_model

    def test_the_checkpoint_LOADS_IN_A_PROCESS_THAT_CANNOT_IMPORT_MIFORGE(
        self, tokenizer_dir: Path, roots: Path
    ) -> None:
        """**Requirement 4, and AC23's strong form.** The load happens in a subprocess
        whose ``sys.path`` has no miForge on it at all, so *"no miForge code was involved"*
        is structural rather than observed - and it generates, so the artifact works rather
        than merely parses.

        This is the assertion the create path never had: ``test_create_run_gpu.py``'s
        equivalent is ``gpu``-marked and, until 2026-10-16, **also skipped without
        ``MIFORGE_TEST_TOKENIZER``** - so criterion 23 was proved for a created checkpoint on
        no machine anywhere. That file now writes its own tokenizer through the product's
        mechanism and no longer skips; **this case remains the unmarked one**, which is what
        keeps the proof in CI rather than on an operator-run gate.
        """
        from tests.support.conformance import assert_loads_in_a_clean_process

        acquirer = ScratchAcquirer()
        acquired = acquirer.acquire(acquirer.preflight(create_spec(tokenizer_dir)))
        outcome = save_one(acquired, root=roots)

        report = assert_loads_in_a_clean_process(outcome.directory)
        assert report["miforge_modules"] == []
        assert report["generated"].strip()

    def test_the_checkpoint_conforms_through_the_SHARED_harness_too(
        self, tokenizer_dir: Path, roots: Path
    ) -> None:
        """The general form, which also asserts the standard file set."""
        from tests.support.conformance import assert_output_conforms

        acquirer = ScratchAcquirer()
        acquired = acquirer.acquire(acquirer.preflight(create_spec(tokenizer_dir)))
        outcome = save_one(acquired, root=roots)

        report = assert_output_conforms(outcome.directory)
        assert report.miforge_modules_imported == ()
        assert "manifest.json" in report.files


class TestTheRefusalIsONTHESUBMITPATH:
    """**ADR-016 #1 reachability: the refusal has to be reachable from where a run is
    admitted**, not only from the acquirer's own unit tests. These are the cases that go
    red if ``ScratchAcquirer.preflight`` stops resolving the tokenizer.
    """

    def test_the_SERVICE_preflight_refuses_a_create_with_no_tokenizer(self, roots: Path) -> None:
        """``services.acquisition.preflight`` is what ``services/run_definition.py``'s
        budget step calls. Refusing here means refusing **before the lease and before any
        allocation** (ADR-008), which is the whole difference between this and the
        ``AttributeError`` it replaces."""
        from services import acquisition as acquisition_service
        from training.run_config import RunConfig

        cfg = RunConfig(
            learning_rate=1e-5, warmup_steps=1, batch_size=1, gradient_accumulation=1, seed=13
        )
        spec = AcquisitionSpec(mode=AcquisitionMode.CREATE, config=dict(LLAMA_TINY), seed=13)
        with pytest.raises(Refusal) as caught:
            acquisition_service.preflight(spec, cfg, guard=_NeverCalledGuard())
        assert caught.value.code == "tokenizer_required"

    def test_the_refusal_lands_BEFORE_THE_BUDGET_GUARD(self, roots: Path) -> None:
        """Ordered most-specific-first, as ``services/acquisition.py``'s own comments say:
        the acquirer's gate runs first, so *"this run names no tokenizer"* is answered
        without resolving a device or estimating anything."""
        from services import acquisition as acquisition_service
        from training.run_config import RunConfig

        guard = _NeverCalledGuard()
        cfg = RunConfig(
            learning_rate=1e-5, warmup_steps=1, batch_size=1, gradient_accumulation=1, seed=13
        )
        with pytest.raises(Refusal):
            acquisition_service.preflight(
                AcquisitionSpec(mode=AcquisitionMode.CREATE, config=dict(LLAMA_TINY), seed=13),
                cfg,
                guard=guard,
            )
        assert guard.calls == 0, "the budget guard must not be reached"

    def test_THE_RUN_DEFINITION_VALIDATOR_collects_it_as_a_submit_failure(
        self, roots: Path
    ) -> None:
        """**The real submit path.** R-02.53's validator runs F1's preflight hook as its
        budget step and *collects* refusals, so an operator posting a ``create`` run with
        no tokenizer gets a named failure from ``POST /runs/validate`` and
        ``POST /runs/define`` rather than a run that dies hours later."""
        from services import run_definition as validator
        from tests.support.jobs import memory_engine, session_for

        engine = memory_engine()
        with session_for(engine) as session:
            outcome = validator.validate(
                _create_definition(),
                session=session,
                gate=lambda *a, **k: None,
                resolve_parts=lambda *a, **k: {"source": "parts", "parts": []},
                services=lambda *a, **k: {"needed": []},
            )
        assert not outcome.ok
        codes = [f.code for f in outcome.failures]
        assert "tokenizer_required" in codes, codes

    def test_a_VOCAB_MISMATCH_is_also_refused_at_submit(self, roots: Path) -> None:
        """Requirement 3. ``LLAMA_TINY`` is 512 tokens; a 32k tokenizer against it is
        ``IndexError`` at the first generation and a named refusal here."""
        from services import run_definition as validator
        from tests.support.jobs import memory_engine, session_for

        big = write_tokenizer_json_only(roots / "big-tok", vocab_size=32000)
        engine = memory_engine()
        with session_for(engine) as session:
            outcome = validator.validate(
                _create_definition(tokenizer_path=str(big)),
                session=session,
                gate=lambda *a, **k: None,
                resolve_parts=lambda *a, **k: {"source": "parts", "parts": []},
                services=lambda *a, **k: {"needed": []},
            )
        assert not outcome.ok
        failure = next(f for f in outcome.failures if f.code == "tokenizer_vocab_mismatch")
        assert failure.details["config_vocab_size"] == LLAMA_TINY["vocab_size"]
        assert failure.details["tokenizer_vocab_size"] == 32000

    def test_a_GOOD_create_definition_PASSES_the_budget_step(self, tokenizer_dir: Path) -> None:
        """The positive control for the four cases above. Without it, a budget step that
        refused *every* create run would satisfy all of them."""
        from services import run_definition as validator
        from tests.support.jobs import memory_engine, session_for

        engine = memory_engine()
        with session_for(engine) as session:
            outcome = validator.validate(
                _create_definition(tokenizer_path=str(tokenizer_dir)),
                session=session,
                gate=lambda *a, **k: None,
                resolve_parts=lambda *a, **k: {"source": "parts", "parts": []},
                services=lambda *a, **k: {"needed": []},
            )
        assert [f.code for f in outcome.failures] == [], outcome.failures
        assert outcome.ok


class TestTheWritersLASTRESORT:
    """**Deliberately a second code, not the gate.**

    The fix for the ``AttributeError`` is the tokenizer reference, which refuses at submit.
    This refusal exists because a crash inside an atomic write, hours into a run, is the
    worst available failure - but it is named ``checkpoint_tokenizer_missing`` rather than
    ``tokenizer_required`` so the two can never be confused, and
    ``TestTheRefusalIsONTHESUBMITPATH`` is what proves nothing reaches it in production. If
    this were the only guard, the missing mechanism would have been made *tidy* rather than
    *present*, which is exactly the wrong fix.
    """

    def test_a_save_with_no_tokenizer_REFUSES_instead_of_raising_AttributeError(
        self, roots: Path
    ) -> None:
        from tests.support.checkpoints import TinyModel, fields
        from tests.support.jobs import insert_run, memory_engine, session_for

        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(session, "run-no-tok", status="running")
            with pytest.raises(Refusal) as caught:
                checkpoint_service.save(
                    TinyModel(),
                    None,
                    session=session,
                    fields=fields(run_id="run-no-tok", step=0),
                    root=roots,
                )
        assert caught.value.code == "checkpoint_tokenizer_missing"
        assert not list(roots.glob("run-no-tok/*")), "nothing is published"


class _NeverCalledGuard:
    def __init__(self) -> None:
        self.calls = 0

    def check(self, spec: Any, cfg: Any) -> None:
        self.calls += 1


def _create_definition(**source: Any) -> Any:
    from schemas.run_definition import RunDefinition

    model_source: dict[str, Any] = {"mode": "create", "config": dict(LLAMA_TINY)}
    model_source.update(source)
    return RunDefinition(
        run_id="run-create-def",
        model_source=model_source,
        run_type="create",
        run_configuration={
            "learning_rate": 1e-5,
            "warmup_steps": 2,
            "batch_size": 2,
            "gradient_accumulation": 1,
        },
        signal_source={"source": "corpus"},
        parts=[{"part_id": "part-1", "kind": "corpus"}],
        cadences={"test_every": 2, "retention_every": 1, "save_every": 10},
        stop_conditions=[{"condition": "step_count", "config": {"total": 100}}],
        domain=None,
        seed=13,
    )
