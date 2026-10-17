"""The ``create`` run type: lineage, ``domain: null``, the tokenizer, the note.

**M10's and M11's targets.** ``parent: null`` and ``base_model`` = the architecture config
are validated **against the real** ``checkpoint-manifest-v1.json``, and the absent-``domain``
case must *fail* that validation - because null is an assertion and an absent key is an
omission, and the schema keeps the field required precisely so the two are different.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from core.errors import Refusal
from models.domain_state import ArtifactKind
from services.checkpoints import manifest as checkpoint_manifest
from services.reindex import load_schema, schema_violations
from tests.support.test_adapter_doubles import (
    TINY_CONFIG,
    FakeModule,
    FakeParameter,
    checkpoint_record,
    creation_record,
    optimizer_block,
)
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.run_config import RunConfig
from training.runtypes import resolve
from training.runtypes.base import SaveContext
from training.runtypes.create import SENSIBLE_AT_STEP_ZERO, random_start_note

CREATE = resolve("create")


def cfg() -> RunConfig:
    return RunConfig(learning_rate=1e-4, warmup_steps=10, batch_size=2, seed=7)


def create_spec(**overrides: Any) -> AcquisitionSpec:
    values: dict[str, Any] = {
        "mode": AcquisitionMode.CREATE,
        "config": dict(TINY_CONFIG),
        "seed": 7,
    }
    values.update(overrides)
    return AcquisitionSpec(**values)


class TestPreflight:
    def test_a_creation_run_with_a_config_passes(self) -> None:
        CREATE.preflight(create_spec(), cfg())

    def test_a_creation_run_with_no_architecture_config_is_refused(self) -> None:
        spec = AcquisitionSpec(mode=AcquisitionMode.CREATE, config_path=Path("/tmp/c.json"))
        CREATE.preflight(spec, cfg())  # a path is an input too

        with pytest.raises(Refusal) as caught:
            CREATE.preflight(
                AcquisitionSpec.model_construct(
                    mode=AcquisitionMode.CREATE, config=None, config_path=None
                ),
                cfg(),
            )
        assert caught.value.code == "architecture_config_required"

    def test_a_non_create_acquisition_mode_is_refused_rather_than_overridden(self) -> None:
        """Overriding would leave the two disagreeing about what was trained, and the
        acquisition record is what a manifest cites."""
        spec = AcquisitionSpec(mode=AcquisitionMode.LOAD, repo="org/model")
        with pytest.raises(Refusal) as caught:
            CREATE.preflight(spec, cfg())
        assert caught.value.code == "run_type_mode_mismatch"


class TestTheSignalSourceIsNotedAndNeverRefused:
    """E-3, FR-008.6, R-02.41 bullet 4."""

    @pytest.mark.parametrize("source", ["self_generation", "preference_pairs", "reward"])
    def test_a_non_corpus_source_produces_a_note(self, source: str) -> None:
        note = random_start_note(source)
        assert note is not None
        assert note["code"] == "create_run_starts_from_random_weights"
        assert note["signal_source"] == source

    def test_a_corpus_source_produces_no_note(self) -> None:
        assert random_start_note(SENSIBLE_AT_STEP_ZERO) is None

    def test_an_unstated_source_produces_no_note(self) -> None:
        """``None`` means *not stated here*, which is not the same as *a corpus*."""
        assert random_start_note(None) is None

    @pytest.mark.parametrize("source", ["self_generation", "preference_pairs", "reward"])
    def test_preflight_LOGS_and_does_NOT_refuse(
        self, source: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A refusal would be miForge overruling an operator; silence would be miForge
        letting a run learn from noise without saying so."""
        with caplog.at_level(logging.INFO, logger="training.runtypes.create"):
            CREATE.preflight(create_spec(), cfg(), signal_source=source)
        assert any(
            getattr(record, "code", None) == "create_run_starts_from_random_weights"
            for record in caplog.records
        ), "the note must reach the run log"


class TestPrepare:
    def test_every_parameter_is_trainable(self) -> None:
        module = FakeModule({"w": FakeParameter(), "b": FakeParameter()})
        acquired = type("Acquired", (), {"module": module, "tokenizer": object()})()
        plan = CREATE.prepare(acquired)
        assert len(plan.trainable) == 2
        assert plan.base_digest is None, "a creation run is supposed to change its parameters"

    def test_a_creation_run_with_NO_TOKENIZER_is_refused(self) -> None:
        """E-2, FR-008.7. F1's instantiation deliberately returns ``tokenizer=None`` and
        says the caller supplies it; F8 is that caller and refuses rather than inventing a
        default, which would train happily and generate nonsense."""
        module = FakeModule({"w": FakeParameter()})
        acquired = type("Acquired", (), {"module": module, "tokenizer": None})()
        with pytest.raises(Refusal) as caught:
            CREATE.prepare(acquired)
        assert caught.value.code == "tokenizer_required"
        assert "does not train one" in caught.value.message


class TestTheLineageIsWrittenAgainstTheRealSchema:
    """M10 and M11. Validated against the vendored file, not against a description of it."""

    def _manifest(self, *, domain: str | None, record: Any = None) -> dict[str, Any]:
        from training.acquisition.base import lineage_for_creation

        used = record if record is not None else creation_record()
        base_model, parent = lineage_for_creation(config_identity=used.base_model)
        return checkpoint_manifest.build(
            run_id="run-create",
            step=0,
            domain=domain,
            signal_source="corpus",
            base_model=base_model,
            parent=parent,
            optimizer=optimizer_block(),
            weights_sha256="f" * 64,
        )

    def test_parent_is_null_and_base_model_is_the_config_identity(self) -> None:
        doc = self._manifest(domain=None)
        assert doc["parent"] is None
        assert doc["base_model"].startswith("config:")
        assert "+seed=7" in doc["base_model"]

    def test_two_seeds_give_two_different_base_models(self) -> None:
        """The seed is folded into the identity, so a creation is identifiable rather than
        only describable. F1's rule, used rather than re-derived."""
        from training.acquisition.scratch import created_base_model_identity

        seven = created_base_model_identity(dict(TINY_CONFIG), 7)
        eight = created_base_model_identity(dict(TINY_CONFIG), 8)
        none = created_base_model_identity(dict(TINY_CONFIG), None)
        assert seven != eight
        assert none.endswith("+seed=none"), "an absent seed is written out, not omitted"

    def test_domain_null_VALIDATES_against_the_real_schema(self) -> None:
        doc = self._manifest(domain=None)
        assert doc["domain"] is None
        assert schema_violations(doc, ArtifactKind.CHECKPOINT) == []

    def test_domain_ABSENT_FAILS_the_real_schema(self) -> None:
        """M11's control at the contract layer. ``domain`` is nullable and **still
        required**: an absent key is an omission and the schema refuses it."""
        doc = self._manifest(domain=None)
        del doc["domain"]
        violations = schema_violations(doc, ArtifactKind.CHECKPOINT)
        assert violations, "the schema must refuse a manifest with no 'domain' key"
        assert any("domain" in violation for violation in violations)

    def test_the_schema_still_lists_domain_as_required(self) -> None:
        """Read from the file. If a re-vendor ever relaxed this, the test above would pass
        for the wrong reason and nothing would say so."""
        schema = load_schema(ArtifactKind.CHECKPOINT)
        root = schema["$defs"]["CheckpointManifestV1"]
        assert "domain" in root["required"]
        assert {"type": "null"} in root["properties"]["domain"]["anyOf"]


class TestSaveRefusesALineageThatWouldRootTheTreeWrongly:
    """M10's code-level control: the only line that could write a non-null parent."""

    def _ctx(self, record: Any, tmp_path: Path) -> SaveContext:
        return SaveContext(
            run_id="run-create",
            step=0,
            session=object(),
            module=object(),
            domain=None,
            signal_source="corpus",
            record=record,
            optimizer=optimizer_block(),
            tokenizer=object(),
            root=tmp_path,
        )

    def test_a_creation_record_naming_a_parent_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            CREATE.save(self._ctx(checkpoint_record(), tmp_path))
        assert caught.value.code == "create_lineage_invalid"

    def test_a_save_with_no_tokenizer_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        ctx = SaveContext(
            run_id="run-create",
            step=0,
            session=object(),
            module=object(),
            domain=None,
            signal_source="corpus",
            record=creation_record(),
            optimizer=optimizer_block(),
            tokenizer=None,
            root=tmp_path,
        )
        with pytest.raises(Refusal) as caught:
            CREATE.save(ctx)
        assert caught.value.code == "tokenizer_required"
        assert not list(tmp_path.iterdir()), "nothing is written on a refusal"


class TestCreateIsSmallOnPurpose:
    def test_create_py_is_not_larger_than_change_py_by_much(self) -> None:
        """*"If* ``create.py`` *ends up larger than a screen, the loop has been duplicated
        somewhere."* A crude measure, deliberately: it is a design tripwire, not a style
        rule, and it counts **statements** rather than lines so the long explanatory
        docstrings this project wants do not trip it.
        """
        import ast

        src = Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "runtypes"
        counts = {}
        for name in ("change", "create"):
            tree = ast.parse((src / f"{name}.py").read_text(encoding="utf-8"))
            counts[name] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))
        assert counts["create"] < 4 * counts["change"], counts

    def test_create_prepare_delegates_to_the_shared_helper(self) -> None:
        """Not a second implementation of "all parameters"."""
        source = (
            Path(__file__).resolve().parents[3] / "backend/src/training/runtypes/create.py"
        ).read_text(encoding="utf-8")
        assert "all_parameters(acquired)" in source


class TestTheConfigIsNotInstantiatedToCountIt:
    def test_create_py_never_builds_a_model(self) -> None:
        """R-02.42's correction: the budget is estimated from the config **before** the
        model is instantiated, so nothing on this path may build one to measure it."""
        source = (
            Path(__file__).resolve().parents[3] / "backend/src/training/runtypes/create.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("from_config", "AutoModelForCausalLM", "from_pretrained"):
            assert forbidden not in source, forbidden

    def test_the_tiny_config_is_a_real_transformers_shaped_config(self) -> None:
        """A fixture that was not config-shaped would make the arithmetic tests vacuous."""
        assert json.loads(json.dumps(TINY_CONFIG))["model_type"] == "llama"
        for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
            assert key in TINY_CONFIG


class TestTheLineagePairComesFromF1sFunctionAndNotFromTheRecord:
    """A **surviving mutation**, found in F8's sweep and closed here.

    The mutation was ``base_model, parent = ctx.record.base_model, ctx.record.parent`` in
    place of ``lineage_for_creation(config_identity=ctx.record.base_model)``. Against a
    *conforming* creation record the two are equivalent - ``record.parent`` is already
    ``None`` and ``record.base_model`` is already the config identity - so every assertion
    in this file stayed green, including the one that refuses a record naming a parent.

    **Why that mattered rather than being a tie.** ``parent: null`` for a creation run is
    then a property of the record F1 happened to build, not a property of F8's save: the
    guarantee moves from *structural* to *incidental*. The day F1's lineage rule changes -
    and it has changed once already, when the carry-forward defect rooted every continued
    run's tree at its parent - a creation save would silently follow the record instead of
    the rule. So the assertion is on the **seam**, with the payload and the call count: *"was
    called" passes against a call sending the wrong arguments*, and here it would also pass
    against a call not made at all.
    """

    def _save(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record: Any) -> Any:
        from core.config import reset_settings
        from tests.support.checkpoints import TinyModel, TinyTokenizer, create_feature_tables
        from tests.support.jobs import insert_run, memory_engine, session_for

        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
        reset_settings()
        engine = memory_engine()
        create_feature_tables(engine)
        with session_for(engine) as session:
            insert_run(session, "run-create", status="running")
            outcome = CREATE.save(
                SaveContext(
                    run_id="run-create",
                    step=0,
                    session=session,
                    module=TinyModel(),
                    domain=None,
                    signal_source="corpus",
                    record=record,
                    optimizer=optimizer_block(),
                    tokenizer=TinyTokenizer(),
                    root=checkpoints,
                )
            )
            session.commit()
        reset_settings()
        return outcome

    def test_a_creation_save_writes_the_lineage_the_FUNCTION_returns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The spy returns a sentinel the record does not carry, so a save that read the
        record instead of calling the function writes the wrong ``base_model`` and this
        goes red."""
        from training.runtypes import create as create_module

        calls: list[dict[str, Any]] = []
        sentinel = "config:sentinel@" + "0" * 64 + "+seed=none"

        def spy(*, config_identity: str) -> tuple[str, str | None]:
            calls.append({"config_identity": config_identity})
            return sentinel, None

        monkeypatch.setattr(create_module, "lineage_for_creation", spy)
        record = creation_record()
        outcome = self._save(monkeypatch, tmp_path, record)

        assert outcome.manifest["base_model"] == sentinel, (
            "the save wrote the record's own base_model rather than the lineage function's "
            "answer, so parent: null is incidental rather than structural"
        )
        assert outcome.manifest["parent"] is None
        # Payload AND call count. One call, with the record's base_model as the config
        # identity - not the record's `config_identity` field, which is the config's hash
        # and not the identity string the manifest's base_model takes.
        assert calls == [{"config_identity": record.base_model}]

    def test_the_function_is_called_EXACTLY_ONCE_per_save(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from training.runtypes import create as create_module

        calls: list[str] = []
        real = create_module.lineage_for_creation

        def spy(*, config_identity: str) -> tuple[str, str | None]:
            calls.append(config_identity)
            return real(config_identity=config_identity)

        monkeypatch.setattr(create_module, "lineage_for_creation", spy)
        self._save(monkeypatch, tmp_path, creation_record())
        assert len(calls) == 1

    def test_the_parent_written_is_the_FUNCTIONS_second_value_even_if_it_changed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The structural half. If F1's rule ever returned a parent for a creation, F8 must
        follow the rule rather than the record - and a save reading ``ctx.record.parent``
        would write ``None`` here and go red."""
        from training.runtypes import create as create_module

        monkeypatch.setattr(
            create_module,
            "lineage_for_creation",
            lambda *, config_identity: (config_identity, "some:other:step-1"),
        )
        outcome = self._save(monkeypatch, tmp_path, creation_record())
        assert outcome.manifest["parent"] == "some:other:step-1"
