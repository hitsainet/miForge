"""With ``training/hotupdate/`` **absent**, F8's adapter run and merge take their own
paths and no §5.10 variant is offered  (FR-011.62, FR-011.47; FTASKS 10a.10).

**Why this file is dynamic and not another AST scan.** Two static guards already live in
``test_hotupdate_adapter_read.py``: no F8 module imports the fenced package, and no F8
module branches on the ``attach`` mode. Both are necessary and neither is sufficient -
*a grep proves a call does not exist; it does not prove the feature works without it.*
What 10a.10 states is a **behavioural** property: with §5.10 gone, an adapter run still
trains and a merge still writes, and the mode is reported unavailable **by name** rather
than silently missing.

**How the absence is made real.** The package is made genuinely unimportable by a
``sys.meta_path`` finder that raises ``ImportError`` for it, with every already-imported
submodule removed from ``sys.modules`` first. That is the same failure
``importlib.import_module("training.hotupdate")`` produces when the directory is deleted,
and it is forced rather than borrowed from the machine: *a test whose outcome turns on
whether an import happens to succeed is a test about the environment, not the code* - the
``pyarrow`` and ``torch`` findings of 2026-10-14, in both directions. The fixture carries
its own **positive check** that the block bites before any assertion is made.

**What this file does NOT claim.** It does not replace the acceptance control of FTASKS
17.6, which moves the real directory out of the tree and runs the whole suite; nothing
inside one pytest process can prove a package's files are gone. It proves the weaker and
still load-bearing half: that F8's two paths execute to completion while the import of
§5.10 raises, and that nothing on those paths pulls it in - asserted by reading
``sys.modules`` after the run rather than by trusting the finder.

**Real ``torch`` and real ``peft``.** The adapter run goes through ``tasks.drive_run``, the
production dispatch, over a real tiny ``AutoModelForCausalLM``; the merge goes through
``merge.merge``, over a base checkpoint written by **F5's own writer** and an adapter
directory written by **F8's own writer**. A double cannot express the defect: a stubbed
optimizer or a hand-built manifest would agree with this file by construction, which is
this project's recorded reason a suite stays green over a real bug.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from core.config import Settings, reset_settings
from core.errors import Refusal
from tests.support.conformance import ml_runtime_available
from tests.support.jobs import insert_run, memory_engine, session_for
from tests.support.test_adapter_doubles import TINY_CONFIG, adapter_config, optimizer_block
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    SourceKind,
)

FENCED = "training.hotupdate"

_available, _why = ml_runtime_available()


def _ml_ready() -> tuple[bool, str]:
    if not _available:
        return False, _why
    try:
        import peft  # noqa: F401
    except ImportError as exc:  # pragma: no cover - peft is declared (ADR-002)
        return False, f"peft is not installed in this environment ({exc})"
    return True, ""


_ok, _skip_why = _ml_ready()

RUN = "run-f8-without-5-10"


# ======================================================================================
# The absence, forced rather than borrowed
# ======================================================================================


class _FenceOff:
    """A meta-path finder that makes the fenced package unimportable.

    ``find_spec`` **raises** rather than returning ``None``: returning ``None`` only means
    *"not my business"* and the next finder on the path would then import it from disk,
    which would make this whole file a no-op that looked exactly like a pass.
    """

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == FENCED or fullname.startswith(FENCED + "."):
            raise ImportError(
                f"{fullname} is absent (ADR-022's fence control). This is the error "
                f"`importlib.import_module` raises when training/hotupdate/ is deleted."
            )
        return None


@pytest.fixture
def fenced_absent(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """§5.10 unimportable, with a positive check that the block actually bites."""
    for name in [n for n in sys.modules if n == FENCED or n.startswith(FENCED + ".")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_FenceOff(), *sys.meta_path])

    # THE POSITIVE CHECK. Without it a finder that silently failed to install would make
    # every assertion below pass against a build that still has the package - the
    # false-measurement family's signature, where the broken measurement and the real
    # result are indistinguishable.
    with pytest.raises(ImportError):
        importlib.import_module(FENCED)
    assert FENCED not in sys.modules

    yield
    reset_settings()


def _assert_never_imported() -> None:
    pulled = sorted(n for n in sys.modules if n == FENCED or n.startswith(FENCED + "."))
    assert pulled == [], f"the fenced package was imported after all: {pulled}"


# ======================================================================================
# FR-011.47 — the mode is present, unavailable, and NAMED
# ======================================================================================


class TestNoSection510VariantIsReportedAvailable:
    """The registry's answer with the package gone, read through the **live** registry."""

    @pytest.fixture
    def registry(self, fenced_absent: None, monkeypatch: pytest.MonkeyPatch) -> Any:
        """``_install_defaults`` re-run over a **copy** of the registry.

        A copy rather than the live dict so nothing leaks into another test in this worker
        process: ``register`` writes through the module global, so replacing the attribute
        is enough and ``monkeypatch`` puts the real one back.
        """
        from training import acquisition

        monkeypatch.setattr(acquisition, "_REGISTRY", {})
        acquisition._install_defaults()
        return acquisition

    def test_the_mode_is_unavailable_and_says_why(self, registry: Any) -> None:
        state = registry.availability(AcquisitionMode.ATTACH)
        assert state.available is False
        assert state.reason == "hot_update_mode_not_available"
        # Says WHY in operator words (2026-10-20 copy review took "§5.10" out of this message):
        # the package is not installed, and load mode is the way to train the same weights.
        assert "hot-update package" in state.message
        assert "not installed" in state.message
        assert "load mode" in state.message
        assert state.unavailable_capabilities, "an unavailable mode must name what is lost"

    def test_it_is_still_LISTED_rather_than_missing(self, registry: Any) -> None:
        """*Absent-and-normal is reported, never hidden* (ADR-010). A mode that vanishes
        from the payload is indistinguishable from a build that forgot to register it."""
        payload = {entry["mode"]: entry for entry in registry.modes()}
        assert "attach" in payload
        assert payload["attach"]["available"] is False
        assert payload["attach"]["reason"] == "hot_update_mode_not_available"

    def test_requesting_it_refuses_rather_than_substituting_a_load(self, registry: Any) -> None:
        """R-02.48. A silent fall-back to ``load`` trains a different set of tensors than
        was asked for, and records the run as though it had attached."""
        acquirer = registry.resolve(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
        assert type(acquirer).__name__ == "_UnavailableMode"
        with pytest.raises(Refusal) as caught:
            acquirer.acquire(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
        assert caught.value.code == "acquisition_mode_unavailable"
        assert caught.value.details["reason"] == "hot_update_mode_not_available"
        _assert_never_imported()

    def test_the_other_modes_are_untouched(self, registry: Any) -> None:
        """The fence confines the loss to §5.10 (ADR-022, risk R-1)."""
        modes = {entry["mode"]: entry for entry in registry.modes()}
        assert modes["load"]["available"] is True
        assert "create" in modes


# ======================================================================================
# F8's two paths, driven for real with §5.10 unimportable
# ======================================================================================

pytestmark = pytest.mark.skipif(not _ok, reason=_skip_why or "ML runtime unavailable")


def _tiny_module() -> Any:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        TINY_CONFIG["model_type"], **{k: v for k, v in TINY_CONFIG.items() if k != "model_type"}
    )
    torch.manual_seed(11)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=False)


class _Tokenizer:
    """Writes the tokenizer files a checkpoint carries **at its root** (R-02.21)."""

    def save_pretrained(self, directory: Any) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")
        (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")


@dataclass
class _Batch:
    objective: str = "sft"
    record: dict[str, Any] = field(default_factory=lambda: {"batch_sha256": "b" * 64})


@dataclass
class _Source:
    def next(self) -> _Batch:
        return _Batch()


@dataclass
class _RealObjective:
    """A **real** forward and backward through whichever module the step was handed.

    The losses are F2's, so the placeholder lives in the test as it does in
    ``test_step_loop.py`` and ``test_runtype_dispatch.py``. The tensor is not faked: the
    loss is the model's own, so ``backward()`` populates real ``.grad`` tensors and the
    gradient assertion below is a measurement rather than a restatement of the fixture.
    """

    name: str = "sft"

    def loss(self, model: Any, batch: Any) -> Any:
        import torch

        del batch
        ids = torch.tensor([[1, 2, 3, 4]])
        return model(input_ids=ids, labels=ids).loss


@dataclass
class _Registry:
    objective: _RealObjective

    def get(self, name: str) -> _RealObjective:
        assert name == self.objective.name
        return self.objective


HUB_RECORD = AcquisitionRecord(
    mode=AcquisitionMode.LOAD,
    source_kind=SourceKind.HUB,
    source_repo="LiquidAI/LFM2.5-1.2B-Instruct",
    source_revision_requested="main",
    source_revision_resolved="c" * 40,
    starting_weights_sha256="b" * 64,
    base_model="LiquidAI/LFM2.5-1.2B-Instruct",
    parent=None,
    param_count=512_000,
)


class TestAnAdapterRunTakesF8sDefaultPath:
    """AC14's default half, with §5.10 gone. R-02.43 is **F8's** requirement (task 17.2a)."""

    def test_it_trains_the_adapter_alone_and_publishes_a_peft_directory(
        self, fenced_absent: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch  # noqa: F401  - imported here so the skip is the marker's, not an error

        from training import tasks
        from training.run_config import RunConfig
        from training.runtypes.adapter import base_parameter_digest
        from training.runtypes.adapters import paths as adapter_paths

        checkpoints = tmp_path / "checkpoints"
        states = tmp_path / "trainer-state"
        for directory in (checkpoints, states):
            directory.mkdir()
        monkeypatch.setattr(
            "core.config._settings",
            Settings(  # type: ignore[call-arg]
                _env_file=None, trainer_state_root=states, checkpoint_root=checkpoints
            ),
        )

        module = _tiny_module()
        before = base_parameter_digest(module)
        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(session, RUN, status="running")

        acquired = AcquiredModel(module=module, tokenizer=_Tokenizer(), record=HUB_RECORD)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(tasks, "acquire_for_run", lambda spec: acquired)
            with session_for(engine) as session:
                tasks.drive_run(
                    run_id=RUN,
                    spec=AcquisitionSpec(repo="LiquidAI/LFM2.5-1.2B-Instruct"),
                    # 32-bit on purpose: `bitsandbytes` is an extra with no pure-CPU wheel
                    # everywhere (ADR-023) and `build` REFUSES the `8bit` default rather
                    # than downgrading, because the recorded identity would otherwise name
                    # a precision that was not used.
                    cfg=RunConfig(
                        learning_rate=1e-2,
                        warmup_steps=0,
                        batch_size=1,
                        optimizer_state_precision="32bit",
                    ),
                    batch_source=_Source(),
                    objectives=_Registry(_RealObjective()),
                    session=session,
                    log_root=tmp_path,
                    trainer_state_root=states,
                    max_steps=1,
                    save_every=1,
                    signal_source="corpus",
                    domain="legal",
                    run_type="adapter",
                    adapter=adapter_config(),
                )

        # 1. The artifact is a PEFT directory, not a full checkpoint of the base.
        directory = adapter_paths.adapter_dir(RUN, 1, root=checkpoints)
        assert (directory / adapter_paths.MANIFEST_NAME).is_file()
        assert (directory / "adapter_model.safetensors").is_file()
        assert not (
            directory / "model.safetensors"
        ).exists(), "an adapter save wrote a full parameter set; R-02.44's artifact is the adapter"

        # 2. AC14: the base is byte-unchanged. Over CANONICAL base names, so the pre-wrap
        #    and post-wrap hashes of one untouched base are comparable.
        assert base_parameter_digest(module) == before, "an adapter run moved the base's bytes"

        # 3. And none of it reached §5.10.
        _assert_never_imported()


class TestAMergeTakesF8sDefaultPath:
    """AC16's default half, with §5.10 gone. The bracketed form is F11's and is absent."""

    def test_a_real_merge_writes_a_merged_checkpoint_with_both_ancestors(
        self, fenced_absent: None, tmp_path: Path
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        from core.canonical import sha256_of
        from services.checkpoints import ManifestFields
        from services.checkpoints import manifest as checkpoint_manifest
        from services.checkpoints import service as checkpoint_service
        from services.checkpoints.manifest import MANIFEST_NAME as CHECKPOINT_MANIFEST_NAME
        from training.runtypes.adapters import manifest as adapter_manifest
        from training.runtypes.adapters import merge as merge_module
        from training.runtypes.adapters import paths as adapter_paths
        from training.runtypes.adapters import save as adapter_save
        from training.runtypes.adapters.build import build as build_adapter

        root = tmp_path / "checkpoints"
        root.mkdir()
        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(session, "run-base", status="running")
            insert_run(session, "run-merge", status="running")

        # ---- 1. A base checkpoint, through F5'S OWN WRITER ------------------------
        # Not a hand-built manifest: `resolve_base_checkpoint` re-hashes what is on disk
        # and compares it with what the adapter recorded, so a fixture that wrote both
        # from one literal would agree with itself by construction.
        base_module = _tiny_module()
        with session_for(engine) as session:
            saved = checkpoint_service.save(
                base_module,
                _Tokenizer(),
                session=session,
                fields=ManifestFields(
                    run_id="run-base",
                    step=10,
                    domain="legal",
                    signal_source="corpus",
                    base_model="LiquidAI/LFM2.5-1.2B-Instruct",
                    parent=None,
                    optimizer=optimizer_block(),
                ),
                root=root,
            )
            session.commit()

        # Read back through F5's OWN reader and hashed with the project's one canonical
        # hasher, because `resolve_base_checkpoint` re-hashes what is on disk and compares:
        # a fixture that recorded `sha256_of(the_dict_it_just_built)` would agree with
        # itself and say nothing about the file.
        base_sha = sha256_of(checkpoint_manifest.read(saved.directory / CHECKPOINT_MANIFEST_NAME))

        # ---- 2. An adapter over that base, through F8'S OWN WRITER ----------------
        wrapped, validated = build_adapter(
            AutoModelForCausalLM.from_pretrained(str(saved.directory), trust_remote_code=False),
            adapter_config(rank=4, alpha=8.0, target_modules=("q_proj", "v_proj")),
        )
        # `peft` initialises `lora_B` to zeros, so a freshly built adapter is the IDENTITY
        # and every merge would look correct. Perturbing B is what makes the merge
        # observable at all.
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if "lora_B" in name:
                    parameter.add_(torch.randn_like(parameter) * 0.05)
        doc = adapter_manifest.build(
            run_id="run-merge",
            step=10,
            base={"name": saved.name, "manifest_sha256": base_sha},
            adapter=validated,
            optimizer=optimizer_block(),
        )
        adapter_directory = adapter_paths.adapter_dir("run-merge", 10, root=root)
        adapter_save.save(wrapped, doc, adapter_directory)

        # ---- 3. The merge itself, F8's entry point -------------------------------
        with session_for(engine) as session:
            outcome = merge_module.merge(
                session=session,
                adapter_name=doc["name"],
                base_module=AutoModelForCausalLM.from_pretrained(
                    str(saved.directory), trust_remote_code=False
                ),
                tokenizer=_Tokenizer(),
                run_id="run-merge",
                step=11,
                domain="legal",
                signal_source="corpus",
                optimizer=optimizer_block(),
                held_base_name=saved.name,
                root=root,
            )
            session.commit()

        assert outcome.outcome == "written"
        assert outcome.base_checkpoint_name == saved.name
        assert outcome.result_checkpoint_name

        # R-02.45/FR-008.36: a merged checkpoint has TWO ancestors, and without the second
        # field two checkpoints at one step with one parent - merged and not - are
        # indistinguishable.
        merged_manifest = json.loads(
            (root / "run-merge" / "step-000011" / "manifest.json").read_text("utf-8")
        )
        assert merged_manifest["parent"] == saved.name
        assert merged_manifest["merged_adapter"]["name"] == doc["name"]

        # The adapter survives its own merge: a merged adapter cannot be taken off again.
        assert (adapter_directory / adapter_paths.MANIFEST_NAME).is_file()

        _assert_never_imported()
