"""The loop is routed through ``RunType``, so an adapter run *is* an adapter run.

**ADR-016 #1, the strong form.** ``RunType.prepare`` and ``RunType.save`` were built,
unit-tested, documented and called by **nothing** in ``backend/src/`` until 2026-10-16 -
``services.run_definition`` called ``preflight`` and the loop went straight to
:func:`training.step.run_one` over ``acquired.module`` and to F5's checkpoint writer. Both
halves of the claim in ``run_driver.py``'s own docstring (*"this module calls prepare and
save"*) were false, and ``grep -rn "run_type\\.save\\|\\.save(ctx\\|SaveContext(" src/``
found the one ``SaveContext(`` there was: a projection built to reach F8's manifest-score
join, never handed to a run type.

So a capability that is three registry entries and a shared loop behaved as **one** run
type, and two requirements were not implemented by anything a caller could reach:

1. **R-02.43's economy did not happen.** ``optimizer_module.build`` was handed
   ``acquired.module`` whole and ``assert_only_adapter_requires_grad`` never ran, so an
   adapter run built an AdamW over **every parameter** - full-parameter gradients, a full
   FP32 master copy, the memory profile BRD-02 §6 reasons about for the *dearest* run type.
   The cheapest run type was the dearest, which is the backwards reading R-02.43's own
   section heading warns about.
2. **R-02.44's artifact was not what any run produced.** An adapter run's save point wrote
   a **full Hugging Face checkpoint of its base** through F5's writer. Acceptance criterion
   15 is about a PEFT directory that loads under plain ``peft``; nothing produced one.

## Why a test is the only instrument that could catch either

Neither defect raises, and neither produces an invalid artifact. The adapter run trained,
its loss fell, its steps were recorded, its manifests validated against
``checkpoint-manifest-v1.json`` and its checkpoints loaded. **The suite was 5,343 green
with the dispatch absent and 5,343 green with it present** - which is the definition of an
unshipped capability, and the reason this file asserts the *optimizer's parameter set* and
the *directory's contents* rather than that a save happened.

## The controls that were run, with their exact outcome

Each mutation was applied with ``PYTHONDONTWRITEBYTECODE=1``, byte-compared against a
backup to prove it had applied (*a ``replace()`` that matches nothing is indistinguishable
from a survivor*), and the ``__pycache__`` trees purged after the restore (*a stale ``.pyc``
outlived a restore once and produced a red from correct source*).

* **MUT-1** - ``plan = run_type_impl.prepare(acquired, adapter=adapter)`` reverted to
  ``plan = runtypes_base.all_parameters(acquired)``, which is exactly the pre-dispatch
  state. **6 failed** - every assertion in :class:`TestAnAdapterRunTrainsTheAdapterAlone`
  and the create run's refusal. Recorded in the feature report.
* **MUT-2** - ``run_type_impl.save(ctx)`` reverted to
  ``change_runtype.save_checkpoint(ctx, base_model=…, parent=…)``, i.e. F5's writer
  unconditionally. **2 failed**, both in
  :class:`TestAnAdapterRunPublishesAPeftDirectory`.
* **MUT-3** - ``runtypes.resolve(run_type)`` forced to ``runtypes.resolve("adapter")``, so
  the dispatch sends *every* run to the wrong run type. **The change path goes red too**,
  which is what proves :class:`TestAChangeRunIsUnregressed` is sensitive to the artifact
  kind rather than merely to a save having occurred.

## Real ``torch`` and real ``peft``, no doubles

A double cannot express either defect: *a double that cannot express the defect makes its
assertion vacuous*, and this project has paid for that twice in one week - an
``object()`` model that only worked on a short-circuit path, and a source-string
reachability test that survived its own mutation. The optimizer's parameter set is the
subject, so the optimizer is the real one over a real ``peft`` wrapper; the published
directory is the subject, so ``peft``'s own ``save_pretrained`` writes it.

``torch``, ``peft`` and ``transformers`` are declared dependencies (ADR-002) and CI
installs them, so nothing here skips and nothing here is ``gpu``-marked: the dispatch is
device-independent and reproduces on the CPU. *A skipping test is not a passing test* - and
these three cases were nearly written inside ``test_adapter_run_gpu.py``, which carries
``pytestmark = pytest.mark.gpu`` and would have deselected them on every push.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from core.config import Settings, reset_settings
from core.errors import Refusal
from tests.support.jobs import insert_run, memory_engine, session_for
from tests.support.test_adapter_doubles import TINY_CONFIG, adapter_config
from training import tasks
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    SourceKind,
)
from training.run_config import RunConfig
from training.runtypes.adapters import paths as adapter_paths

RUN = "run-dispatch"

#: 32-bit optimizer states on purpose. ``bitsandbytes`` has no pure-CPU wheel everywhere -
#: which is exactly why it is an extra (ADR-023) - and ``build`` **refuses**
#: ``optimizer_precision_unavailable`` on the ``8bit`` default rather than downgrading,
#: because the recorded identity would otherwise name a precision that was not used. The
#: subject here is *which parameters* the optimizer covers, not their precision.
CFG = RunConfig(
    learning_rate=1e-2,
    warmup_steps=0,
    batch_size=1,
    optimizer_state_precision="32bit",
)

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

CREATE_RECORD = AcquisitionRecord(
    mode=AcquisitionMode.CREATE,
    source_kind=SourceKind.CONFIG,
    config_identity="d" * 64,
    starting_weights_sha256="e" * 64,
    seed=7,
    base_model="config:llama@" + "d" * 64 + "+seed=7",
    parent=None,
    param_count=512_000,
)


# --------------------------------------------------------------------------------------
# A real tiny model, and the narrowest batch/objective that drives a REAL backward pass
# --------------------------------------------------------------------------------------


def _tiny_module() -> Any:
    """A real ``AutoModelForCausalLM`` from ``TINY_CONFIG``, on the CPU.

    The same construction ``test_adapter_step_backward.py`` and
    ``test_create_run_type.py`` already use, so this is the established convention here
    rather than a new one.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        TINY_CONFIG["model_type"], **{k: v for k, v in TINY_CONFIG.items() if k != "model_type"}
    )
    torch.manual_seed(11)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=False)


@dataclass
class Batch:
    objective: str = "sft"
    record: dict[str, Any] = field(default_factory=lambda: {"batch_sha256": "b" * 64})


@dataclass
class Source:
    calls: int = 0

    def next(self) -> Batch:
        self.calls += 1
        return Batch()


@dataclass
class RealObjective:
    """A **real** forward and backward through whichever module the step was handed.

    F1 defines no objective and implements no loss - the losses are F2's - so the
    placeholder lives here, as it does in ``test_step_loop.py``. What is *not* faked is the
    tensor: the loss is the model's own, so ``loss.backward()`` populates real ``.grad``
    tensors and the gradient assertions below are measurements rather than restatements of
    the fixture.
    """

    name: str = "sft"
    models_seen: list[Any] = field(default_factory=list)

    def loss(self, model: Any, batch: Any) -> Any:
        import torch

        del batch
        self.models_seen.append(model)
        ids = torch.tensor([[1, 2, 3, 4]])
        return model(input_ids=ids, labels=ids).loss


@dataclass
class Registry:
    objective: RealObjective

    def get(self, name: str) -> RealObjective:
        assert name == self.objective.name
        return self.objective


class Tokenizer:
    """Writes the tokenizer files a checkpoint carries **at its root** (R-02.21)."""

    def __init__(self) -> None:
        self.saved_to: list[Path] = []

    def save_pretrained(self, directory: Any) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")
        (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        self.saved_to.append(target)


class RealModelSaver:
    """``save_pretrained`` for the ``change`` path, writing what ``safetensors`` would.

    Not ``transformers``' own saver: this file's subject is *which writer the dispatch
    reaches and what lands on disk*, and a real ``save_pretrained`` of a tiny model would
    add a shard-naming dependency to an assertion about the tokenizer's position.
    """

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The forward pass is the REAL model's - the step must drive real autograd.

        ``__getattr__`` does not cover this: ``__call__`` is looked up on the *type* for
        the ``model(...)`` syntax, so a delegating ``__getattr__`` alone makes the object
        uncallable and the step raises ``TypeError`` instead of stepping.
        """
        return self._module(*args, **kwargs)

    def parameters(self) -> Any:
        return self._module.parameters()

    def named_parameters(self) -> Any:
        return self._module.named_parameters()

    def save_pretrained(self, directory: Any, **_kwargs: Any) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"full-parameter-tensors")
        (target / "config.json").write_text(json.dumps(TINY_CONFIG), encoding="utf-8")


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real model, real optimizer builder, a spy that records what ``build`` was given."""
    from training import optimizer as optimizer_module
    from training import trainer_state

    checkpoints = tmp_path / "checkpoints"
    states = tmp_path / "trainer-state"
    runs = tmp_path / "runs"
    for directory in (checkpoints, states, runs):
        directory.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(  # type: ignore[call-arg]
            _env_file=None, trainer_state_root=states, checkpoint_root=checkpoints
        ),
    )

    module = _tiny_module()
    tokenizer = Tokenizer()
    objective = RealObjective()
    built: list[dict[str, Any]] = []
    real_build = optimizer_module.build

    def build_spy(model: Any, cfg: Any, state: str = "fresh") -> Any:
        """Delegates. *A test that reimplements what it tests cannot fail* - the optimizer
        asserted on below is the real one, this only records which module it was built
        over."""
        optimizer, identity = real_build(model, cfg, state=state)
        built.append({"module": model, "optimizer": optimizer})
        return optimizer, identity

    monkeypatch.setattr(tasks.optimizer_module, "build", build_spy)

    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(session, RUN, status="running")

    yield {
        "engine": engine,
        "module": module,
        "tokenizer": tokenizer,
        "objectives": Registry(objective),
        "objective": objective,
        "source": Source(),
        "built": built,
        "checkpoints": checkpoints,
        "states": states,
        "log_root": tmp_path,
        "trainer_state": trainer_state,
    }
    reset_settings()


def _drive(rig, *, run_type: str, record: AcquisitionRecord, **overrides):
    """Drive the real loop for one run type, through the real dispatch."""
    monkeypatch_target = overrides.pop("acquired", None)
    acquired = monkeypatch_target or AcquiredModel(
        module=rig["module"], tokenizer=rig["tokenizer"], record=record
    )
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as patch:
        patch.setattr(tasks, "acquire_for_run", lambda spec: acquired)
        kwargs: dict[str, Any] = {
            "run_id": RUN,
            "spec": AcquisitionSpec(repo="LiquidAI/LFM2.5-1.2B-Instruct"),
            "cfg": CFG,
            "batch_source": rig["source"],
            "objectives": rig["objectives"],
            "log_root": rig["log_root"],
            "trainer_state_root": rig["states"],
            "max_steps": 1,
            "save_every": 1,
            "signal_source": "corpus",
            "domain": "legal",
            "run_type": run_type,
        }
        kwargs.update(overrides)
        with session_for(rig["engine"]) as session:
            kwargs.setdefault("session", session)
            return tasks.drive_run(**kwargs)


def _named(module: Any) -> dict[str, Any]:
    return dict(module.named_parameters())


# ======================================================================================


class TestAnAdapterRunTrainsTheAdapterAlone:
    """R-02.43, FR-008.19. **The expensive half of the defect.**"""

    def test_the_optimizer_covers_the_adapters_parameters_and_NOTHING_else(self, rig) -> None:
        """Set equality by object identity, in **both** directions.

        Not ``len(...) < len(all)``, and not *"some adapter parameter is in there"*: a
        superset passes both of those while carrying the whole base, which is precisely the
        state this file closes. The two directions say *the adapter is all in* and *nothing
        else is*.
        """
        _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=adapter_config())

        assert len(rig["built"]) == 1, (
            f"the optimizer was built {len(rig['built'])} times; one acquisition, one " f"optimizer"
        )
        wrapped = rig["built"][0]["module"]
        optimizer = rig["built"][0]["optimizer"]

        in_optimizer = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        named = _named(wrapped)
        adapter_params = {id(p) for name, p in named.items() if "lora_" in name}
        base_params = {id(p) for name, p in named.items() if "lora_" not in name}

        assert adapter_params, (
            "the module the optimizer was built over exposes no `lora_` parameter at all, "
            "so `prepare` did not wrap the base and this assertion would be vacuous"
        )
        assert base_params, "a base with no parameters makes the exclusion vacuous too"
        assert in_optimizer == adapter_params, (
            f"the optimizer covers {len(in_optimizer)} parameters; the adapter has "
            f"{len(adapter_params)} and the base has {len(base_params)}. An optimizer over "
            f"the base writes to the bytes acceptance criterion 14 requires to be "
            f"unchanged, and carries a full-parameter gradient and FP32 master copy for the "
            f"run type R-02.43 exists to make cheap."
        )

    def test_only_the_adapters_parameters_RECEIVE_GRADIENTS(self, rig) -> None:
        """The optimizer's membership is construction; this is what the backward pass did.

        Asserted separately because the two can disagree: ``requires_grad`` left on a base
        tensor populates ``.grad`` whether or not the optimizer covers it, and a
        ``clip_grad_norm_`` or a second optimizer would then move it.
        """
        _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=adapter_config())

        wrapped = rig["built"][0]["module"]
        leaked = [
            name
            for name, parameter in wrapped.named_parameters()
            if "lora_" not in name and getattr(parameter, "grad", None) is not None
        ]
        assert not leaked, (
            f"{len(leaked)} base parameter(s) carry a gradient after a real step: " f"{leaked[:5]}"
        )

    def test_the_base_is_byte_unchanged_after_a_real_step(self, rig) -> None:
        """AC14, over the loop rather than over ``prepare`` alone.

        The digest is taken over **canonical** base names, so the pre-wrap and post-wrap
        hashes of one untouched base agree - see
        :func:`training.runtypes.adapter.canonical_base_name`.
        """
        from training.runtypes.adapter import base_parameter_digest

        before = base_parameter_digest(rig["module"])
        _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=adapter_config())
        after = base_parameter_digest(rig["built"][0]["module"])

        assert after == before, "an adapter run moved the base's bytes"

    def test_the_step_forwarded_through_the_WRAPPER_not_the_bare_base(self, rig) -> None:
        """The module handed to the objective is ``plan.module``.

        Without it the adapter's matrices are in the graph only by accident of ``peft``
        replacing modules in place, and nothing would notice a ``prepare`` that returned a
        module the step never saw.
        """
        _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=adapter_config())

        seen = rig["objective"].models_seen
        assert len(seen) == 1
        assert seen[0] is rig["built"][0]["module"]
        assert seen[0] is not rig["module"], (
            "the step forwarded through the unwrapped base; `prepare`'s wrapper was built "
            "and then not used"
        )

    def test_assert_only_adapter_requires_grad_RUNS_on_this_runs_module(
        self, rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The production assertion is **on the dispatch path**, once, over the wrapper.

        A call-count assertion on its own is weak - *"was called" passes against a call
        sending the wrong arguments* - which is why the payload is asserted too and why the
        two tests above measure the optimizer and the gradients instead of the call. This
        pins the production guard's *presence*: it is what turns a base tensor left
        trainable into a refusal **at construction** rather than into a hash comparison that
        fails at the end of a run, having already written to the bytes acceptance criterion
        14 requires to be unchanged.

        Patched on ``training.runtypes.adapter``'s own binding, not on
        ``training.runtypes.adapters.build``'s: ``adapter.py`` imports the function by name,
        so patching the defining module would leave the live reference untouched and this
        test would pass against nothing.
        """
        from training.runtypes import adapter as adapter_runtype

        real = adapter_runtype.assert_only_adapter_requires_grad
        seen: list[Any] = []

        def spy(wrapped: Any) -> Any:
            seen.append(wrapped)
            return real(wrapped)

        monkeypatch.setattr(adapter_runtype, "assert_only_adapter_requires_grad", spy)
        _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=adapter_config())

        assert len(seen) == 1, (
            f"the frozen-base assertion ran {len(seen)} time(s) on the dispatch path; once "
            f"per acquisition, over the module the optimizer is then built from"
        )
        assert seen[0] is rig["built"][0]["module"], (
            "it ran over a different module than the one the optimizer was built over, so "
            "it asserted nothing about the parameter set that trains"
        )

    def test_an_adapter_run_with_no_adapter_block_is_a_NAMED_refusal(self, rig) -> None:
        """``prepare`` reached with no configuration refuses rather than defaulting a rank:
        *a defaulted rank trains an adapter the operator did not ask for and records it in a
        manifest as though they had.*"""
        with pytest.raises(Refusal) as raised:
            _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=None)
        assert raised.value.code == "adapter_config_required"


class TestAnAdapterRunPublishesAPeftDirectory:
    """R-02.44, acceptance criterion 15. **The artifact half of the defect.**"""

    def test_the_save_point_writes_the_three_files_and_no_checkpoint(self, rig) -> None:
        """The directory's **contents**, not that a save happened.

        ``adapter_config.json`` is the file ``PeftConfig.from_pretrained`` reads, so a
        directory without it is not loadable by the standard tooling R-02.51 promises -
        which is why F8 writes three files and treats the normative layout's two-file
        enumeration as incomplete rather than exclusive.
        """
        result = _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=adapter_config())

        assert len(result.checkpoints) == 1, (
            f"expected one published artifact, got {result.checkpoints}. If this is empty "
            f"the save point published nothing and the run produced no artifact at all."
        )
        published = Path(result.checkpoints[0])
        assert published.is_dir(), published
        names = {entry.name for entry in published.iterdir()}
        assert adapter_paths.WEIGHTS_NAME in names, names
        assert adapter_paths.PEFT_CONFIG_NAME in names, names
        assert adapter_paths.MANIFEST_NAME in names, names

        assert "model.safetensors" not in names, (
            "a full parameter set was written where an adapter belongs - this is the "
            "pre-dispatch behaviour, in which an adapter run published a checkpoint of its "
            "base through F5's writer"
        )
        # F5's checkpoint manifest must not be there either: the two artifact kinds have
        # two schemas and `kind` is a const in each, so one directory carrying both would
        # be a node with two identities in R-02.29's tree.
        assert "manifest.json" not in names, names

    def test_the_tensor_keys_follow_the_PEFT_CONVENTION(self, rig) -> None:
        """FR-008.30: the adapter loads in standard tooling, not only in this suite.

        Read off the written ``safetensors`` file rather than off the in-memory module,
        because the file is what travels.
        """
        from safetensors import safe_open

        result = _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=adapter_config())
        weights = Path(result.checkpoints[0]) / adapter_paths.WEIGHTS_NAME

        with safe_open(str(weights), framework="pt") as handle:
            keys = list(handle.keys())

        assert keys, "the adapter file carries no tensors"
        assert all("lora_" in key for key in keys), (
            f"a non-adapter tensor was written into the adapter file: "
            f"{[k for k in keys if 'lora_' not in k][:5]}"
        )
        assert any(".lora_A" in key for key in keys) and any(
            ".lora_B" in key for key in keys
        ), f"PEFT's convention is `<module>.lora_A.weight` / `.lora_B.weight`; got {keys[:4]}"

    def test_the_manifest_on_disk_names_the_rank_the_run_was_GIVEN(self, rig) -> None:
        """One object produces both the manifest's numbers and ``save_pretrained``'s, because
        miLLM's loader cross-checks them and refuses a mismatch (task 9.2)."""
        result = _drive(
            rig,
            run_type="adapter",
            record=HUB_RECORD,
            adapter=adapter_config(rank=8, alpha=16.0),
        )
        directory = Path(result.checkpoints[0])
        manifest = json.loads((directory / adapter_paths.MANIFEST_NAME).read_text())
        peft_config = json.loads((directory / adapter_paths.PEFT_CONFIG_NAME).read_text())

        assert manifest["adapter"]["rank"] == 8
        assert manifest["adapter"]["alpha"] == 16.0
        # PEFT's own spellings, which is the whole reason a cross-check is needed.
        assert peft_config["r"] == 8
        assert peft_config["lora_alpha"] == 16.0
        assert manifest["run_id"] == RUN and manifest["step"] == 1
        assert manifest["base"]["name"] == HUB_RECORD.base_model
        assert (
            manifest["base"]["manifest_sha256"] is None
        ), "a Hub base was never a saved checkpoint, and null there is what says so"


class TestAChangeRunIsUnregressed:
    """The direction ten of eleven features depend on (R-02.21, ADR-015).

    A mutation that routed *everything* through the adapter writer would pass every
    assertion above and break the whole product, which is what these two are for.
    """

    def test_a_change_run_writes_an_HF_CHECKPOINT_with_the_tokenizer_at_the_ROOT(self, rig) -> None:
        """Acceptance criterion 2's precondition. A checkpoint whose tokenizer files sit one
        level down cannot be loaded by ``transformers``, vLLM, TGI or miLLM's loader at all -
        the defect miLLM's Feature 032 review found in the original layout."""
        result = _drive(
            rig,
            run_type="change",
            record=HUB_RECORD,
            acquired=AcquiredModel(
                module=RealModelSaver(rig["module"]),
                tokenizer=rig["tokenizer"],
                record=HUB_RECORD,
            ),
        )

        assert len(result.checkpoints) == 1, result.checkpoints
        directory = Path(result.checkpoints[0])
        names = {entry.name for entry in directory.iterdir()}
        assert "manifest.json" in names, names
        assert "model.safetensors" in names, names
        # AT THE ROOT, not in a `tokenizer/` subdirectory.
        assert "tokenizer.json" in names, names
        assert "tokenizer_config.json" in names, names
        assert adapter_paths.MANIFEST_NAME not in names, (
            "an adapter manifest was written for a `change` run - the dispatch sent it to "
            "the wrong run type"
        )
        assert adapter_paths.WEIGHTS_NAME not in names, names

    def test_the_change_manifest_carries_F1s_lineage_pair_verbatim(self, rig) -> None:
        """``base_model`` is the lineage **root** and ``parent`` is what was loaded. F1's
        own lineage functions derived both; F8 does not re-derive them, which is what fixed
        the carry-forward defect where a continue rooted the tree at its parent."""
        result = _drive(
            rig,
            run_type="change",
            record=HUB_RECORD,
            acquired=AcquiredModel(
                module=RealModelSaver(rig["module"]),
                tokenizer=rig["tokenizer"],
                record=HUB_RECORD,
            ),
        )
        manifest = json.loads((Path(result.checkpoints[0]) / "manifest.json").read_text())
        assert manifest["base_model"] == HUB_RECORD.base_model
        assert manifest["parent"] is None
        assert manifest["domain"] == "legal"
        assert manifest["signal_source"] == "corpus"


class TestACreateRunsTokenizerRefusalIsLIVE:
    """``CreateRunType.prepare`` raises ``tokenizer_required`` and was unreachable.

    Wiring ``prepare`` makes that refusal live, which is why **both directions** are here:
    a create run that *was* given a tokenizer must not be refused, or every creation run
    breaks. The tokenizer supply mechanism (``tokenizer_path`` / ``tokenizer_repo`` on
    ``AcquisitionSpec``) landed hours before this dispatch did, so the refusal now fires
    only on the case it names.
    """

    def test_a_create_run_WITHOUT_a_tokenizer_refuses_by_name(self, rig) -> None:
        """miForge does not train a tokenizer (BRD-02 §8) and will not invent a default: a
        mismatched tokenizer produces a model that trains happily and generates nonsense."""
        with pytest.raises(Refusal) as raised:
            _drive(
                rig,
                run_type="create",
                record=CREATE_RECORD,
                acquired=AcquiredModel(module=rig["module"], tokenizer=None, record=CREATE_RECORD),
            )
        assert raised.value.code == "tokenizer_required"

    def test_a_create_run_WITH_a_tokenizer_runs_and_saves(self, rig) -> None:
        """The half that matters more. A refusal that fires on every create run is worse
        than one that fires on none, and this is the assertion that tells them apart."""
        result = _drive(
            rig,
            run_type="create",
            record=CREATE_RECORD,
            acquired=AcquiredModel(
                module=RealModelSaver(rig["module"]),
                tokenizer=rig["tokenizer"],
                record=CREATE_RECORD,
            ),
        )
        assert result.steps_completed == 1
        assert len(result.checkpoints) == 1
        manifest = json.loads((Path(result.checkpoints[0]) / "manifest.json").read_text())
        # R-02.41's three lineage facts: the config identity is the root and nothing is
        # the parent, derived by F1's `lineage_for_creation` rather than spelled in F8.
        assert manifest["base_model"] == CREATE_RECORD.base_model
        assert manifest["parent"] is None
        names = {entry.name for entry in Path(result.checkpoints[0]).iterdir()}
        assert "tokenizer.json" in names, names


class TestTheDispatchGoesThroughTheLiveRegistry:
    def test_an_unknown_run_type_refuses_LISTING_the_three(self, rig) -> None:
        """Resolved through :func:`training.runtypes.resolve`, so a bare ``KeyError`` cannot
        reach the worker as a 500 where a named refusal belongs (ADR-013)."""
        with pytest.raises(Refusal) as raised:
            _drive(rig, run_type="lora", record=HUB_RECORD)
        assert raised.value.code == "run_type_unknown"
        assert set(raised.value.details["known"]) == {"change", "create", "adapter"}

    def test_removing_a_registration_makes_the_dispatch_unreachable(self, rig) -> None:
        """The removal control, as a test rather than as a note.

        ``adapter`` deleted from the live registry and the loop refuses by name - which is
        what proves the loop reaches the **registry** rather than importing the module.
        """
        from training import runtypes

        registry = runtypes._REGISTRY
        removed = registry.pop("adapter")
        try:
            with pytest.raises(Refusal) as raised:
                _drive(rig, run_type="adapter", record=HUB_RECORD, adapter=adapter_config())
            assert raised.value.code == "run_type_unknown"
        finally:
            registry["adapter"] = removed
        assert set(runtypes.known()) == {"change", "create", "adapter"}
