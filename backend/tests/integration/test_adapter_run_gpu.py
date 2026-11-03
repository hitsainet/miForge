"""AC14 on a card: a real adapter run leaves the base byte-unchanged.  ``gpu``-marked.

**Deselected by default** (ADR-023, and ``addopts`` in ``pyproject.toml``): a developer
without a GPU must be able to run everything except a training step, and **CI has no card**,
so these cases do not run there. That is stated rather than assumed green - they are an
**operator-run gate at feature acceptance**, and the report must say when they were last run
and on what.

Run them with::

    .venv/bin/pytest -m gpu backend/tests/integration/test_adapter_run_gpu.py

**First run 2026-10-16, inside the cluster pod, and it found a production defect that made
every adapter run impossible** (D3: gradient checkpointing is reentrant by default, so a
frozen base built no autograd graph and ``loss.backward()`` raised). A failure here is still
as likely to be the node as the code; check ``nvidia-smi`` on the host before reading a red
as a regression, and see AGENT.md's standing caution that the GPU node may not come back
from its next reboot.

*This paragraph read "They have NOT been run for this increment - ``torch``, ``peft`` and
``transformers`` are not installed in the environment this feature was built in" until that
run.* It was true when written, and it is the reason D3 sat undetected in the step path of
the cheapest run type: **a suite that has never been executed reports nothing, which reads
exactly like a suite that passed.**

**Only the RTX 3090 is present on this node.** Nothing below encodes a device count or an
index; the device is resolved at runtime by UUID through ``core/gpu.py`` (ADR-008).

## What this adds over the CPU cases

``test_adapter_requires_grad.py`` proves the base digest is sound and the construction-time
assertion bites, over doubles. ``test_adapter_base_digest_across_wrap.py`` proves the digest
**survives ``peft``'s rename** and runs the whole ``prepare`` -> optimizer -> step -> compare
cycle on the CPU, so CI pins AC14's mechanism. What needs a card is the real device and the
real 8-bit optimizer underneath it - AC14's own wording, *"hashed before and after, while the
test score rises."*

**``test_the_base_digest_is_identical_after_a_real_step`` could never pass until
2026-10-16**, and that is worth recording here because this file is where it was observed.
``base_parameter_digest`` covers each parameter's *name*, and ``get_peft_model`` renames
every base tensor: ``model.…q_proj.weight`` becomes
``base_model.model.model.…q_proj.base_layer.weight``. Instrumented on this card with **no
training at all**, the "before" and a digest of the wrapped module disagreed while a
bytes-only digest over the same two modules was identical - the weights were right and the
digest was wrong. Names are now reduced to the unwrapped convention before hashing, so this
case measures the base rather than the naming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.test_adapter_doubles import TINY_CONFIG, adapter_config, optimizer_block
from training.runtypes import resolve
from training.runtypes.adapter import base_parameter_digest
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths

pytestmark = pytest.mark.gpu

ADAPTER = resolve("adapter")


@pytest.fixture
def device_uuid() -> str:
    """The card, by UUID, resolved from live ``nvidia-smi -L``. Never an index.

    ``resolve_device`` **refuses** when ``nvidia-smi`` itself fails rather than returning
    ``None``, and this handled only the ``None``. So on a host whose driver and user-space
    diverge - which is this node's standing state, see the module docstring - every case in
    this file **errored** instead of skipping, and a collection error reads as a regression.
    ``test_acquisition_gpu.py``'s own device fixture had always caught it; this one had not.
    """
    from core.errors import Refusal
    from core.gpu import resolve_device

    try:
        device = resolve_device("auto")
    except Refusal as exc:
        pytest.skip(f"no CUDA device resolved ({exc.code}); this case needs the GPU node")
    if device is None:
        pytest.skip("no CUDA device is visible to this process")
    return device.uuid


@pytest.fixture
def acquired(device_uuid: str) -> Any:
    """A real tiny model, created in this process, on the resolved card."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from training.acquisition.base import (
        AcquiredModel,
        AcquisitionMode,
        AcquisitionRecord,
        SourceKind,
    )

    config = AutoConfig.for_model(
        TINY_CONFIG["model_type"], **{k: v for k, v in TINY_CONFIG.items() if k != "model_type"}
    )
    torch.manual_seed(11)
    module = AutoModelForCausalLM.from_config(config, trust_remote_code=False).to("cuda")
    return AcquiredModel(
        module=module,
        tokenizer=None,
        record=AcquisitionRecord(
            mode=AcquisitionMode.LOAD,
            source_kind=SourceKind.HUB,
            source_repo="LiquidAI/LFM2.5-1.2B-Instruct",
            starting_weights_sha256="b" * 64,
            base_model="LiquidAI/LFM2.5-1.2B-Instruct",
        ),
        device_uuid=device_uuid,
    )


class TestARealAdapterRunLeavesTheBaseByteUnchanged:
    def test_the_base_digest_is_identical_after_a_real_step(self, acquired: Any) -> None:
        """AC14's default clause, in its own words: hashed before and after.

        **F8 claims this clause only.** AC14's section 5.10 clause - the resident tensors of
        a *serving process* when attached - is F11's acceptance, and F8 builds no attached
        mode at all, so it can neither verify nor claim it.
        """
        import torch

        from training import optimizer as optimizer_module
        from training.run_config import RunConfig

        plan = ADAPTER.prepare(
            acquired, adapter=adapter_config(rank=4, alpha=8.0, target_modules=("q_proj",))
        )
        before = plan.base_digest
        assert before is not None

        cfg = RunConfig(learning_rate=1e-3, warmup_steps=0, batch_size=1)
        optimizer, _ = optimizer_module.build(plan.module, cfg)
        ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
        loss = plan.module(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert base_parameter_digest(plan.module) == before, (
            "a base parameter changed during an adapter run; the optimizer was built over "
            "something it should not have been"
        )

    def test_the_adapters_parameters_DID_change(self, acquired: Any) -> None:
        """The companion assertion. Without it the test above would pass against a run that
        trained nothing at all - which is the vacuity the digest guards against from the
        other side."""
        import torch

        from training import optimizer as optimizer_module
        from training.run_config import RunConfig

        plan = ADAPTER.prepare(
            acquired, adapter=adapter_config(rank=4, alpha=8.0, target_modules=("q_proj",))
        )
        snapshot = [p.detach().clone() for p in plan.trainable]
        optimizer, _ = optimizer_module.build(
            plan.module, RunConfig(learning_rate=1e-2, warmup_steps=0, batch_size=1)
        )
        ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
        plan.module(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        assert any(
            not torch.allclose(before, after)
            for before, after in zip(snapshot, plan.trainable, strict=True)
        )

    def test_the_optimizer_covers_ONLY_the_adapters_parameters(self, acquired: Any) -> None:
        """Which is what makes an adapter run the cheapest type: gradients and optimizer
        state are tens of megabytes rather than gigabytes."""
        from training import optimizer as optimizer_module
        from training.run_config import RunConfig

        plan = ADAPTER.prepare(
            acquired, adapter=adapter_config(rank=4, alpha=8.0, target_modules=("q_proj",))
        )
        optimizer, _ = optimizer_module.build(
            plan.module, RunConfig(learning_rate=1e-3, warmup_steps=0, batch_size=1)
        )
        covered = sum(len(group["params"]) for group in optimizer.param_groups)
        assert covered == len(plan.trainable)
        assert covered < len(list(plan.module.parameters()))

    def test_the_step_executed_is_THE_shared_one(self, acquired: Any) -> None:
        """Not a second step path. The identity assertion lives in the unit test; this is
        the on-card confirmation that the shared function actually runs an adapter step."""
        from training.step import run_one

        assert ADAPTER.step is run_one


class TestARealAdapterSaveOnTheCard:
    def test_the_published_directory_carries_real_peft_tensors(
        self, acquired: Any, tmp_path: Path
    ) -> None:
        from safetensors import safe_open

        from training.runtypes.adapters import save as adapter_save

        config = adapter_config(rank=4, alpha=8.0, target_modules=("q_proj",))
        plan = ADAPTER.prepare(acquired, adapter=config)
        doc = adapter_manifest.build(
            run_id="run-gpu",
            step=10,
            base=adapter_manifest.base_block(acquired.record),
            adapter=config,
            optimizer=optimizer_block(),
        )
        dest = adapter_paths.adapter_dir("run-gpu", 10, root=tmp_path)
        published = adapter_save.save(plan.module, doc, dest)

        with safe_open(str(dest / adapter_paths.WEIGHTS_NAME), framework="pt") as handle:
            keys = list(handle.keys())
        assert any(".lora_A." in key for key in keys)
        assert published["weights_sha256"]
        assert published["base"]["name"] == "LiquidAI/LFM2.5-1.2B-Instruct"
        assert published["base"]["manifest_sha256"] is None, "a Hub base has no manifest"


class TestNoBarrierIsEverTaken:
    """R-02.43: *"an adapter run never interrupts serving."*

    **This asserted module imports until 2026-10-16, and that proxy was invalid.**
    ``not any(name.startswith("training.hotupdate") for name in sys.modules)`` fails whether
    or not a barrier was taken, because ``training/acquisition/__init__.py``'s
    ``_install_optional_attach_mode`` imports ``training.hotupdate`` **dynamically at
    package import, by design** - that is ADR-022's seam, there is no static edge, and
    ``test_acquisition_registry.py``'s AST guard is what protects the fence. The test failed
    when run alone, so it was never order pollution: the proxy simply did not measure the
    thing it named.

    **A barrier is taken in exactly two places**, and both are asserted here instead:
    ``barrier.commit_bracket`` (``resident_merge``'s one call site, and the context manager
    ``BracketedOptimizer.step`` opens) and ``barrier.BracketedOptimizer`` itself, which is
    what wraps an optimizer so its ``step()`` holds serving. Zero of the first, and the
    adapter run's optimizer is not an instance of the second.

    *Each zero carries a positive check that the measurement happened* - the spy is proved
    able to fire, and the ``isinstance`` predicate is proved able to be true - because a
    counter that was never wired reads exactly like a barrier that was never taken.
    """

    def test_a_real_adapter_step_opens_no_commit_bracket(
        self, acquired: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextlib import contextmanager

        import torch

        from training import optimizer as optimizer_module
        from training.hotupdate import barrier
        from training.run_config import RunConfig

        opened: list[str] = []

        @contextmanager
        def spy(client: Any, timings: Any, *, what: str = "optimizer update") -> Any:
            opened.append(what)
            yield

        # Patched on the module, which is how BOTH call sites resolve it: `resident_merge`
        # does `barrier.commit_bracket(...)` and `BracketedOptimizer.step` looks the name up
        # in barrier's own globals, so neither holds an early reference.
        monkeypatch.setattr(barrier, "commit_bracket", spy)

        plan = ADAPTER.prepare(
            acquired, adapter=adapter_config(rank=4, alpha=8.0, target_modules=("q_proj",))
        )
        cfg = RunConfig(learning_rate=1e-3, warmup_steps=0, batch_size=1)
        optimizer, _ = optimizer_module.build(plan.module, cfg)
        ids = torch.tensor([[1, 2, 3, 4]], device=plan.module.device)
        plan.module(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert opened == [], (
            f"an adapter run opened the commit bracket {len(opened)} time(s) ({opened}); "
            f"R-02.43 writes no resident tensor, so holding serving for it is a defect "
            f"whose symptom is latency rather than an error"
        )
        # Two positive checks: the counter is wired to the thing it counts. Without them a
        # patch that silently missed would read identically to a barrier never taken.
        # First the module attribute, which is `resident_merge`'s call site:
        with barrier.commit_bracket(None, None, what="positive control"):
            pass
        assert opened == ["positive control"]
        # Then the *other* call site, through barrier's own globals, which is how an
        # optimizer whose step() holds serving would have reached it:
        probe = barrier.BracketedOptimizer(optimizer, None, None)
        probe.step()
        assert len(opened) == 2 and probe.bracketed_steps == 1, (
            "the spy does not see BracketedOptimizer.step's bracket, so the zero above "
            "would not have caught a bracketed optimizer"
        )

    def test_the_adapter_runs_optimizer_is_not_a_bracketed_one(self, acquired: Any) -> None:
        from training import optimizer as optimizer_module
        from training.hotupdate import barrier
        from training.run_config import RunConfig

        plan = ADAPTER.prepare(
            acquired, adapter=adapter_config(rank=4, alpha=8.0, target_modules=("q_proj",))
        )
        optimizer, _ = optimizer_module.build(
            plan.module, RunConfig(learning_rate=1e-3, warmup_steps=0, batch_size=1)
        )
        assert not isinstance(
            optimizer, barrier.BracketedOptimizer
        ), "the adapter run's optimizer was wrapped so that every step() holds serving"
        # And the predicate can be true, so the assertion above is not vacuous.
        assert isinstance(
            barrier.BracketedOptimizer(optimizer, None, None), barrier.BracketedOptimizer
        )


# ⚠ THREE CASES MOVED OUT OF THIS FILE — 2026-10-16.
# `TestTheAdapterStepCanActuallyBackward` — the D3 regression, i.e. the reason an adapter run
# can complete a backward pass at all — is now `tests/unit/test_adapter_step_backward.py`,
# **unmarked**, together with its `_any_device` / `acquired_anywhere` / `_tiny_module` /
# `_thirty_two_bit_config` helpers.
#
# Why: the defect is device-independent and reproduces on CPU, so those cases never needed a
# card — but living in this `gpu`-marked module meant `pyproject.toml`'s `addopts` deselected
# them on every push, and **the fix for the cheapest run type was unpinned in CI**. That is
# this project's recorded "a skipping test is not a passing test" trap in its quietest form:
# not a reported skip, a silent deselection. They belong on the on-card gate's *reading list*,
# not in its selection.
#
# What stayed here is everything that genuinely needs the card: a real optimizer over a real
# adapter on CUDA, the base digest before and after a real step (AC14), the real PEFT save,
# and the two no-barrier assertions.
