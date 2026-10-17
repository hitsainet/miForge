"""D3: the adapter step could not complete a single backward pass.  **Unmarked, on purpose.**

``optimizer.build`` turns gradient checkpointing on (R-02.13), and it is **reentrant by
default**. Reentrant ``torch.utils.checkpoint`` builds no autograd graph for a segment whose
**inputs** all have ``requires_grad=False`` - which on a frozen PEFT base is every input to
the first checkpointed block. So ``loss.backward()`` raised *"element 0 of tensors does not
require grad and does not have a grad_fn"* and **no adapter run could complete one step** -
the cheapest run type, the one that holds no barrier, and the one BRD-02 R-02.43 leans on.

``optimizer._needs_input_grad_hook`` plus a conditional ``enable_input_require_grads()`` is
the fix. The second case below is what pins that **conditionality**, and the third is what
stops the first two covering for each other.

## Why this file is unmarked, and why it was not

These three cases were written inside ``tests/integration/test_adapter_run_gpu.py``, which
carries ``pytestmark = pytest.mark.gpu``, and that was wrong for them: *the defect is
device-independent and reproduces on CPU*. ``pyproject.toml``'s
``addopts = "-m 'not gpu and not upstream'"`` therefore deselected them on every push, so the
fix for the run type R-02.43 leans on was **unpinned in CI** from the moment it was written.
Moved out 2026-10-16. The on-card module keeps a pointer to here.

That is this project's recorded *"a skipping test is not a passing test"* trap in its
quietest form - not a reported skip that ``-rs`` would list, a **silent deselection** that
reports nothing. ``-rs`` on this file lists nothing: ``torch``, ``transformers`` and ``peft``
are declared dependencies (ADR-002), CI installs them, and nothing here reaches for a device.

## Why unit rather than integration

The subject is one function's behaviour - ``training.optimizer``'s decision about whether to
install the input-gradient hook - exercised over a real tiny model because autograd is the
thing under test and a double cannot express the defect. No database, no API, no
orchestration. It sits beside ``test_adapter_requires_grad.py``, which covers the frozen-base
half over doubles, so a reader looking for *"does the adapter step backward"* finds it next to
*"is the base actually frozen"*. ``tests/unit/test_create_run_type.py`` already builds real
``AutoModelForCausalLM`` instances here, so this is the established convention, not a new one.

**What is NOT here, deliberately:** the whole-loop assertions that genuinely need a card -
a real optimizer over a real adapter on CUDA, the base digest hashed before and after a real
step (AC14), the real PEFT save - stay ``gpu``-marked in
``tests/integration/test_adapter_run_gpu.py``. Split rather than watered down.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.test_adapter_doubles import TINY_CONFIG, adapter_config
from training.runtypes import resolve

ADAPTER = resolve("adapter")


def _any_device() -> str:
    """``"cuda"`` when a card answers, otherwise ``"cpu"``. Never an index (ADR-008)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def acquired_anywhere() -> Any:
    """A tiny model on whatever device this process has. See the note above."""
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
    module = AutoModelForCausalLM.from_config(config, trust_remote_code=False).to(_any_device())
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
        device_uuid=None,
    )


def _tiny_module() -> Any:
    """A bare tiny model - nothing frozen, i.e. the full-parameter path."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        TINY_CONFIG["model_type"], **{k: v for k, v in TINY_CONFIG.items() if k != "model_type"}
    )
    torch.manual_seed(11)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=False).to(_any_device())


#: 32-bit states on purpose: ``bitsandbytes`` has no pure-CPU wheel everywhere, and these two
#: cases are about the backward pass, not about the optimizer's precision. The 8-bit path is
#: covered on the card by ``test_acquisition_gpu.py``.
def _thirty_two_bit_config() -> Any:
    from training.run_config import RunConfig

    return RunConfig(
        learning_rate=1e-2,
        warmup_steps=0,
        batch_size=1,
        optimizer_state_precision="32bit",
    )


class TestTheAdapterStepCanActuallyBackward:
    """``optimizer.build`` turns gradient checkpointing on (R-02.13), and it is **reentrant
    by default**. Reentrant ``torch.utils.checkpoint`` builds no autograd graph for a segment
    whose inputs all have ``requires_grad=False`` - which on a frozen base is every input to
    the first checkpointed block. So ``loss.backward()`` raised *"element 0 of tensors does
    not require grad and does not have a grad_fn"* and **no adapter run could complete one
    step**. Found by running this module on the card for the first time, 2026-10-16.

    ``optimizer._needs_input_grad_hook`` plus ``enable_input_require_grads()`` is the fix,
    conditional on the frozen-input case so the full-parameter path is untouched. The second
    case below is what pins that conditionality.
    """

    def test_a_real_backward_moves_the_adapters_parameters(self, acquired_anywhere: Any) -> None:
        import torch

        from training import optimizer as optimizer_module

        plan = ADAPTER.prepare(
            acquired_anywhere,
            adapter=adapter_config(rank=4, alpha=8.0, target_modules=("q_proj",)),
        )
        before = [parameter.detach().clone() for parameter in plan.trainable]
        optimizer, _ = optimizer_module.build(plan.module, _thirty_two_bit_config())

        ids = torch.tensor([[1, 2, 3, 4]], device=plan.module.device)
        # The assertion is that this line does not raise. Named rather than left implicit,
        # because the defect was an exception and not a wrong number.
        plan.module(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()

        assert plan.module.config.use_cache is False, "R-02.13: use_cache goes off with it"
        assert all(
            parameter.grad is not None for parameter in plan.trainable
        ), "gradient checkpointing built no graph reaching the adapter"
        moved = [
            not torch.allclose(was, now) for was, now in zip(before, plan.trainable, strict=True)
        ]
        assert any(moved), (
            "a backward pass and an optimizer step left every adapter parameter where it "
            "was, so the step trained nothing"
        )

    def test_the_full_parameter_path_is_not_touched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No hook is installed when nothing is frozen, and the step still works.

        The call count is the assertion, not just the outcome: ``enable_input_require_grads``
        is harmless on a full-parameter model, so an unconditional fix would pass an
        outcome-only test while having changed the path BRD-02 §6's memory budget was
        reasoned about.
        """
        import torch

        from training import optimizer as optimizer_module

        module = _tiny_module()
        assert optimizer_module._needs_input_grad_hook(module) is None

        calls: list[int] = []
        original = type(module).enable_input_require_grads
        monkeypatch.setattr(
            type(module),
            "enable_input_require_grads",
            lambda self, *a, **k: (calls.append(1), original(self, *a, **k))[1],
        )

        optimizer, _ = optimizer_module.build(module, _thirty_two_bit_config())
        ids = torch.tensor([[1, 2, 3, 4]], device=module.device)
        module(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()

        assert calls == [], (
            "the frozen-base fix reached the full-parameter path; it must be conditional, "
            "so that path executes exactly the code it executed before"
        )
        assert any(parameter.grad is not None for parameter in module.parameters())

    def test_a_partly_frozen_model_whose_EMBEDDING_is_trainable_needs_no_hook(self) -> None:
        """The condition is *"the checkpointed segment's input carries no gradient"*, not
        *"something is frozen"*.

        A run that freezes some blocks and keeps the embedding trainable already has a
        gradient-carrying input, so the hook is unnecessary there too. Without this case the
        trainable/frozen test and the embedding test cover for each other - mutating either
        one alone leaves the suite green, which is how a redundant pair reads as two
        protections and behaves as none.
        """
        from training import optimizer as optimizer_module

        module = _tiny_module()
        frozen = 0
        for name, parameter in module.named_parameters():
            if "layers.1." in name:
                parameter.requires_grad_(False)
                frozen += 1
        assert frozen > 0, "the tiny config changed shape; this case froze nothing"
        assert any(p.requires_grad for p in module.get_input_embeddings().parameters())

        assert optimizer_module._needs_input_grad_hook(module) is None
