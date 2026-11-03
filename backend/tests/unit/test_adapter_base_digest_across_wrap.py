"""AC14's comparison is possible at all: one base, hashed either side of a real PEFT wrap.

**Unmarked, and that is the point.** This is the control for the defect that made BRD-02
acceptance criterion 14 *structurally unverifiable* - the base digest could never match
across the wrap, whatever happened to the weights - and it needs **no card**: ``peft``'s
rename is device-independent and reproduces on CPU in under a second.

## What was wrong, measured rather than reasoned

``base_parameter_digest`` hashes each parameter's **name** as well as its bytes, and
``get_peft_model`` renames every base tensor::

    model.layers.0.self_attn.q_proj.weight
    base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight

With **no training at all** - ``prepare()`` and nothing else - the recorded "before" and a
digest of the returned wrapped module disagreed, while a bytes-only digest over the same two
modules was byte-identical. So ``test_adapter_run_gpu.py``'s
``test_the_base_digest_is_identical_after_a_real_step`` could only ever fail, and it would
have failed for a reason that has nothing to do with the base.

## Why the suite did not see it

``test_adapter_requires_grad.py``'s digest cases ran entirely against ``FakeModule``s whose
parameter names were **already in PEFT form**, so the double never underwent the rename and
the guarantee was tested exactly where the bug cannot exist - AGENT.md's *"a fixture whose
fields agree by construction"*. And the one case that looked like it covered the ordering
asserted on the **source text** that ``base_parameter_digest(`` appears before
``build_adapter(``, which pins a call order rather than checking that the comparison works;
any correct fix turned it red. Both are repaired in that file; this module is the positive
check that the comparison itself holds, through the real library.

## The property that decides the fix

The digest exists so that *someone else* can verify the base was untouched. A consumer holds
the base (``manifest.base.name`` - a Hub repo id or a checkpoint directory) and the adapter;
they load the base with ``from_pretrained`` and **do not wrap it**. So the canonical naming
has to be the unwrapped one, and the digest must not depend on the adapter configuration or
on the installed ``peft`` version.
``test_two_different_adapter_configurations_hash_the_same_base_identically`` is that
property as an assertion; it is exactly what a post-wrap digest could not give, because
``base_layer`` appears only on **targeted** modules.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from tests.support.test_adapter_doubles import TINY_CONFIG, adapter_config
from training.runtypes import resolve
from training.runtypes.adapter import (
    PEFT_SEGMENT,
    base_parameter_digest,
    canonical_base_name,
)

ADAPTER = resolve("adapter")


def _tiny_base() -> Any:
    """A real tiny causal LM on the CPU. The base, unwrapped, as a consumer would hold it."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        TINY_CONFIG["model_type"], **{k: v for k, v in TINY_CONFIG.items() if k != "model_type"}
    )
    torch.manual_seed(11)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=False)


def _wrap(module: Any, **overrides: Any) -> Any:
    """Through **production's** builder, not a hand-rolled ``get_peft_model`` call.

    A test that wrapped the model itself would pass against a ``prepare`` that wrapped it
    differently, which is the whole class of defect this file exists for.
    """
    from training.runtypes.adapters.build import build as build_adapter

    wrapped, _ = build_adapter(module, adapter_config(**overrides))
    return wrapped


def _acquired(module: Any) -> Any:
    from training.acquisition.base import (
        AcquiredModel,
        AcquisitionMode,
        AcquisitionRecord,
        SourceKind,
    )

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


class TestTheRenameIsReal:
    """First the measurement, so the cases below are not asserting against a remembered
    naming scheme. If ``peft`` ever stops renaming, this goes red and says so - and the
    normalisation can then be removed deliberately rather than discovered to be dead."""

    def test_peft_renames_every_base_tensor_on_wrap(self) -> None:
        module = _tiny_base()
        before = {name for name, _ in module.named_parameters()}
        wrapped = _wrap(module, target_modules=("q_proj",))
        after = {name for name, _ in wrapped.named_parameters() if PEFT_SEGMENT not in name}

        assert before and after
        assert before.isdisjoint(after), (
            "peft no longer renames the base, so the normalisation this file protects is "
            "dead code rather than a fix - remove it on purpose, do not leave it"
        )
        assert any(".base_layer." in name for name in after), "no base_layer indirection"

    def test_canonicalising_the_WRAPPED_names_recovers_the_UNWRAPPED_ones_exactly(self) -> None:
        """Derived from what ``peft`` does, not restated. A set equality, so a reduction
        that is too aggressive (collapsing two names) or too timid (leaving a prefix on)
        both fail - which a per-name spot check would not catch."""
        module = _tiny_base()
        before = {name for name, _ in module.named_parameters()}
        wrapped = _wrap(module, target_modules=("q_proj", "down_proj"))
        recovered = {
            canonical_base_name(name)
            for name, _ in wrapped.named_parameters()
            if PEFT_SEGMENT not in name
        }
        assert recovered == before

    def test_canonicalising_an_UNWRAPPED_name_is_the_identity(self) -> None:
        """Or the pre-wrap digest would change value, and F11's ``starting_weights_sha256``
        over an attached base - computed through this same function - would change with it."""
        for name, _ in _tiny_base().named_parameters():
            assert canonical_base_name(name) == name


class TestTheBaseDigestSurvivesTheWrap:
    def test_the_digest_is_identical_either_side_of_a_real_peft_wrap(self) -> None:
        """AC14's comparison, with **no training at all**. This is the case that was red
        before the fix and that nothing in the suite could express."""
        module = _tiny_base()
        before = base_parameter_digest(module)
        wrapped = _wrap(module, target_modules=("q_proj",))
        assert base_parameter_digest(wrapped) == before

    def test_two_different_adapter_configurations_hash_the_same_base_identically(self) -> None:
        """The property that chose name-normalisation over a post-wrap digest.

        ``base_layer`` appears only on **targeted** modules, so post-wrap names are a
        function of the ``LoraConfig``. A digest that varied with the adapter configuration
        could not be reproduced by anyone holding only the base, which is the one job it has.
        """
        narrow = base_parameter_digest(_wrap(_tiny_base(), target_modules=("q_proj",)))
        wide = base_parameter_digest(
            _wrap(
                _tiny_base(), rank=16, alpha=32.0, target_modules=("q_proj", "v_proj", "down_proj")
            )
        )
        assert narrow == wide == base_parameter_digest(_tiny_base())

    def test_a_CHANGED_base_tensor_still_breaks_the_comparison(self) -> None:
        """The companion that stops the equality above being trivially true. Without it a
        normalisation that reduced every name to ``""`` would pass the two cases above."""
        import torch

        module = _tiny_base()
        before = base_parameter_digest(module)
        wrapped = _wrap(module, target_modules=("q_proj",))
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if PEFT_SEGMENT not in name:
                    parameter.add_(1.0)
                    break
        assert base_parameter_digest(wrapped) != before

    def test_a_changed_ADAPTER_tensor_does_NOT_break_it(self) -> None:
        """The adapter's tensors are supposed to move. Hashing them would make AC14 fail on
        every successful run - the false-negative mirror, and just as useless."""
        import torch

        module = _tiny_base()
        before = base_parameter_digest(module)
        wrapped = _wrap(module, target_modules=("q_proj",))
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if PEFT_SEGMENT in name:
                    parameter.add_(1.0)
        assert base_parameter_digest(wrapped) == before


class TestAC14ThroughTheProductionPrepareAndAStep:
    """Not ``base_parameter_digest`` in isolation - the value ``prepare`` actually records,
    against the module it actually returns. On CPU, so **CI runs it**; the card's own version
    in ``tests/integration/test_adapter_run_gpu.py`` stays, for the real optimizer and the
    real device."""

    def test_the_digest_prepare_records_is_comparable_to_the_module_it_returns(self) -> None:
        plan = ADAPTER.prepare(_acquired(_tiny_base()), adapter=adapter_config())
        assert plan.base_digest is not None
        assert base_parameter_digest(plan.module) == plan.base_digest, (
            "the digest prepare() records cannot be compared against the module it hands "
            "back, so acceptance criterion 14 is unverifiable whatever the weights do"
        )

    def test_the_base_digest_is_identical_after_a_real_optimizer_step(self) -> None:
        """AC14 in its own words - *hashed before and after* - on the CPU.

        32-bit optimizer states on purpose: ``bitsandbytes`` has no pure-CPU wheel
        everywhere (ADR-023), and the subject here is the base's bytes, not the optimizer's
        precision.
        """
        import torch

        from training import optimizer as optimizer_module
        from training.run_config import RunConfig

        plan = ADAPTER.prepare(_acquired(_tiny_base()), adapter=adapter_config())
        before = plan.base_digest
        cfg = RunConfig(
            learning_rate=1e-2,
            warmup_steps=0,
            batch_size=1,
            optimizer_state_precision="32bit",
        )
        optimizer, _ = optimizer_module.build(plan.module, cfg)
        snapshot = [p.detach().clone() for p in plan.trainable]
        ids = torch.tensor([[1, 2, 3, 4]])
        plan.module(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert any(
            not torch.allclose(a, b) for a, b in zip(snapshot, plan.trainable, strict=True)
        ), "the adapter did not move, so the assertion below would hold against a no-op run"
        assert base_parameter_digest(plan.module) == before, (
            "a base parameter changed during an adapter run; the optimizer was built over "
            "something it should not have been"
        )


class TestTheCollisionRefusal:
    def test_two_tensors_reducing_to_ONE_canonical_name_are_refused(self) -> None:
        """A ``dict`` would drop one of them and the digest would stop covering a tensor,
        silently. ``modules_to_save`` is the shape that would produce this - miForge's
        builder sets none, so this is the guard rather than a reachable path today."""
        from tests.support.test_adapter_doubles import FakeModule, FakeParameter

        module = FakeModule(
            {
                "model.layers.0.q_proj.weight": FakeParameter(b"\x01"),
                "base_model.model.model.layers.0.q_proj.base_layer.weight": FakeParameter(b"\x02"),
            }
        )
        with pytest.raises(Refusal) as caught:
            base_parameter_digest(module)
        assert caught.value.code == "base_digest_name_collision"
        details = caught.value.details or {}
        assert details["canonical_name"] == "model.layers.0.q_proj.weight"
        assert len(details["parameters"]) == 2
