"""AC14: the base is byte-unchanged, and the guarantee is **enforced at construction**.

**M9's target** - *set* ``requires_grad`` *on the base in an adapter run*.

The enforcement is deliberately not "hash at the end and compare". An optimizer built over a
base parameter *will* write to it, and a hash comparison would find that out at the end of a
run rather than at the start of one, so
:func:`training.runtypes.adapters.build.assert_only_adapter_requires_grad` is a **production
assertion** and :func:`training.runtypes.adapter.base_parameter_digest` is the independent
check on top.

## The vacuity guard is the part worth reading

A digest that skipped parameters it could not read would compare **two empty hashes as
equal**, and AC14 would pass against a run that rewrote every byte of the base. That is the
construction-green failure this project keeps finding, in the one place where it would make a
headline acceptance criterion meaningless. So the digest **refuses** an unreadable parameter
and refuses a module with no base parameters at all, and both refusals are tested.

**What is NOT covered here**, stated rather than implied: the real before-and-after hash
over a trained model **on a card** is ``gpu``-marked in ``test_adapter_run_gpu.py``. The
pieces below are the real production functions over doubles whose bytes are genuinely
readable.

*That paragraph also said the full ``prepare()`` path and the real before-and-after hash
were on the card and therefore unrun - which is why nobody noticed that the before-and-after
hash **could never match**.* ``peft`` renames every base tensor on wrap and the digest covers
names, so AC14's comparison was structurally unverifiable and
``test_the_base_digest_is_identical_after_a_real_step`` could only ever fail, for a reason
that has nothing to do with the base. The rename needs **no card**: it is now exercised over
the doubles below, and over a real tiny model wrapped by production's own builder in
``test_adapter_base_digest_across_wrap.py``, which also carries the whole ``prepare`` ->
optimizer -> step -> compare cycle on the CPU. **The card's remaining job is the real
device and the real optimizer, not the mechanism.**

*That paragraph read "``torch`` and ``peft`` are not installed in this environment" until
2026-10-16, and it was stale in the way that matters: both are declared core dependencies
(``torch>=2.4,<3``, ``peft>=0.13,<1``) and both are installed here and in CI. What is
missing is a **card**, not a library - so the reason the real path is unrun is the GPU
marker, not an absent import. The ``ml_runtime_unavailable`` test below was written on the
absent-library premise and skipped on every machine because of it; see its docstring.*
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from tests.support.test_adapter_doubles import (
    FakeModule,
    FakeParameter,
    OpaqueParameter,
    base_module,
)
from training.runtypes.adapter import PEFT_SEGMENT, base_parameter_digest, canonical_base_name
from training.runtypes.adapters.build import assert_only_adapter_requires_grad

# ======================================================================================
# The rename the doubles in this file had never undergone
# ======================================================================================
# `base_module()`'s names are **already in PEFT form**, so every digest case below used to
# run on the far side of the rename and the before/after guarantee was tested exactly where
# the defect cannot exist. The real-library control is
# `test_adapter_base_digest_across_wrap.py`; what the doubles need is to undergo the rename,
# so `peft_renamed` below applies it - and is itself checked against `peft` further down,
# because a rename double that diverged from the thing it stands for would conceal the
# defect a second time.


def unwrapped_base(payload: bytes = b"\x01\x02\x03\x04") -> FakeModule:
    """A base in the names an **unwrapped** Hugging Face model carries.

    One targeted module (``q_proj``) and one untargeted (``down_proj``), because ``peft``
    renames the two differently and a single-case fixture could not tell them apart.
    """
    return FakeModule(
        {
            "model.layers.0.self_attn.q_proj.weight": FakeParameter(payload),
            "model.layers.0.mlp.down_proj.weight": FakeParameter(payload[::-1]),
        }
    )


def peft_renamed(module: FakeModule, *, targeted: tuple[str, ...] = ("q_proj",)) -> FakeModule:
    """The same base tensors under the names ``get_peft_model`` gives them, plus a LoRA pair.

    The **same objects**, not copies: the point of the before/after comparison is that the
    bytes did not change, so a helper that copied them would be asserting something weaker
    than the production path does.
    """
    renamed: dict[str, Any] = {}
    for name, parameter in module.named_parameters():
        pieces = name.split(".")
        stem, leaf, owner = ".".join(pieces[:-1]), pieces[-1], pieces[-2]
        if owner in targeted:
            renamed[f"base_model.model.{stem}.base_layer.{leaf}"] = parameter
            renamed[f"base_model.model.{stem}.lora_A.default.{leaf}"] = FakeParameter(
                b"\xaa", requires_grad=True
            )
            renamed[f"base_model.model.{stem}.lora_B.default.{leaf}"] = FakeParameter(
                b"\xbb", requires_grad=True
            )
        else:
            renamed[f"base_model.model.{name}"] = parameter
    return FakeModule(renamed)


class TestOnlyTheAdapterIsTrainable:
    def test_a_correctly_wrapped_module_passes_and_returns_the_adapters_parameters(self) -> None:
        trainable = assert_only_adapter_requires_grad(base_module())
        assert len(trainable) == 2, "the two LoRA matrices, and nothing else"

    def test_a_TRAINABLE_BASE_PARAMETER_is_refused_by_name(self) -> None:
        """M9. The optimizer would update the base and AC14's guarantee would be false."""
        module = base_module()
        module.parameters_by_name["base_model.model.layers.0.q_proj.weight"].requires_grad = True
        with pytest.raises(Refusal) as caught:
            assert_only_adapter_requires_grad(module)
        assert caught.value.code == "adapter_base_is_trainable"
        details = caught.value.details or {}
        assert details["leaked_count"] == 1
        assert "q_proj.weight" in details["parameters"][0]

    def test_the_refusal_names_the_parameters_rather_than_only_counting_them(self) -> None:
        """An operator cannot act on *"3 parameters leaked"*."""
        module = base_module()
        for name, parameter in module.parameters_by_name.items():
            if PEFT_SEGMENT not in name:
                parameter.requires_grad = True
        with pytest.raises(Refusal) as caught:
            assert_only_adapter_requires_grad(module)
        assert len((caught.value.details or {})["parameters"]) == 2

    def test_a_module_with_NOTHING_trainable_is_refused(self) -> None:
        """Every step would report a falling loss of zero while the model did not change -
        the silent-failure shape, so it refuses rather than training nothing."""
        module = FakeModule({"w": FakeParameter(), "b": FakeParameter()})
        with pytest.raises(Refusal) as caught:
            assert_only_adapter_requires_grad(module)
        assert caught.value.code == "adapter_has_no_trainable_parameters"

    def test_the_assertion_is_production_code_not_a_test_helper(self) -> None:
        """*"The cheapest place to enforce it is at construction."* So it must be called by
        the builder, not only by this file."""
        import inspect

        from training.runtypes import adapter as adapter_run_type
        from training.runtypes.adapters import build as build_module

        assert "assert_only_adapter_requires_grad(wrapped)" in inspect.getsource(build_module.build)
        assert "assert_only_adapter_requires_grad(wrapped)" in inspect.getsource(
            adapter_run_type.AdapterRunType.prepare
        )


class TestTheBaseDigestSurvivesPeftsRename:
    """**The guarantee these doubles used to be unable to express.**

    Every case in the class below ran against names already in PEFT form, so none of them
    crossed the rename and AC14's before-and-after comparison was tested where it cannot
    break. It could never pass: the digest covers the parameter *name*, and ``peft`` renames
    every base tensor on wrap. The real-library version of these two cases - a tiny model
    wrapped by production's own builder - is ``test_adapter_base_digest_across_wrap.py``.
    """

    def test_the_rename_does_not_change_the_digest(self) -> None:
        base = unwrapped_base()
        assert base_parameter_digest(peft_renamed(base)) == base_parameter_digest(base)

    def test_a_base_tensor_CHANGED_under_the_rename_is_still_caught(self) -> None:
        """So the equality above is a comparison rather than a constant. Without this a
        normalisation that reduced every name to the empty string would pass it."""
        before = base_parameter_digest(unwrapped_base())
        touched = unwrapped_base()
        touched.parameters_by_name["model.layers.0.mlp.down_proj.weight"] = FakeParameter(b"\xff")
        assert base_parameter_digest(peft_renamed(touched)) != before

    def test_the_canonical_name_round_trips_the_whole_name_set(self) -> None:
        """A set equality, so a reduction that is too aggressive - collapsing two names into
        one - fails as loudly as one that leaves a prefix behind."""
        base = unwrapped_base()
        expected = {name for name, _ in base.named_parameters()}
        recovered = {
            canonical_base_name(name)
            for name, _ in peft_renamed(base).named_parameters()
            if PEFT_SEGMENT not in name
        }
        assert recovered == expected


class TestTheRenameDoubleMatchesTheRealLibrary:
    """``peft_renamed`` is a double, and a double that diverged from ``peft`` would conceal
    the defect a second time - so it is checked against the library, in the file it lives in,
    which is the convention ``tests/support/test_adapter_doubles.py`` sets for its own.

    ``peft`` and ``transformers`` are **core** dependencies (ADR-002) and no device is
    touched here, so this runs in CI rather than skipping."""

    def test_it_produces_the_names_peft_actually_produces(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        from tests.support.test_adapter_doubles import TINY_CONFIG, adapter_config
        from training.runtypes.adapters.build import build as build_adapter

        config = AutoConfig.for_model(
            TINY_CONFIG["model_type"],
            **{k: v for k, v in TINY_CONFIG.items() if k != "model_type"},
        )
        torch.manual_seed(11)
        real = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        mirror = FakeModule({name: FakeParameter() for name, _ in real.named_parameters()})

        wrapped, _ = build_adapter(real, adapter_config(target_modules=("q_proj",)))
        from_library = {name for name, _ in wrapped.named_parameters() if PEFT_SEGMENT not in name}
        from_double = {
            name
            for name, _ in peft_renamed(mirror, targeted=("q_proj",)).named_parameters()
            if PEFT_SEGMENT not in name
        }
        assert from_double == from_library


class TestTheBaseDigestCoversTheBaseAndNothingElse:
    def test_the_same_base_hashes_the_same(self) -> None:
        assert base_parameter_digest(base_module()) == base_parameter_digest(base_module())

    def test_a_CHANGED_base_parameter_changes_the_digest(self) -> None:
        """Without this, AC14's comparison would hold against any run at all."""
        before = base_parameter_digest(base_module(b"\x01\x02\x03\x04"))
        after = base_parameter_digest(base_module(b"\x09\x02\x03\x04"))
        assert before != after

    def test_a_changed_ADAPTER_parameter_does_NOT_change_the_digest(self) -> None:
        """The adapter's tensors are *supposed* to change. Including them would make the
        comparison fail on every successful run - the false-negative mirror of the vacuity
        failure, and just as useless."""
        module = base_module()
        before = base_parameter_digest(module)
        module.parameters_by_name["base_model.model.layers.0.q_proj.lora_A.weight"] = FakeParameter(
            b"\xff\xff", requires_grad=True
        )
        assert base_parameter_digest(module) == before

    def test_the_digest_covers_parameter_NAMES_as_well_as_bytes(self) -> None:
        """Two bases with the same bytes under different names are different bases.

        *This case swapped the two payloads between two fixed names until 2026-10-16, and it
        could not fail against a digest that hashed no name at all*: swapping the bytes
        changes the concatenation order, so the two digests differ whether or not the names
        contribute. Found by mutating the name out of the hash and watching all 33 cases
        stay green - the fixture distinguished the right things by construction. The payload
        is the **same** on both sides now, so only the name can separate them.
        """
        left = FakeModule({"a.weight": FakeParameter(b"\x01"), "b.weight": FakeParameter(b"\x01")})
        right = FakeModule({"a.weight": FakeParameter(b"\x01"), "z.weight": FakeParameter(b"\x01")})
        assert base_parameter_digest(left) != base_parameter_digest(right)

    def test_the_order_of_named_parameters_does_not_change_the_digest(self) -> None:
        """Sorted, so two readings of one module agree. An unsorted digest would make the
        before/after comparison fail for a reason that is not the base changing."""
        forward = FakeModule({"a": FakeParameter(b"\x01"), "b": FakeParameter(b"\x02")})
        reverse = FakeModule({"b": FakeParameter(b"\x02"), "a": FakeParameter(b"\x01")})
        assert base_parameter_digest(forward) == base_parameter_digest(reverse)


class TestTheVacuityGuards:
    """The two ways a digest could silently cover nothing."""

    def test_an_UNREADABLE_parameter_is_refused_not_skipped(self) -> None:
        module = FakeModule({"w": OpaqueParameter()})
        with pytest.raises(Refusal) as caught:
            base_parameter_digest(module)
        assert caught.value.code == "base_digest_unreadable"
        assert "compares equal to another digest over nothing" in caught.value.message

    def test_a_module_with_NO_BASE_PARAMETERS_is_refused(self) -> None:
        """An all-adapter module. Hashing an empty input returns a digest that compares
        equal to itself and reports the guarantee as held."""
        module = FakeModule({"base_model.model.q_proj.lora_A.weight": FakeParameter()})
        with pytest.raises(Refusal) as caught:
            base_parameter_digest(module)
        assert caught.value.code == "base_digest_empty"

    def test_skipping_an_unreadable_parameter_WOULD_have_made_two_bases_equal(self) -> None:
        """The counterfactual, made concrete: with the refusal removed, these two modules -
        which differ in exactly the unreadable parameter - would hash identically."""
        import hashlib

        def digest_that_skips(module: FakeModule) -> str:
            hasher = hashlib.sha256()
            for name, parameter in sorted(module.named_parameters()):
                try:
                    hasher.update(bytes(memoryview(parameter)))  # type: ignore[arg-type]
                except TypeError:
                    continue
                hasher.update(name.encode())
            return hasher.hexdigest()

        left = FakeModule({"w": OpaqueParameter()})
        right = FakeModule({"w": OpaqueParameter()})
        assert digest_that_skips(left) == digest_that_skips(right) == hashlib.sha256().hexdigest()


class TestTheDigestPrepareRecordsCanActuallyBeCompared:
    """**This class asserted the source text until 2026-10-16**, and that assertion pinned
    the defect as a requirement.

    It read ``source.index("base_parameter_digest(") < source.index("build_adapter(")`` -
    *"asserted on the body's order, because there is no other way to see it."* There is: run
    ``prepare`` and compare. The source assertion proves two calls appear in an order and
    says nothing about whether the two hashes are comparable, which is the only property
    AC14 needs - and because the ordering is *how* the defect was expressed, **any correct
    fix turned it red.** Same family as the ``TestRingPruningIsWired`` AGENT.md records:
    asserting an entry point exists rather than that the mechanism works.

    The ordering still matters, and the third case below is what makes it bite: a wrap that
    touched a base tensor must be **caught**, which is the reason to hash before it rather
    than after. That is the ordering's purpose asserted as behaviour.
    """

    @staticmethod
    def _prepared(monkeypatch: pytest.MonkeyPatch, build: Any) -> Any:
        """``prepare`` over doubles, with ``build`` standing in for ``peft``'s wrap.

        Patched on the module ``prepare`` imports from, which is how its function-local
        ``from … import build as build_adapter`` resolves it.
        """
        from tests.support.test_adapter_doubles import adapter_config
        from training.runtypes import resolve
        from training.runtypes.adapters import build as build_module

        monkeypatch.setattr(build_module, "build", build)
        acquired = type("Acquired", (), {"module": unwrapped_base(), "tokenizer": None})()
        return resolve("adapter").prepare(acquired, adapter=adapter_config())

    def test_the_recorded_digest_matches_the_module_prepare_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole of AC14's mechanism: the "before" value and the module it will be
        compared against are the same hash. This is what the source-order assertion could
        not see, and it is red against the unnormalised digest."""
        plan = self._prepared(monkeypatch, lambda module, config: (peft_renamed(module), config))
        assert plan.base_digest is not None
        assert base_parameter_digest(plan.module) == plan.base_digest

    def test_a_base_tensor_changed_AFTER_prepare_breaks_the_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the equality above would hold against a run that rewrote the base."""
        plan = self._prepared(monkeypatch, lambda module, config: (peft_renamed(module), config))
        name = next(n for n, _ in plan.module.named_parameters() if PEFT_SEGMENT not in n)
        plan.module.parameters_by_name[name] = FakeParameter(b"\xff\xff\xff\xff")
        assert base_parameter_digest(plan.module) != plan.base_digest

    def test_a_WRAP_that_touches_the_base_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ordering's **purpose**, as behaviour rather than as a line number. Hashing
        after the wrap would make this undetectable: the digest would simply record whatever
        the wrap left behind and compare equal to itself for the rest of the run."""

        def destructive(module: FakeModule, config: Any) -> Any:
            wrapped = peft_renamed(module)
            victim = next(n for n, _ in wrapped.named_parameters() if PEFT_SEGMENT not in n)
            wrapped.parameters_by_name[victim] = FakeParameter(b"\x00\x00\x00\x00")
            return wrapped, config

        plan = self._prepared(monkeypatch, destructive)
        assert base_parameter_digest(plan.module) != plan.base_digest, (
            "a wrap that rewrote a base tensor went unnoticed, so the digest is not taken "
            "before the wrap and AC14 cannot see anything the wrap does"
        )

    def test_the_plan_carries_the_digest_for_change_and_create_as_NONE(self) -> None:
        """They are supposed to change the base, so a digest there would be a guarantee
        nobody wants and a comparison that always fails."""
        from training.runtypes import resolve

        module = FakeModule({"w": FakeParameter()})
        acquired = type("Acquired", (), {"module": module, "tokenizer": object()})()
        assert resolve("change").prepare(acquired).base_digest is None
        assert resolve("create").prepare(acquired).base_digest is None


class TestPeftIsRefusedByNameWhenAbsentRatherThanHandRolled:
    def test_building_an_adapter_without_peft_refuses_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal is **forced**, not waited for. It used to skip, and that was a hole.

        This test said *"On this host ``peft`` is genuinely not installed, so this is the
        real path"* and skipped otherwise. ``peft>=0.13,<1`` is a **core** dependency in
        ``pyproject.toml``, so it is installed here and in CI - the skip therefore fired on
        **every** machine at once and ``ml_runtime_unavailable`` was asserted nowhere. Same
        shape as F2's ``parquet_reader_unavailable``, which skipped itself out of existence
        the day ``pyarrow`` was declared: *a test whose outcome turns on whether an import
        succeeds is a test about the environment, not the code*, and **a skipping test is
        not a passing test**.

        A refusal path does not stop mattering because the dependency is declared: an
        install can still lack the wheel, and then this refusal is the only thing between
        an operator and a bare ``ImportError`` they cannot act on. So the import is blocked
        - the remedy the ``pyarrow`` finding prescribes, in the same shape - and the
        refusal is asserted unconditionally, whether or not ``peft`` is present.
        """
        import builtins

        real_import = builtins.__import__

        def no_peft(name: str, *args: object, **kwargs: object) -> object:
            if name == "peft" or name.startswith("peft."):
                raise ImportError("blocked by the test: peft is absent")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", no_peft)

        from training.runtypes.adapters.build import build

        with pytest.raises(Refusal) as caught:
            build(base_module(), {"rank": 4, "alpha": 8.0, "target_modules": ("q_proj",)})
        assert caught.value.code == "ml_runtime_unavailable"
        assert "will not hand-roll LoRA" in caught.value.message
        assert "peft" in caught.value.message, (
            "the message must name the dependency, or the operator is told an adapter "
            "could not be built without being told what is missing"
        )

    def test_the_config_is_validated_BEFORE_peft_is_reached(self) -> None:
        """So an invalid rank refuses with ``adapter_config_invalid`` on a machine with no
        ML stack, rather than with ``ml_runtime_unavailable`` - the refusal an operator can
        act on, not the one about the environment (ADR-013's most-specific-first ordering)."""
        from training.runtypes.adapters.build import build

        with pytest.raises(Refusal) as caught:
            build(base_module(), {"rank": 0, "alpha": 8.0, "target_modules": ("q_proj",)})
        assert caught.value.code == "adapter_config_invalid"
