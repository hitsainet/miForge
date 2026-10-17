"""The optimizer and its recorded identity  (R-02.13; FR-001.21 - FR-001.23, AC-001.7).

Four things R-02.13 fixes and one it leaves configurable, and each gets an assertion here
because **three of the four fail silently**:

* **FP32 master weights.** A silently FP16 master set still trains, the loss still falls,
  and small updates are lost to rounding for the whole run. No reading catches it.
* **Gradient checkpointing on, ``use_cache`` off.** The two conflict and the library only
  *warns*; a missed warning in a worker's log means activations are cached and the memory
  gradient checkpointing was enabled to save is spent anyway.
* **8-bit is a default, not the only value**, and the configured precision is part of the
  recorded identity - which is only meaningful if it is actually recorded.
* **The four run-configuration values are refused, never clamped.** A clamped learning rate
  trains a model the operator did not ask for and reports success.

``torch`` and ``bitsandbytes`` are not installed here (ADR-023), so the imports are faked at
their one seam each. The config validation, the identity and the config hash all run through
the real code.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from training import optimizer as optimizer_module
from training.optimizer import (
    OptimizerIdentity,
    assert_fp32_master_weights,
    build,
    enable_gradient_checkpointing,
)
from training.run_config import MAX_LEARNING_RATE, RunConfig


class FakeDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


FLOAT32 = FakeDtype("float32")
FLOAT16 = FakeDtype("float16")


class FakeTorch:
    float32 = FLOAT32
    float16 = FLOAT16

    class optim:
        class AdamW:
            def __init__(self, params: Any, lr: float) -> None:
                self.params = list(params)
                self.lr = lr


class Param:
    def __init__(self, dtype: Any = FLOAT32, requires_grad: bool = True, numel: int = 10) -> None:
        self.dtype = dtype
        self.requires_grad = requires_grad
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class Config:
    def __init__(self) -> None:
        self.use_cache = True


class Model:
    def __init__(self, params: list[Param] | None = None) -> None:
        self.params = params if params is not None else [Param(), Param()]
        self.config = Config()
        self.checkpointing_enabled = 0

    def parameters(self) -> list[Param]:
        return self.params

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing_enabled += 1


CFG = RunConfig(learning_rate=1e-5, warmup_steps=100, batch_size=2, gradient_accumulation=8)


@pytest.fixture
def faked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(optimizer_module, "_torch", lambda: FakeTorch)
    return FakeTorch


class AdamW8bit:
    """``bitsandbytes``' 8-bit optimizer, by the name the real one has.

    The name matters: :func:`assert_fp32_master_weights` recognises an 8-bit optimizer by the
    ``8bit`` suffix, because such an optimizer keeps its FP32 master copy internally and
    exposes it only after the first step.
    """

    def __init__(self, params: Any, lr: float) -> None:
        self.params = list(params)
        self.lr = lr


@pytest.fixture
def with_bnb(monkeypatch: pytest.MonkeyPatch, faked):
    monkeypatch.setattr(
        optimizer_module, "_adamw_8bit", lambda params, cfg: AdamW8bit(params, cfg.learning_rate)
    )


class TestTheIdentityIsRecorded:
    def test_a_default_run_records_adamw_8bit_with_a_config_hash(self, with_bnb) -> None:
        """AC-001.7. The ``name`` shape is the one ``checkpoint-manifest-v1.json``'s
        ``optimizer.name`` field exemplifies, so F5 writes this value through unchanged
        rather than deriving its own spelling."""
        _, identity = build(Model(), CFG)
        assert identity.name == "adamw-8bit"
        assert identity.state == "fresh"
        assert identity.config_hash == CFG.config_hash()
        assert len(identity.config_hash or "") == 64

    def test_a_configured_non_default_precision_is_recorded_AS_CONFIGURED(self, faked) -> None:
        """**8-bit is a default, not the only value** (FR-001.23). R-02.42 estimates
        "optimizer states at their configured precision", which only means something if the
        precision is configurable *and* recorded."""
        cfg = CFG.model_copy(update={"optimizer_state_precision": "32bit"})
        optimizer, identity = build(Model(), cfg)
        assert identity.name == "adamw-32bit"
        assert isinstance(optimizer, FakeTorch.optim.AdamW)

    def test_the_state_provenance_is_passed_in_not_inferred(self, with_bnb) -> None:
        """Only the caller knows whether trainer state was found. Inferring it from whether
        the optimizer holds any state would read ``fresh`` for a *restore that failed* -
        and R-02.22 requires the distinction be recorded, because a fresh optimizer on a
        mid-run checkpoint produces a loss discontinuity that is otherwise unexplained."""
        _, identity = build(Model(), CFG, state="restored")
        assert identity.state == "restored"

    def test_the_identity_carries_exactly_the_schemas_three_fields(self) -> None:
        """R-02.35: a manifest's fields are *exactly* the schema's, and a dataclass with a
        fourth field is how a writer ends up asserting one that does not exist."""
        block = OptimizerIdentity(name="adamw-8bit", state="fresh", config_hash="a" * 64)
        assert set(block.as_manifest_block()) == {"name", "state", "config_hash"}

    def test_the_config_hash_covers_every_hyperparameter(self) -> None:
        """Over the model dump rather than a hand-built dict: a hand-built dict is a second
        list of the fields and would stop including one the moment one was added."""
        other = CFG.model_copy(update={"warmup_steps": CFG.warmup_steps + 1})
        assert other.config_hash() != CFG.config_hash()

    def test_two_runs_differing_only_by_seed_have_different_config_hashes(self) -> None:
        """Two runs that differ only by seed are two different runs, and a hash that
        collapsed them would make the manifests claim they were the same."""
        assert (
            CFG.model_copy(update={"seed": 1}).config_hash()
            != CFG.model_copy(update={"seed": 2}).config_hash()
        )


class TestFp32MasterWeights:
    def test_a_32_bit_optimizer_over_fp16_parameters_REFUSES(self, faked) -> None:
        """**The defect no reading catches.** There is no FP32 master copy anywhere, so
        small updates are lost to rounding for the whole run - while training proceeds and
        the loss falls."""
        model = Model([Param(dtype=FLOAT16), Param(dtype=FLOAT16)])
        cfg = CFG.model_copy(update={"optimizer_state_precision": "32bit"})
        with pytest.raises(Refusal) as caught:
            build(model, cfg)
        assert caught.value.code == "fp32_master_weights_absent"
        assert "lost to rounding" in caught.value.message
        assert caught.value.details["non_fp32_parameters"] == 2

    def test_a_32_bit_optimizer_over_fp32_parameters_passes(self, faked) -> None:
        cfg = CFG.model_copy(update={"optimizer_state_precision": "32bit"})
        build(Model(), cfg)

    def test_an_8_bit_optimizer_over_fp16_parameters_is_the_EXPECTED_state(self, with_bnb) -> None:
        """``bitsandbytes``' 8-bit optimizers keep an FP32 master copy internally and expose
        it only after the first step, so this is the correct configuration - and refusing it
        would make the default path unusable."""
        model = Model([Param(dtype=FLOAT16)])
        build(model, CFG)

    def test_the_check_is_called_as_part_of_build_not_left_to_a_caller(
        self, faked, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared mechanism nothing calls is the defect class this project has shipped
        three times. Asserted by removing the behaviour and requiring a signal."""
        calls: list[Any] = []
        monkeypatch.setattr(
            optimizer_module,
            "assert_fp32_master_weights",
            lambda optimizer, params: calls.append((optimizer, params)),
        )
        cfg = CFG.model_copy(update={"optimizer_state_precision": "32bit"})
        build(Model(), cfg)
        assert len(calls) == 1, "build() must run the master-weight check itself"

    def test_the_check_can_be_used_directly_by_a_caller(self, faked) -> None:
        assert_fp32_master_weights(AdamW8bit([], 1e-5), [Param(dtype=FLOAT16)])


class TestGradientCheckpointing:
    def test_it_is_enabled_and_use_cache_is_turned_OFF_with_it(self, with_bnb) -> None:
        """The two conflict and the library only warns - and a warning in a worker's log is
        easy to miss, after which the memory this was enabled to save is spent anyway."""
        model = Model()
        build(model, CFG)
        assert model.checkpointing_enabled == 1
        assert model.config.use_cache is False

    def test_it_is_NOT_configurable(self) -> None:
        """R-02.13 fixes it on, and BRD-02 §6's memory budget assumes it. There is no field
        for it in the run config, and that is the assertion."""
        assert "gradient_checkpointing" not in RunConfig.model_fields

    def test_a_model_that_cannot_checkpoint_is_REFUSED(self, faked) -> None:
        class NoCheckpointing(Model):
            gradient_checkpointing_enable = None  # type: ignore[assignment]

        with pytest.raises(Refusal) as caught:
            enable_gradient_checkpointing(NoCheckpointing())
        assert caught.value.code == "gradient_checkpointing_unavailable"
        assert "refusing rather than training without it" in caught.value.message.lower()


class TestWhatIsRefusedRatherThanDowngraded:
    def test_a_model_with_no_trainable_parameter_refuses(self, faked) -> None:
        """Otherwise the optimizer has nothing to update and every step reports a falling
        loss of zero. R-02.13's updates are full-parameter."""
        model = Model([Param(requires_grad=False)])
        with pytest.raises(Refusal) as caught:
            build(model, CFG)
        assert caught.value.code == "model_has_no_trainable_parameters"

    def test_only_trainable_parameters_reach_the_optimizer(self, with_bnb) -> None:
        model = Model([Param(), Param(requires_grad=False), Param()])
        optimizer, _ = build(model, CFG)
        assert len(optimizer.params) == 2

    def test_a_partly_frozen_model_with_no_input_grad_hook_is_NOT_refused(self, with_bnb) -> None:
        """**The regression the D3 fix nearly shipped, pinned so it cannot come back.**

        D3's fix calls ``enable_input_require_grads()`` when a frozen input would leave a
        reentrant checkpointed block without an autograd graph. Its first form **refused**
        when that method was absent - and this class's own partly-frozen double has no such
        method, so a supported shape became a build failure. That is a worse defect than D3
        itself: D3 failed loudly at ``backward()``, while this failed a configuration that
        may train perfectly well at construction, before a single step.

        All ``enable_gradient_checkpointing`` can establish for a model it cannot introspect
        is *"some parameters are frozen"*, which is not *"backward will fail"*. So it warns,
        proceeds, and leaves autograd's own loud ``RuntimeError`` as the failure if one
        comes. Asserted here as **no refusal and checkpointing still enabled**, because
        proceeding without turning checkpointing on would breach R-02.13 just as quietly.
        """
        model = Model([Param(), Param(requires_grad=False), Param()])
        assert not hasattr(model, "enable_input_require_grads")

        optimizer, _ = build(model, CFG)

        assert len(optimizer.params) == 2
        assert model.checkpointing_enabled == 1, "R-02.13: checkpointing must still be on"
        assert model.config.use_cache is False

    def test_absent_bitsandbytes_REFUSES_rather_than_silently_using_32_bit(
        self, faked, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**A silent downgrade quadruples the optimizer's memory** on a card budgeted to
        the megabyte, and the recorded identity would then name a precision that was not
        used - which is the fabricated value ADR-018 forbids."""
        with pytest.raises(Refusal) as caught:
            build(Model(), CFG)  # `_adamw_8bit` is unpatched here, so the real import runs
        assert caught.value.code in {"optimizer_precision_unavailable", "ml_runtime_unavailable"}
        if caught.value.code == "optimizer_precision_unavailable":
            assert "will not downgrade silently" in caught.value.message
            assert caught.value.details["requested"] == "8bit"

    def test_an_absent_torch_refuses_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ADR-023: everything in miForge except a training step runs without it, and saying
        so is what keeps a developer from hunting a configuration problem.

        **The absence is FORCED. It used to be borrowed from the environment, and that made
        this test a statement about the machine rather than about the code.**

        It took a `monkeypatch` fixture and never used it — the intent was there and the
        mechanism was not — so it passed only where `torch` happened to be uninstalled. My
        own venv was missing it, a *declared* dependency (`torch>=2.4,<3`), so this was green
        locally for the wrong reason while CI, which installs the CPU wheel, ran the real
        path and went red. **Thirteen tests across F8 and F11 were in that state**, which
        means every "0 failed" I read locally was true of my environment and not of a correct
        one.

        Same shape as the `pyarrow` refusal earlier the same day, inverted: there a declared
        dependency made a refusal test **skip** and stop asserting; here a *missing* one made
        tests **pass** against a degradation path that is not the product. The common root is
        that **a test whose outcome turns on whether an import succeeds is a test about the
        environment.** So the import is blocked here, and the refusal is asserted whether or
        not `torch` is installed.
        """
        import builtins

        real_import = builtins.__import__

        def no_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch" or name.startswith("torch."):
                raise ImportError("blocked by the test: torch is absent")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", no_torch)

        with pytest.raises(Refusal) as caught:
            build(Model(), CFG)
        assert caught.value.code == "ml_runtime_unavailable"
        assert "ADR-023" in caught.value.message, (
            "the refusal must say that everything but a training step runs without torch, "
            "or a developer reads it as a broken install rather than a stated boundary"
        )

    def test_with_torch_PRESENT_the_optimizer_actually_builds(self) -> None:
        """The other half, and the half that was never covered.

        While `torch` was absent from my venv, `build()` could only ever be observed
        refusing — so the suite asserted the boundary and never the behaviour. This runs the
        real path, which is what ships.
        """
        torch = pytest.importorskip("torch")

        # A REAL `nn.Module` with real fp32 parameters, exposing the one method R-02.13
        # obliges: `gradient_checkpointing_enable()`. A bare `nn.Linear` is refused
        # (`gradient_checkpointing_unavailable`) and refused correctly - §6's memory budget
        # assumes checkpointing is on, so training without it is refused rather than
        # silently accepted. That refusal is asserted elsewhere in this class; here the
        # module satisfies it so the optimizer path itself can be observed.
        class _Checkpointable(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
                self.enabled = False

            def gradient_checkpointing_enable(self, **_: object) -> None:
                self.enabled = True

        # 32-BIT STATES, DELIBERATELY. The default is `8bit`, which needs `bitsandbytes` -
        # an extra that is NOT installed here and has no pure-CPU wheel on every platform,
        # which is exactly why it is an extra. `build()` therefore refuses
        # `optimizer_precision_unavailable` on the default config rather than downgrading
        # silently, *"because that quadruples the optimizer's memory and the recorded
        # identity would name a precision that was not used."*
        #
        # **That refusal is what CI actually reported** when it ran this file with torch
        # present: `'optimizer_precision_unavailable' == 'ml_runtime_unavailable'`. So the
        # 32-bit path is the one a developer without the extra can execute, which ADR-023
        # makes a requirement on the layout rather than an accident.
        cfg = CFG.model_copy(update={"optimizer_state_precision": "32bit"})

        module = _Checkpointable()
        # `build` returns (optimizer, identity) - read from the source, not guessed.
        optimizer, identity = build(module, cfg)
        assert module.enabled, (
            "build() did not turn gradient checkpointing on. R-02.13 requires it and §6's "
            "memory budget assumes it, so a run without it exceeds a budget nobody checked."
        )
        assert optimizer is not None
        # The learning rate the config asked for reached the optimizer, rather than a
        # default that happens to look plausible.
        assert optimizer.param_groups[0]["lr"] == cfg.learning_rate
        # AND THE RECORDED IDENTITY NAMES THE PRECISION ACTUALLY USED, which is the whole
        # reason `build` refuses rather than downgrading: a manifest claiming `8bit` over a
        # run that used 32-bit states would misattribute both the memory and the result.
        assert "32bit" in str(
            identity
        ), f"the optimizer identity does not name the precision used: {identity!r}"


class TestTheFourRunConfigurationValues:
    """FR-001.22: refused with the field named, **never clamped**."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("learning_rate", 0.0),
            ("learning_rate", -1e-5),
            ("learning_rate", MAX_LEARNING_RATE * 2),
            ("warmup_steps", -1),
            ("batch_size", 0),
            ("gradient_accumulation", 0),
        ],
    )
    def test_an_out_of_range_value_is_refused(self, field: str, value: Any) -> None:
        from pydantic import ValidationError

        payload = {
            "learning_rate": 1e-5,
            "warmup_steps": 10,
            "batch_size": 1,
            "gradient_accumulation": 1,
        }
        payload[field] = value
        with pytest.raises(ValidationError) as caught:
            RunConfig(**payload)
        assert any(field in str(error["loc"]) for error in caught.value.errors())

    def test_a_non_finite_learning_rate_is_refused(self) -> None:
        """NaN passes ``gt=0`` in some orderings and would then poison the first update
        silently - and ``canonical_bytes`` refuses NaN too, so it would also make the config
        unhashable several steps later, far from the cause."""
        from pydantic import ValidationError

        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValidationError):
                RunConfig(learning_rate=bad, warmup_steps=1, batch_size=1, gradient_accumulation=1)

    def test_nothing_is_clamped(self) -> None:
        """A clamped learning rate trains a model the operator did not ask for and reports
        success. The accepted value must come back exactly as given."""
        cfg = RunConfig(learning_rate=0.999, warmup_steps=0, batch_size=1, gradient_accumulation=1)
        assert cfg.learning_rate == 0.999

    def test_an_unknown_key_is_refused_not_ignored(self) -> None:
        """A misspelled ``grad_accum`` must not silently leave the real field at its
        default - that trains a different run than the one described."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunConfig(
                learning_rate=1e-5,
                warmup_steps=1,
                batch_size=1,
                gradient_accumulation=1,
                grad_accum=8,  # type: ignore[call-arg]
            )

    def test_the_effective_batch_size_is_computed_in_one_place(self) -> None:
        """ADR-012: a loss without its effective batch size is not a measurement, and the
        two factors are configured separately - so the product has one definition."""
        assert CFG.effective_batch_size == CFG.batch_size * CFG.gradient_accumulation == 16
