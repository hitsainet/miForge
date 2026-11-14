"""``base_parameter_digest`` must be able to read a **bfloat16** base. Regression for a defect.

Found 2026-10-16 while re-scaling ``tests/performance/test_adapter_step_overhead.py`` off its
toy fixture: the re-scaled benchmark instantiates in ``bfloat16`` - what
``tests.support.bench_model``, ``tests.support.tiny_hf_model.write_tiny_model`` and the
checkpoint writer all emit - and ``resolve("adapter").prepare`` refused
``base_digest_unreadable`` before a single step ran.

## What was wrong, measured rather than inferred

``_parameter_bytes`` tried three routes and refused if none worked. Probed on the card
against a real ``torch.nn.Parameter`` of each dtype:

=============  =====================================  ==========================
dtype          ``.cpu().contiguous().numpy()``        ``bytes(memoryview(p))``
=============  =====================================  ==========================
``float32``    OK                                     ``TypeError``
``float16``    OK                                     ``TypeError``
``bfloat16``   ``TypeError: Got unsupported            ``TypeError``
               ScalarType BFloat16``
=============  =====================================  ==========================

**``bfloat16`` was the one dtype that failed**, because numpy has no bfloat16 scalar type -
and it is the dtype every real base is in. So an adapter run against a real checkpoint
refused at ``prepare``, and R-02.43's base-unchanged guarantee (AC14) was *uncomputable on
the main path*. It survived review because the adapter fixtures instantiated at **default
fp32** - a fixture that agreed with the reader by construction, which is this project's named
reason a suite stays green over a real bug.

**And one thing this file got wrong on its first pass, corrected here rather than quietly:**
it recorded ``bytes(memoryview(p))`` as *"dead for every dtype"*. That overstated it. The
route fails for a torch ``Tensor`` or ``Parameter`` of any dtype, but it **works** for a
numpy array, ``bytes`` and a ``bytearray`` - which is exactly the no-ML-stack stub its
docstring line is about. It was never going to rescue a torch tensor, and that is a different
claim from being useless. Both are asserted below.

## The fix, and why it changes no hash that already exists

``_reinterpret_as_uint8``: ``p.detach().cpu().contiguous().view(torch.uint8).numpy()``.
A ``uint8`` view *reinterprets* the storage instead of converting it, so the element type
never crosses into numpy. It returns ``element_size() * numel()`` bytes for every dtype, and
for ``float32`` it is **byte-identical to the route it precedes** - asserted below, because
the alternative would silently move every ``base_parameter_digest`` value already written
down.

``_parameter_bytes`` has exactly **one** caller in ``backend/src/``
(``base_parameter_digest``). ``starting_weights_sha256`` is computed by ``weights_sha256``
over safetensors shards or by ``created_weights_sha256(config, seed)`` and never reaches
here, so no published R-02.47 hash moves. Checked by grep, stated because
``hotupdate/attach.py`` consumes that field.

**Unmarked, so CI runs it**: nothing here needs a card, and a regression deselected by
default is a regression nobody catches.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


class TestBfloat16IsReadable:
    """The defect itself. These are the cases that go red if the fix is reverted."""

    def test_a_bfloat16_parameter_READS(self) -> None:
        import torch

        from training.runtypes.adapter import _parameter_bytes

        parameter = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        value = _parameter_bytes(parameter)
        assert len(value) == parameter.numel() * parameter.element_size() == 32, (
            "a bfloat16 parameter did not read as element_size * numel bytes; numpy has no "
            "bfloat16 scalar type, so this must go through the uint8 view"
        )

    def test_a_WHOLE_bfloat16_MODULE_digests_which_is_the_production_shape(self) -> None:
        """The shape that matters: ``prepare`` calls ``base_parameter_digest`` on the module.

        Asserted over a real module rather than one parameter, because the per-parameter case
        could be fixed while the module walk still refused on, say, a buffer.
        """
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from transformers import AutoConfig, AutoModelForCausalLM

        from tests.support.test_adapter_doubles import TINY_CONFIG
        from training.runtypes.adapter import base_parameter_digest

        config = AutoConfig.for_model(
            TINY_CONFIG["model_type"],
            **{key: value for key, value in TINY_CONFIG.items() if key != "model_type"},
        )
        module = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
        digest = base_parameter_digest(module)
        assert len(digest) == 64 and int(digest, 16) >= 0, digest

    def test_the_bfloat16_digest_is_SENSITIVE_to_a_changed_base(self) -> None:
        """A digest that reads is not yet a digest that *works*.

        Without this, ``_reinterpret_as_uint8`` returning a constant - or hashing the shape
        rather than the bytes - would pass every case above, and AC14 would hold vacuously
        over bfloat16 exactly as it previously failed loudly over it.
        """
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from transformers import AutoConfig, AutoModelForCausalLM

        from tests.support.test_adapter_doubles import TINY_CONFIG
        from training.runtypes.adapter import base_parameter_digest

        config = AutoConfig.for_model(
            TINY_CONFIG["model_type"],
            **{key: value for key, value in TINY_CONFIG.items() if key != "model_type"},
        )
        torch.manual_seed(11)
        module = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
        before = base_parameter_digest(module)
        assert base_parameter_digest(module) == before, "the digest is not deterministic"

        with torch.no_grad():
            first = next(iter(module.parameters()))
            first.add_(torch.ones_like(first))
        assert base_parameter_digest(module) != before, (
            "mutating a base parameter did not change the bfloat16 digest, so AC14 would "
            "pass against a run that rewrote the base - the failure mode the refusal this "
            "replaced was written to prevent"
        )


class TestTheFixChangedNoExistingDigest:
    """The half that protects what was already correct."""

    def test_float32_and_float16_still_read(self) -> None:
        import torch

        from training.runtypes.adapter import _parameter_bytes

        for dtype in (torch.float32, torch.float16):
            parameter = torch.nn.Parameter(torch.ones(4, 4, dtype=dtype))
            assert len(_parameter_bytes(parameter)) == 16 * parameter.element_size()

    def test_the_uint8_VIEW_is_byte_identical_to_the_numpy_route_for_float32(self) -> None:
        """So adopting the new route is known not to move a digest already recorded.

        Compared against the numpy expression **written out here** rather than against
        ``_parameter_bytes`` - which now *uses* the view, so comparing the two would compare
        the fix with itself and could not fail.
        """
        import torch

        from training.runtypes.adapter import _reinterpret_as_uint8

        for dtype in (torch.float32, torch.float16):
            parameter = torch.nn.Parameter(torch.randn(8, 8).to(dtype))
            legacy = parameter.detach().cpu().contiguous().numpy().tobytes()
            assert _reinterpret_as_uint8(parameter) == legacy, (
                f"the uint8 view disagrees with the numpy route for {dtype}; adopting it "
                f"would silently change every base_parameter_digest already written down"
            )

    def test_a_NON_CONTIGUOUS_parameter_reads(self) -> None:
        """``.view(torch.uint8)`` requires a contiguous last dimension, so the
        ``.contiguous()`` in front of it is load-bearing rather than defensive."""
        import torch

        from training.runtypes.adapter import _parameter_bytes

        parameter = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16).t())
        assert not parameter.is_contiguous()
        assert len(_parameter_bytes(parameter)) == 64 * parameter.element_size()


class TestTheRefusalStillRefuses:
    """The refusal is the point of the function and must survive the fix."""

    def test_something_that_is_no_tensor_at_all_is_REFUSED_BY_NAME(self) -> None:
        from core.errors import Refusal
        from training.runtypes.adapter import _parameter_bytes

        class NotATensor:
            pass

        with pytest.raises(Refusal) as caught:
            _parameter_bytes(NotATensor())
        assert caught.value.code == "base_digest_unreadable"

    def test_the_refusal_NAMES_WHY_EACH_ROUTE_FAILED(self) -> None:
        """Added with the fix. The old refusal carried only the parameter's type name, which
        is why a one-dtype numpy gap read as a generic unreadable parameter and cost a probe
        script to diagnose. *Name what failed* (ADR-013)."""
        from core.errors import Refusal
        from training.runtypes.adapter import _parameter_bytes

        class NotATensor:
            pass

        with pytest.raises(Refusal) as caught:
            _parameter_bytes(NotATensor())
        tried = caught.value.details.get("routes_tried")
        assert isinstance(tried, dict), caught.value.details
        assert set(tried) == {"uint8_view", "numpy_contiguous", "numpy", "memoryview"}, tried
        assert all(isinstance(reason, str) and reason for reason in tried.values()), tried


class TestTheMemoryviewRouteIsForNonTorchStubs:
    """The claim this file originally overstated, now asserted in both directions."""

    def test_it_fails_for_a_torch_parameter_of_every_dtype(self) -> None:
        import torch

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            parameter = torch.nn.Parameter(torch.ones(2, 2, dtype=dtype))
            with pytest.raises(TypeError):
                bytes(memoryview(parameter))  # type: ignore[arg-type]

    def test_but_it_WORKS_for_the_stub_shapes_its_docstring_is_about(self) -> None:
        """So it is a live fallback and not dead code, which is a different finding from the
        one this file first recorded. Driven through ``_parameter_bytes`` so the assertion is
        about reachability and not about ``memoryview`` itself."""
        from training.runtypes.adapter import _parameter_bytes

        numpy = pytest.importorskip("numpy")
        assert len(_parameter_bytes(numpy.ones(4, dtype=numpy.float32))) == 16
        assert _parameter_bytes(b"abcd") == b"abcd"
        assert _parameter_bytes(bytearray(b"wxyz")) == b"wxyz"
