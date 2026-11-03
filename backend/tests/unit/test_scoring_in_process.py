"""The default provider, and the refusal that happens **before** allocation  (FR-004.2, 11; E-24).

ADR-009's in-process provider is what makes acceptance criterion 20 true for scoring: it
needs no endpoint, no credential and no sibling. Everything here runs against
:class:`providers.fakes.FakeModelHandle`, which is the point of FR-004.11's port - *a
developer without a GPU runs all of F4 except one file.*

**E-24 is the sharp one.** A scoring pass allocates a KV cache on the card already holding
gradients, FP32 master weights and 8-bit optimizer states (BRD-02 §6). Failing in the
allocator partway through a pass kills a run **at an evaluation** rather than at a step, with
a torch OOM instead of a sentence an operator can act on. So the pass declares what it needs
and refuses by name, naming **both** numbers.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from providers.base import GenerationParams
from providers.fakes import FakeModelHandle
from providers.in_process import DEFAULT_KV_BYTES_PER_TOKEN, InProcessProvider


class TestTheIdentitySaysWhichParametersAnswered:
    def test_the_parameter_set_digest_is_on_the_identity(self) -> None:
        """*"ours" is not an identity.* Two steps of one run are two different parameter
        sets and their scores are not interchangeable."""
        provider = InProcessProvider(FakeModelHandle(digest="abc@step900"))
        assert provider.identity.engine_version == "abc@step900"
        assert provider.parameter_set_digest == "abc@step900"

    def test_the_base_url_is_none_AND_that_is_the_distinction(self) -> None:
        """Not a missing value - the interface documents it as the distinction itself."""
        assert InProcessProvider(FakeModelHandle()).identity.base_url is None

    def test_the_rendered_provider_string_omits_an_endpoint(self) -> None:
        assert InProcessProvider(FakeModelHandle()).identity.record() == "in_process:fake-model"

    def test_the_device_is_reported_by_uuid(self) -> None:
        """ADR-008. An index inverted once on this cluster; a bus address unbound the root
        NVMe."""
        assert InProcessProvider(FakeModelHandle()).device_uuid == "GPU-fake-0000"

    def test_it_declares_generation_and_nothing_else(self) -> None:
        assert InProcessProvider(FakeModelHandle()).capabilities() == frozenset({"generation"})

    def test_availability_needs_no_network_and_no_allocation(self) -> None:
        """A listing that probed would time out; one that allocated could OOM a run."""
        assert InProcessProvider(FakeModelHandle()).available() is True

    def test_an_unidentifiable_parameter_set_is_unavailable_rather_than_raising(self) -> None:
        class Nameless(FakeModelHandle):
            def model_identity(self) -> str:
                raise RuntimeError("no record")

        assert InProcessProvider(Nameless()).available() is False

    def test_a_handle_with_no_starting_hash_refuses_to_produce_a_digest(self) -> None:
        """A score that cannot say which parameters produced it is not a measurement."""
        from providers.model_handle import AcquiredModelHandle

        class Bare:
            record = None
            device_uuid = None

        with pytest.raises(Refusal) as caught:
            AcquiredModelHandle(Bare()).parameter_set_digest()
        assert caught.value.code == "inference_provider_unavailable"


class TestTheBudgetRefusalHappensBeforeAnyAllocation:
    def test_a_pass_that_does_not_fit_refuses_naming_both_numbers(self) -> None:
        handle = FakeModelHandle(free_vram_bytes=1024)
        provider = InProcessProvider(handle, batch_bound=4)
        with pytest.raises(Refusal) as caught:
            provider.generate(["a", "b", "c", "d"], GenerationParams(max_tokens=512))
        assert caught.value.code == "provider_budget_exceeded"
        assert caught.value.details["free_bytes"] == 1024
        assert caught.value.details["estimate_bytes"] > 1024
        assert caught.value.details["device_uuid"] == "GPU-fake-0000"

    def test_the_handle_is_never_asked_to_generate(self) -> None:
        """**Before** allocating. Asserted by call count on the handle, which is the only
        thing that distinguishes "refused early" from "refused after trying"."""
        handle = FakeModelHandle(free_vram_bytes=1)
        with pytest.raises(Refusal):
            InProcessProvider(handle).generate(["a"], GenerationParams(max_tokens=64))
        assert handle.calls == []

    def test_the_message_says_what_to_do_next(self) -> None:
        """ADR-021. Three concrete levers, not a description of the problem."""
        with pytest.raises(Refusal) as caught:
            InProcessProvider(FakeModelHandle(free_vram_bytes=1)).generate(
                ["a"], GenerationParams(max_tokens=64)
            )
        assert "Lower the batch bound" in caught.value.message

    def test_the_estimate_is_a_named_function_a_benchmark_can_quote(self) -> None:
        """ADR-024: *a figure without its parameters is not a measurement.*"""
        provider = InProcessProvider(FakeModelHandle(), kv_bytes_per_token=1000, batch_bound=4)
        assert provider.kv_cache_estimate_bytes(8, 16) == 4 * 16 * 1000

    def test_a_cpu_handle_skips_the_check_rather_than_passing_a_fabricated_one(self) -> None:
        """A CPU-resident parameter set has **no device budget to spend**, and checking
        against a fabricated figure would be ADR-018's silent default - in the direction that
        refuses a legitimate pass."""
        handle = FakeModelHandle(device_uuid=None, free_vram_bytes=0)
        assert InProcessProvider(handle).generate(["a"], GenerationParams()) != []

    def test_the_default_coefficient_is_a_stated_constant(self) -> None:
        assert DEFAULT_KV_BYTES_PER_TOKEN == 128 * 1024


class TestGenerationIsOrderedAndRefusesRatherThanGuessing:
    def test_one_completion_per_prompt_in_order(self) -> None:
        handle = FakeModelHandle(answers=lambda prompt, index: f"{index}:{prompt}")
        provider = InProcessProvider(handle, batch_bound=2)
        texts = [c.text for c in provider.generate(["a", "b", "c"], GenerationParams())]
        assert texts == ["0:a", "1:b", "0:c"]

    def test_every_completion_carries_the_identity_and_the_params(self) -> None:
        params = GenerationParams(max_tokens=8)
        completion = InProcessProvider(FakeModelHandle()).generate(["a"], params)[0]
        assert completion.provider.kind == "in_process"
        assert completion.params is params

    def test_an_unknown_option_refuses_rather_than_being_ignored(self) -> None:
        """FR-004.8. *A silently dropped option is a request the caller believes it made*,
        and a score taken under a parameter the engine ignored is not the score it claims."""
        with pytest.raises(Refusal) as caught:
            InProcessProvider(FakeModelHandle()).generate(
                ["a"], GenerationParams(), response_format="json"
            )
        assert caught.value.code == "inference_provider_option_unknown"
        assert caught.value.details["unknown"] == ["response_format"]

    def test_no_prompts_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            InProcessProvider(FakeModelHandle()).generate([], GenerationParams())
        assert caught.value.code == "evaluation_set_empty"

    def test_a_short_reply_list_refuses_rather_than_padding(self) -> None:
        """*Results are joined to prompts positionally, so a short list misattributes every
        score after the gap.*"""

        class Short(FakeModelHandle):
            def generate(self, prompts, params):
                return list(super().generate(prompts, params))[:-1]

        with pytest.raises(Refusal) as caught:
            InProcessProvider(Short(), batch_bound=4).generate(["a", "b", "c"], GenerationParams())
        assert caught.value.code == "inference_provider_reply_unparseable"

    def test_a_batch_bound_below_one_refuses_rather_than_clamping(self) -> None:
        """*A clamped bound is a configuration the operator did not choose.*"""
        with pytest.raises(Refusal) as caught:
            InProcessProvider(FakeModelHandle(), batch_bound=0)
        assert caught.value.code == "inference_provider_unavailable"


class TestTheFakeHandleIsAStubAndNotAMock:
    """*Over-mocked stubs that never run the real function* are named in ADR-016 among the
    reasons a suite stays green over a real bug. The fake replaces the tensors; the real
    provider's budget check, identity assembly and ordering guarantee all still run."""

    def test_it_records_the_payload_and_the_count(self) -> None:
        handle = FakeModelHandle()
        InProcessProvider(handle, batch_bound=2).generate(["a", "b", "c"], GenerationParams())
        assert len(handle.calls) == 2
        assert handle.prompts_seen == ["a", "b", "c"]

    def test_a_mis_sized_answer_list_refuses_rather_than_wrapping(self) -> None:
        """A modulo would answer prompt 7 with answer 0 and the test would still pass."""
        handle = FakeModelHandle(answers=["one"])
        with pytest.raises(AssertionError):
            handle.generate(["a", "b"], GenerationParams())
