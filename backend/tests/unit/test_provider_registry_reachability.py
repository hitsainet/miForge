"""Both providers in the **live** registry  (ADR-016 #1; AC-004.2, AC-004.4; FR-004.1).

*A capability is not shipped until a test FAILS when its wiring is removed.* Importable is
not registered: a provider implementation can be complete, unit-tested and documented and
still be reachable by no caller - which is the sixteen-unregistered-MCP-tools failure the
whole review discipline was written after.

So nothing here imports an implementation and asserts on it. Every assertion goes through
:func:`providers.registry.get`, which is the path the scoring service actually resolves from,
and through :func:`providers.registry.list_rows`, which is the path ``GET /api/v1/providers``
serves.

**The control:** delete ``register(OPENAI_COMPATIBLE_KIND, ...)`` in
:func:`providers.registry.ensure_registered` and this file must go red. Observed and recorded.

The generation call is asserted on **payload and call count** - field by field - because
*"was called" passes against a call sending the wrong arguments*, and for a scoring pass the
arguments **are** what the number means.
"""

from __future__ import annotations

import httpx
import pytest

from core.errors import Refusal
from providers import registry
from providers.base import GenerationParams
from providers.fakes import FakeModelHandle
from tests.support.selector_stubs import StubJudgeServer


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch):
    registry.bind_model_handle(None)
    for key in list(registry.http_provider_rows.__globals__["os"].environ):
        if key.startswith("PROVIDER_"):
            monkeypatch.delenv(key, raising=False)
    yield
    registry.bind_model_handle(None)


class TestBothImplementationsAreInTheLiveRegistry:
    def test_the_http_kind_is_registered(self) -> None:
        """Not *importable* - **registered**, in the mapping ``get`` consults."""
        assert "openai_compatible" in registry.registered_kinds()

    def test_the_http_row_resolves_through_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "judge-7b")
        assert registry.get("judge").identity.kind == "openai_compatible"

    def test_the_in_process_row_resolves_through_get(self) -> None:
        registry.bind_model_handle(FakeModelHandle())
        assert registry.get(registry.IN_PROCESS_NAME).identity.kind == "in_process"

    def test_an_unregistered_kind_refuses_rather_than_resolving_another(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The mutation's shape, asserted directly.**

        With the HTTP kind unregistered, a configured row must refuse by name - not silently
        resolve to the in-process provider, which would be M-9 shipped on purpose.
        """
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "judge-7b")
        monkeypatch.delitem(registry._FACTORIES, "openai_compatible")
        # `ensure_registered` is called by `list_rows` and `get`, so it would re-register
        # the kind this test just removed. Stubbed out for the duration, which is what
        # deleting the `register(...)` line in the source actually does - and that source
        # mutation is the authoritative control; this is its in-test rehearsal.
        monkeypatch.setattr(registry, "ensure_registered", lambda: None)
        row = registry.row_for("judge")
        assert row is not None and row.available is False
        with pytest.raises(Refusal) as caught:
            registry.get("judge")
        assert caught.value.code == "inference_provider_unavailable"
        registry.ensure_registered()


class TestTheGenerationCallIsAssertedOnPayloadAndCount:
    def test_the_http_payload_field_by_field_and_the_count(self) -> None:
        """Through F3's **real** request builder; only the socket is replaced.

        A mock would never run the builder, so a wrong field, a missing one or a mishandled
        status would pass it - ADR-016 names over-mocked stubs among the reasons a suite
        stays green over a real bug.
        """
        server = StubJudgeServer(reply="an answer")
        provider = server.provider(base_url="http://judge.test:8000", model="judge-7b")
        params = GenerationParams(max_tokens=32, temperature=0.0, top_p=1.0)

        completions = provider.generate(["one", "two"], params)

        assert server.call_count == 2, "one call per prompt, counted - not 'was called'"
        body = server.calls[0]
        assert body["model"] == "judge-7b"
        assert body["max_tokens"] == 32
        # Greedy is temperature ZERO, sent explicitly. F4's own documents say to assert
        # `"temperature" not in body`; F3's provider - which F4 reuses and does not fork -
        # always sends it, and sending `temperature: 0` is how an OpenAI-compatible endpoint
        # is told to decode deterministically. Asserting the VALUE is the assertion that
        # actually protects FR-004.5; asserting absence would protect nothing and would
        # require forking F3's module to satisfy.
        assert body["temperature"] == 0.0
        assert body["top_p"] == 1.0
        assert body["stream"] is False
        # Sent only when set: `seed: null` is accepted by some engines and rejected by
        # others, and sending a null the caller did not ask for is a request they did not
        # make.
        assert "seed" not in body
        assert body["messages"] == [{"role": "user", "content": "one"}]
        assert [completion.text for completion in completions] == ["an answer", "an answer"]

    def test_every_completion_carries_the_provider_and_the_params(self) -> None:
        """R-02.16, at the smallest unit: a completion knows what produced it."""
        server = StubJudgeServer(reply="x")
        provider = server.provider()
        params = GenerationParams(max_tokens=8)
        completion = provider.generate(["p"], params)[0]
        assert completion.provider == provider.identity
        assert completion.params == params

    def test_the_in_process_payload_and_count(self) -> None:
        """Same assertion on the other implementation, so neither can drift alone."""
        handle = FakeModelHandle(answers=lambda prompt, index: f"a{index}")
        registry.bind_model_handle(handle)
        params = GenerationParams(max_tokens=16, temperature=0.0)

        completions = registry.get(registry.IN_PROCESS_NAME).generate(["p0", "p1"], params)

        assert len(handle.calls) == 1, "one batched call, counted"
        assert handle.calls[0][0] == ("p0", "p1")
        assert handle.calls[0][1] is params
        assert [completion.text for completion in completions] == ["a0", "a1"]

    def test_order_is_preserved_across_batch_boundaries(self) -> None:
        """Results are joined to prompts positionally; a reorder misattributes every score
        after the gap, silently."""
        handle = FakeModelHandle(answers=lambda prompt, index: prompt.upper())
        registry.bind_model_handle(handle)
        from providers.in_process import InProcessProvider

        provider = InProcessProvider(handle, batch_bound=2)
        texts = [c.text for c in provider.generate(["a", "b", "c", "d", "e"], GenerationParams())]
        assert texts == ["A", "B", "C", "D", "E"]


class TestATransportFailureRefusesRatherThanFabricating:
    def test_an_unreachable_endpoint_refuses(self) -> None:
        server = StubJudgeServer(raise_error=httpx.ConnectError("refused"))
        with pytest.raises(Refusal):
            server.provider().generate(["p"], GenerationParams())

    def test_an_unparseable_reply_refuses_with_no_default_completion(self) -> None:
        server = StubJudgeServer(body={"nothing": "useful"})
        with pytest.raises(Refusal) as caught:
            server.provider().generate(["p"], GenerationParams())
        assert "unparseable" in caught.value.code or "unavailable" in caught.value.code
