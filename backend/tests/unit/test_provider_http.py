"""The OpenAI-compatible HTTP provider  (ADR-009, R-02.49; F3's module, F4 reuses it).

One implementation covers **miLLM, vLLM, Ollama and the OpenAI API** with one ``base_url``.
These tests exist to pin the three properties the rest of the system leans on:

* **miLLM is never a default.** The endpoint is configuration. A named default here is the
  thing R-02.49 was written to remove - *"a test path wired to one sibling by name is what
  made the whole document depend on it."*
* **Order is part of the contract.** Callers join completions to prompts positionally, so a
  provider that reordered or dropped one would silently misattribute every score after the
  gap.
* **No fabricated reply, ever.** Every failure is a named refusal.

Run over an ``httpx.MockTransport`` stub rather than a mock client, so the **real** request
builder and the **real** response parser run and only the socket is replaced.
"""

from __future__ import annotations

import httpx
import pytest

from core.errors import Refusal
from providers.base import Completion, GenerationParams, InferenceProvider
from providers.openai_compatible import (
    CAPABILITY_GENERATION,
    PROVIDER_KIND,
    OpenAICompatibleProvider,
)
from tests.support.selector_stubs import StubJudgeServer


def provider(server: StubJudgeServer, **kwargs) -> OpenAICompatibleProvider:
    defaults = {
        "base_url": "http://any-endpoint:8000",
        "model": "some-model",
        "connect_timeout_s": 1.0,
        "read_timeout_s": 2.0,
        "client": httpx.Client(transport=httpx.MockTransport(server.handler)),
    }
    defaults.update(kwargs)
    return OpenAICompatibleProvider(**defaults)  # type: ignore[arg-type]


class TestItSatisfiesTheFoundationInterface:
    def test_it_is_an_inference_provider(self) -> None:
        """Against the runtime-checkable Protocol, not against a docstring."""
        assert isinstance(provider(StubJudgeServer()), InferenceProvider)

    def test_the_identity_names_the_family_and_never_a_sibling(self) -> None:
        identity = provider(StubJudgeServer()).identity
        assert identity.kind == PROVIDER_KIND == "openai_compatible"
        for sibling in ("millm", "mistudio", "midataworks", "ollama", "vllm"):
            assert sibling not in identity.kind

    def test_the_endpoint_lives_in_base_url_not_in_the_type_system(self) -> None:
        identity = provider(StubJudgeServer(), base_url="http://millm:8000").identity
        assert identity.base_url == "http://millm:8000"
        assert identity.kind == "openai_compatible"

    def test_the_record_string_is_what_score_record_provider_takes(self) -> None:
        """``ScoreRecord.provider`` is ``anyOf [string, null]`` in both vendored manifests,
        so the schema decides the shape, not this module."""
        record = provider(StubJudgeServer()).identity.record()
        assert record == "openai_compatible:some-model@http://any-endpoint:8000"

    def test_capabilities_are_declared(self) -> None:
        """Declared, not inferred from a 404: a reverse proxy, a wrong path prefix and an
        unsupported capability all produce one. Next-token log-probabilities joined on
        2026-10-16 for R-02.2's reconstruction check."""
        from providers.openai_compatible import CAPABILITY_NEXT_TOKEN_LOGPROBS

        assert provider(StubJudgeServer()).capabilities() == frozenset(
            {CAPABILITY_GENERATION, CAPABILITY_NEXT_TOKEN_LOGPROBS}
        )
        assert "embeddings" not in provider(StubJudgeServer()).capabilities()


class TestConstructionRefusesRatherThanGuessing:
    def test_no_base_url_refuses_by_name(self) -> None:
        with pytest.raises(Refusal) as exc:
            OpenAICompatibleProvider(
                base_url="", model="m", connect_timeout_s=1.0, read_timeout_s=1.0
            )
        assert exc.value.code == "judge_provider_not_configured"

    def test_no_model_refuses_because_a_score_needs_a_named_provider(self) -> None:
        with pytest.raises(Refusal) as exc:
            OpenAICompatibleProvider(
                base_url="http://x:1", model="", connect_timeout_s=1.0, read_timeout_s=1.0
            )
        assert exc.value.code == "judge_provider_not_configured"

    def test_the_timeouts_are_required_keyword_arguments(self) -> None:
        """A missing timeout is how a judge outage becomes a hung step. No default exists
        to forget."""
        with pytest.raises(TypeError):
            OpenAICompatibleProvider(base_url="http://x:1", model="m")  # type: ignore[call-arg]

    def test_a_trailing_slash_in_the_base_url_is_normalised_away(self) -> None:
        assert provider(StubJudgeServer(), base_url="http://x:1/").base_url == "http://x:1"


class TestGeneration:
    def test_one_completion_per_prompt_in_input_order(self) -> None:
        server = StubJudgeServer(reply="ok")
        results = provider(server).generate(["first", "second", "third"], GenerationParams())
        assert len(results) == 3
        assert server.call_count == 3
        assert [call["messages"][-1]["content"] for call in server.calls] == [
            "first",
            "second",
            "third",
        ]

    def test_the_completion_carries_the_provenance_a_score_needs(self) -> None:
        server = StubJudgeServer(reply="an answer")
        result = provider(server).generate(["q"], GenerationParams(temperature=0.7))[0]
        assert isinstance(result, Completion)
        assert result.text == "an answer"
        assert result.provider.model == "some-model"
        assert result.params.temperature == 0.7
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 7

    def test_the_decode_parameters_are_on_the_wire(self) -> None:
        """They are part of what a score means (ADR-009): a score at temperature 0 is not
        comparable with one at 0.8."""
        server = StubJudgeServer()
        params = GenerationParams(max_tokens=64, temperature=0.3, top_p=0.9, stop=("\n\n",))
        provider(server).generate(["q"], params)
        body = server.calls[0]
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.9
        assert body["stop"] == ["\n\n"]

    def test_a_seed_is_sent_only_when_set(self) -> None:
        """Sending a null the caller did not ask for is a request they did not make, and
        some engines reject it."""
        server = StubJudgeServer()
        provider(server).generate(["q"], GenerationParams())
        assert "seed" not in server.calls[0]
        provider(server).generate(["q"], GenerationParams(seed=7))
        assert server.calls[1]["seed"] == 7

    def test_a_system_message_is_sent_first_when_given(self) -> None:
        server = StubJudgeServer()
        provider(server).generate(["q"], GenerationParams(), system="be terse")
        roles = [message["role"] for message in server.calls[0]["messages"]]
        assert roles == ["system", "user"]

    def test_an_unknown_option_is_refused_rather_than_dropped(self) -> None:
        """A silently dropped option is a request the caller believes it made."""
        with pytest.raises(Refusal) as exc:
            provider(StubJudgeServer()).generate(["q"], GenerationParams(), top_k=40)
        assert exc.value.code == "judge_provider_option_unknown"
        assert exc.value.details["unknown"] == ["top_k"]

    def test_an_api_key_becomes_a_bearer_header_and_is_not_in_the_body(self) -> None:
        """ADR-020: no secret in a record, a hash input or a log line."""
        server = StubJudgeServer()
        provider(server, api_key="sk-secret").generate(["q"], GenerationParams())
        assert server.headers[0]["authorization"] == "Bearer sk-secret"
        assert "sk-secret" not in str(server.calls[0])

    def test_the_legacy_completions_shape_is_also_read(self) -> None:
        server = StubJudgeServer(body={"choices": [{"text": "legacy answer"}]})
        assert provider(server).generate(["q"], GenerationParams())[0].text == "legacy answer"


class TestFailuresAreNamedRefusals:
    @pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
    def test_a_non_2xx_refuses_with_the_status(self, status: int) -> None:
        server = StubJudgeServer(status=status)
        with pytest.raises(Refusal) as exc:
            provider(server).generate(["q"], GenerationParams())
        assert exc.value.code == "judge_provider_unavailable"
        assert exc.value.details["status"] == status

    def test_a_body_that_is_not_json_refuses(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json at all")

        instance = OpenAICompatibleProvider(
            base_url="http://x:1",
            model="m",
            connect_timeout_s=1.0,
            read_timeout_s=1.0,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(Refusal) as exc:
            instance.generate(["q"], GenerationParams())
        assert exc.value.code == "judge_reply_unparseable"

    def test_a_json_array_body_refuses(self) -> None:
        server = StubJudgeServer(body=[1, 2, 3])
        with pytest.raises(Refusal) as exc:
            provider(server).generate(["q"], GenerationParams())
        assert exc.value.details["parsed_type"] == "list"

    def test_an_empty_choices_list_refuses(self) -> None:
        server = StubJudgeServer(body={"choices": []})
        with pytest.raises(Refusal) as exc:
            provider(server).generate(["q"], GenerationParams())
        assert exc.value.code == "judge_reply_unparseable"

    def test_a_choice_with_no_text_refuses(self) -> None:
        server = StubJudgeServer(body={"choices": [{"index": 0}]})
        with pytest.raises(Refusal) as exc:
            provider(server).generate(["q"], GenerationParams())
        assert exc.value.code == "judge_reply_unparseable"

    def test_the_error_body_is_truncated_in_the_details(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="y" * 4000)

        instance = OpenAICompatibleProvider(
            base_url="http://x:1",
            model="m",
            connect_timeout_s=1.0,
            read_timeout_s=1.0,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(Refusal) as exc:
            instance.generate(["q"], GenerationParams())
        assert len(exc.value.details["body"]) < 600

    def test_no_failure_path_returns_a_completion(self) -> None:
        """The one invariant that matters most: a transport failure is not a reply.

        Asserted over every failure mode at once, because the dangerous edit is a single
        ``except: return []`` added later in one of them.
        """
        failures = [
            StubJudgeServer(status=500),
            StubJudgeServer(raise_error=httpx.ConnectError("x")),
            StubJudgeServer(raise_error=httpx.ReadTimeout("x")),
            StubJudgeServer(body={"choices": []}),
        ]
        for server in failures:
            with pytest.raises(Refusal):
                provider(server).generate(["q"], GenerationParams())


class TestAvailabilityDoesNotProbeTheNetwork:
    def test_available_is_true_once_constructed(self) -> None:
        """A liveness probe here would make listing the selector types depend on a remote
        host being up, and a slow endpoint would make the listing slow."""
        server = StubJudgeServer(raise_error=httpx.ConnectError("down"))
        assert provider(server).available() is True
        assert server.call_count == 0
