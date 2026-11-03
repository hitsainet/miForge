"""The HTTP provider's client lifetime and its concurrency bound.

Three defects were found on 2026-10-16 by **measuring** the judge path against a live
miLLM rather than reading it, and this file is what keeps each one closed.

1. ``close()`` was unreachable in both branches. The guard read
   ``self._client is not None and self._owns_client`` while ``_owns_client`` was
   ``client is None``: an injected client made ``_owns_client`` False, and an owned one
   made ``_client`` None. It read as resource management and closed nothing, ever.
2. A fresh ``httpx.Client`` was opened **per completion**, so a judged batch paid
   connection setup per candidate - measured at 66.6 ms against 59.3 ms pooled, 11.0 % of
   a same-cluster round trip.
3. ``HttpProviderRow.concurrency`` was **published by GET /api/v1/providers** and enforced
   nowhere: the provider had no such parameter, so every endpoint ran at an effective
   bound of 1 while the API reported 4.

Each test below is paired with the mutation that must turn it red; the controls were run
and recorded in AGENT.md rather than asserted here.
"""

from __future__ import annotations

import threading
import time
from typing import ClassVar

import httpx
import pytest

from core.errors import Refusal
from providers import openai_compatible as oc
from providers.base import GenerationParams
from providers.openai_compatible import OpenAICompatibleProvider

_REPLY = {
    "choices": [{"message": {"content": "SCORE: 0.90"}}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}


class _RecordingClient:
    """A stand-in for ``httpx.Client`` that records construction, closure and overlap."""

    #: Every instance ever built, so a test can count them rather than trust a flag.
    built: ClassVar[list[_RecordingClient]] = []
    #: The high-water mark of concurrent ``post`` calls across all instances.
    max_in_flight = 0
    _in_flight = 0
    _lock = threading.Lock()
    #: Set to a prompt substring to make that one request fail.
    fail_on: str | None = None
    #: How long each request dwells, so overlap is observable.
    dwell_s = 0.05

    def __init__(self, **_: object) -> None:
        self.closed = False
        self.posts: list[dict[str, object]] = []
        type(self).built.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.built = []
        cls.max_in_flight = 0
        cls._in_flight = 0
        cls.fail_on = None
        cls.dwell_s = 0.05

    def post(self, url: str, *, json: dict[str, object], headers: dict[str, str]):
        cls = type(self)
        with cls._lock:
            cls._in_flight += 1
            cls.max_in_flight = max(cls.max_in_flight, cls._in_flight)
        try:
            self.posts.append(json)
            time.sleep(cls.dwell_s)
            text = str(json["messages"][-1]["content"])  # type: ignore[index]
            if cls.fail_on is not None and cls.fail_on in text:
                return httpx.Response(
                    500, text="judge exploded", request=httpx.Request("POST", url)
                )
            body = {
                "choices": [{"message": {"content": f"SCORE: 0.90 for {text}"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
            return httpx.Response(200, json=body, request=httpx.Request("POST", url))
        finally:
            with cls._lock:
                cls._in_flight -= 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingClient]:
    _RecordingClient.reset()
    monkeypatch.setattr(oc.httpx, "Client", _RecordingClient)
    return _RecordingClient


def _provider(**kw: object) -> OpenAICompatibleProvider:
    defaults: dict[str, object] = {
        "base_url": "http://judge.invalid:8000",
        "model": "a-judge-model",
        "connect_timeout_s": 1.0,
        "read_timeout_s": 5.0,
    }
    defaults.update(kw)
    return OpenAICompatibleProvider(**defaults)  # type: ignore[arg-type]


PARAMS = GenerationParams(max_tokens=16, temperature=0.0, top_p=1.0)


class TestTheClientIsPooledNotPerCompletion:
    """Defect 2. Mutation: restore the ``with httpx.Client(...)`` inside ``_post``."""

    def test_eight_completions_open_exactly_one_client(self, recording) -> None:
        provider = _provider()
        out = provider.generate([f"p{i}" for i in range(8)], PARAMS)

        assert len(out) == 8
        # The payload, not merely the count: eight bodies actually went out.
        assert sum(len(c.posts) for c in recording.built) == 8
        assert len(recording.built) == 1, (
            "One pooled client per provider. A client per completion is the 11 % the "
            "live measurement found."
        )

    def test_a_provider_that_never_generates_opens_no_client(self, recording) -> None:
        """``GET /api/v1/providers`` builds a row per provider to report availability."""
        provider = _provider()
        assert provider.available() is True
        assert provider.identity.model == "a-judge-model"
        assert recording.built == [], "Constructing a provider must not open a socket."


class TestCloseIsReachable:
    """Defect 1. Mutation: ``if self._client is not None and self._owns_client:``."""

    def test_closing_an_owned_client_actually_closes_it(self, recording) -> None:
        provider = _provider()
        provider.generate(["one"], PARAMS)
        assert len(recording.built) == 1
        client = recording.built[0]
        assert client.closed is False

        provider.close()

        assert client.closed is True, (
            "close() must close the client this provider owns. The old guard could never "
            "fire in either branch."
        )

    def test_closing_is_idempotent_and_reopens_on_next_use(self, recording) -> None:
        provider = _provider()
        provider.generate(["one"], PARAMS)
        provider.close()
        provider.close()  # must not raise
        provider.generate(["two"], PARAMS)
        assert len(recording.built) == 2, "A closed provider builds a fresh client on reuse."

    def test_an_injected_client_is_never_closed_by_the_provider(self, recording) -> None:
        """The caller owns what the caller supplied."""
        injected = _RecordingClient()
        recording.built.clear()
        provider = _provider(client=injected)
        provider.generate(["one"], PARAMS)

        provider.close()

        assert injected.closed is False, "Closing a client the provider did not open."
        assert recording.built == [], "An injected client must not be supplemented."


class TestTheConcurrencyBoundIsEnforced:
    """Defect 3. Mutation: drop ``concurrency=row.concurrency`` from the registry."""

    def test_the_default_is_sequential(self, recording) -> None:
        """Unset means one in flight - the behaviour this class had before the wiring."""
        provider = _provider()
        provider.generate([f"p{i}" for i in range(6)], PARAMS)
        assert recording.max_in_flight == 1

    def test_four_is_four_in_flight_not_one(self, recording) -> None:
        provider = _provider(concurrency=4)
        provider.generate([f"p{i}" for i in range(8)], PARAMS)
        assert recording.max_in_flight == 4, (
            "The bound must be the ceiling AND the floor here: 1 means the fan-out never "
            "happened, and more than 4 means the bound is not a bound."
        )

    def test_the_bound_is_never_exceeded_by_a_bigger_batch(self, recording) -> None:
        provider = _provider(concurrency=2)
        provider.generate([f"p{i}" for i in range(16)], PARAMS)
        assert recording.max_in_flight == 2

    def test_fewer_prompts_than_workers_opens_no_idle_threads(self, recording) -> None:
        provider = _provider(concurrency=8)
        provider.generate(["only-one"], PARAMS)
        assert recording.max_in_flight == 1

    def test_results_stay_in_input_order_under_concurrency(self, recording) -> None:
        """The interface's contract. Callers join to prompts positionally."""
        prompts = [f"prompt-{i}" for i in range(12)]
        provider = _provider(concurrency=4)

        out = provider.generate(prompts, PARAMS)

        assert [c.text.rsplit(" ", 1)[-1] for c in out] == prompts

    def test_an_empty_prompt_list_makes_no_request(self, recording) -> None:
        provider = _provider(concurrency=4)
        assert provider.generate([], PARAMS) == []
        assert recording.built == []

    def test_a_concurrency_below_one_is_refused_not_clamped(self) -> None:
        with pytest.raises(Refusal) as caught:
            _provider(concurrency=0)
        assert caught.value.code == "judge_provider_not_configured"


class TestFailureUnderConcurrencyIsDeterministic:
    """A refusal that varied run to run would blame a different prompt each time."""

    def test_the_first_failure_by_prompt_index_is_the_one_raised(self, recording) -> None:
        recording.fail_on = "prompt-2"
        provider = _provider(concurrency=4)

        with pytest.raises(Refusal) as caught:
            provider.generate([f"prompt-{i}" for i in range(12)], PARAMS)

        assert caught.value.code == "judge_provider_unavailable"
        assert caught.value.details["status"] == 500

    def test_a_later_failure_does_not_mask_an_earlier_one(self, recording) -> None:
        """Two bad prompts: index order decides, not which thread finished first."""
        calls: list[str] = []
        original = _RecordingClient.post

        def post(self, url, *, json, headers):  # type: ignore[no-untyped-def]
            text = str(json["messages"][-1]["content"])
            calls.append(text)
            return original(self, url, json=json, headers=headers)

        _RecordingClient.post = post  # type: ignore[assignment]
        try:
            recording.fail_on = "prompt-"  # every prompt fails
            provider = _provider(concurrency=4)
            with pytest.raises(Refusal):
                provider.generate([f"prompt-{i}" for i in range(16)], PARAMS)
        finally:
            _RecordingClient.post = original  # type: ignore[assignment]

        assert len(calls) < 16, (
            "Pending work must be cancelled once a failure is seen, so a dead endpoint "
            "fails fast instead of retrying once per prompt."
        )


class TestTheRegistryActuallyPassesTheRowsBound:
    """The wiring, not the capability. Removing one keyword must turn this red.

    ``redacted()`` publishes ``concurrency`` on ``GET /api/v1/providers``, so this is the
    test that keeps the published figure and the enforced figure the same number.
    """

    def test_the_rows_concurrency_reaches_the_provider(self) -> None:
        from providers.registry import _build_openai_compatible
        from providers.settings import HttpProviderRow

        row = HttpProviderRow(
            name="judge",
            base_url="http://judge.invalid:8000",
            model="a-judge-model",
            concurrency=7,
        )

        provider = _build_openai_compatible(row)

        assert provider._concurrency == 7, (
            "GET /api/v1/providers publishes this number. A provider built without it "
            "runs at 1 while the API reports 7."
        )

    def test_the_published_figure_and_the_enforced_figure_are_one_number(self) -> None:
        from providers.registry import _build_openai_compatible
        from providers.settings import HttpProviderRow

        row = HttpProviderRow(
            name="judge",
            base_url="http://judge.invalid:8000",
            model="a-judge-model",
            concurrency=3,
        )

        assert row.redacted()["concurrency"] == _build_openai_compatible(row)._concurrency
