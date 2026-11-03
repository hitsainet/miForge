"""The reconstruction check: the quantity, the token set, the tolerance, the provider.

**MUT-14's target, and it is the shape the FPRD singles out:** widen the tolerance to accept
anything and *"a wrongly reconstructed graph trains silently and every score is against a
different model than the one served."* Nothing raises, the loss moves, and every test score
in the run is against the wrong model.

Two properties are asserted that a careless implementation drops:

* **the statistic is the one R-02.2 names** - maximum absolute difference, not a mean and
  not ``allclose`` with its own hidden ``rtol``. *A different statistic is a different bar
  at the same number;*
* **a token missing from either side is a mismatch**, never skipped. Skipping is how a
  comparison over an empty intersection returns "matched", which is the vacuity failure this
  project keeps finding.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from providers.base import ProviderIdentity
from training.hotupdate import verify
from training.hotupdate.verify import LocalLogprobs

TOKENS = ("A", "B", "C")


def local(**values: float) -> LocalLogprobs:
    return LocalLogprobs(values=dict(values))


def result(local_values: dict[str, float], remote: dict[str, float], tol: float = 1e-2):
    return verify.check(
        LocalLogprobs(values=local_values),
        remote,
        token_set=TOKENS,
        tolerance=tol,
        tolerance_is_provisional=True,
        provider="openai_compatible:m@http://millm.test",
    )


class TestTheComparison:
    def test_agreement_within_the_tolerance_matches(self) -> None:
        out = result({"A": -1.0, "B": -2.0, "C": -3.0}, {"A": -1.001, "B": -2.0, "C": -3.005})
        assert out.matched is True
        assert out.max_abs_difference == pytest.approx(0.005)

    def test_one_token_outside_the_tolerance_is_a_mismatch(self) -> None:
        """**MUT-14.** The statistic is the maximum, so one bad token is enough - which is
        the point: a wrongly rebuilt tensor moves one or two log-probabilities a long way."""
        out = result({"A": -1.0, "B": -2.0, "C": -3.0}, {"A": -1.0, "B": -2.0, "C": -9.0})
        assert out.matched is False
        assert out.max_abs_difference == pytest.approx(6.0)

    def test_the_statistic_is_the_maximum_and_not_a_mean(self) -> None:
        """A mean over three tokens would pass this: (0 + 0 + 6) / 3 = 2 is still over, but
        over thirty tokens it would not be. The requirement names the statistic."""
        values = dict.fromkeys(TOKENS, 0.0)
        remote = dict(values)
        remote["C"] = 6.0
        out = verify.check(
            LocalLogprobs(values=values),
            remote,
            token_set=TOKENS,
            tolerance=2.5,
            tolerance_is_provisional=True,
            provider="p",
        )
        assert out.statistic == "max_absolute_difference"
        assert out.matched is False

    def test_a_token_missing_from_either_side_is_a_mismatch_rather_than_skipped(self) -> None:
        """A comparison over an empty intersection must not return "matched"."""
        out = result({"A": -1.0}, {"A": -1.0})
        assert out.matched is False
        assert out.missing_tokens == ("B", "C")

    def test_an_empty_token_set_cannot_match(self) -> None:
        out = verify.check(
            local(), {}, token_set=(), tolerance=1e-2, tolerance_is_provisional=True, provider="p"
        )
        assert out.matched is False
        assert out.max_abs_difference is None


class TestTheTermsAreRecorded:
    def test_every_term_is_in_the_record(self) -> None:
        """*A verification whose terms are not recorded cannot be re-run later against the
        same bar.*"""
        record = result(
            {"A": -1.0, "B": -2.0, "C": -3.0}, {"A": -1.0, "B": -2.0, "C": -3.0}
        ).record()
        assert record["quantity"] == "log_probability"
        assert record["statistic"] == "max_absolute_difference"
        assert record["token_set"] == list(TOKENS)
        assert record["tolerance"] == 1e-2
        assert record["tolerance_is_provisional"] is True
        assert record["provider"].startswith("openai_compatible:")

    def test_the_provisional_label_travels_into_the_refusal(self) -> None:
        """R-02.2's value is *"a starting point for measurement, not a measured result"*, and
        a provisional bar that stops reading as provisional is MUT-14 shipping without a
        mutation."""
        out = result({"A": 0.0, "B": 0.0, "C": 0.0}, {"A": 9.0, "B": 0.0, "C": 0.0})
        with pytest.raises(Refusal) as caught:
            verify.abort_on_mismatch(out)
        assert caught.value.code == "reconstruction_mismatch"
        assert "PROVISIONAL" in caught.value.message
        assert caught.value.details["tolerance_is_provisional"] is True

    def test_a_measured_tolerance_is_not_labelled_provisional(self) -> None:
        out = verify.check(
            local(A=0.0, B=0.0, C=0.0),
            {"A": 9.0, "B": 0.0, "C": 0.0},
            token_set=TOKENS,
            tolerance=0.004,
            tolerance_is_provisional=False,
            provider="p",
        )
        with pytest.raises(Refusal) as caught:
            verify.abort_on_mismatch(out)
        assert "PROVISIONAL" not in caught.value.message

    def test_the_settings_know_whether_the_bar_is_still_the_provisional_one(self) -> None:
        from core.config import reset_settings
        from training.hotupdate import settings as settings_module

        reset_settings()
        cfg = settings_module.load()
        # The native bar defaults to D41's 0.5 (2026-10-18; it was R-02.2's 1e-2, which refused
        # every clean bf16 reconstruction) and stays PROVISIONAL.
        assert cfg.verification_tolerance == 0.5
        assert cfg.tolerance_is_provisional is True

    def test_the_default_is_float32_at_the_measured_bar(self) -> None:
        """F11 14.3b / D61: miLLM scores in float32 (FR-25.15), and 5e-4 was measured on the
        3090 (clean <= 4.96e-5, smallest perturbation 3.62e-3). So the default comparison is
        float32 at 5e-4, recorded as MEASURED - not the provisional 1e-2 it replaced."""
        from core.config import reset_settings
        from training.hotupdate import settings as settings_module

        reset_settings()
        cfg = settings_module.load()
        assert cfg.verification_precision == "float32"
        assert cfg.verification_tolerance_in_force == 5e-4
        assert cfg.tolerance_in_force_is_provisional is False
        # The dataclass's own defaults agree with the process configuration.
        bare = settings_module.HotUpdateSettings(
            base_url=None,
            trainer_token=None,
            audit_label=None,
            verification_tolerance=0.5,
            provisional_tolerance=0.01,
            heartbeat_interval_seconds=None,
            request_timeout_seconds=10.0,
            checkpoint_timeout_seconds=600.0,
            commit_end_timeout_seconds=30.0,
            commit_watchdog_ms=800.0,
            commit_creep_fraction=0.5,
        )
        assert (bare.verification_precision, bare.verification_tolerance_in_force) == (
            "float32",
            5e-4,
        )
        assert bare.tolerance_in_force_is_provisional is False

    def test_a_different_value_is_still_provisional_until_declared_measured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D41 (2026-10-17): 0.5 was set from one session's numbers. It must be recorded as
        PROVISIONAL in every verify result; only an explicit `..._MEASURED=true` changes that."""
        from core.config import reset_settings
        from training.hotupdate import settings as settings_module

        monkeypatch.setenv("HOTUPDATE_VERIFICATION_TOLERANCE", "0.5")
        reset_settings()
        cfg = settings_module.load()
        assert cfg.verification_tolerance == 0.5
        assert cfg.tolerance_is_provisional is True
        monkeypatch.setenv("HOTUPDATE_VERIFICATION_TOLERANCE_MEASURED", "true")
        reset_settings()
        assert settings_module.load().tolerance_is_provisional is False
        monkeypatch.delenv("HOTUPDATE_VERIFICATION_TOLERANCE")
        monkeypatch.delenv("HOTUPDATE_VERIFICATION_TOLERANCE_MEASURED")
        reset_settings()


class TestTheProviderPath:
    """FR-011.10: the serving side is reached through R-02.49's provider - never a second
    private HTTP path. *Rewritten 2026-10-16:* the provider used to fill no log-probabilities
    and this class pinned the refusal; it now has a scoring capability against miLLM's served
    ``/v1/completions`` scoring mode, and these tests drive the real provider over a stub."""

    class _Scorer:
        identity = ProviderIdentity(kind="openai_compatible", model="m", base_url="http://x")

        def __init__(self) -> None:
            self.calls: list[tuple[str, list[int], bool]] = []

        def next_token_logprobs(
            self, prompt: str, token_ids: list[int], *, add_special_tokens: bool = True
        ) -> dict[int, float]:
            self.calls.append((prompt, token_ids, add_special_tokens))
            return {i: -float(n + 1) for n, i in enumerate(token_ids)}

    def test_the_serving_side_is_reached_through_the_provider(self) -> None:
        scorer = self._Scorer()
        got = verify.logprobs_from_provider(scorer, "fixed prompt", [7, 3])
        assert got == {"7": -1.0, "3": -2.0}
        assert scorer.calls == [("fixed prompt", [7, 3], True)]

    def test_a_provider_without_the_capability_is_refused(self) -> None:
        class _GenerationOnly:
            identity = "generation-only"

        with pytest.raises(Refusal) as caught:
            verify.logprobs_from_provider(_GenerationOnly(), "p", [1])
        assert caught.value.code == "provider_logprobs_unavailable"
        assert "will NOT open a second HTTP path" in caught.value.message

    def _http(self, handler):  # type: ignore[no-untyped-def]
        import httpx

        from providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            base_url="http://millm.test:8000",
            model="LiquidAI/LFM2.5-1.2B-Instruct",
            connect_timeout_s=1.0,
            read_timeout_s=1.0,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    def test_the_http_provider_speaks_millms_scoring_mode(self) -> None:
        """The request miLLM's ``TextCompletionRequest`` accepts in scoring mode, field by
        field (``millm/api/schemas/openai.py`` on its main), and its response shape."""
        import json

        import httpx

        seen: list[tuple[str, dict]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.url.path, json.loads(request.content)))
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-x",
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": "Paris",
                            "finish_reason": "length",
                            "logprobs": {
                                "tokens": ["token_id:7"],
                                "token_logprobs": [-0.1],
                                "top_logprobs": [{"token_id:7": -0.1, "token_id:3": -2.4}],
                                "text_offset": [0],
                            },
                        }
                    ],
                },
            )

        got = self._http(handler).next_token_logprobs("The capital of France is", [7, 3])
        assert got == {7: -0.1, 3: -2.4}
        path, body = seen[0]
        assert path == "/v1/completions"
        assert body == {
            "model": "LiquidAI/LFM2.5-1.2B-Instruct",
            "prompt": "The capital of France is",
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 2,
            "allowed_token_ids": [7, 3],
            "return_tokens_as_token_ids": True,
            "add_special_tokens": True,
            "stream": False,
        }

    def test_a_partial_answer_is_refused_not_narrowed(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"logprobs": {"top_logprobs": [{"token_id:7": -0.1}]}}]},
            )

        with pytest.raises(Refusal) as caught:
            self._http(handler).next_token_logprobs("p", [7, 3])
        assert caught.value.code == "provider_logprobs_incomplete"

    def test_an_endpoint_without_scoring_mode_is_refused_by_name(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"text": "Paris"}]})

        with pytest.raises(Refusal) as caught:
            self._http(handler).next_token_logprobs("p", [7])
        assert caught.value.code == "provider_logprobs_unavailable"

    def test_more_than_twenty_tokens_is_refused(self) -> None:
        with pytest.raises(Refusal) as caught:
            self._http(lambda r: None).next_token_logprobs("p", list(range(21)))
        assert caught.value.code == "provider_logprobs_token_set_invalid"
