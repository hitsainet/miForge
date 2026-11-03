"""Every refusal code F3 produces, in the envelope  (AC-003.22; FR-003.19, ADR-013).

**A reason code with no test is a code that exists only in prose.** So this file is the
roll-call: each code F3 can raise is produced by a test, in ADR-013's envelope, carrying the
``details`` the FPRD's edge-case table says it carries - *"both digests, both identities,
both numbers, the failing pointer or the cycle path as applicable."* A refusal whose
``details`` omits the figures makes the operator guess, which is the thing ADR-018 forbids
the code from doing and is no better when the reader has to do it.

The three categories are kept apart, in code and here:

1. **configuration refusals** - at resolve/``prepare``, before any completion;
2. **evaluation refusals** - they fail the whole evaluation;
3. **verdicts** - normal outcomes, never errors.

**Category 2 must never collapse into category 3.** A transport failure is not a verdict;
defaulting it to accept turns this feature into the accept-everything selector it exists to
prevent.
"""

from __future__ import annotations

import json

import httpx
import pytest

from core.errors import Refusal
from services.selectors import availability, provider_port
from services.selectors.plugins import human_review
from services.selectors.resolve import resolve
from services.selectors.service import evaluate
from services.selectors.settings import reset_selector_settings
from services.selectors.types import EvaluationContext
from tests.support.selector_stubs import (
    StubJudgeServer,
    StubProviderResolver,
    StubReviewQueue,
    composite_config,
    handoff_version,
    human_config,
    judge_config,
    reference_config,
    resolved,
    rule_config,
)

#: Every code this feature raises, with the category it belongs to. **The roll-call.**
#: A code added to the source without a line here, or a line here with no raising test, is
#: the drift this file exists to catch.
CONFIGURATION_CODES = frozenset(
    {
        "selector_type_unknown",
        "selector_config_invalid",
        "selector_config_hash_mismatch",
        "selector_config_contract_malformed",
        "selector_config_source_missing",
        "composite_cycle",
        "composite_child_unknown",
        "composite_children_missing",
        "rule_expression_invalid",
        "rule_pattern_invalid",
        "reference_match_threshold_missing",
        "reference_match_threshold_not_applicable",
        "judge_threshold_missing",
        "judge_template_not_configured",
        "judge_template_invalid",
        "judge_is_resident_model",
        "judge_provider_not_configured",
        "judge_run_target_unreadable",
        "midataworks_not_configured",
        # Added 2026-10-16 with the review-queue port's production binding. A **different
        # fact** from the line above it: `midataworks_not_configured` means the sibling is
        # absent, this means in-run review was reached with no run to attach a queue to -
        # R-02.39's authoring path, which has no binding row to record a queue id on. One
        # code for both states would have put the operator back where the defect left
        # them, told that a configured service is missing.
        "review_queue_run_unknown",
        "schema_not_vendored",
    }
)

EVALUATION_CODES = frozenset(
    {
        "judge_provider_unavailable",
        "judge_reply_unparseable",
        "judge_provider_option_unknown",
        "grader_score_out_of_range",
        "grader_form_unavailable",
        "reference_missing",
        "review_decision_unrecognised",
        "selector_text_too_long",
        "evaluation_cancelled",
        "evaluation_too_large",
        "evaluation_prompt_count_mismatch",
        "selector_not_prepared",
        # Added 2026-10-16 with the provider's next-token scoring capability (R-02.2's
        # reconstruction check reads it; F11 FR-011.10). They live in the shared provider
        # module this roll call scans, so they are produced through it here.
        "provider_logprobs_unavailable",
        "provider_logprobs_incomplete",
        "provider_logprobs_token_set_invalid",
        # F11 14.3b: a float32 scoring request the endpoint did not confirm.
        "provider_scoring_precision_unconfirmed",
        # miLLM's 409 scoring_precision_unavailable (FR-25.15): it refused float32 by name.
        "provider_scoring_precision_unavailable",
    }
)

ALL_CODES = CONFIGURATION_CODES | EVALUATION_CODES

TEMPLATE = "Rate {prompt}/{completion}. Reply 'SCORE: <0..1>'."


@pytest.fixture(autouse=True)
def _clean():
    reset_selector_settings()
    yield
    provider_port.bind_provider_resolver(None)
    human_review.bind_review_queue(None)
    reset_selector_settings()


def _raise(code: str, monkeypatch) -> Refusal:
    """Produce ``code`` through the real code path, and hand back the refusal.

    One function so the roll-call below is a parametrisation rather than fifty near-copies -
    and so a code with no way to be produced is impossible to list.
    """
    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
    reset_selector_settings()

    def judge_with(server: StubJudgeServer, **kwargs):
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        return resolved(judge_config(**kwargs))

    with pytest.raises(Refusal) as exc:
        match code:
            case "selector_type_unknown":
                from services.selectors import registry

                registry.get("nope")
            case "selector_config_invalid":
                resolve({"kind": "rule", "name": "x"}, source="inline")
            case "selector_config_hash_mismatch":
                config = rule_config(min_length=1)
                config["config_sha256"] = "b" * 64
                resolve(config, handoff=handoff_version("a" * 64))
            case "selector_config_contract_malformed":
                broken = handoff_version("a" * 64)
                broken["content"]["config"].pop("sha256")
                resolve(rule_config(min_length=1), handoff=broken)
            case "selector_config_source_missing":
                resolve(rule_config(min_length=1), source="")
            case "composite_cycle":
                loop = composite_config("loop", ["loop"], "all")
                resolved(loop, catalog=[loop])
            case "composite_child_unknown":
                config = composite_config("c", ["ghost"], "all")
                resolved(config, catalog=[config])
            case "composite_children_missing":
                config = composite_config("c", ["a"], "all")
                selector = resolved(config, catalog=[config, rule_config("a", min_length=1)])
                selector.children = ()
                selector.plugin.prepare(selector)
            case "rule_expression_invalid":
                resolve(
                    {
                        "kind": "rule",
                        "name": "x",
                        "config_sha256": None,
                        "expression": "not json",
                    },
                    source="inline",
                )
            case "rule_pattern_invalid":
                resolved(rule_config(patterns=[{"regex": "(a+)+$", "must_match": True}]))
            case "reference_match_threshold_missing":
                resolved(reference_config(mode="fuzzy", threshold=None))
            case "reference_match_threshold_not_applicable":
                resolved(reference_config(mode="exact", threshold=0.5))
            case "judge_threshold_missing":
                judge_with(StubJudgeServer(), threshold=None)
            case "judge_template_not_configured":
                monkeypatch.delenv("JUDGE_PROMPT_TEMPLATE", raising=False)
                reset_selector_settings()
                judge_with(StubJudgeServer())
            case "judge_template_invalid":
                monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", "Rate {nope}")
                reset_selector_settings()
                judge_with(StubJudgeServer())
            case "judge_is_resident_model":
                from providers.base import ProviderIdentity

                target = ProviderIdentity(
                    kind="openai_compatible",
                    model="judge-7b",
                    base_url="http://judge.test:8000",
                )
                provider_port.bind_provider_resolver(
                    StubProviderResolver(StubJudgeServer().provider())
                )
                resolved(judge_config(), run_target=target)
            case "judge_provider_not_configured":
                provider_port.bind_provider_resolver(None)
                resolved(judge_config())
            case "judge_run_target_unreadable":
                provider_port.bind_provider_resolver(
                    StubProviderResolver(StubJudgeServer().provider())
                )
                resolved(judge_config(), run_target=3.14)
            case "midataworks_not_configured":
                human_review.bind_review_queue(None)
                resolved(human_config())
            case "review_queue_run_unknown":
                # The scope's own guard, which is the reachable half from here: F10's
                # `RunScopedReviewQueue` raises the same code for a queue operation with
                # no run in scope, and `test_review_queue_wiring.py` covers that half
                # through the real port.
                with human_review.review_run(""):  # pragma: no cover - refuses first
                    pass
            case "schema_not_vendored":
                import services.selectors.resolve as resolve_module

                monkeypatch.setattr(resolve_module, "_schema", None)
                monkeypatch.setenv("MIFORGE_SCHEMA_DIR", "/nonexistent/schemas")
                try:
                    resolve_module.load_schema()
                finally:
                    monkeypatch.delenv("MIFORGE_SCHEMA_DIR", raising=False)
                    resolve_module._schema = None
            case "judge_provider_unavailable":
                selector = judge_with(StubJudgeServer(raise_error=httpx.ConnectError("x")))
                selector.plugin.evaluate(prompt="q", completion="a", context={})
            case "judge_reply_unparseable":
                selector = judge_with(StubJudgeServer(reply="no score here"))
                selector.plugin.evaluate(prompt="q", completion="a", context={})
            case "judge_provider_option_unknown":
                from providers.base import GenerationParams

                StubJudgeServer().provider().generate(["q"], GenerationParams(), top_k=5)
            case "grader_score_out_of_range":
                selector = judge_with(StubJudgeServer(reply="SCORE: 9"))
                selector.plugin.evaluate(prompt="q", completion="a", context={})
            case "grader_form_unavailable":
                from services.graders import grader_for

                grader_for(resolved(rule_config(min_length=1)))
            case "reference_missing":
                selector = resolved(reference_config(mode="exact", threshold=None))
                selector.plugin.evaluate(prompt="", completion="x", context={})
            case "review_decision_unrecognised":
                queue = StubReviewQueue()
                human_review.bind_review_queue(queue)
                selector = resolved(human_config())
                selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
                queue.decide(human_review.derive_external_id("b", 0, "A"), "escalate")
                selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
                selector.plugin.evaluate(
                    prompt="Q", completion="A", context={"index": 0, "batch_id": "b"}
                )
            case "selector_text_too_long":
                selector = judge_with(StubJudgeServer())
                selector.plugin.evaluate(prompt="q", completion="x" * 70_000, context={})
            case "selector_not_prepared":
                # Reachable exactly when a caller builds a selector without going through
                # `resolve` - which is what a hand-built fixture does. Refusing by name
                # beats an AttributeError, and beats judging with no threshold.
                monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
                reset_selector_settings()
                selector = judge_with(StubJudgeServer())
                selector.prepared = None
                selector.plugin.evaluate(prompt="q", completion="a", context={})
            case "evaluation_cancelled":
                evaluate(
                    ["a"],
                    resolved(rule_config(min_length=1)),
                    EvaluationContext(is_cancelled=lambda: True),
                )
            case "evaluation_too_large":
                monkeypatch.setenv("EVALUATE_MAX_COMPLETIONS", "1")
                reset_selector_settings()
                evaluate(["a", "b"], resolved(rule_config(min_length=1)))
            case "evaluation_prompt_count_mismatch":
                evaluate(
                    ["a", "b"],
                    resolved(rule_config(min_length=1)),
                    EvaluationContext(),
                    prompts=["one"],
                )
            case "provider_logprobs_unavailable":
                StubJudgeServer(
                    body={"choices": [{"text": "Paris"}]}
                ).provider().next_token_logprobs("q", [1])
            case "provider_logprobs_incomplete":
                StubJudgeServer(
                    body={"choices": [{"logprobs": {"top_logprobs": [{"token_id:1": -0.1}]}}]}
                ).provider().next_token_logprobs("q", [1, 2])
            case "provider_logprobs_token_set_invalid":
                StubJudgeServer().provider().next_token_logprobs("q", list(range(21)))
            case "provider_scoring_precision_unconfirmed":
                # A complete scoring answer with no `scoring_precision` echo - today's miLLM.
                StubJudgeServer(
                    body={"choices": [{"logprobs": {"top_logprobs": [{"token_id:1": -0.1}]}}]}
                ).provider().next_token_logprobs("q", [1], precision="float32")
            case "provider_scoring_precision_unavailable":
                # miLLM's real envelope for a bitsandbytes model (its test_float32_scoring.py).
                StubJudgeServer(
                    status=409,
                    body={
                        "error": {
                            "message": "float32 scoring is unavailable "
                            "(reason: model_not_float32_castable): quantized",
                            "type": "invalid_request_error",
                            "param": "scoring_precision",
                            "code": "scoring_precision_unavailable",
                        }
                    },
                ).provider().next_token_logprobs("q", [1], precision="float32")
            case _:  # pragma: no cover - the roll-call is exhaustive by construction
                raise AssertionError(f"no production path written for {code!r}")
    return exc.value


@pytest.mark.parametrize("code", sorted(ALL_CODES))
def test_every_code_is_produced_by_a_real_path(code: str, monkeypatch) -> None:
    """The roll-call. A code only in prose fails here."""
    refusal = _raise(code, monkeypatch)
    assert refusal.code == code


@pytest.mark.parametrize("code", sorted(ALL_CODES))
def test_every_refusal_uses_adr_013s_envelope(code: str, monkeypatch) -> None:
    """AC-003.22: always these three keys, always in this shape."""
    envelope = _raise(code, monkeypatch).envelope()
    assert set(envelope) == {"error"}
    assert set(envelope["error"]) == {"code", "message", "details"}
    assert envelope["error"]["code"] == code
    assert isinstance(envelope["error"]["details"], dict)


@pytest.mark.parametrize("code", sorted(ALL_CODES))
def test_every_message_says_what_to_do_next_rather_than_how_to_feel(code: str, monkeypatch) -> None:
    """ADR-021. "Set JUDGE_PROVIDER_BASE_URL" is useful; "something went wrong" is not."""
    message = _raise(code, monkeypatch).message
    assert len(message) > 30, f"{code}'s message is too short to say what to do next"
    for apology in ("sorry", "oops", "something went wrong", "unexpected error"):
        assert apology not in message.lower()


@pytest.mark.parametrize("code", sorted(ALL_CODES))
def test_every_refusal_is_json_serialisable(code: str, monkeypatch) -> None:
    """A refusal that cannot be rendered fails while reporting a failure."""
    json.dumps(_raise(code, monkeypatch).envelope(), default=str)


def test_the_roll_call_is_complete() -> None:
    """**Every code raised in F3's source is listed above.** The drift-catcher.

    Without this, a code added to the source with no test here would be a code that exists
    only in prose - and the three parametrised tests above would still be green, because
    they only check what the list already names. Scanned with AST rather than by reading,
    and in **both** directions: a listed code nobody raises is dead prose too.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    roots = [
        src / "services" / "selectors",
        src / "services" / "graders",
        src / "providers" / "openai_compatible.py",
        src / "api" / "v1" / "endpoints" / "selectors.py",
    ]
    files = [p for root in roots for p in (root.rglob("*.py") if root.is_dir() else [root])]

    raised: set[str] = set()
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "refuse" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    raised.add(first.value)
                elif isinstance(first, ast.Attribute):
                    # `availability.JUDGE_PROVIDER_NOT_CONFIGURED` and friends.
                    raised.add(getattr(availability, first.attr, first.attr))
            elif name == "not_configured" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    raised.add(f"{first.value}_not_configured")

    # Two codes are named rather than discovered, each for a stated reason:
    #   `evaluation_cancelled` - an `EvaluationCancelled` subclass builds its code in
    #     `__init__`, so no `refuse(...)` literal exists to find;
    #   `review_decision_unrecognised` - raised by the CONTRACT's `map_decision`, which is
    #     not F3's file. It is listed because F3's `human_review` is what reaches it, and
    #     an unrecognised decision kind arriving from miDataworks must be a named failure
    #     an operator can see rather than a verdict invented on its behalf.
    raised |= {"evaluation_cancelled", "review_decision_unrecognised"}

    unlisted = raised - ALL_CODES
    assert unlisted == set(), (
        f"these codes are raised but have no test in this file: {sorted(unlisted)}. A "
        f"reason code with no test is a code that exists only in prose."
    )
    unraised = ALL_CODES - raised
    assert unraised == set(), (
        f"these codes are listed but raised nowhere in F3's source: {sorted(unraised)}. A "
        f"listed code nobody raises is dead prose in the other direction."
    )


class TestTheDetailsCarryTheFigures:
    """*"both digests, both identities, both numbers, the failing pointer, the cycle path."*"""

    def test_a_hash_mismatch_carries_both_digests(self, monkeypatch) -> None:
        details = _raise("selector_config_hash_mismatch", monkeypatch).details
        assert {"configuration_cites", "contract_declares"} <= set(details)

    def test_a_self_judging_refusal_carries_both_identities(self, monkeypatch) -> None:
        details = _raise("judge_is_resident_model", monkeypatch).details
        assert details["judge"] and details["run_target"]

    def test_a_schema_failure_carries_the_failing_pointer(self, monkeypatch) -> None:
        details = _raise("selector_config_invalid", monkeypatch).details
        assert "pointer" in details

    def test_a_cycle_carries_the_cycle_path(self, monkeypatch) -> None:
        assert _raise("composite_cycle", monkeypatch).details["cycle"] == ["loop", "loop"]

    def test_an_out_of_range_score_carries_the_value_and_the_range(self, monkeypatch) -> None:
        details = _raise("grader_score_out_of_range", monkeypatch).details
        assert details["value"] == 9.0
        assert details["range"] == [0.0, 1.0]

    def test_a_bound_refusal_names_the_bound(self, monkeypatch) -> None:
        details = _raise("selector_text_too_long", monkeypatch).details
        assert details["limit"] == 65_536
        assert details["length"] == 70_000

    def test_an_unknown_type_lists_the_registered_ones(self, monkeypatch) -> None:
        details = _raise("selector_type_unknown", monkeypatch).details
        assert len(details["registered"]) == 5

    def test_a_not_configured_refusal_names_the_service_and_the_capabilities(
        self, monkeypatch
    ) -> None:
        details = _raise("midataworks_not_configured", monkeypatch).details
        assert details["service"] == "midataworks"
        assert details["capabilities_unavailable"]


class TestTheCategoriesStayApart:
    def test_the_two_code_sets_do_not_overlap(self) -> None:
        assert CONFIGURATION_CODES & EVALUATION_CODES == frozenset()

    def test_a_not_configured_code_follows_adr_010s_spelling(self) -> None:
        """``<service>_not_configured``, which the app maps to 503 **by suffix** - so a new
        optional service does not need an edit there to be reported as "not configured"
        rather than "your request was bad"."""
        from api.app import status_for

        for code in ALL_CODES:
            if code.endswith("_not_configured"):
                assert status_for(code) == 503

    def test_the_refusal_codes_the_app_maps_specially_are_still_mapped(self) -> None:
        from api.app import status_for

        assert status_for(availability.MIDATAWORKS_NOT_CONFIGURED) == 503
        assert status_for(availability.JUDGE_PROVIDER_NOT_CONFIGURED) == 503

    def test_no_evaluation_failure_returns_a_verdict(self, monkeypatch) -> None:
        """**The invariant category 2 exists for.** Mutation control 3's target.

        A transport failure is not a verdict; defaulting it to accept turns this feature
        into the accept-everything selector it exists to prevent.
        """
        for code in (
            "judge_provider_unavailable",
            "judge_reply_unparseable",
            "grader_score_out_of_range",
        ):
            refusal = _raise(code, monkeypatch)
            assert refusal.code == code, "a refusal, not a verdict"
