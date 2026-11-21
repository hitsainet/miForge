"""The gateway's pure parts (feature 016; FTASKS 6.3b, 6.7c in part; FR-016.15; R1B-A5, R1B-A13,
R3A-6): the allow-list's imports, the locality copy, the worst case, the cap and the pricing."""

from __future__ import annotations

import ast
import math
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from wsp import endpoints
from wsp_gateway import allowlist, meter, upstream

SRC = Path(__file__).resolve().parents[3] / "src"


def test_the_allowlist_imports_only_the_standard_library() -> None:
    """R3A-6: the worker image's test target carries this one file beside the real producer."""
    tree = ast.parse((SRC / "wsp_gateway" / "allowlist.py").read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import reaches the package"
            modules.add((node.module or "").split(".")[0])
    assert modules, "the scan saw no import"
    assert modules <= set(sys.stdlib_module_names) | {"__future__"}, sorted(modules)


@pytest.mark.parametrize(
    "url",
    [
        "http://miserver-backend.miserver.svc:8000/v1",
        "http://miserver-backend.miserver.svc.cluster.local:8000",
        "http://10.1.2.3:8000",
        "http://192.168.244.52",
        "http://172.16.0.9",
        "http://127.0.0.1:9",
        "http://localhost",
        "http://[fd00::1]:80",
        "https://api.anthropic.com",
        "https://api.openai.com/v1",
        "https://93.184.216.34",
        "https://x.svc.evil.example",
    ],
)
def test_the_gateways_locality_copy_agrees_with_the_backend(url: str) -> None:
    assert upstream.derive_locality(url) == endpoints.derive_locality(url)


PRICES = meter.Prices(Decimal(3), Decimal(15), Decimal("3.75"), Decimal("0.3"))


def test_the_worst_case_prices_every_input_byte_as_a_cache_write() -> None:
    usd, tokens = meter.worst_case(1000, 2000, PRICES)
    assert tokens == 3000
    assert usd == Decimal(1000) * Decimal("3.75") / 1_000_000 + Decimal(2000) * 15 / 1_000_000


def test_the_cap_is_what_the_budget_can_pay_for() -> None:
    cap = meter.clamp(
        None, body_bytes=1000, remaining_usd=Decimal("0.03"), remaining_tokens=None, prices=PRICES
    )
    assert cap is not None
    usd, _ = meter.worst_case(1000, cap, PRICES)
    assert usd <= Decimal("0.03") < meter.worst_case(1000, cap + 1, PRICES)[0]
    assert (
        meter.clamp(8000, body_bytes=10, remaining_usd=None, remaining_tokens=1000, prices=PRICES)
        == 990
    )
    assert (
        meter.clamp(None, body_bytes=900, remaining_usd=None, remaining_tokens=1000, prices=PRICES)
        is None
    )


def test_a_request_with_no_cap_is_sized_to_the_window_not_32000() -> None:
    """Fix 2026-11-20 (defect 1b), resized 2026-11-21: the estimated prompt, plus its 10% margin
    and the template's 512, and the cap fit the window; 8,192 when it states none; never above
    32,000 nor below the 256 floor."""
    assert meter.uncapped_default(32_768, prompt_tokens=10_000) == 32_768 - 11_000 - 512
    assert meter.uncapped_default(None, prompt_tokens=1_000) == 8_192 - 1_100 - 512
    assert meter.DEFAULT_WINDOW_TOKENS == 8_192 and meter.TEMPLATE_TOKENS == 512
    assert meter.ESTIMATE_MARGIN == 0.10
    for bad in (0, -1, True, "32768", 1.5):
        assert meter.uncapped_default(bad, prompt_tokens=1_000) == 8_192 - 1_100 - 512
    assert meter.uncapped_default(1_000_000, prompt_tokens=10) == meter.MAX_OUTPUT_TOKENS
    assert meter.uncapped_default(32_768, prompt_tokens=40_000) == meter.MIN_OUTPUT_TOKENS


def test_a_tight_window_gives_up_the_margin_before_the_floor_and_never_its_own_room() -> None:
    """Fix 2026-11-21: under 1,024 tokens of margined room, the cap is 1,024 while the estimate
    itself leaves that much (a thought and one action), else what the estimate leaves, else the
    budget's 256 floor: the floor is never applied past the estimate's own room."""
    assert meter.UNCAPPED_FLOOR_TOKENS == 1_024
    # Margined room 32,768 - 29,700 - 512 = 2,556: above the floor, unchanged.
    assert meter.uncapped_default(32_768, prompt_tokens=27_000) == 2_556
    # Margined room 32,768 - 31,240 - 512 < 1,024, but 32,768 - 28,400 = 4,368 leaves it.
    assert meter.uncapped_default(32_768, prompt_tokens=28_400) == 1_024
    # The estimate leaves 600: 600, not the floor.
    assert meter.uncapped_default(32_768, prompt_tokens=32_168) == 600
    assert meter.uncapped_default(32_768, prompt_tokens=32_700) == meter.MIN_OUTPUT_TOKENS


def _conversation(*contents: str) -> dict[str, object]:
    return {
        "model": "m",
        "tools": [{"type": "function", "function": {"name": "python"}}],
        "messages": [{"role": "user", "content": c} for c in contents],
    }


def test_the_first_call_is_estimated_at_the_fixed_ratio() -> None:
    """No count seen yet: 0.75 tokens per body byte (measured on run wsp_79660d1613a1: 0.24 for
    the task's prose, 0.60 for the whole conversation, 0.95 for a tensor dump)."""
    assert meter.FIRST_TOKENS_PER_BYTE == 0.75
    assert meter.PromptMeter().estimate(_conversation("hi"), 10_000) == 7_500


def test_a_grown_conversation_is_its_last_count_plus_its_new_bytes() -> None:
    """The upstream counted 18,459 tokens for 31,212 bytes (run wsp_79660d1613a1, call 10, its
    bytes from miServer's max_tokens of 1,044): the next body, the same conversation and 1,888
    bytes more, is 18,459 + 1,888 tokens, the new bytes at a token each."""
    m = meter.PromptMeter()
    first = _conversation("task", "result")
    m.observe(first, 31_212, 18_459)
    grown = _conversation("task", "result", "more", "and more")
    assert m.estimate(grown, 33_100) == 18_459 + 1_888
    # A field that changes per call (the cap, the steering set) is not a new conversation.
    assert m.estimate({**grown, "max_tokens": 5, "steering": {"features": []}}, 33_100) == 20_347


def test_a_rewritten_conversation_is_its_bytes_at_the_largest_recent_ratio() -> None:
    """A handoff rewrites the conversation: the last count no longer covers its prefix, so the
    body's bytes are taken at the largest of the last three observed ratios."""
    m = meter.PromptMeter()
    for n, (b, t) in enumerate([(1_000, 900), (10_000, 3_000), (20_000, 8_000), (30_000, 18_000)]):
        m.observe(_conversation(*["x"] * (n + 1)), b, t)
    # Ratios kept: 0.30, 0.40, 0.60 (0.90 aged out); the rewritten body is at 0.60.
    assert m.estimate(_conversation("summary"), 10_000) == 6_000
    # A changed earlier message is a rewrite too, though the body grew.
    assert m.estimate(_conversation("y", "x", "x", "x", "x"), 40_000) == 24_000
    # So is a changed tool list.
    changed = {**_conversation("x", "x", "x", "x", "x"), "tools": []}
    assert m.estimate(changed, 40_000) == 24_000


def test_the_incident_session_gets_a_reply_not_256_tokens() -> None:
    """Run wsp_79660d1613a1, replayed from miServer's log and the run's calls: calls 7-10 sent
    30,114 / 30,306 / 30,850 / 31,212 bytes (32,768 - 512 - their max_tokens of 2,142 / 1,950 /
    1,406 / 1,044) for 18,182 / 18,199 / 18,397 / 18,459 prompt tokens; call 11 sent about
    33,100 bytes for 19,177 tokens. Counting a byte as a token gave call 11 the 256 floor (its
    reply was cut at 192 tokens, 128 of them reasoning); the counts give it about 9,900, and the
    true prompt and that cap fit the window."""
    window, true_prompt = 32_768, 19_177
    bodies = [_conversation(*["m"] * n) for n in range(1, 6)]
    counted = [(30_114, 18_182), (30_306, 18_199), (30_850, 18_397), (31_212, 18_459)]
    m = meter.PromptMeter()
    for body, (b, t) in zip(bodies[:4], counted, strict=True):
        m.observe(body, b, t)
    estimate = m.estimate(bodies[4], 33_100)
    cap = meter.uncapped_default(window, prompt_tokens=estimate)
    assert cap == 32_768 - (20_347 + 2_035) - 512 == 9_874
    assert true_prompt + cap <= window
    # The rule it replaces: every byte a token.
    assert max(meter.MIN_OUTPUT_TOKENS, window - 33_100 - meter.TEMPLATE_TOKENS) == 256
    # The same body after a rewrite, at the session's largest recent ratio (0.60): about 10.4k.
    rewritten = meter.uncapped_default(window, prompt_tokens=m.estimate(_conversation("s"), 33_100))
    assert 10_000 < rewritten < 11_500 and true_prompt + rewritten <= window


def test_a_tensor_dump_added_this_turn_is_not_under_counted() -> None:
    """Run wsp_79660d1613a1's call 4 added a 4,096-float read_tensor dump: 15,197 bytes that
    tokenize at 0.84 per byte (Qwen3 tokenizer: 12,794 tokens; the prompt went 2,993 -> 15,958).
    A whole-body ratio from the turn before (about 0.25) would put the prompt near 7,000 and size
    a cap past the window; the new bytes at a token each keep it inside."""
    window, before_bytes, true_prompt = 32_768, 12_000, 15_958
    m = meter.PromptMeter()
    m.observe(_conversation("task", "early"), before_bytes, 2_993)
    grown = _conversation("task", "early", "dump")
    cap = meter.uncapped_default(window, prompt_tokens=m.estimate(grown, before_bytes + 15_197))
    assert true_prompt + cap <= window
    whole_body = (before_bytes + 15_197) * 2_993 / before_bytes
    assert true_prompt + meter.uncapped_default(window, prompt_tokens=int(whole_body)) > window


def test_nothing_is_learned_from_an_empty_count() -> None:
    m = meter.PromptMeter()
    m.observe(_conversation("a"), 1_000, 0)
    m.observe(_conversation("a"), 0, 50)
    assert m.estimate(_conversation("a"), 1_000) == 750


TRANSFORMERS_REFUSAL = (
    "This model's maximum context length is 32768 tokens. However, you requested 41000 tokens "
    "(19177 in the prompt, 21823 for the completion). Shorten the prompt or ask for fewer "
    "max_tokens."
)
LLAMACPP_REFUSAL = (
    "Requested tokens (32800) exceed context window of 32768. The model is serving a smaller "
    "context than its file declares because the larger one did not fit in VRAM"
)


def test_a_context_refusal_implies_the_cap_to_retry_with() -> None:
    """Fix 2026-11-21: miServer's transformers engine names the window and the prompt (the room
    between them); its llama.cpp engine names only what it was handed, reasoning included (half
    the cap); anything else, half the cap. Never under 256, always smaller than the cap."""
    assert meter.retry_cap(TRANSFORMERS_REFUSAL, 21_823) == 32_768 - 19_177
    assert meter.retry_cap(LLAMACPP_REFUSAL, 12_000) == 6_000
    assert meter.retry_cap("context_length_exceeded", 2_000) == 1_000
    assert meter.retry_cap(None, 2_000) == 1_000
    assert meter.retry_cap("x", 511) is None  # half is under the floor
    assert meter.retry_cap(TRANSFORMERS_REFUSAL, 13_000) is None  # not smaller than the cap
    full = TRANSFORMERS_REFUSAL.replace("19177 in the prompt", "32700 in the prompt")
    assert meter.retry_cap(full, 21_823) is None  # the prompt leaves no room


@pytest.mark.parametrize(
    ("declared", "served", "window"),
    [(16_384, 32_768, 16_384), (None, 32_768, 32_768), (None, None, 8_192), (None, 0, 8_192)],
    ids=["endpoint-first", "then-served", "then-default", "bad-served"],
)
def test_the_uncapped_window_is_the_endpoints_then_the_served_then_the_default(
    declared: int | None, served: int | None, window: int
) -> None:
    """2026-11-21: an uncapped request (the output cap off) is sized in the engine's own order
    (``context.window_tokens``), so the worker and the gateway agree on one figure."""
    from wsp_worker.miwsp_engine import context

    assert meter.window_for({"context_window_tokens": declared}, served) == window
    editor = {"context_window_tokens": declared, "served_context_tokens": served}
    assert context.window_tokens(editor) == window


def test_the_gateways_default_window_is_the_engines() -> None:
    """One stated default (F18-D12), not two that drift: the gateway's copy equals the engine's."""
    from wsp_worker.miwsp_engine import context

    assert meter.DEFAULT_WINDOW_TOKENS == context.DEFAULT_WINDOW_TOKENS
    assert meter.MIN_OUTPUT_TOKENS == context.MIN_OUTPUT_TOKENS


def test_usage_is_priced_like_weightspace_with_cache_reads_and_writes() -> None:
    m = meter.metered(
        "anthropic",
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 10,
            "output_tokens": 50,
        },
        PRICES,
    )
    assert (m.input_tokens, m.output_tokens) == (1110, 50)
    assert m.cost_usd == (
        100 * 3 + Decimal(1000) * Decimal("0.3") + Decimal(10) * Decimal("3.75") + 50 * 15
    ) / Decimal(1_000_000)
    o = meter.metered(
        "openai_compatible",
        {
            "prompt_tokens": 100,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
        PRICES,
    )
    assert (o.input_tokens, o.output_tokens) == (100, 7)
    assert o.cost_usd == (60 * 3 + Decimal(40) * Decimal("0.3") + 7 * 15) / Decimal(1_000_000)


def test_the_cap_replaces_the_other_spelling() -> None:
    body = allowlist.insert_cap(
        "openai_compatible", {"max_tokens": 9}, "max_completion_tokens", 300
    )
    assert body == {"max_completion_tokens": 300}


def test_a_meter_seeded_from_the_persisted_evidence_estimates_as_the_original() -> None:
    """2026-11-21: a restarted gateway recovers the session's last served calls (oldest first)
    from the backend; the same estimate follows. An entry without its digest restores its ratio
    only (never the grown-conversation path); a malformed one is skipped."""
    first = _conversation("task", "result")
    grown = _conversation("task", "result", "more", "and more")
    original = meter.PromptMeter()
    original.observe(first, 31_212, 18_459)
    evidence = [
        {"sent_bytes": 30_850, "prompt_tokens": 18_397, "messages": 1, "conversation_sha256": None},
        {"sent_bytes": "x", "prompt_tokens": 5},
        {
            "sent_bytes": 31_212,
            "prompt_tokens": 18_459,
            "messages": 2,
            "conversation_sha256": meter.conversation_key(first),
        },
    ]
    recovered = meter.PromptMeter()
    recovered.seed(evidence)
    assert recovered.observed and not meter.PromptMeter().observed
    assert recovered.estimate(grown, 33_100) == original.estimate(grown, 33_100) == 20_347
    ratio_only = meter.PromptMeter()
    ratio_only.seed([{"sent_bytes": 31_212, "prompt_tokens": 18_459, "messages": 2}])
    assert ratio_only.estimate(grown, 33_100) == math.ceil(33_100 * 18_459 / 31_212)
