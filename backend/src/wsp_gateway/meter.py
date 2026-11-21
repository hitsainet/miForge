"""Metering: worst cases, the output cap, prices, and the backend's reserve and settle (FTID §3.4).

``worst_case`` and ``clamp`` are pure. ``Ledger.reserve`` and ``Ledger.settle`` POST to the internal
application and raise on any non-2xx; the caller then refuses, or withholds the response (FTDD §5.5:
the gateway fails closed). Usage is normalised as weightspace's ``Usage.from_anthropic`` /
``from_openai`` and priced as ``Pricing.cost`` ([ws] ``usage.py#L33-87``): cache reads and writes
are counted, at their own prices when the endpoint has them, else at the input price.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import threading
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypeGuard

import httpx

ZERO = Decimal("0")
MILLION = Decimal(1_000_000)
#: Fewer affordable output tokens than this and the request is refused (PF-15).
MIN_OUTPUT_TOKENS = 256
#: The cap when nothing else bounds it (no price, no token budget).
MAX_OUTPUT_TOKENS = 32_000
#: The window assumed for a request that names no cap, when the endpoint states none: the
#: engine's default window (``miwsp_engine.context.DEFAULT_WINDOW_TOKENS``, F18-D12, operator
#: 2026-11-18). The gateway never reads miServer's served window (the backend does, at push).
DEFAULT_WINDOW_TOKENS = 8_192
#: Tokens a chat template adds that the request body does not carry (a tools preamble, the
#: role markers), on top of the prompt's estimate.
TEMPLATE_TOKENS = 512
#: An uncapped request's cap is kept at least this large while the prompt's evidence-based
#: count (its margin given up) still leaves the room (fix 2026-11-21, run ``wsp_79660d1613a1``):
#: a thinking turn needs room for its reasoning AND one action, and miServer gives the reasoning
#: half of ``max_tokens`` when no allowance is sent. The budget's floor stays
#: :data:`MIN_OUTPUT_TOKENS`.
UNCAPPED_FLOOR_TOKENS = 1_024
#: The least reasoning allowance worth sending (fix 2026-11-21): when the cap clamps a sent
#: ``max_reasoning_tokens`` under it, the gateway sends the turn with thinking off instead
#: (``app.THINKING_SKIPPED_HEADER``). Pinned equal to ``miwsp_engine.context.MIN_REASONING_TOKENS``.
MIN_REASONING_TOKENS = 256
#: The prompt's tokens per body byte on a session's FIRST call, before any upstream count is
#: seen (fix 2026-11-21). Measured on run ``wsp_79660d1613a1`` (Qwen3 tokenizer): the system and
#: task messages 0.24, a tool's JSON 0.33-0.53, the whole conversation 0.60 (miServer's own
#: counts agree: 18,182 tokens for a 30,114-byte body), and a tensor dump 0.95 (one token per
#: digit). 0.75 is above every whole-body figure seen, and under the 1.0 that shrank the cap.
FIRST_TOKENS_PER_BYTE = 0.75
#: Tokens per byte for the bytes a conversation added since its last counted call: at most one
#: token per byte (a byte-level BPE token is at least one byte, and a JSON escape is two bytes
#: for one character), which a tensor dump nearly reaches. So new content is never under-counted.
NEW_TOKENS_PER_BYTE = 1.0
#: The estimate's margin: the prompt is taken as ``(1 + margin)`` times the evidence, plus
#: :data:`TEMPLATE_TOKENS`.
ESTIMATE_MARGIN = 0.10
#: How many of a session's last observed ratios the fallback estimate takes the largest of.
RATIO_MEMORY = 3
#: Request fields that change from call to call without changing the conversation.
_VOLATILE = frozenset({"max_tokens", "max_completion_tokens", "max_reasoning_tokens", "steering"})


@dataclass(frozen=True)
class Prices:
    """USD per million tokens; an absent price is 0 (a local endpoint)."""

    input: Decimal = ZERO
    output: Decimal = ZERO
    cache_write: Decimal | None = None
    cache_read: Decimal | None = None

    @classmethod
    def of(cls, endpoint: dict[str, Any]) -> Prices:
        def d(name: str) -> Decimal | None:
            value = endpoint.get(name)
            return None if value is None else Decimal(str(value))

        return cls(
            input=d("price_in_per_mtok") or ZERO,
            output=d("price_out_per_mtok") or ZERO,
            cache_write=d("price_cache_write_per_mtok"),
            cache_read=d("price_cache_read_per_mtok"),
        )

    @property
    def worst_input(self) -> Decimal:
        """Every input token could be a cache write (R1B-A13)."""
        return max(self.input, self.cache_write if self.cache_write is not None else ZERO)


def worst_case(body_bytes: int, cap: int, prices: Prices) -> tuple[Decimal, int]:
    """``(usd, tokens)``: a byte is at least a token, plus the whole output cap."""
    usd = (
        Decimal(body_bytes) * prices.worst_input / MILLION + Decimal(cap) * prices.output / MILLION
    )
    return usd, body_bytes + cap


def window_for(endpoint: dict[str, Any], served: Any) -> int:
    """The window an uncapped request is sized to (2026-11-21), in the engine's own order
    (``miwsp_engine.context.window_tokens``): the endpoint's ``context_window_tokens``, then the
    window miServer served at push (the session's ``served_context_tokens``), then
    :data:`DEFAULT_WINDOW_TOKENS`. So the worker and the gateway agree on one figure."""
    for value in (endpoint.get("context_window_tokens"), served):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return DEFAULT_WINDOW_TOKENS


def conversation_key(body: dict[str, Any], messages: int | None = None) -> str:
    """A digest of the conversation a body carries: every field but the per-call ones
    (:data:`_VOLATILE`), and its first ``messages`` messages (all of them when ``None``). Two
    calls whose keys agree on the earlier call's messages are one conversation that grew."""
    rest = {k: v for k, v in body.items() if k not in _VOLATILE}
    found = body.get("messages")
    listed = list(found) if isinstance(found, list) else []
    if messages is not None:
        listed = listed[:messages]
    rest.pop("messages", None)
    text = json.dumps([rest, listed], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _count(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _message_count(body: dict[str, Any]) -> int:
    found = body.get("messages")
    return len(found) if isinstance(found, list) else 0


@dataclass(frozen=True)
class Observation:
    """One served call's evidence: the bytes sent, the prompt tokens the upstream counted, and
    the conversation they belong to."""

    sent_bytes: int
    prompt_tokens: int
    messages: int
    key: str


class PromptMeter:
    """A session's estimate of its next prompt's tokens, from the upstream's own counts (fix
    2026-11-21, run ``wsp_79660d1613a1``: counting every body byte as a token shrank an uncapped
    reply to 256 tokens as the session grew; 19,177 real tokens were about 32,000 bytes).

    In order:

    - **The conversation grew** (its earlier messages and every other field are unchanged): the
      last counted prompt, plus the bytes added since at :data:`NEW_TOKENS_PER_BYTE`. Only the new
      content is a worst case, so a tensor dump added this turn is never under-counted (a
      whole-body ratio from the turn before would put it at a third of its tokens).
    - **Otherwise** (the conversation was rewritten, a handoff): the body's bytes at the largest
      of the last :data:`RATIO_MEMORY` observed ratios.
    - **No count yet** (the first call, or a gateway restarted mid-session): the body's bytes at
      :data:`FIRST_TOKENS_PER_BYTE`.

    Only the window's room uses this estimate. The cost and token reservation
    (:func:`worst_case`, :func:`clamp`) keep a byte as at least a token: those stay upper bounds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ratios: deque[float] = deque(maxlen=RATIO_MEMORY)
        self._last: Observation | None = None

    def estimate(self, body: dict[str, Any], body_bytes: int) -> int:
        """The prompt's tokens as the evidence says, before the margin and the template."""
        with self._lock:
            last, ratios = self._last, list(self._ratios)
        if (
            last is not None
            and body_bytes >= last.sent_bytes
            and last.messages >= 0
            and _message_count(body) >= last.messages
            and conversation_key(body, last.messages) == last.key
        ):
            grown = body_bytes - last.sent_bytes
            return last.prompt_tokens + math.ceil(grown * NEW_TOKENS_PER_BYTE)
        ratio = max(ratios) if ratios else FIRST_TOKENS_PER_BYTE
        return math.ceil(body_bytes * ratio)

    def observe(
        self, body: dict[str, Any], sent_bytes: int, prompt_tokens: int
    ) -> Observation | None:
        """Record what the upstream counted for a body of ``sent_bytes``; the observation, for
        the call's settle to persist (migration 0033), or ``None`` when nothing was learned."""
        if sent_bytes <= 0 or prompt_tokens <= 0:
            return None
        seen = Observation(sent_bytes, prompt_tokens, _message_count(body), conversation_key(body))
        self._record(seen)
        return seen

    def _record(self, seen: Observation) -> None:
        with self._lock:
            self._ratios.append(seen.prompt_tokens / seen.sent_bytes)
            self._last = seen

    def seed(self, evidence: Any) -> None:
        """Recover the evidence a session's served calls persisted (the backend's session entry,
        ``prompt_evidence``, oldest first), so a restarted gateway estimates as before it
        restarted (fix 2026-11-21). Malformed entries are skipped: they only cost precision."""
        for item in evidence if isinstance(evidence, list) else []:
            if not isinstance(item, dict):
                continue
            sent, tokens = item.get("sent_bytes"), item.get("prompt_tokens")
            messages, key = item.get("messages"), item.get("conversation_sha256")
            if not (_count(sent) and _count(tokens)) or sent <= 0 or tokens <= 0:
                continue
            if _count(messages) and isinstance(key, str) and len(key) == 64:
                self._record(Observation(sent, tokens, messages, key))
            else:
                # Its ratio only: messages -1 never matches the grown-conversation path.
                self._record(Observation(sent, tokens, -1, ""))

    @property
    def observed(self) -> bool:
        with self._lock:
            return bool(self._ratios)


def margined(prompt_tokens: int) -> int:
    """The prompt the window must hold: the estimate times ``1 +`` :data:`ESTIMATE_MARGIN`, plus
    :data:`TEMPLATE_TOKENS`."""
    return math.ceil(Decimal(prompt_tokens) * (1 + Decimal(str(ESTIMATE_MARGIN)))) + TEMPLATE_TOKENS


def uncapped_default(window: Any, *, prompt_tokens: int) -> int:
    """The cap for a request that names none (fix 2026-11-20; sized from evidence 2026-11-21).

    ``prompt_tokens`` is the session's estimate (:meth:`PromptMeter.estimate`). The cap is what
    the window leaves after :func:`margined` (the estimate plus 10% plus the template's 512):

    - ``window`` is the endpoint's ``context_window_tokens``, else the window miServer served at
      push, else :data:`DEFAULT_WINDOW_TOKENS` (:func:`window_for`);
    - never above :data:`MAX_OUTPUT_TOKENS`;
    - when that room is under :data:`UNCAPPED_FLOOR_TOKENS`, the margin is given up before the
      floor is: the cap is the floor if the estimate itself leaves room for it, else what the
      estimate leaves. Never past the estimate's own room, except
    - never below :data:`MIN_OUTPUT_TOKENS`: a prompt that leaves less is the endpoint's to
      refuse (``editor_context_exceeded`` is budget-like: the run keeps its result).

    The result is then the REQUESTED cap :func:`clamp` lowers to what the budget can pay for, so
    the budget's rules (and ``wsp_budget_exhausted``) are unchanged. A request that names its own
    cap is not changed here.
    """
    known = isinstance(window, int) and not isinstance(window, bool) and window > 0
    size = int(window) if known else DEFAULT_WINDOW_TOKENS
    room = size - margined(prompt_tokens)
    if room < UNCAPPED_FLOOR_TOKENS:
        room = min(UNCAPPED_FLOOR_TOKENS, size - prompt_tokens)
    return max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS, room))


#: miServer's context refusals (``ContextLengthExceededError``): the transformers engine names
#: the window and the prompt; llama.cpp names the tokens it was handed and its window.
_TRANSFORMERS_CONTEXT = re.compile(
    r"maximum context length is (\d+) tokens.*?\((\d+) in the (?:prompt|input)", re.DOTALL
)
_LLAMACPP_CONTEXT = re.compile(r"Requested tokens \((\d+)\) exceeds? context window of (\d+)")


def retry_cap(message: Any, cap: int) -> int | None:
    """The cap to send ONCE more after miServer refused ``cap`` as past its context (fix
    2026-11-21), or ``None`` when no smaller cap fits.

    - The transformers form names the window and the prompt: the room between them.
    - The llama.cpp form names only the window and the tokens it was handed, which include any
      reasoning already generated (miServer's GGUF path clamps ``max_tokens`` to the window and
      refuses only once the prompt and the text so far fill it, so an over-sized cap's reasoning
      half can fill it): half the cap, which halves the reasoning share.
    - Any other message: half the cap.

    Never under :data:`MIN_OUTPUT_TOKENS`, and always smaller than ``cap``."""
    text = message if isinstance(message, str) else ""
    found = _TRANSFORMERS_CONTEXT.search(text)
    if found:
        smaller = min(int(found.group(1)) - int(found.group(2)), MAX_OUTPUT_TOKENS)
    else:
        smaller = cap // 2
    if smaller < MIN_OUTPUT_TOKENS or smaller >= cap:
        return None
    return smaller


def clamp(
    requested: int | None,
    *,
    body_bytes: int,
    remaining_usd: Decimal | None,
    remaining_tokens: int | None,
    prices: Prices,
) -> int | None:
    """The output cap: the request's own, lowered to what the remaining budget can pay for.

    ``None`` when fewer than :data:`MIN_OUTPUT_TOKENS` remain affordable (``wsp_budget_exhausted``).
    """
    cap = MAX_OUTPUT_TOKENS if requested is None else min(int(requested), MAX_OUTPUT_TOKENS)
    if remaining_tokens is not None:
        cap = min(cap, remaining_tokens - body_bytes)
    if remaining_usd is not None and prices.output > ZERO:
        left = remaining_usd - Decimal(body_bytes) * prices.worst_input / MILLION
        cap = min(cap, int(left * MILLION / prices.output))
    elif (
        remaining_usd is not None
        and remaining_usd - Decimal(body_bytes) * prices.worst_input / MILLION < ZERO
    ):
        return None
    return cap if cap >= MIN_OUTPUT_TOKENS else None


@dataclass(frozen=True)
class Metered:
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal

    def as_usage(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": format(self.cost_usd, "f"),
        }


#: Transport errors raised before the request left the gateway (fix 2026-11-21): nothing was
#: sent, so nothing was generated or billed, and the call settles at zero. Every other transport
#: error (a read timeout, a dropped connection, a write that may have reached the server) leaves
#: the outcome unknown and settles at the reservation's worst case.
NOT_SENT: tuple[type[httpx.HTTPError], ...] = (httpx.ConnectError, httpx.ConnectTimeout)


def refused_unserved(status: int, usage: Any) -> bool:
    """A 4xx answer with no ``usage`` object (fix 2026-11-21): the upstream refused the request
    and generated nothing, so it is settled at zero. With a ``usage`` object, the usage is the
    settlement (:func:`metered`); a 5xx, or a 2xx without usage, may have been billed and keeps
    the worst case."""
    return 400 <= status < 500 and usage is None


def metered(kind: str, usage: Any, prices: Prices) -> Metered:
    """The upstream ``usage`` object, normalised and priced; an absent one meters as zero."""
    u = usage if isinstance(usage, dict) else {}

    def n(value: Any) -> int:
        # A negative count would refund spend (review OR1-016-8): clamped at zero. A non-finite
        # one cannot be priced: ValueError, and the caller settles at the worst case.
        if isinstance(value, int | float) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("a non-finite usage count")
            return max(int(value), 0)
        return 0

    if kind == "anthropic":
        read, write = n(u.get("cache_read_input_tokens")), n(u.get("cache_creation_input_tokens"))
        fresh = n(u.get("input_tokens"))
        out = n(u.get("output_tokens"))
        total_in = fresh + read + write
    else:
        details = (
            u.get("prompt_tokens_details")
            if isinstance(u.get("prompt_tokens_details"), dict)
            else {}
        )
        total_in = n(u.get("prompt_tokens"))
        read, write = n((details or {}).get("cached_tokens")), 0
        fresh = max(total_in - read, 0)
        out = n(u.get("completion_tokens"))
    cached = prices.input if prices.cache_read is None else prices.cache_read
    writes = prices.input if prices.cache_write is None else prices.cache_write
    cost = (
        Decimal(fresh) * prices.input
        + Decimal(read) * cached
        + Decimal(write) * writes
        + Decimal(out) * prices.output
    ) / MILLION
    return Metered(total_in, out, cost)


#: A call that generated nothing and cost nothing (fix 2026-11-21).
NOTHING = Metered(0, 0, Decimal(0))


class LedgerRefused(Exception):  # noqa: N818 - mirrors core.errors.Refusal
    """The backend refused (a named code), or could not be reached (``code`` is ``None``)."""

    def __init__(self, code: str | None, message: str, backend_message: str | None = None) -> None:
        self.code = code
        #: The backend's own ``error.message`` (no secret: the ledger's refusals name a budget).
        self.message = backend_message
        super().__init__(message)


class Ledger:
    """The backend's reserve and settle, over the internal application (FTDD §5.4)."""

    def __init__(self, internal: httpx.Client) -> None:
        self._internal = internal

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._internal.post(path, json=body)
        except httpx.HTTPError as exc:
            raise LedgerRefused(None, f"the backend did not answer {path}") from exc
        if response.status_code // 100 != 2:
            code = message = None
            with contextlib.suppress(ValueError, KeyError, TypeError):
                code = response.json()["error"]["code"]
            with contextlib.suppress(ValueError, KeyError, TypeError):
                found = response.json()["error"]["message"]
                message = found[:500] if isinstance(found, str) else None
            raise LedgerRefused(
                code, f"the backend refused {path} ({response.status_code})", message
            )
        data: dict[str, Any] = response.json()
        return data

    def reserve(
        self,
        *,
        run_id: str,
        token_sha256: str,
        worst_case_usd: Decimal,
        worst_case_tokens: int,
        external: bool,
    ) -> int:
        data = self._post(
            "/wsp-internal/v1/gateway/reserve",
            {
                "run_id": run_id,
                "token_sha256": token_sha256,
                "worst_case_usd": format(worst_case_usd, "f"),
                "worst_case_tokens": worst_case_tokens,
                "external": external,
            },
        )
        return int(data["reservation_id"])

    def settle(
        self, reservation_id: int, usage: Metered, call: dict[str, Any] | None = None
    ) -> None:
        """Settle ``usage``; ``call`` is what became of the call (counts and codes only: outcome,
        upstream status, latency, finish reason, refusal code), for the run's call timeline."""
        body: dict[str, Any] = {"reservation_id": reservation_id, "usage": usage.as_usage()}
        if call is not None:
            body["call"] = call
        self._post("/wsp-internal/v1/gateway/settle", body)

    def refused(self, *, run_id: str, token_sha256: str, code: str) -> None:
        """Record a call refused before any spend: its code, never the request (live progress,
        2026-11-20)."""
        self._post(
            "/wsp-internal/v1/gateway/refused",
            {"run_id": run_id, "token_sha256": token_sha256, "code": code},
        )


def finish_reason(kind: str, data: Any) -> str | None:
    """Why the editor stopped, as its answer says: Anthropic's ``stop_reason``, OpenAI's first
    ``choices[].finish_reason``. Only the field is read, never a message; the backend keeps it
    only when it is a short identifier (``gateway_ledger.call_columns``)."""
    if not isinstance(data, dict):
        return None
    if kind == "anthropic":
        found = data.get("stop_reason")
    else:
        choices = data.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        found = first.get("finish_reason") if isinstance(first, dict) else None
    return found if isinstance(found, str) else None
