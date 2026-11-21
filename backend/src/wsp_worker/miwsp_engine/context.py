"""The context guard (feature 018; FR-018.12; FTDD §6.4; TD-4; F18-D12).

A miServer context overflow is unrecoverable (the gateway ends the session), so the guard must
PREVENT one; it cannot react to one.

- **The window** is the endpoint's ``context_window_tokens``, else the ``served_context_tokens``
  the backend read from miServer at push, else :data:`DEFAULT_WINDOW_TOKENS` (8,192: OQ-18-2,
  answered by the operator 2026-11-18, F18-D12).
- **The estimate** before each request is the GATEWAY's rule (``wsp_gateway.meter.PromptMeter``;
  amended 2026-11-21, the paging branch, from TD-4's third of a token per byte): the editor's last
  reported ``prompt_tokens`` plus the bytes added since at :data:`NEW_TOKENS_PER_BYTE` (1.0, a
  token per byte: real WSP prompts measured 0.59-0.60 tokens per byte overall and 0.95 for a
  ``read_tensor`` dump, so the old third let the handoff fire late); after the conversation was
  rewritten (a handoff, a page event) every byte at the largest of the last :data:`RATIO_MEMORY`
  observed ratios. Then times 1.1 (:data:`FACTOR`, the gateway's margin), plus the answer the
  guard reserves. **Only before the session's first count** do the two differ: the gateway sizes
  a reply at :data:`GATEWAY_FIRST_TOKENS_PER_BYTE` (0.75; over-estimating there only shrinks a
  cap), while the guard, which refuses a task that does not fit, counts a third of a token per
  byte (:data:`GUARD_FIRST_TOKENS_PER_BYTE`, TD-4's figure, which matches the system, task and
  tool text measured at 0.24 to 0.53): at 0.75 an 8,192-token window with thinking on could not
  even send its first request.
- **The handoff:** when the estimate passes :data:`HANDOFF_SHARE` (85%) of the window, the
  conversation restarts: the system prompt and one user message carrying the task, the editor's
  notes, the edit summary and the budget state.
- **Thinking (2026-11-21):** with thinking on, each request asks for the action's cap PLUS a
  reasoning allowance (:func:`thinking_allowance`), and the guard counts both as the answer. The
  reasoning is never sent back (``client.openai_messages`` drops it), so it is not counted in the
  conversation's bytes either (:func:`message_bytes`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

#: The window when neither the endpoint nor miServer gives one (F18-D12, operator 2026-11-18).
DEFAULT_WINDOW_TOKENS = 8_192
HANDOFF_SHARE = 0.85
#: The estimate's margin (``wsp_gateway.meter.ESTIMATE_MARGIN``, 10%).
FACTOR = 1.1
#: Tokens per byte for what a conversation added since its last count: at most one token a byte
#: (``wsp_gateway.meter.NEW_TOKENS_PER_BYTE``; ``test_context.py`` pins the two equal).
NEW_TOKENS_PER_BYTE = 1.0
#: Tokens per byte before the session's first count, for the guard only (see the module doc).
GUARD_FIRST_TOKENS_PER_BYTE = 1 / 3
#: How many of the last observed ratios a rewritten conversation's estimate takes the largest of
#: (``wsp_gateway.meter.RATIO_MEMORY``).
RATIO_MEMORY = 3
#: The gateway refuses a request that can afford fewer output tokens than this (meter.py:22).
MIN_OUTPUT_TOKENS = 256


def window_tokens(editor: Mapping[str, Any]) -> int:
    """The editor's context window: the endpoint's, else miServer's served one, else 8,192."""
    for key in ("context_window_tokens", "served_context_tokens"):
        value = editor.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return DEFAULT_WINDOW_TOKENS


#: The share of the window one answer may take (so the prompt keeps the rest).
CAP_SHARE = 8
#: The engine's default output cap (FTDD §6.7), before :func:`output_cap` bounds it by the window.
DEFAULT_OUTPUT_CAP = 2_048


def output_cap(window: int, default: int) -> int:
    """The engine's output cap: its default (2,048), but at most an eighth of the window, and
    never below the gateway's floor of 256. At the default window that is 1,024 tokens: one
    action per turn needs no more, and the prompt keeps room before the 85% handoff."""
    return max(MIN_OUTPUT_TOKENS, min(default, window // CAP_SHARE))


def editor_output_cap(editor: Mapping[str, Any]) -> int:
    """The per-turn output cap for ``editor`` (the pushed ``spec.editor``): its window
    (:func:`window_tokens`) through :func:`output_cap` at :data:`DEFAULT_OUTPUT_CAP`.

    The ONE rule both providers send (fix 2026-11-20): the engine's session and the
    ``weightspace`` adapter call this, so a bake-off compares the two at the same cap, and
    neither leaves the cap to the gateway (whose fallback once asked a 32,768-token miServer
    window for 32,000 output tokens, and miServer ended the session ``context_length_exceeded``).
    """
    return output_cap(window_tokens(editor), DEFAULT_OUTPUT_CAP)


#: The engine's reasoning allowance per turn when the request names none (2026-11-21); the
#: backend freezes the same figure (``wsp.endpoints.DEFAULT_THINKING_TOKENS``, a test pins them).
DEFAULT_THINKING_TOKENS = 4_096
#: The answer (the reasoning allowance plus the action's cap) takes at most this share of the
#: window, so the prompt keeps the other half before the 85% handoff.
THINKING_WINDOW_DIVISOR = 2


#: The gateway's sizing of a request that names no cap (``wsp_gateway.meter``, fix 2026-11-21),
#: mirrored here so the engine knows the room an uncapped turn will get (the worker does not import
#: the gateway; ``test_engine_room.py`` pins each figure, and the rule case by case, to the
#: gateway's). Its estimate of the prompt: the last count plus the bytes added since at a token
#: each, or, before any count, every byte at :data:`GATEWAY_FIRST_TOKENS_PER_BYTE`.
GATEWAY_FIRST_TOKENS_PER_BYTE = 0.75
GATEWAY_MARGIN_PERCENT = 10
GATEWAY_TEMPLATE_TOKENS = 512
GATEWAY_UNCAPPED_FLOOR_TOKENS = 1_024
GATEWAY_MAX_OUTPUT_TOKENS = 32_000
#: The gateway's share of an uncapped cap a reasoning allowance may take (``allowlist``'s
#: ``UNCAPPED_REASONING_DIVISOR``), and what it leaves the action (``ACTION_FLOOR_TOKENS``).
GATEWAY_REASONING_DIVISOR = 2
#: The least reasoning share worth asking for (fix 2026-11-21, run ``wsp_855a74f820ff``: a
#: 256-token cap left a reasoning allowance of 1 and every action was cut mid-JSON). Below it the
#: turn is sent with thinking off, and the transcript says so.
MIN_REASONING_TOKENS = MIN_OUTPUT_TOKENS
#: The least an action needs: a turn whose room cannot hold it is not sent.
MIN_ACTION_TOKENS = MIN_OUTPUT_TOKENS


def gateway_cap(window: int, prompt_tokens: int) -> int:
    """The cap the gateway sizes for an uncapped request (``meter.uncapped_default``)."""
    margined = -(-prompt_tokens * (100 + GATEWAY_MARGIN_PERCENT) // 100) + GATEWAY_TEMPLATE_TOKENS
    room = window - margined
    if room < GATEWAY_UNCAPPED_FLOOR_TOKENS:
        room = min(GATEWAY_UNCAPPED_FLOOR_TOKENS, window - prompt_tokens)
    return max(MIN_OUTPUT_TOKENS, min(GATEWAY_MAX_OUTPUT_TOKENS, room))


def reasoning_share(allowance: int, cap: int) -> int:
    """What the gateway leaves of ``allowance`` beside an uncapped ``cap``
    (``allowlist.insert_cap``): at most half the cap, and the action's floor below it."""
    return max(min(allowance, max(cap // GATEWAY_REASONING_DIVISOR, 1), cap - MIN_ACTION_TOKENS), 1)


class ThinkingDoesNotFitError(ValueError):
    """``editor_context_window_too_small``: the window leaves no room to think."""

    code = "editor_context_window_too_small"


def capped(request: Mapping[str, Any]) -> bool:
    """The frozen request's ``output_cap`` (2026-11-21): absent or ``on`` names the per-turn
    cap; ``off`` names none. Both providers read it."""
    return bool(request.get("output_cap", "on") != "off")


def thinking_allowance(window: int, cap: int, requested: int | None) -> int:
    """The reasoning tokens a thinking turn may use: the request's allowance (else
    :data:`DEFAULT_THINKING_TOKENS`), at most half the window less the action's cap. Below
    :data:`MIN_OUTPUT_TOKENS` the window cannot hold a thought and the action together, which is
    refused by name rather than sent as thinking that eats the action's tokens."""
    asked = DEFAULT_THINKING_TOKENS if requested is None else int(requested)
    allowance = min(asked, window // THINKING_WINDOW_DIVISOR - cap)
    if allowance < MIN_OUTPUT_TOKENS:
        raise ThinkingDoesNotFitError(
            f"thinking needs at least {MIN_OUTPUT_TOKENS} reasoning tokens beside a {cap}-token "
            f"action in half of a {window}-token window"
        )
    return allowance


#: Neutral message keys that are recorded but never sent back to the editor.
NOT_SENT = frozenset({"reasoning", "reasoning_tokens"})


def message_bytes(messages: Sequence[Mapping[str, Any]]) -> int:
    """The bytes the messages add to a request: a message's ``reasoning`` is recorded in the
    transcript but never sent back (:data:`NOT_SENT`), so it does not count."""
    return sum(
        len(
            json.dumps({k: v for k, v in m.items() if k not in NOT_SENT}, default=str).encode(
                "utf-8"
            )
        )
        for m in messages
    )


def byte_estimate(messages: Sequence[Mapping[str, Any]]) -> int:
    """The messages' size in tokens from their bytes alone, at :data:`NEW_TOKENS_PER_BYTE` (the
    worst case, never an under-count): a deterministic function of the history, so a replay,
    which has no usage, computes the same value (paging's batching reads it,
    ``paging.Pager.due``)."""
    return math.ceil(message_bytes(messages) * NEW_TOKENS_PER_BYTE)


class Guard:
    """Tracks the last reported prompt size, and says when the next request must hand off."""

    def __init__(self, window: int, cap: int) -> None:
        self.window, self.cap = window, cap
        self.last_prompt_tokens: int | None = None
        self.counted_messages = 0
        #: The last :data:`RATIO_MEMORY` observed ``prompt_tokens / bytes sent``.
        self.ratios: list[float] = []

    def reported(self, prompt_tokens: int, messages_sent: int, sent_bytes: int = 0) -> None:
        """After a reply: the endpoint's own count of the prompt it was just sent (``sent_bytes``:
        the messages' and tool fields' bytes, for the ratio a rewritten conversation uses)."""
        self.last_prompt_tokens = prompt_tokens
        self.counted_messages = messages_sent
        if sent_bytes > 0 and prompt_tokens > 0:
            self.ratios = [*self.ratios, prompt_tokens / sent_bytes][-RATIO_MEMORY:]

    def reset(self) -> None:
        """The conversation was rewritten (a handoff, a page event): the last count no longer
        describes it. The observed ratios are kept."""
        self.last_prompt_tokens = None
        self.counted_messages = 0

    def _prompt(self, messages: Sequence[Mapping[str, Any]], tools_bytes: int, first: float) -> int:
        if self.last_prompt_tokens is None:
            every = message_bytes(messages) + tools_bytes
            # A ratio above a token a byte is not a tokenizer's (a chat template's own tokens on a
            # tiny prompt): kept at most NEW_TOKENS_PER_BYTE, the most new bytes can cost.
            ratio = min(max(self.ratios), NEW_TOKENS_PER_BYTE) if self.ratios else first
            return math.ceil(every * ratio)
        grown = message_bytes(messages[self.counted_messages :])
        return self.last_prompt_tokens + math.ceil(grown * NEW_TOKENS_PER_BYTE)

    def prompt_estimate(
        self, messages: Sequence[Mapping[str, Any]], *, tools_bytes: int = 0
    ) -> int:
        """The prompt's tokens by the gateway's rule (see the module doc), before the margin."""
        return self._prompt(messages, tools_bytes, GUARD_FIRST_TOKENS_PER_BYTE)

    def estimate(self, messages: Sequence[Mapping[str, Any]], *, tools_bytes: int = 0) -> int:
        """The prompt with the margin, plus the answer the guard reserves."""
        return math.ceil(self.prompt_estimate(messages, tools_bytes=tools_bytes) * FACTOR) + (
            self.cap
        )

    def gateway_prompt(self, messages: Sequence[Mapping[str, Any]], *, tools_bytes: int = 0) -> int:
        """The prompt as the gateway estimates it (fix 2026-11-21): the guard's rule, with the
        gateway's own figure before the first count (see the module doc)."""
        return self._prompt(messages, tools_bytes, GATEWAY_FIRST_TOKENS_PER_BYTE)

    def must_hand_off(self, messages: Sequence[Mapping[str, Any]], *, tools_bytes: int = 0) -> bool:
        return self.estimate(messages, tools_bytes=tools_bytes) > HANDOFF_SHARE * self.window
