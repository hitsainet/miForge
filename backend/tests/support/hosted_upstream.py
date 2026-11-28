"""Scripted hosted upstreams that enforce the Messages API's rules (feature 020; FR-020.23;
TDD §6.7; FTID §3.7). Test fixtures, permanent (F20-D11): the rehearsal runs on every push.

:class:`HostedUpstream` is an ``httpx`` transport, like ``FakeUpstream``, standing where
``api.anthropic.com`` (personality ``anthropic``) or a hosted OpenAI-compatible server
(``openai``) stands, behind the REAL gateway. Each rule is one function citing what it imitates:
FTID §0 (the live documentation read on 2026-11-25) or the bundled reference.

**Conversations.** The API checks each thinking block against its own record, never against the
"last request" (FTID §0, L3b). This fixture does the same (:func:`signatures_ok`) and, to check the
append-only rule on every request as the rehearsal requires (FR-020.23), it also links each request
to the conversation that issued the first marker it carries (a thinking signature, or for OpenAI
a tool call id) and applies ``miwsp_engine.prefix.check`` against that conversation's previous
request (:func:`prefix_ok`). The rule is the engine's own module, so the two cannot drift (TD-3).

**The cache** (:func:`cache_reads`): an entry is written at each breakpoint a request carries (the
system block's marker covers tools and system; the top-level marker covers the whole request,
FTDD §3 "prompt-caching.md lines 164 and 176"), and a request reads the longest written prefix it
starts with, at the positions its own markers allow: the system boundary, and, with the top-level
marker, each message boundary (the automatic breakpoint's lookback). Tokens are ``ceil(bytes / 4)``
of the canonical normal form. A request with no marker reads and writes nothing.

A step (``scripts``, consumed one per request in order): ``{"calls": [(name, args)], "thinking":
str}``; ``{"refusal": category | None}``; ``{"drop": True}`` (the connection dies, a lost reply);
``{"delay_s": float}`` (alongside a reply); ``{"output_tokens": n}``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from wsp_worker.miwsp_engine import prefix

ANTHROPIC = "anthropic"
OPENAI = "openai"
VERSION = "2026-10-21"
#: FTID §0, L3: the API's answer to a thinking block bound to another conversation, without the
#: beta header (whose sentence it then adds).
BOUND_TEXT = (
    "{path}: Invalid `signature` in `thinking` block. The block is bound to a different "
    "conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to "
    '"drop_block". That setting requires the `thinking-binding-controls-2026-08-01` value in the '
    "`anthropic-beta` header."
)
SAMPLING = ("temperature", "top_p", "top_k")
EFFORTS = ("low", "medium", "high")


class Rejected(Exception):  # noqa: N818 - an answer, not a failure of the fixture
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message, self.status = message, status


def tokens(value: Any) -> int:
    return math.ceil(len(prefix.canonical(value).encode("utf-8")) / 4)


# ----------------------------------------------------------------------------- the rules


def version_ok(headers: dict[str, str]) -> None:
    """``anthropic-version: 2026-10-21`` (FR-020.19; the bundled ``api-version-migration.md``)."""
    if headers.get("anthropic-version") != VERSION:
        raise Rejected(f"anthropic-version must be {VERSION}")


def no_beta(headers: dict[str, str]) -> None:
    """No ``anthropic-beta`` (F20-D9: 020 uses no beta feature)."""
    if "anthropic-beta" in headers:
        raise Rejected("anthropic-beta is not expected")


def no_sampling(body: dict[str, Any]) -> None:
    """Version 2026-10-21 refuses ``temperature``, ``top_p`` and ``top_k`` on every model, even at
    their defaults (bundled ``api-version-migration.md``, line 99)."""
    for name in SAMPLING:
        if name in body:
            raise Rejected(f"{name}: sampling parameters are not supported on {VERSION}")


def no_forced_tool_choice(body: dict[str, Any]) -> None:
    """``tool_choice`` ``any`` and ``tool`` return a 400 (bundled ``tool-use-concepts.md``, 106)."""
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") not in ("auto", "none"):
        raise Rejected("tool_choice: forced tool use is not supported with thinking")


def signature_for(body: dict[str, Any], blocks_before: list[dict[str, Any]]) -> str:
    """The digest part of a signature (see :func:`signatures_ok`)."""
    return _digest(body, blocks_before)


def _digest(body: dict[str, Any], blocks_before: list[dict[str, Any]]) -> str:
    """What a thinking block's signature records: the system, the tools and every message
    before it, and the blocks before it in its own message (FTID §0, L3b; the bundled
    ``preserved-thinking-migration.md`` line 7). Formatting, key order and ``cache_control``
    are not part of it (``prefix.normal``)."""
    normal = prefix.normal(body)
    record = {
        **normal,
        "messages": [*normal["messages"], {"role": "assistant", "content": blocks_before}],
    }
    return "sig_" + hashlib.sha256(prefix.canonical(record).encode()).hexdigest()


def signatures_ok(body: dict[str, Any]) -> list[str]:
    """Every thinking block's signature still matches what precedes it here; the signatures it
    carries, in order. A mismatch is L3's 400 with the block's path."""
    seen: list[str] = []
    messages = body.get("messages") or []
    for i, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, list):
            continue
        before = {**body, "messages": messages[:i]}
        for j, block in enumerate(content):
            if block.get("type") != "thinking":
                continue
            expected = _digest(before, [prefix._strip(b) for b in content[:j]])
            # ``sig_<digest>_<serial>``: the serial makes each reply's signature its own, as the
            # API's are (two conversations that open alike never share one); the digest is what
            # is checked.
            signature = str(block.get("signature", ""))
            if signature.rsplit("_", 1)[0] != expected:
                raise Rejected(BOUND_TEXT.format(path=f"messages.{i}.content.{j}"))
            seen.append(str(block["signature"]))
    return seen


def prefix_ok(previous: dict[str, Any] | None, body: dict[str, Any]) -> None:
    """The append-only rule against this conversation's previous request (F20-D15), by the
    engine's own definition (TD-3); a change is a 400 naming the first changed path."""
    changed = prefix.check(previous, body)
    if changed is not None:
        raise Rejected(f"{changed}: the request edits the conversation it continues")


def openai_effort_ok(body: dict[str, Any]) -> None:
    """``reasoning_effort`` when set is one OpenAI documents (FTID §0, L5); 020 sends at most
    ``high`` (W3-D9)."""
    if "reasoning_effort" in body and body["reasoning_effort"] not in EFFORTS:
        raise Rejected("reasoning_effort: unsupported value")


@dataclass
class Entry:
    digest: str
    tokens: int
    written_at: float


def _segments(body: dict[str, Any]) -> tuple[Any, list[Any]]:
    normal = prefix.normal(body)
    return {"system": normal["system"], "tools": normal["tools"]}, normal["messages"]


def cache_reads(
    entries: dict[str, Entry], body: dict[str, Any], now: float, ttl_s: float = 300.0
) -> tuple[int, int, int]:
    """``(read, written, total)`` tokens for ``body``, updating ``entries`` (module doc). The
    5-minute lifetime is measured from the start of the request that wrote or read an entry
    (bundled ``prompt-caching.md`` line 148)."""
    static, messages = _segments(body)
    total = tokens([static, messages])
    system = body.get("system")
    system_marked = isinstance(system, list) and any(
        isinstance(b, dict) and "cache_control" in b for b in system
    )
    automatic = "cache_control" in body
    positions: list[tuple[str, int]] = []
    if system_marked:
        positions.append((prefix.canonical([static, []]), tokens([static, []])))
    if automatic:
        for k in range(len(messages) + 1):
            positions.append(
                (prefix.canonical([static, messages[:k]]), tokens([static, messages[:k]]))
            )
    for key in [k for k, e in entries.items() if now - e.written_at > ttl_s]:
        del entries[key]
    read = 0
    for text, count in positions:
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in entries and count > read:
            read = count
            entries[digest].written_at = now  # a read refreshes the entry
    # Entries are written at the request's own breakpoints: the system boundary and the end.
    writes: list[tuple[str, int]] = []
    if system_marked:
        writes.append((prefix.canonical([static, []]), tokens([static, []])))
    if automatic:
        writes.append((prefix.canonical([static, messages]), total))
    last = 0
    for text, count in writes:
        digest = hashlib.sha256(text.encode()).hexdigest()
        entries.setdefault(digest, Entry(digest, count, now))
        last = max(last, count)
    written = max(last - read, 0)
    return read, written, total


# ----------------------------------------------------------------------------- the transport


@dataclass
class Seen:
    """One request as the upstream received it."""

    conversation: int
    body: dict[str, Any]
    headers: dict[str, str]
    started: float
    status: int = 200
    message: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class HostedUpstream:
    personality: str
    scripts: list[dict[str, Any]] = field(default_factory=list)
    requests: list[Seen] = field(default_factory=list)
    #: Switch one rule off (the self-test's control): ``{"prefix_ok"}``.
    disabled: set[str] = field(default_factory=set)
    output_tokens: int = 40
    _conversations: dict[str, int] = field(default_factory=dict)
    _last: dict[int, dict[str, Any]] = field(default_factory=dict)
    _entries: dict[str, Entry] = field(default_factory=dict)
    _ids: int = 0

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))

    def conversations(self) -> dict[int, list[Seen]]:
        out: dict[int, list[Seen]] = {}
        for seen in self.requests:
            out.setdefault(seen.conversation, []).append(seen)
        return out

    def _markers(self, body: dict[str, Any]) -> list[str]:
        found: list[str] = []
        for message in body.get("messages") or []:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, list):
                found += [str(b["signature"]) for b in content if b.get("type") == "thinking"]
                found += [str(b["id"]) for b in content if b.get("type") == "tool_use"]
            found += [str(c["id"]) for c in message.get("tool_calls") or []]
        return found

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        headers = {k.lower(): v for k, v in request.headers.items()}
        markers = self._markers(body)
        if markers and markers[0] in self._conversations:
            conversation = self._conversations[markers[0]]
        elif markers:
            conversation = -1  # a marker this upstream never issued
        else:
            conversation = len(self._last) + 1
            self._last.setdefault(conversation, {})
        seen = Seen(conversation, body, headers, time.monotonic())
        self.requests.append(seen)
        try:
            if conversation == -1:
                raise Rejected("the request carries a block this upstream never produced")
            if self.personality == ANTHROPIC:
                version_ok(headers)
                no_beta(headers)
                no_sampling(body)
                no_forced_tool_choice(body)
                signatures_ok(body)
            else:
                openai_effort_ok(body)
            previous = self._last.get(conversation) or None
            if "prefix_ok" not in self.disabled:
                prefix_ok(previous, body)
        except Rejected as exc:
            seen.status, seen.message = exc.status, exc.message
            return httpx.Response(
                exc.status,
                json={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": exc.message},
                },
            )
        self._last[conversation] = body
        step = self.scripts.pop(0) if self.scripts else {"calls": [("submit", {"answer": "end"})]}
        if step.get("delay_s"):
            time.sleep(float(step["delay_s"]))
        if step.get("drop"):
            raise httpx.ReadError("the connection was closed with no answer", request=request)
        read, written, total = (
            cache_reads(self._entries, body, seen.started)
            if self.personality == ANTHROPIC
            else self._openai_cache(conversation, previous, body)
        )
        out = int(step.get("output_tokens", self.output_tokens))
        if self.personality == ANTHROPIC:
            seen.usage = {
                "input_tokens": total - read - written,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": written,
                "output_tokens": out,
            }
            return httpx.Response(200, json=self._anthropic(body, step, seen, conversation))
        seen.usage = {
            "prompt_tokens": total,
            "prompt_tokens_details": {"cached_tokens": read},
            "completion_tokens": out,
        }
        return httpx.Response(200, json=self._openai(step, seen, conversation))

    def _openai_cache(
        self, _conversation: int, previous: dict[str, Any] | None, body: dict[str, Any]
    ) -> tuple[int, int, int]:
        """``prompt_tokens_details.cached_tokens``: the unchanged prefix of the previous request
        (FTID §0, L5); an OpenAI-compatible usage reports no writes."""
        static, messages = _segments(body)
        total = tokens([static, messages])
        if not previous:
            return 0, 0, total
        _ps, before = _segments(previous)
        return tokens([static, messages[: len(before)]]), 0, total

    def _issue(self, conversation: int, marker: str) -> None:
        self._conversations.setdefault(marker, conversation)

    def _anthropic(
        self, body: dict[str, Any], step: dict[str, Any], seen: Seen, conversation: int
    ) -> dict[str, Any]:
        if "refusal" in step:
            category = step["refusal"]
            return {
                "id": f"msg_{len(self.requests)}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "stop_reason": "refusal",
                "stop_details": (
                    None if category is None else {"type": "refusal", "category": category}
                ),
                "usage": seen.usage,
            }
        blocks: list[dict[str, Any]] = []
        thinking = {
            "type": "thinking",
            "thinking": step.get("thinking", f"plan {len(self.requests)}"),
        }
        thinking["signature"] = f"{signature_for(body, [])}_{len(self.requests):04d}"
        blocks.append(thinking)
        self._issue(conversation, thinking["signature"])
        for name, args in step.get("calls", []):
            self._ids += 1
            blocks.append(
                {"type": "tool_use", "id": f"toolu_{self._ids:04d}", "name": name, "input": args}
            )
        return {
            "id": f"msg_{len(self.requests)}",
            "type": "message",
            "role": "assistant",
            "content": blocks,
            "stop_reason": "tool_use",
            "usage": seen.usage,
        }

    def _openai(self, step: dict[str, Any], seen: Seen, conversation: int) -> dict[str, Any]:
        if "refusal" in step:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "refusal": "I can't help with that.",
            }
        else:
            calls = []
            for name, args in step.get("calls", []):
                self._ids += 1
                call_id = f"call_{self._ids:04d}"
                self._issue(conversation, call_id)
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                )
            message = {"role": "assistant", "content": None, "tool_calls": calls}
        return {
            "id": f"cmpl_{len(self.requests)}",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": seen.usage,
        }
