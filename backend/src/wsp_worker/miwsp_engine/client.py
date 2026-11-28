"""The engine's own model client, over ``httpx2`` to the session's gateway only (feature 018;
FR-018.10; FTDD §6.7; FTID §3.3).

- **OpenAI-compatible:** ``POST /v1/chat/completions``; system and user content as strings, never
  part arrays (the gateway refuses arrays); assistant ``tool_calls`` as ``{id, type: function,
  function: {name, arguments: <str>}}``; tool messages ``{role, tool_call_id, content}``.
- **Anthropic:** ``POST /v1/messages``; ``system`` at the top; content blocks ``text``,
  ``tool_use``, ``tool_result`` (and ``thinking`` blocks given back as received).
- **The token** goes as ``Authorization: Bearer`` (OpenAI) or ``x-api-key`` with
  ``anthropic-version`` (Anthropic). The gateway accepts either.
- **The output cap** is the engine's own (``context.editor_output_cap``), sent as
  ``max_tokens`` (both kinds); the gateway moves it under the endpoint's ``output_cap_field`` and
  clamps it to the budget. With the run's ``output_cap`` off (2026-11-21) no cap is named, and
  the gateway's window-aware default for an uncapped request applies.
- **No retries, ever.** A non-2xx answer raises :class:`ClientError` with the answer's
  ``error.code`` (the gateway's own, e.g. ``editor_context_exceeded``; or a generic endpoint's,
  passed through), and the session ends ``error`` with that code, except a LOST reply (a
  transport failure, ``wsp_gateway_unreachable``, or the gateway's upstream failing,
  ``wsp_gateway_upstream_failed``: :data:`LOST_REPLY_CODES`), which the session survives once
  (``session.py`` step 3c).
- **The timeout** is ``call_time.CLIENT_TIMEOUT_S``, beyond the gateway's upstream timeout, so
  the gateway answers first (fix 2026-11-22).
- **Usage** is read as the gateway meters it (``meter.metered``): OpenAI ``prompt_tokens`` (cached
  ``prompt_tokens_details.cached_tokens``) and ``completion_tokens``; Anthropic ``input_tokens``
  plus cache reads and writes. Reasoning tokens are completion tokens (the OpenAI convention), so
  a thinking turn is metered, budgeted and priced like any other output.
- **Reasoning** (2026-11-21, :func:`split_reasoning`): an OpenAI reply's
  ``message.reasoning_content`` (miServer's field), else a leading ``<think>...</think>`` block in
  ``content`` (a server that inlines it), is taken OUT of the text the action is parsed from and
  kept as the neutral message's ``reasoning {text, source}``: recorded in the transcript, never
  parsed as an action and never sent back (:func:`openai_messages` copies role, content and
  tool calls only).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx2

from wsp_worker import call_time
from wsp_worker.miwsp_engine.protocol import ANTHROPIC, CUT_OFF, REFUSAL, Reply, ToolCall, Usage

#: Feature 020 (FR-020.19, HR-7): the Messages API version, pinned equal to the gateway's
#: ``wsp_gateway.allowlist.ANTHROPIC_VERSION`` (the gateway's is what reaches the API: it builds
#: forwarded headers from nothing). ``2026-10-21`` refuses ``temperature``, ``top_p`` and ``top_k``.
ANTHROPIC_VERSION = "2026-10-21"
#: Feature 020 (FR-020.12, F20-D7): the cache marker, the 5-minute default (no ``ttl``), pinned
#: equal to the gateway's ``allowlist.CACHE_CONTROL``.
CACHE_CONTROL = {"type": "ephemeral"}
#: Strictly longer than the gateway's upstream timeout (fix 2026-11-22): the two were both 900 s
#: and raced, and a lost race dropped the connection while the gateway still held the session's
#: in-flight reservation. ``call_time`` orders them.
TIMEOUT_S = call_time.CLIENT_TIMEOUT_S


#: The codes of a call whose reply was lost rather than refused (fix 2026-11-22): the client
#: could not reach the gateway or gave up waiting, or the gateway's own upstream call failed (a
#: reply that ran past its time limit). Nothing in the answer refused the request.
LOST_REPLY_CODES = frozenset({"wsp_gateway_unreachable", "wsp_gateway_upstream_failed"})


class ClientError(Exception):
    """The endpoint (through the gateway) answered non-2xx, or could not be reached."""

    def __init__(self, status: int, code: str, message: str = "") -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


class ModelClient(Protocol):
    """What the session needs from a client: one completion, no retries."""

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> Reply: ...

    def body(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> dict[str, Any]: ...


def _n(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(int(value), 0)


def _error_code(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for holder in (data.get("error"), data.get("detail"), data):
        if isinstance(holder, dict) and isinstance(holder.get("code"), str):
            return str(holder["code"])
    return None


# ----------------------------------------------------------------------------- OpenAI


def openai_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            out.append({"role": role, "content": str(m["content"])})
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content")}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": str(c["id"]),
                        "type": "function",
                        "function": {
                            "name": str(c["name"]),
                            "arguments": (
                                c["arguments"]
                                if isinstance(c["arguments"], str)
                                else json.dumps(c["arguments"])
                            ),
                        },
                    }
                    for c in calls
                ]
            out.append(msg)
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(m["tool_call_id"]),
                    "content": str(m["content"]),
                }
            )
    return out


#: A leading think block: optional whitespace, an optional ``<think>`` (a template may put the
#: opening tag in the prompt, so the reply starts inside it), the reasoning, ``</think>``.
_THINK = re.compile(r"\A\s*(?:<think>)?(?P<reasoning>.*?)</think>\s*", re.DOTALL)
#: A reply that opened a think block and never closed it (the cap ran out mid-thought).
_UNCLOSED = re.compile(r"\A\s*<think>(?P<reasoning>.*)\Z", re.DOTALL)
REASONING_CONTENT = "reasoning_content"
INLINE_THINK = "inline_think"


def split_reasoning(
    content: str | None, reasoning_content: Any
) -> tuple[str | None, dict[str, str] | None]:
    """``(content without its reasoning, {text, source} or None)``.

    ``reasoning_content`` (a non-empty string) is the reasoning, source ``reasoning_content``. A
    leading ``<think>...</think>`` in ``content`` is stripped either way (and is the reasoning,
    source ``inline_think``, when no ``reasoning_content`` came); an opened block that never
    closes makes the WHOLE content reasoning, so nothing in it can be read as an action."""
    parts: list[str] = []
    source: str | None = None
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        parts.append(reasoning_content)
        source = REASONING_CONTENT
    if isinstance(content, str):
        closed = _THINK.match(content) if "</think>" in content else None
        unclosed = None if closed else _UNCLOSED.match(content)
        found = closed or unclosed
        if found is not None:
            inline = found.group("reasoning").strip()
            if inline:
                parts.append(inline)
                source = source or INLINE_THINK
            content = content[found.end() :] if closed else ""
    if source is None:
        return content, None
    return content, {"text": "\n".join(parts), "source": source}


def openai_reply(data: Mapping[str, Any]) -> Reply:
    choices = data.get("choices") or []
    message = (choices[0] or {}).get("message") or {} if choices else {}
    calls = [
        ToolCall(
            str(c.get("id") or f"call_{i}"),
            str((c.get("function") or {}).get("name", "")),
            (c.get("function") or {}).get("arguments", ""),
        )
        for i, c in enumerate(message.get("tool_calls") or [])
    ]
    u: dict[str, Any] = data["usage"] if isinstance(data.get("usage"), dict) else {}
    raw_details = u.get("prompt_tokens_details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    raw_completion = u.get("completion_tokens_details")
    completion: dict[str, Any] = raw_completion if isinstance(raw_completion, dict) else {}
    usage = Usage(
        _n(u.get("prompt_tokens")),
        _n(u.get("completion_tokens")),
        _n(details.get("cached_tokens")),
        0,
        _n(completion.get("reasoning_tokens")),
    )
    raw = message.get("content")
    text, reasoning = split_reasoning(
        raw if isinstance(raw, str) else None, message.get(REASONING_CONTENT)
    )
    neutral: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        neutral["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls
        ]
    if reasoning is not None:
        neutral["reasoning"] = reasoning
    if usage.reasoning_tokens:
        # Recorded beside the reasoning (miServer's usage figure); never sent back.
        neutral["reasoning_tokens"] = usage.reasoning_tokens
    if choices and (choices[0] or {}).get("finish_reason") == "length":
        neutral[CUT_OFF] = True  # recorded, never sent back (``openai_messages``)
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        # Feature 020 (FR-020.17; FTID §0 L5): the session ends ``editor_refused``. Recorded,
        # never sent back (``openai_messages`` copies role, content and tool calls only).
        neutral[REFUSAL] = {"category": None, "text": refusal[:500]}
    return Reply(text, calls, usage, neutral, None if reasoning is None else reasoning["text"])


# ----------------------------------------------------------------------------- Anthropic


def _anthropic_block(block: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": str(block.get("text", ""))}
    if kind == "tool_use":
        return {
            "type": "tool_use",
            "id": str(block["id"]),
            "name": str(block["name"]),
            "input": dict(block.get("input") or {}),
        }
    if kind == "thinking":
        return {
            "type": "thinking",
            "thinking": str(block.get("thinking", "")),
            "signature": str(block.get("signature", "")),
        }
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": str(block.get("data", ""))}
    return None


def anthropic_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    system: list[str] = []
    out: list[dict[str, Any]] = []

    def user_blocks(blocks: list[dict[str, Any]]) -> None:
        if out and out[-1]["role"] == "user":
            last = out[-1]["content"]
            if isinstance(last, str):
                last = [{"type": "text", "text": last}]
            out[-1]["content"] = last + blocks
        else:
            out.append({"role": "user", "content": blocks})

    for m in messages:
        role = m["role"]
        if role == "system":
            system.append(str(m["content"]))
        elif role == "user":
            if out and out[-1]["role"] == "user":
                user_blocks([{"type": "text", "text": str(m["content"])}])
            else:
                out.append({"role": "user", "content": str(m["content"])})
        elif role == "tool":
            user_blocks(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": str(m["tool_call_id"]),
                        "content": str(m["content"]),
                    }
                ]
            )
        elif role == "assistant":
            raw = m.get("anthropic_content")
            if isinstance(raw, list):
                blocks = [b for b in (_anthropic_block(x) for x in raw) if b is not None]
            else:
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": str(m["content"])})
                for c in m.get("tool_calls") or []:
                    args = c["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(c["id"]),
                            "name": str(c["name"]),
                            "input": args if isinstance(args, dict) else {},
                        }
                    )
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})
    return "\n\n".join(system), out


def anthropic_reply(data: Mapping[str, Any]) -> Reply:
    blocks = [b for b in data.get("content") or [] if isinstance(b, dict)]
    texts = [str(b.get("text", "")) for b in blocks if b.get("type") == "text"]
    calls = [
        ToolCall(str(b.get("id", "")), str(b.get("name", "")), b.get("input"))
        for b in blocks
        if b.get("type") == "tool_use"
    ]
    u: dict[str, Any] = data["usage"] if isinstance(data.get("usage"), dict) else {}
    read, write = _n(u.get("cache_read_input_tokens")), _n(u.get("cache_creation_input_tokens"))
    usage = Usage(_n(u.get("input_tokens")) + read + write, _n(u.get("output_tokens")), read, write)
    text = "".join(texts) if texts else None
    neutral: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        neutral["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls
        ]
    neutral["anthropic_content"] = [b for b in (_anthropic_block(x) for x in blocks) if b]
    if data.get("stop_reason") == "max_tokens":
        neutral[CUT_OFF] = True  # recorded, never sent back (``anthropic_messages``)
    if data.get("stop_reason") == "refusal":
        # Feature 020 (FR-020.17, W3-D8): HTTP 200 with ``stop_reason: refusal``; the category is
        # informational and may be null. Recorded, never sent back (``anthropic_messages`` sends
        # ``anthropic_content`` only).
        details = data.get("stop_details")
        category = details.get("category") if isinstance(details, dict) else None
        neutral[REFUSAL] = {"category": category if isinstance(category, str) else None}
    return Reply(text, calls, usage, neutral)


# ----------------------------------------------------------------------------- the client


#: The gateway's header on an answer it sent with thinking off (``wsp_gateway.app``; the worker
#: does not import the gateway, a test pins the two equal).
THINKING_SKIPPED_HEADER = "X-miForge-Thinking-Skipped"


class GatewayClient:
    """One editor endpoint, reached only through the session's gateway, with the session token."""

    def __init__(
        self,
        kind: str,
        model: str,
        gateway: Mapping[str, Any],
        *,
        transport: httpx2.BaseTransport | None = None,
        cache: bool = False,
    ) -> None:
        self.kind, self.model = kind, model
        #: Feature 020 (FR-020.12): mark the system text and the growing tail for caching (an
        #: Anthropic-kind endpoint in an append-only session; the session decides).
        self.cache = cache and kind == ANTHROPIC
        self._token = str(gateway["token"])
        self._http = httpx2.Client(
            base_url=f"http://{gateway['ip']}:{int(gateway['port'])}",
            timeout=TIMEOUT_S,
            trust_env=False,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def headers(self) -> dict[str, str]:
        if self.kind == ANTHROPIC:
            return {"x-api-key": self._token, "anthropic-version": ANTHROPIC_VERSION}
        return {"authorization": f"Bearer {self._token}"}

    def path(self) -> str:
        return "/v1/messages" if self.kind == ANTHROPIC else "/v1/chat/completions"

    def body(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> dict[str, Any]:
        if self.kind == ANTHROPIC:
            system, wire = anthropic_messages(messages)
            body: dict[str, Any] = {"model": self.model, "messages": wire}
            if system and self.cache:
                # One breakpoint on the static prefix (tools render before system, so it caches
                # both), and the top-level marker for the growing tail: two of four slots.
                body["system"] = [
                    {"type": "text", "text": system, "cache_control": dict(CACHE_CONTROL)}
                ]
            elif system:
                body["system"] = system
            if self.cache:
                body["cache_control"] = dict(CACHE_CONTROL)
        else:
            body = {"model": self.model, "messages": openai_messages(messages)}
        if max_tokens is not None:
            # ``None``: the output cap is off (2026-11-21); the request names none, and the
            # gateway inserts its window-aware default under the endpoint's cap field.
            body["max_tokens"] = max_tokens
        body.update(fields)
        return body

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> Reply:
        return self.complete_body(self.body(messages, fields=fields, max_tokens=max_tokens))

    def complete_body(self, body: Mapping[str, Any]) -> Reply:
        """Send ``body`` as built (feature 020: the session checks it first, F20-D15)."""
        try:
            response = self._http.post(self.path(), json=body, headers=self.headers())
        except httpx2.HTTPError as exc:
            raise ClientError(0, "wsp_gateway_unreachable", type(exc).__name__) from exc
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.status_code >= 300:
            code = _error_code(data) or f"http_{response.status_code}"
            message = ""
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                message = str(data["error"].get("message", ""))[:500]
            raise ClientError(response.status_code, code, message)
        if not isinstance(data, dict):
            raise ClientError(response.status_code, "editor_reply_invalid", "not a JSON object")
        reply = anthropic_reply(data) if self.kind == ANTHROPIC else openai_reply(data)
        skipped = response.headers.get(THINKING_SKIPPED_HEADER)
        if skipped:
            reply.thinking_skipped = skipped[:32]
        return reply
