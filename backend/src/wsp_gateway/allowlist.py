"""The gateway's request allow-list (feature 016; FR-016.15 step 4; FTDD §5.5; P50, P61, R1B-M6,
R2A-5, R2A-23, R3A-6, R3A-17).

**Standard library only** (an AST test pins it): the worker image's ``test`` target carries this
one file so the real producer, ``miwsp_engine``'s client, and this consumer run in one process
(018 FTASKS 10.1, ``tests/image/in_container/test_engine_bodies.py``; until 018-P4 the producer
was weightspace's ``AnthropicChat``/``OpenAIChat`` on the pinned SDKs, 016 FTASKS 6.7c).

The request is REBUILT from what is allowed, never filtered by what is known to be bad: a field,
block type, tool type or header that is not listed here is refused ``wsp_request_field_refused``
before any upstream call. Every content block is checked at any depth against the five shapes
an editor's producer sends (``text``, ``tool_use``, ``tool_result``, ``thinking``,
``redacted_thinking``; weightspace's set, which the engine keeps),
each with an optional ``cache_control: {"type": "ephemeral"}``; any ``image``, ``document``,
``search_result``, server-tool block or ``source`` object is refused, so no URL fetch can be
smuggled in a nested block and "a byte is at least a token" holds for the worst case. OpenAI
messages are pinned to what weightspace's ``to_openai_messages`` emitted, which the engine's
``openai_messages`` keeps; a content-part array is refused. Headers are built from scratch:
incoming ones are only checked, never forwarded.
"""

from __future__ import annotations

import re
from typing import Any

#: A refused name is echoed only when it matches this; otherwise ``<other>`` (R2A-23).
SAFE_NAME = re.compile(r"[A-Za-z0-9_.]{1,64}\Z")

ANTHROPIC = "anthropic"
OPENAI = "openai_compatible"
MISERVER = "miserver"

#: Feature 020 (FR-020.13, FR-020.19; FTDD §6.4): no ``temperature`` or ``top_p`` (version
#: ``2026-10-21`` refuses the sampling fields on every model, even at their defaults; ``top_k`` was
#: never admitted), and the top-level ``cache_control`` (automatic caching) at one exact value.
ANTHROPIC_TOP = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "stop_sequences",
        "cache_control",
        "thinking",
        "output_config",
        "tools",
        "tool_choice",
        "stream",
    }
)
OPENAI_TOP = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "stream",
    }
)
#: Admitted for an OpenAI-kind endpoint only when it is served by miServer (P61, R2A-5).
#: Feature 018 (FR-018.9, F18-D11): ``response_format`` (``json_schema`` only, the engine's
#: structured mode) and ``parallel_tool_calls`` (``false`` only, one call per reply).
#: The reasoning allowance ``miwsp_engine`` sends with thinking on in the structured protocol
#: (2026-11-21; miServer's field: the share of the cap the reasoning block may use): an integer
#: above 0, clamped below the inserted cap (:func:`insert_cap`). The name is the engine's
#: ``protocol.REASONING_ALLOWANCE_FIELD`` (the gateway does not import the worker; a test pins
#: the two equal), so changing it is one line in each, caught if only one moves.
REASONING_ALLOWANCE_FIELD = "max_reasoning_tokens"
#: The output tokens a capped request keeps for the action beside its reasoning allowance: the
#: meter's own floor (``meter.MIN_OUTPUT_TOKENS``, pinned equal by the same test).
ACTION_FLOOR_TOKENS = 256
#: With the run's output cap off (2026-11-21) the request names no cap and the gateway sizes one
#: to the window; the reasoning allowance then takes at most this share of it (half), and never
#: more than the run asked for (its ``thinking_tokens``, 4,096 unless set).
UNCAPPED_REASONING_DIVISOR = 2
OPENAI_MISERVER_TOP = frozenset(
    {
        "reasoning_effort",
        "chat_template_kwargs",
        "response_format",
        "parallel_tool_calls",
        REASONING_ALLOWANCE_FIELD,
    }
)
#: miServer's own ``json_schema`` shape (``miserver/api/schemas/openai.py``, ``extra=forbid``).
JSON_SCHEMA_REQUIRED = {"name", "schema"}
JSON_SCHEMA_OPTIONAL = {"description", "strict"}

#: Anthropic thinking effort above ``high`` is refused (the SDK refuses it non-streaming; R1B-A5).
ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high"})
#: Feature 020 (FR-020.16, W3-D9): the efforts a hosted OpenAI-compatible endpoint may declare
#: (``wsp.models.EFFORTS``, pinned equal), admitted as ``reasoning_effort`` only for an endpoint
#: whose served ``thinking`` is that effort.
HOSTED_EFFORTS = frozenset({"low", "medium", "high"})
#: Feature 020 (FR-020.16): adaptive thinking only, with an optional display. ``enabled`` (with
#: ``budget_tokens``) and ``disabled`` are refused: the engine sends neither, and the newest models
#: reject both (FTID §0).
THINKING_DISPLAYS = frozenset({"summarized", "omitted"})
#: Feature 020 (FR-020.19, HR-3): ``auto`` and ``none`` only; a forced tool choice (``any``,
#: ``tool``) returns a 400 on the newest models, and the engine never sends one.
ANTHROPIC_TOOL_CHOICES = ("auto", "none")
OPENAI_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high"})
CACHE_CONTROL = {"type": "ephemeral"}

#: Incoming headers that are tolerated (and dropped). Anything else, ``anthropic-beta`` included,
#: is refused. The SDKs' own ``x-stainless-*`` telemetry headers are tolerated by prefix.
TOLERATED_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "anthropic-version",
        "authorization",
        "connection",
        "content-length",
        "content-type",
        "host",
        "user-agent",
        "x-api-key",
    }
)
TOLERATED_HEADER_PREFIXES = ("x-stainless-",)
#: Feature 020 (FR-020.19, HR-7): what the gateway forwards (it builds headers from nothing, so
#: this is what reaches the API); pinned equal to ``miwsp_engine.client.ANTHROPIC_VERSION``.
ANTHROPIC_VERSION = "2026-10-21"


class FieldRefused(ValueError):  # noqa: N818 - a refusal, like core.errors.Refusal
    """``wsp_request_field_refused``: ``field`` is already sanitised for echoing."""

    def __init__(self, name: object) -> None:
        self.field = safe_name(name)
        super().__init__(self.field)


class StreamingRefused(ValueError):  # noqa: N818 - a refusal, like core.errors.Refusal
    """``wsp_streaming_unsupported``: metering is exact only on a complete response (PF-14)."""


def safe_name(name: object) -> str:
    return name if isinstance(name, str) and SAFE_NAME.match(name) else "<other>"


# ------------------------------------------------------------------------------------- helpers


def _keys(obj: dict[str, Any], required: set[str], optional: set[str], where: str) -> None:
    for key in obj:
        if key not in required and key not in optional:
            raise FieldRefused(key)
    for key in required:
        if key not in obj:
            raise FieldRefused(where)


def _str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise FieldRefused(name)
    return value


def _dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FieldRefused(name)
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise FieldRefused(name)
    return value


def _number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FieldRefused(name)


def _int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FieldRefused(name)


def _cache_control(block: dict[str, Any]) -> None:
    if "cache_control" in block and block["cache_control"] != CACHE_CONTROL:
        raise FieldRefused("cache_control")


# ----------------------------------------------------------------------------------- Anthropic


def _text_block(block: dict[str, Any]) -> None:
    _keys(block, {"type", "text"}, {"cache_control"}, "text")
    _str(block["text"], "text")
    _cache_control(block)


def _anthropic_block(block: object, allowed: frozenset[str]) -> None:
    b = _dict(block, "content")
    kind = b.get("type")
    if not isinstance(kind, str) or kind not in allowed:
        raise FieldRefused(kind if isinstance(kind, str) else "type")
    if kind == "text":
        _text_block(b)
    elif kind == "tool_use":
        _keys(b, {"type", "id", "name", "input"}, {"cache_control"}, "tool_use")
        _str(b["id"], "id")
        _str(b["name"], "name")
        _dict(b["input"], "input")  # the model's own arguments: data, never a content block
    elif kind == "tool_result":
        _keys(b, {"type", "tool_use_id", "content"}, {"is_error", "cache_control"}, "tool_result")
        _str(b["tool_use_id"], "tool_use_id")
        if "is_error" in b and not isinstance(b["is_error"], bool):
            raise FieldRefused("is_error")
        content = b["content"]
        if not isinstance(content, str):
            for inner in _list(content, "content"):  # nested: text blocks only (R2A-5)
                _anthropic_block(inner, frozenset({"text"}))
    elif kind == "thinking":
        _keys(b, {"type", "thinking", "signature"}, {"cache_control"}, "thinking")
        _str(b["thinking"], "thinking")
        _str(b["signature"], "signature")
    else:  # redacted_thinking
        _keys(b, {"type", "data"}, {"cache_control"}, "redacted_thinking")
        _str(b["data"], "data")
    _cache_control(b)


USER_BLOCKS = frozenset({"text", "tool_result"})
ASSISTANT_BLOCKS = frozenset({"text", "tool_use", "thinking", "redacted_thinking"})


def _anthropic_message(message: object) -> None:
    m = _dict(message, "messages")
    _keys(m, {"role", "content"}, set(), "messages")
    role = m["role"]
    if role not in ("user", "assistant"):
        raise FieldRefused("role")
    content = m["content"]
    if isinstance(content, str):
        return
    allowed = USER_BLOCKS if role == "user" else ASSISTANT_BLOCKS
    for block in _list(content, "content"):
        _anthropic_block(block, allowed)


def _anthropic_tool(tool: object) -> None:
    t = _dict(tool, "tools")
    _keys(t, {"name", "input_schema"}, {"description", "type", "cache_control"}, "tools")
    if t.get("type", "custom") != "custom":  # server tools (web_fetch, web_search, ...) refused
        raise FieldRefused(t.get("type"))
    _str(t["name"], "name")
    if "description" in t:
        _str(t["description"], "description")
    _dict(t["input_schema"], "input_schema")
    _cache_control(t)


def _anthropic(body: dict[str, Any]) -> dict[str, Any]:
    for key in body:
        if key not in ANTHROPIC_TOP:
            raise FieldRefused(key)
    for message in _list(body.get("messages"), "messages"):
        _anthropic_message(message)
    if "system" in body and not isinstance(body["system"], str):
        for block in _list(body["system"], "system"):
            _anthropic_block(block, frozenset({"text"}))
    for tool in _list(body.get("tools", []), "tools"):
        _anthropic_tool(tool)
    if "cache_control" in body and body["cache_control"] != CACHE_CONTROL:
        # Exactly the 5-minute default (F20-D7): a ``ttl`` or any other value is refused.
        raise FieldRefused("cache_control")
    if "tool_choice" in body:
        choice = _dict(body["tool_choice"], "tool_choice")
        _keys(choice, {"type"}, {"disable_parallel_tool_use"}, "tool_choice")
        if choice["type"] not in ANTHROPIC_TOOL_CHOICES:
            raise FieldRefused("tool_choice")
        if "disable_parallel_tool_use" in choice and not isinstance(
            choice["disable_parallel_tool_use"], bool
        ):
            raise FieldRefused("disable_parallel_tool_use")
    if "thinking" in body:
        thinking = _dict(body["thinking"], "thinking")
        if thinking.get("type") != "adaptive":
            raise FieldRefused("thinking")
        _keys(thinking, {"type"}, {"display"}, "thinking")
        if "display" in thinking and thinking["display"] not in THINKING_DISPLAYS:
            raise FieldRefused("display")
    if "output_config" in body:
        config = _dict(body["output_config"], "output_config")
        _keys(config, set(), {"effort"}, "output_config")
        if "effort" in config and config["effort"] not in ANTHROPIC_EFFORTS:
            raise FieldRefused("effort")
    for name in ("max_tokens",):
        if name in body:
            _int(body[name], name)
    if "stop_sequences" in body:
        for stop in _list(body["stop_sequences"], "stop_sequences"):
            _str(stop, "stop_sequences")
    return body


# -------------------------------------------------------------------------------------- OpenAI


def _openai_message(message: object) -> None:
    m = _dict(message, "messages")
    role = m.get("role")
    if role in ("system", "user"):
        _keys(m, {"role", "content"}, set(), "messages")
        _str(m["content"], "content")  # a content-part array is refused (R3A-17)
    elif role == "assistant":
        _keys(m, {"role", "content"}, {"tool_calls"}, "messages")
        if m["content"] is not None:
            _str(m["content"], "content")
        for call in _list(m.get("tool_calls", []), "tool_calls"):
            c = _dict(call, "tool_calls")
            _keys(c, {"id", "type", "function"}, set(), "tool_calls")
            _str(c["id"], "id")
            if c["type"] != "function":
                raise FieldRefused(c["type"])
            fn = _dict(c["function"], "function")
            _keys(fn, {"name", "arguments"}, set(), "function")
            _str(fn["name"], "name")
            _str(fn["arguments"], "arguments")
    elif role == "tool":
        _keys(m, {"role", "tool_call_id", "content"}, set(), "messages")
        _str(m["tool_call_id"], "tool_call_id")
        _str(m["content"], "content")
    else:
        raise FieldRefused("role")


def _response_format(body: dict[str, Any]) -> None:
    """``{type: json_schema, json_schema: {name, schema, description?, strict?}}`` and nothing
    else (FR-018.9). miServer refuses ``response_format`` beside ``tools`` or ``stop``, so such a
    body is refused here, before it is metered."""
    fmt = _dict(body["response_format"], "response_format")
    _keys(fmt, {"type", "json_schema"}, set(), "response_format")
    if fmt["type"] != "json_schema":
        raise FieldRefused("response_format")
    schema = _dict(fmt["json_schema"], "json_schema")
    _keys(schema, JSON_SCHEMA_REQUIRED, JSON_SCHEMA_OPTIONAL, "json_schema")
    _str(schema["name"], "name")
    _dict(schema["schema"], "schema")
    if "description" in schema:
        _str(schema["description"], "description")
    if "strict" in schema and not isinstance(schema["strict"], bool):
        raise FieldRefused("strict")
    for other in ("tools", "tool_choice", "stop"):
        if other in body:
            raise FieldRefused(other)


#: Feature 019 P2 (FR-019.13): the one miServer field a generic OpenAI-compatible endpoint may
#: carry, and only when the backend says it is admitted (a current protocol-test pass, or the
#: endpoint's own test session). Every other miServer-only field stays miServer-only.
STRUCTURED_TOP = frozenset({"response_format"})


def _declared_effort(serving: str, endpoint: dict[str, Any] | None) -> str | None:
    """Feature 020 (FR-020.16): the effort a GENERIC (hosted) endpoint declares, from the
    endpoint document the backend serves on every request (its ``thinking``), else None."""
    if serving == MISERVER or not isinstance(endpoint, dict):
        return None
    thinking = endpoint.get("thinking")
    return thinking if isinstance(thinking, str) and thinking in HOSTED_EFFORTS else None


def _openai(
    body: dict[str, Any],
    serving: str,
    structured_admitted: bool,
    endpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effort = _declared_effort(serving, endpoint)
    if serving == MISERVER:
        extra = OPENAI_MISERVER_TOP
    else:
        extra = STRUCTURED_TOP if structured_admitted else frozenset()
        if effort is not None:
            extra = extra | {"reasoning_effort"}
    allowed = OPENAI_TOP | extra
    for key in body:
        if key not in allowed:
            raise FieldRefused(key)
    for message in _list(body.get("messages"), "messages"):
        _openai_message(message)
    for tool in _list(body.get("tools", []), "tools"):
        t = _dict(tool, "tools")
        _keys(t, {"type", "function"}, set(), "tools")
        if t["type"] != "function":
            raise FieldRefused(t["type"])
        fn = _dict(t["function"], "function")
        _keys(fn, {"name", "parameters"}, {"description"}, "function")
        _str(fn["name"], "name")
        _dict(fn["parameters"], "parameters")
    if "tool_choice" in body:
        choice = body["tool_choice"]
        if isinstance(choice, str):
            if choice not in ("auto", "none", "required"):
                raise FieldRefused("tool_choice")
        else:
            c = _dict(choice, "tool_choice")
            _keys(c, {"type", "function"}, set(), "tool_choice")
            if c["type"] != "function":
                raise FieldRefused("tool_choice")
            _keys(_dict(c["function"], "function"), {"name"}, set(), "function")
    if "reasoning_effort" in body and body["reasoning_effort"] not in OPENAI_EFFORTS:
        raise FieldRefused("reasoning_effort")
    if effort is not None and "reasoning_effort" in body and body["reasoning_effort"] != effort:
        # A generic endpoint carries exactly the effort it declares, nothing else.
        raise FieldRefused("reasoning_effort")
    if "chat_template_kwargs" in body:
        kwargs = _dict(body["chat_template_kwargs"], "chat_template_kwargs")
        _keys(kwargs, set(), {"enable_thinking"}, "chat_template_kwargs")
        if "enable_thinking" in kwargs and not isinstance(kwargs["enable_thinking"], bool):
            raise FieldRefused("enable_thinking")
    if REASONING_ALLOWANCE_FIELD in body:
        _int(body[REASONING_ALLOWANCE_FIELD], REASONING_ALLOWANCE_FIELD)
        if body[REASONING_ALLOWANCE_FIELD] < 1:  # miServer: an integer above 0
            raise FieldRefused(REASONING_ALLOWANCE_FIELD)
    if "parallel_tool_calls" in body and body["parallel_tool_calls"] is not False:
        raise FieldRefused("parallel_tool_calls")
    if "response_format" in body:
        _response_format(body)
    for name in ("max_tokens", "max_completion_tokens"):
        if name in body:
            _int(body[name], name)
    for name in ("temperature", "top_p"):
        if name in body:
            _number(body[name], name)
    if "stop" in body and not isinstance(body["stop"], str):
        for stop in _list(body["stop"], "stop"):
            _str(stop, "stop")
    return body


# ---------------------------------------------------------------------------------- the entry


def check_headers(headers: dict[str, str]) -> None:
    """Refuse any incoming header outside the tolerated set; none is ever forwarded."""
    for name in headers:
        lower = name.lower()
        if lower in TOLERATED_HEADERS or lower.startswith(TOLERATED_HEADER_PREFIXES):
            continue
        raise FieldRefused(lower)


def base_headers(kind: str) -> dict[str, str]:
    """The forwarded header set, built from an empty dict (credentials are added upstream)."""
    headers = {"content-type": "application/json"}
    if kind == ANTHROPIC:
        headers["anthropic-version"] = ANTHROPIC_VERSION
    return headers


def rebuild(
    kind: str,
    serving: str,
    body: object,
    headers: dict[str, str],
    *,
    structured_admitted: bool,
    endpoint: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The allow-listed body (a new dict; ``stream: false`` dropped) and the base headers.

    ``structured_admitted`` (feature 019 P2, keyword-only with no default, so a caller that forgets
    it fails rather than admits): whether a GENERIC OpenAI-compatible endpoint may carry
    ``response_format``. miServer's set is unchanged. ``endpoint`` (feature 020, FR-020.16): the
    backend's endpoint document, whose ``thinking`` admits a hosted ``reasoning_effort``.

    Raises :class:`StreamingRefused` for ``stream: true`` and :class:`FieldRefused` for anything
    outside the allow-list, before any upstream call.
    """
    check_headers(headers)
    b = dict(_dict(body, "body"))
    if "stream" in b:
        if b["stream"] is True:
            raise StreamingRefused("stream")
        if b["stream"] is not False:
            raise FieldRefused("stream")
        del b["stream"]
    if "model" in b:
        _str(b["model"], "model")
    if kind == ANTHROPIC:
        return _anthropic(b), base_headers(kind)
    if kind == OPENAI:
        return _openai(b, serving, structured_admitted, endpoint), base_headers(kind)
    raise FieldRefused("kind")


def insert_cap(
    kind: str, body: dict[str, Any], output_cap_field: str, cap: int, *, uncapped: bool = False
) -> dict[str, Any]:
    """Set the output cap under the endpoint's field, even when the request had none (R1B-A5).

    The other spelling is removed, and an Anthropic ``thinking.budget_tokens`` is clamped below
    the cap. An OpenAI-kind :data:`REASONING_ALLOWANCE_FIELD` is clamped to leave
    :data:`ACTION_FLOOR_TOKENS` of the cap for the action (2026-11-21): when the budget lowers the
    cap, the reasoning gives way, never the action.
    """
    out = dict(body)
    for name in ("max_tokens", "max_completion_tokens"):
        if name != output_cap_field:
            out.pop(name, None)
    out[output_cap_field] = cap
    thinking = out.get("thinking")
    if kind == ANTHROPIC and isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = int(thinking.get("budget_tokens", 0))
        if budget >= cap:
            out["thinking"] = {**thinking, "budget_tokens": max(cap - 1, 0)}
    allowance = out.get(REASONING_ALLOWANCE_FIELD)
    if uncapped and kind == OPENAI and isinstance(allowance, int):
        # The request named no cap: the reasoning takes at most half of the one sized here.
        allowance = min(allowance, max(cap // UNCAPPED_REASONING_DIVISOR, 1))
        out[REASONING_ALLOWANCE_FIELD] = allowance
    if kind == OPENAI and isinstance(allowance, int) and allowance > cap - ACTION_FLOOR_TOKENS:
        # At least 1 (miServer's floor), and below the cap: the cap is never under the floor.
        out[REASONING_ALLOWANCE_FIELD] = max(cap - ACTION_FLOOR_TOKENS, 1)
    return out
