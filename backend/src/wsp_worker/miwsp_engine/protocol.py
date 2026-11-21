"""One action per turn, in two modes (feature 018; FR-018.7; D25; FTDD §6.3; FTID §3.2).

- **``tool_calls``** (the default): the endpoint's own tool calling. A reply carrying more than
  one tool call is refused ``multiple_actions`` and NOTHING in it runs; a reply with none is a
  ``no_action`` (the session nudges); a call whose arguments are not a JSON object is
  ``unparseable``.
- **``structured``** (an OpenAI-kind endpoint served by miServer only): the request carries no
  ``tools`` and no ``stop``; ``response_format`` constrains the reply to :func:`one_action_schema`,
  an ``anyOf`` over one closed object per tool. The reply is parsed and validated again here
  before anything runs.

The neutral message shapes (``weightspace``'s, ``ws:llm/base.py``) are what the session keeps and
the transcript records; :mod:`client` turns them into each endpoint kind's wire format.

**Thinking (2026-11-21), in both modes.** The request's frozen ``thinking`` decides
:func:`thinking_fields`: ``on`` sends ``chat_template_kwargs: {enable_thinking: true}`` (and the
cap the request names is then the reasoning allowance plus the action's cap); in the structured
mode it also sends the allowance under :data:`REASONING_ALLOWANCE_FIELD`. ``off`` sends
``enable_thinking: false`` (explicitly: unset is the template's default, which is ON for Qwen3.6);
``null`` sends nothing.
Only a miServer OpenAI-kind endpoint carries either (the gateway refuses ``chat_template_kwargs``
elsewhere), so ``on`` anywhere else is refused (:func:`check_thinking_supported`). **The reasoning
is NEVER an action:** a reply's ``reasoning`` (``reasoning_content``, or a leading ``<think>``
block stripped from ``content`` by the client) is recorded and nothing else; :func:`one_action`
reads only ``text`` and ``tool_calls``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from wsp_worker.miwsp_engine import prompts
from wsp_worker.miwsp_engine.tools import ToolSpec, validate

TOOL_CALLS = "tool_calls"
STRUCTURED = "structured"
PROTOCOLS = (TOOL_CALLS, STRUCTURED)
OPENAI = "openai_compatible"
ANTHROPIC = "anthropic"
MISERVER = "miserver"
THINKING_ON = "on"
THINKING_OFF = "off"
#: The request field that carries the reasoning allowance to miServer (2026-11-21; miServer's
#: decision). The request's cap (``max_completion_tokens`` / ``max_tokens``) bounds the WHOLE
#: completion, reasoning included, as OpenAI defines it; this field is the share of it the
#: reasoning block may use (its ``</think>`` included), an integer above 0 and below the cap.
#: miServer honours it ONLY beside ``response_format`` on a GGUF engine (it refuses it
#: ``field_not_honoured`` with ``tools``), so the engine sends it in the structured mode only:
#: the tool-calls mode thinks inside the total cap with no allowance (deferred: miServer's tool
#: path has none yet). When the allowance runs out, miServer closes the block and still produces
#: the constrained JSON. **Change it here only**: the gateway admits the same name
#: (``wsp_gateway.allowlist.REASONING_ALLOWANCE_FIELD``, pinned equal by ``test_thinking.py``).
REASONING_ALLOWANCE_FIELD = "max_reasoning_tokens"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    #: A dict (Anthropic, or a scripted reply) or the raw JSON text (OpenAI).
    arguments: Any


@dataclass(frozen=True)
class Usage:
    """Tokens as the gateway meters them (``wsp_gateway/meter.py:metered``)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    #: ``completion_tokens_details.reasoning_tokens`` when the server reports it: a share of
    #: ``output_tokens`` (never added to it), recorded only.
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Reply:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    #: The neutral assistant message to append (and to record in the transcript).
    message: dict[str, Any] = field(default_factory=dict)
    #: The editor's reasoning, recorded only: never parsed as an action, never sent back.
    reasoning: str | None = None
    #: The gateway sent this turn with thinking off (fix 2026-11-21): its cap left the reasoning
    #: allowance no useful room. The gateway's ``X-miForge-Thinking-Skipped`` header.
    thinking_skipped: str | None = None


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class Refusal:
    #: ``multiple_actions`` | ``no_action`` | ``unparseable`` | ``not_offered``.
    kind: str
    n: int = 0
    detail: str = ""

    def text(self) -> str:
        if self.kind == "multiple_actions":
            return prompts.MULTIPLE_ACTIONS.format(n=self.n)
        if self.kind == "not_offered":
            return prompts.NOT_OFFERED.format(name=self.detail)
        if self.kind == "unparseable":
            return prompts.UNPARSEABLE.format(detail=self.detail)
        return prompts.NUDGE


class ProtocolUnsupportedError(ValueError):
    """``wsp_endpoint_protocol_unsupported``: structured output on an endpoint that cannot do it."""

    code = "wsp_endpoint_protocol_unsupported"


class ThinkingUnsupportedError(ValueError):
    """``wsp_endpoint_thinking_unsupported``: thinking on for an endpoint that cannot carry it."""

    code = "wsp_endpoint_thinking_unsupported"


def thinking_carried(kind: str, serving: str) -> bool:
    """``chat_template_kwargs`` reaches the editor: an OpenAI-kind endpoint served by miServer."""
    return kind == OPENAI and serving == MISERVER


def check_thinking_supported(thinking: Any, kind: str, serving: str) -> None:
    if thinking == THINKING_ON and not thinking_carried(kind, serving):
        raise ThinkingUnsupportedError(
            "thinking on needs an openai_compatible endpoint served by miServer"
        )


def thinking_fields(
    thinking: Any, kind: str, serving: str, allowance: int, protocol: str = STRUCTURED
) -> dict[str, Any]:
    """The body fields that switch thinking (see the module docstring); ``{}`` when the frozen
    value is ``null`` (or any value but ``on``/``off``) or the endpoint cannot carry them. The
    allowance field goes with ``on`` in the structured protocol only, and only above 0."""
    if thinking not in (THINKING_ON, THINKING_OFF) or not thinking_carried(kind, serving):
        return {}
    fields: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking == THINKING_ON}}
    if thinking == THINKING_ON and protocol == STRUCTURED and allowance > 0:
        fields[REASONING_ALLOWANCE_FIELD] = allowance
    return fields


def check_supported(protocol: str, kind: str, serving: str) -> None:
    if protocol not in PROTOCOLS:
        raise ProtocolUnsupportedError(f"unknown action protocol {protocol!r}")
    if protocol == STRUCTURED and (kind != OPENAI or serving != MISERVER):
        raise ProtocolUnsupportedError(
            "structured output is accepted only for an openai_compatible endpoint served by "
            "miServer"
        )


#: Keywords left out of the schema SENT to the server (2026-11-20). miServer's llama.cpp engine
#: refuses a grammar that cannot enforce them (``field_not_honoured``: "minimum (not enforced by
#: the llama.cpp grammar)"), which made every structured request to a GGUF editor fail. The
#: engine still enforces them: :func:`_one` validates every reply against the FULL schema, so an
#: out-of-range value is a refused turn, exactly as before on a transformers-served editor.
WIRE_DROPPED = frozenset({"minimum", "maximum"})


def _wire(schema: Any) -> Any:
    """``schema`` without :data:`WIRE_DROPPED` keywords, at every depth. A property NAMED like a
    keyword (a tool argument called ``minimum``) is a key of ``properties`` and is kept."""
    if isinstance(schema, list):
        return [_wire(item) for item in schema]
    if not isinstance(schema, Mapping):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, Mapping):
            out[key] = {name: _wire(sub) for name, sub in value.items()}
        elif key not in WIRE_DROPPED:
            out[key] = _wire(value)
    return out


def one_action_schema(specs: Mapping[str, ToolSpec], *, wire: bool = False) -> dict[str, Any]:
    """The structured-mode reply: exactly one ``{"tool", "arguments"}`` object, per tool.

    ``wire=True`` is the form sent in ``response_format`` (:data:`WIRE_DROPPED` removed); the
    default, full form is what every reply is validated against before anything runs."""
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "tool": {"const": name},
                    "arguments": _wire(spec.parameters) if wire else spec.parameters,
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            }
            for name, spec in specs.items()
        ]
    }


def request_fields(
    protocol: str,
    kind: str,
    serving: str,
    specs: Mapping[str, ToolSpec],
    *,
    thinking: Any = None,
    allowance: int = 0,
) -> dict[str, Any]:
    """The body fields that carry the tools, per mode and endpoint kind, and the thinking switch
    (:func:`thinking_fields`) in either mode."""
    return {
        **_tool_fields(protocol, kind, serving, specs),
        **thinking_fields(thinking, kind, serving, allowance, protocol),
    }


def _tool_fields(
    protocol: str, kind: str, serving: str, specs: Mapping[str, ToolSpec]
) -> dict[str, Any]:
    if protocol == STRUCTURED:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "one_action",
                    "schema": one_action_schema(specs, wire=True),
                },
            }
        }
    if kind == ANTHROPIC:
        return {
            "tools": [
                {"name": s.name, "description": s.description, "input_schema": s.parameters}
                for s in specs.values()
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
    fields: dict[str, Any] = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in specs.values()
        ],
        "tool_choice": "auto",
    }
    if serving == MISERVER:
        fields["parallel_tool_calls"] = False  # F18-D11: one call per reply from the server too
    return fields


def structured_tool_list(specs: Mapping[str, ToolSpec]) -> str:
    """The tools, for the structured-mode system prompt (no ``tools`` field is sent)."""
    return "\n".join(
        f"- {s.name}: {s.description} arguments: {json.dumps(s.parameters, sort_keys=True)}"
        for s in specs.values()
    )


def one_action(reply: Reply, protocol: str, specs: Mapping[str, ToolSpec]) -> Action | Refusal:
    """Exactly one action from ``reply``, or the refusal that says why not. Never runs anything."""
    if protocol == STRUCTURED:
        return _structured(reply, specs)
    calls = reply.tool_calls
    if not calls:
        return Refusal("no_action")
    if len(calls) > 1:
        return Refusal("multiple_actions", len(calls))
    call = calls[0]
    args = call.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return Refusal("unparseable", 1, f"the arguments of {call.name!r} are not JSON")
    if not isinstance(args, dict):
        return Refusal("unparseable", 1, f"the arguments of {call.name!r} are not an object")
    if call.name not in specs:
        return Refusal("not_offered", 1, call.name)
    return Action(call.name, args, call.id)


def _structured(reply: Reply, specs: Mapping[str, ToolSpec]) -> Action | Refusal:
    # ``reply.text`` only: the reasoning is never read here, even when it holds a valid action.
    text = (reply.text or "").strip()
    if not text:
        return Refusal("no_action")
    try:
        parsed = json.loads(text)
    except ValueError:
        return Refusal("unparseable", 1, "the reply is not JSON")
    if isinstance(parsed, list):
        return Refusal("multiple_actions", len(parsed)) if len(parsed) > 1 else _one(parsed, specs)
    return _one(parsed, specs)


def _one(parsed: Any, specs: Mapping[str, ToolSpec]) -> Action | Refusal:
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    tool = parsed.get("tool") if isinstance(parsed, dict) else None
    if isinstance(tool, str) and tool not in specs:
        return Refusal("not_offered", 1, tool)
    errors = validate(parsed, one_action_schema(specs))
    if errors:
        return Refusal("unparseable", 1, "; ".join(errors[:3]))
    assert isinstance(parsed, dict)
    return Action(str(parsed["tool"]), dict(parsed["arguments"]), None)


def followups(
    protocol: str, reply: Reply, decision: Action | Refusal, output: str
) -> list[dict[str, Any]]:
    """The messages that answer ``reply``: ``output`` for the action run, the refusal's text for
    every call that was not run (each tool call must be answered, or the next request is
    malformed), or a user message when the reply carried no call."""
    if protocol == TOOL_CALLS and reply.tool_calls:
        ran = decision.call_id if isinstance(decision, Action) else None
        refused = decision.text() if isinstance(decision, Refusal) else ""
        return [
            {
                "role": "tool",
                "tool_call_id": c.id,
                "name": c.name,
                "content": output if c.id == ran else (refused or prompts.MULTIPLE_ACTIONS),
            }
            for c in reply.tool_calls
        ]
    if isinstance(decision, Action):
        return [{"role": "user", "content": f"Result of {decision.name}:\n{output}"}]
    return [{"role": "user", "content": output or decision.text()}]
