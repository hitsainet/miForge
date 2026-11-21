"""Bodies from the REAL producer pass the gateway's allow-list unchanged (feature 016; FTASKS 6.7c;
FR-016.15's real-producer AC; R3A-6, R3A-17).

Runs inside the worker image's ``test`` target, where the pinned ``weightspace``, ``anthropic`` and
``openai`` live beside ``wsp_gateway/allowlist.py`` (standard library only). ``AnthropicChat``
(``cache=True``, non-streaming) runs with a stub client that returns the pinned SDK's own response
types, through two tool turns and a replayed thinking turn; ``OpenAIChat`` through two tool turns.
Every captured request body passes ``allowlist.rebuild`` unchanged; the cap insertion and the
``model`` overwrite are the gateway's own later steps. Control C15s: admit ``cache_control`` on
``text`` blocks only, and the second Anthropic turn (``cache_control`` on a ``tool_result``) goes
red.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

anthropic_types = pytest.importorskip("anthropic.types")
openai_types = pytest.importorskip("openai.types.chat")
llm_anthropic = pytest.importorskip("weightspace.llm.anthropic")
llm_openai = pytest.importorskip("weightspace.llm.openai")

from wsp_gateway import allowlist  # noqa: E402

TOOLS = [
    {
        "name": "run_python",
        "description": "Run code over the weights.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
    }
]


class _Recorder:
    def __init__(self, answers: list[Any]) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._answers = answers

    def create(self, **kw: Any) -> Any:
        self.bodies.append(json.loads(json.dumps(kw)))  # what the SDK would serialise
        return self._answers.pop(0)


def _anthropic_message(content: list[Any], stop: str) -> Any:
    return anthropic_types.Message(
        id="msg",
        type="message",
        role="assistant",
        model="claude-editor",
        content=content,
        stop_reason=stop,
        usage=anthropic_types.Usage(input_tokens=10, output_tokens=5),
    )


def test_anthropic_chat_bodies_pass_unchanged() -> None:
    T = anthropic_types  # noqa: N806
    answers = [
        _anthropic_message(
            [
                T.ThinkingBlock(type="thinking", thinking="plan", signature="sig1"),
                T.TextBlock(type="text", text="first"),
                T.ToolUseBlock(type="tool_use", id="t1", name="run_python", input={"code": "1"}),
            ],
            "tool_use",
        ),
        _anthropic_message(
            [T.ToolUseBlock(type="tool_use", id="t2", name="run_python", input={"code": "2"})],
            "tool_use",
        ),
        _anthropic_message([T.TextBlock(type="text", text="done")], "end_turn"),
    ]
    recorder = _Recorder(answers)
    client = type("Client", (), {"messages": recorder})()
    llm = llm_anthropic.AnthropicChat(
        "claude-editor", client=client, cache=True, stream=False, thinking="high"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You edit weights."},
        {"role": "user", "content": "Repair the probe."},
    ]
    for turn in range(3):
        response = llm.complete(messages, tools=TOOLS)
        messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in response.tool_calls
                ],
                **response.extra,  # the replayed blocks: thinking with its signature
            }
        )
        for call in response.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": f"out {turn}"})
    assert len(recorder.bodies) == 3
    assert any(
        b.get("type") == "tool_result" and "cache_control" in b
        for body in recorder.bodies
        for m in body["messages"]
        for b in m["content"]
    ), "the producer put no cache_control on a tool_result: the test exercised nothing"
    assert any(
        b.get("type") == "thinking" for m in recorder.bodies[-1]["messages"] for b in m["content"]
    ), "no replayed thinking turn"
    for body in recorder.bodies:
        rebuilt, headers = allowlist.rebuild("anthropic", "generic", body, {})
        assert rebuilt == body
        assert headers == {"content-type": "application/json", "anthropic-version": "2023-06-01"}


def _openai_completion(content: str | None, calls: list[tuple[str, str]]) -> Any:
    C = openai_types  # noqa: N806
    tool_calls = [
        C.ChatCompletionMessageToolCall(
            id=i, type="function", function={"name": "run_python", "arguments": a}
        )
        for i, a in calls
    ] or None
    return C.ChatCompletion(
        id="c",
        object="chat.completion",
        created=0,
        model="m",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def test_openai_chat_bodies_pass_unchanged() -> None:
    recorder = _Recorder(
        [
            _openai_completion(None, [("c1", '{"code": "1"}')]),
            _openai_completion("trying", [("c2", '{"code": "2"}')]),
            _openai_completion("done", []),
        ]
    )
    client = type("Client", (), {"chat": type("Chat", (), {"completions": recorder})()})()
    llm = llm_openai.OpenAIChat("m", client=client, base_url="http://gateway:8080/v1")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You edit weights."},
        {"role": "user", "content": "Repair the probe."},
    ]
    for turn in range(3):
        response = llm.complete(messages, tools=TOOLS)
        messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in response.tool_calls
                ],
            }
        )
        for call in response.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": f"out {turn}"})
    assert len(recorder.bodies) == 3
    for body in recorder.bodies:
        rebuilt, _headers = allowlist.rebuild("openai_compatible", "generic", body, {})
        assert rebuilt == body
