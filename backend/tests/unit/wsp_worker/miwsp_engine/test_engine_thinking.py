"""``miwsp_engine``'s thinking switch (2026-11-21; ``0xcc/reviews/018_thinking_2026-11-21.md``).

- The frozen ``thinking`` decides the request: ``on`` sends ``chat_template_kwargs:
  {enable_thinking: true}`` and the reasoning allowance (``REASONING_ALLOWANCE_FIELD``), and the
  cap it names is the allowance PLUS the action's cap; ``off`` sends ``enable_thinking: false``;
  ``null`` sends nothing. Both protocols (``tool_calls``, ``structured``).
- The reply's reasoning (``reasoning_content``, or a leading ``<think>`` block in ``content``)
  is recorded on the transcript's assistant line as ``reasoning {text, source}``, counted in
  ``engine.reasoning_turns``, NEVER parsed as an action and never sent back.
- The allowance is bounded by the window, and the context guard counts it.
- A long-thinking turn is in the graceful end's longest-turn figure (step 4a).

In-process on a scripted client that answers in the OpenAI wire shape through the client's own
parser (``client.openai_reply``), so the parsing path is the production one; plus one real HTTP
round trip under the worker's interpreter (``engine_cases.py``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, result, run_case, transcript
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.support.wsp_scripted_endpoint import ScriptedEndpoint
from wsp_worker.miwsp_engine import context, prompts
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.client import openai_messages, openai_reply, split_reasoning
from wsp_worker.miwsp_engine.tools import ForwardBudget, Toolbox, submit_spec

#: FU-3: the end of an edit session's limit instruction (the weights as they stand are kept).
KEPT = prompts.SUBMIT_NOW_EDIT.format(why="").lstrip()

SPECS = Toolbox([submit_spec(lambda answer, confidence=None: "ok")], ForwardBudget(0)).specs
CTK = "chat_template_kwargs"
FIELD = proto.REASONING_ALLOWANCE_FIELD
SECRET_ACTION = {"tool": "submit", "arguments": {"answer": "FROM-THE-REASONING"}}


class FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class WireClient:
    """Answers each request with the next scripted OpenAI ``message`` (parsed by the client's
    own ``openai_reply``) and records what the session asked for."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        clock: FakeTime | None = None,
        request_s: list[float] | None = None,
    ) -> None:
        self.messages = messages
        self.clock, self.request_s = clock, request_s or []
        self.asked: list[dict[str, Any]] = []

    def complete(
        self, messages: Sequence[Mapping[str, Any]], *, fields: Mapping[str, Any], max_tokens: int
    ) -> proto.Reply:
        n = len(self.asked)
        self.asked.append(
            {"messages": [dict(m) for m in messages], "fields": dict(fields), "max": max_tokens}
        )
        if self.clock is not None and n < len(self.request_s):
            self.clock.now += self.request_s[n]
        message = self.messages[min(n, len(self.messages) - 1)]
        return openai_reply(
            {
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        )

    def body(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}


def call(name: str, args: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": extra.pop("content", None),
        "tool_calls": [
            {
                "id": f"c{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
        **extra,
    }


SUBMIT = call("submit", {"answer": "done"})


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _session(tmp: Path, fx: FixtureBundle, client: Any, **over: Any) -> session_mod.Session:
    line = line_for(tmp / "run", fx, "repair", None, python=False, **over)
    s = session_mod.Session(line, client=client)
    s.writer = None
    return s


def _drive(s: session_mod.Session) -> None:
    """``run``'s thinking step and loop, without its isolation step (never in a test process)."""
    s._set_thinking()
    s._loop()


# ----------------------------------------------------------------------------- the request


@pytest.mark.parametrize("protocol", [proto.TOOL_CALLS, proto.STRUCTURED])
def test_thinking_on_sends_enable_thinking_and_the_allowance_in_both_protocols(
    tmp_path: Path, fx: FixtureBundle, protocol: str
) -> None:
    reply = (
        SUBMIT
        if protocol == proto.TOOL_CALLS
        else {
            "role": "assistant",
            "content": json.dumps({"tool": "submit", "arguments": {"answer": "done"}}),
        }
    )
    client = WireClient([reply])
    s = _session(
        tmp_path,
        fx,
        client,
        thinking="on",
        thinking_tokens=3000,
        editor={"action_protocol": protocol, "context_window_tokens": 32768},
    )
    _drive(s)
    (asked,) = client.asked
    assert asked["fields"][CTK] == {"enable_thinking": True}
    # The allowance field goes beside response_format only (miServer refuses it with tools);
    # either way the cap asked for is the allowance PLUS the action's cap.
    assert asked["fields"].get(FIELD) == (3000 if protocol == proto.STRUCTURED else None)
    assert asked["max"] == s.cap + 3000 == 2048 + 3000
    assert ("response_format" in asked["fields"]) == (protocol == proto.STRUCTURED)
    assert s.guard.cap == s.cap + 3000  # the guard counts the whole answer


def test_the_guard_counts_the_allowance_before_the_first_request(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The guard is sized for reasoning plus action by the thinking step itself, before anything
    is sent (018 16.2, control T-M5b). The test above reads ``s.guard.cap`` after the loop, and
    the loop re-sizes the guard after every reply, so deleting the thinking step's own sizing
    passed it: the first request was then checked against the action's cap alone."""
    s = _session(
        tmp_path,
        fx,
        WireClient([SUBMIT]),
        thinking="on",
        thinking_tokens=3000,
        editor={"context_window_tokens": 32768},
    )
    s._set_thinking()
    assert s.allowance == 3000
    assert s.guard.cap == s.cap + 3000  # before the first request, not only after a reply


def test_thinking_off_sends_enable_thinking_false_and_the_no_thinking_prompt(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    client = WireClient([SUBMIT])
    s = _session(tmp_path, fx, client, thinking="off")
    _drive(s)
    (asked,) = client.asked
    assert asked["fields"][CTK] == {"enable_thinking": False}
    assert FIELD not in asked["fields"] and asked["max"] == s.cap
    assert prompts.NO_THINKING in asked["messages"][0]["content"]


def test_a_null_thinking_sends_nothing(tmp_path: Path, fx: FixtureBundle) -> None:
    client = WireClient([SUBMIT])
    s = _session(tmp_path, fx, client, thinking=None)
    _drive(s)
    (asked,) = client.asked
    assert CTK not in asked["fields"] and FIELD not in asked["fields"]
    assert prompts.NO_THINKING not in asked["messages"][0]["content"]


@pytest.mark.parametrize(
    ("kind", "serving"), [("openai_compatible", "generic"), ("anthropic", "generic")]
)
def test_off_sends_nothing_where_the_gateway_would_refuse_it(kind: str, serving: str) -> None:
    fields = proto.request_fields(proto.TOOL_CALLS, kind, serving, SPECS, thinking="off")
    assert CTK not in fields and FIELD not in fields


def test_thinking_on_for_an_endpoint_that_cannot_carry_it_ends_the_session_by_name(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    client = WireClient([SUBMIT])
    s = _session(tmp_path, fx, client, thinking="on", serving="generic")
    with pytest.raises(proto.ThinkingUnsupportedError) as caught:
        _drive(s)
    assert caught.value.code == "wsp_endpoint_thinking_unsupported" and client.asked == []


def test_a_window_too_small_to_think_is_refused_by_name() -> None:
    with pytest.raises(context.ThinkingDoesNotFitError):
        context.thinking_allowance(1024, 300, 4096)  # half the window is 512: 212 is left


@pytest.mark.parametrize(
    ("window", "cap", "asked", "allowance"),
    [
        (32768, 2048, None, context.DEFAULT_THINKING_TOKENS),
        (32768, 2048, 20000, 32768 // 2 - 2048),
        (8192, 1024, None, 4096 - 1024),
        (8192, 1024, 1000, 1000),
    ],
)
def test_the_allowance_is_bounded_by_the_window(
    window: int, cap: int, asked: int | None, allowance: int
) -> None:
    assert context.thinking_allowance(window, cap, asked) == allowance


# ----------------------------------------------------------------------------- the reply


@pytest.mark.parametrize(
    ("content", "reasoning_content", "text", "reasoning"),
    [
        ("answer", "I thought", "answer", {"text": "I thought", "source": "reasoning_content"}),
        (
            '<think>hmm</think>\n{"a": 1}',
            None,
            '{"a": 1}',
            {"text": "hmm", "source": "inline_think"},
        ),
        ("  <think>\nx\n</think>  ok", None, "ok", {"text": "x", "source": "inline_think"}),
        (
            "half a thought</think>ok",
            None,
            "ok",
            {"text": "half a thought", "source": "inline_think"},
        ),
        (
            '<think>never closed {"tool": "submit"}',
            None,
            "",
            {"text": 'never closed {"tool": "submit"}', "source": "inline_think"},
        ),
        ("plain", None, "plain", None),
        (None, "", None, None),
        ("<think></think>done", None, "done", None),
    ],
)
def test_split_reasoning(
    content: str | None, reasoning_content: Any, text: str | None, reasoning: Any
) -> None:
    assert split_reasoning(content, reasoning_content) == (text, reasoning)


@pytest.mark.parametrize("protocol", [proto.STRUCTURED, proto.TOOL_CALLS])
@pytest.mark.parametrize("where", ["reasoning_content", "inline"])
def test_an_action_in_the_reasoning_is_never_run(protocol: str, where: str) -> None:
    """The negative control of the guard: the reasoning holds a VALID action and the content
    none. It is refused ``no_action``; only ``content`` / ``tool_calls`` are ever parsed."""
    action = json.dumps(SECRET_ACTION)
    message: dict[str, Any] = (
        {"role": "assistant", "content": "", "reasoning_content": action}
        if where == "reasoning_content"
        else {"role": "assistant", "content": f"<think>{action}</think>"}
    )
    reply = openai_reply({"choices": [{"message": message}], "usage": {}})
    assert reply.reasoning == action
    decision = proto.one_action(reply, protocol, SPECS)
    assert isinstance(decision, proto.Refusal) and decision.kind == "no_action"


def test_the_reasoning_is_recorded_counted_and_never_sent_back(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    thought = "The bias looks shifted by one; I will read it first. " * 20
    replies = [
        call("list_tensors", {}, reasoning_content=thought),
        call("submit", {"answer": "done"}, content=f"<think>{thought}</think>"),
    ]
    client = WireClient(replies)
    line = line_for(tmp_path / "run", fx, "repair", None, python=False, thinking="on")
    s = session_mod.Session(line, client=client)
    s.writer = session_mod.TranscriptWriter(s.out / "transcript.jsonl", {"x": 1})
    _drive(s)
    s.writer.close()
    assert s.reasoning_turns == 2 and s.result()["engine"]["reasoning_turns"] == 2
    lines = transcript(tmp_path / "run")
    assistants = [m for m in lines if m.get("role") == "assistant"]
    assert [a["reasoning"]["source"] for a in assistants] == ["reasoning_content", "inline_think"]
    assert all(a["reasoning"]["text"].strip() == thought.strip() for a in assistants)
    assert assistants[1]["content"] == ""  # the think block is out of the parsed text
    (event,) = [m for m in lines if m.get("event") == "thinking"]
    assert event["reasoning_allowance"] == s.allowance and event["sent"][CTK] == {
        "enable_thinking": True
    }
    # The second request carried the first reply back WITHOUT its reasoning, on the wire...
    wire = openai_messages(client.asked[1]["messages"])
    assert thought not in json.dumps(wire)
    # ...and the guard did not count its bytes (it is never sent).
    sent = client.asked[1]["messages"]
    assert context.message_bytes(sent) == context.message_bytes(
        [{k: v for k, v in m.items() if k != "reasoning"} for m in sent]
    )
    assert any("reasoning" in m for m in sent)


def test_a_replay_counts_the_same_reasoning_and_reproduces(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    from wsp_worker.miwsp_engine.transcript import read_script

    replies = [call("submit", {"answer": "done"}, reasoning_content="why")]
    first = tmp_path / "a"
    line = line_for(first, fx, "repair", None, python=False, thinking="on")
    s = session_mod.Session(line, client=WireClient(replies))
    s.writer = session_mod.TranscriptWriter(s.out / "transcript.jsonl", {"x": 1})
    _drive(s)
    s.writer.close()
    script = read_script(s.out / "transcript.jsonl")
    again = line_for(tmp_path / "b", fx, "repair", None, python=False, thinking="on")
    r = session_mod.Session(again, client=script)
    r.writer = None
    r.replaying = True
    r._loop()
    assert (r.reasoning_turns, r.answer, r.replay_divergence) == (1, ("done", None), None)


# ----------------------------------------------------------------------------- the time


def test_a_long_thinking_turn_is_the_longest_turn_and_moves_the_stop(
    tmp_path: Path, fx: FixtureBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 4a with thinking: a 150 s thinking turn against 400 s left. After it, the next turn
    plus the final one (150 + 150 + 10 s) would cross the kill at 250 s left, so the session
    stops ``time_budget`` and offers ``submit`` alone, instead of starting a second long think."""
    clock = FakeTime()
    monkeypatch.setattr(session_mod, "time", clock)
    replies = [call("list_tensors", {}, reasoning_content="long"), SUBMIT]
    client = WireClient(replies, clock, request_s=[150.0, 1.0])
    s = _session(tmp_path, fx, client, thinking="on", time_s=600, time_left_s=400)
    _drive(s)
    assert s._longest_turn >= 150.0 and s._longest_request >= 150.0
    assert s.stop == "time_budget" and len(client.asked) == 2
    assert client.asked[1]["messages"][-1]["content"].endswith(KEPT)


def test_a_long_thinking_turn_makes_a_forward_pass_that_would_not_fit_refused(
    tmp_path: Path, fx: FixtureBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forward guard keeps the final turn's reserve, which is the longest REQUEST: after a
    100 s thinking request, a 30 s forward pass with 135 s left would end inside it."""
    clock = FakeTime()
    monkeypatch.setattr(session_mod, "time", clock)
    tools = [{"name": "practice_metrics", "version": "2", "cost": 1}]
    replies = [
        call("forward_practice_metrics", {"split": "train"}, reasoning_content="think"),
        SUBMIT,
    ]
    client = WireClient(replies, clock, request_s=[100.0, 1.0])
    s = _session(
        tmp_path,
        fx,
        client,
        thinking="on",
        time_s=600,
        time_left_s=235,
        forward_passes=3,
        observation_tools=tools,
    )
    s._forward_prior_s = 30.0
    _drive(s)
    refused = next(m["content"] for m in client.asked[1]["messages"] if m.get("role") == "tool")
    assert refused.startswith("Not run: about 135 s are left")
    assert s.forward.used == 0


# ----------------------------------------------------------------------------- the wire


def test_thinking_reaches_a_real_endpoint_and_the_reasoning_comes_back(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The real child (``engine_cases.py``) against a real HTTP endpoint: what the SERVER
    received carries ``enable_thinking`` and the allowance under the cap field, and the
    ``reasoning_content`` it sent is on the transcript and in ``reasoning_turns``."""
    steps = [
        {"calls": [("list_tensors", {})], "reasoning": "first I looked"},
        {"calls": [("submit", {"answer": "done"})], "reasoning": "then I was sure"},
    ]
    run = tmp_path / "run"
    with ScriptedEndpoint("openai_compatible", steps) as ep:
        line = line_for(
            run,
            fx,
            "repair",
            ep.gateway,
            thinking="on",
            thinking_tokens=2000,
            editor={"context_window_tokens": 32768},
        )
        seen = run_case(run, line)
    body = ep.requests[0]["body"]
    assert FIELD not in body  # tool-calls mode: no allowance field (miServer's tool path has none)
    # The second request carries the first reply back WITHOUT its reasoning, as the server saw it.
    assert "first I looked" not in json.dumps(ep.requests[1]["body"])
    # The child log is named fields only: no reasoning in it.
    assert "first I looked" not in seen["stdout"] and "then I was sure" not in seen["stdout"]
    assert body[CTK] == {"enable_thinking": True}
    assert body["max_tokens"] == 2048 + 2000
    res = result(run)
    assert res["engine"]["reasoning_turns"] == 2 and res["stop_reason"] == "submitted"
    assistants = [m for m in transcript(run) if m.get("role") == "assistant"]
    assert [a["reasoning"] for a in assistants] == [
        {"text": "first I looked", "source": "reasoning_content"},
        {"text": "then I was sure", "source": "reasoning_content"},
    ]


def test_the_reported_reasoning_tokens_are_recorded_and_summed() -> None:
    """miServer's ``usage.completion_tokens_details.reasoning_tokens``: on the assistant line and
    summed in ``result.json``'s tokens, a share of the output tokens, never added to them."""
    reply = openai_reply(
        {
            "choices": [{"message": SUBMIT}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 900,
                "completion_tokens_details": {"reasoning_tokens": 850},
            },
        }
    )
    assert reply.usage == proto.Usage(100, 900, 0, 0, 850) and reply.usage.total == 1000
    assert reply.message["reasoning_tokens"] == 850
    assert "reasoning_tokens" in context.NOT_SENT
    assert "850" not in json.dumps(openai_messages([reply.message]))


def test_the_session_sums_the_reasoning_tokens(tmp_path: Path, fx: FixtureBundle) -> None:
    class Counted(WireClient):
        def complete(self, messages: Any, *, fields: Any, max_tokens: int) -> proto.Reply:
            reply = super().complete(messages, fields=fields, max_tokens=max_tokens)
            reply.usage = proto.Usage(100, 50, 0, 0, 40)
            return reply

    s = _session(tmp_path, fx, Counted([call("list_tensors", {}), SUBMIT]), thinking="on")
    _drive(s)
    assert s.result()["tokens"]["reasoning_tokens"] == 80


# --------------------------------------- feature 020: hosted thinking, caching and the cap (H-9)

HOSTED_A = {"locality": "external", "history_mode": "append_only", "window_tokens": 200_000}


def _hosted_session(
    tmp: Path, fx: FixtureBundle, kind: str, steps: list[dict[str, Any]], **over: Any
) -> ScriptedEndpoint:
    editor = {**HOSTED_A, **over.pop("editor", {})}
    with ScriptedEndpoint(kind, steps) as ep:
        line = line_for(
            tmp,
            fx,
            "repair",
            ep.gateway,
            editor_kind=kind,
            serving="generic",
            editor=editor,
            **over,
        )
        seen = run_case(tmp, line)
    assert seen["exit"] == 0
    return ep


HOSTED_SUBMIT = {"calls": [("submit", {"answer": "done"})]}
HOSTED_LIST = {"calls": [("list_tensors", {})], "thinking": "plan"}


@pytest.mark.parametrize(
    ("thinking", "config"),
    [("medium", {"effort": "medium"}), ("high", {"effort": "high"}), (None, None)],
)
def test_a_hosted_anthropic_request_asks_for_adaptive_thinking_and_its_effort(
    tmp_path: Path, fx: FixtureBundle, thinking: Any, config: Any
) -> None:
    ep = _hosted_session(tmp_path, fx, "anthropic", [HOSTED_LIST, HOSTED_SUBMIT], thinking=thinking)
    bodies = [r["body"] for r in ep.requests]
    assert len(bodies) == 2
    for body in bodies:
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body.get("output_config") == config  # none at default (F20-D13's rows too)
        assert "chat_template_kwargs" not in body and "temperature" not in body
    # The thinking block comes back byte for byte in the next request.
    assert bodies[1]["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "plan",
        "signature": "sig-0",
    }
    # F20-D13: a default-effort row's system text has no thinking-off sentence.
    assert prompts.NO_THINKING not in json.dumps(bodies[0]["system"])


def test_an_append_only_anthropic_body_carries_both_cache_markers(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    ep = _hosted_session(tmp_path, fx, "anthropic", [HOSTED_LIST, HOSTED_SUBMIT], thinking="low")
    for request in ep.requests:
        body = request["body"]
        assert body["cache_control"] == {"type": "ephemeral"}
        [block] = body["system"]
        assert block["cache_control"] == {"type": "ephemeral"} and block["type"] == "text"


def test_a_paging_anthropic_body_carries_no_cache_marker(tmp_path: Path, fx: FixtureBundle) -> None:
    ep = _hosted_session(
        tmp_path,
        fx,
        "anthropic",
        [{"calls": [("list_tensors", {})]}, HOSTED_SUBMIT],
        editor={"history_mode": "paging", "locality": "local"},
    )
    for request in ep.requests:
        body = request["body"]
        assert "cache_control" not in json.dumps(body)
        assert isinstance(body["system"], str)
        assert "thinking" not in body  # a local Anthropic-kind server gets no adaptive thinking


def test_a_hosted_openai_request_carries_its_declared_effort_on_every_request_and_the_retry(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The final turn's retry (an unreadable submit) sends the same thinking fields (FTDD §0
    item 2): it turns thinking off only when thinking is ``on``."""
    steps = [
        {"calls": [("list_tensors", {})]},
        {"calls": [("submit", "not an object")]},  # unparseable on the final turn
        HOSTED_SUBMIT,  # its one retry
    ]
    ep = _hosted_session(tmp_path, fx, "openai_compatible", steps, thinking="medium", max_steps=1)
    bodies = [r["body"] for r in ep.requests]
    assert len(bodies) == 3
    assert [b.get("reasoning_effort") for b in bodies] == ["medium"] * 3
    assert all("chat_template_kwargs" not in b for b in bodies)


@pytest.mark.parametrize(
    ("window", "ceiling", "cap"),
    [(200_000, None, 32_000), (200_000, 64_000, 64_000), (100_000, 64_000, 50_000)],
)
def test_the_hosted_anthropic_cap_is_the_ceiling_bounded_by_the_window(
    tmp_path: Path, fx: FixtureBundle, window: int, ceiling: Any, cap: int
) -> None:
    editor: dict[str, Any] = {"window_tokens": window}
    if ceiling is not None:
        editor["output_ceiling_tokens"] = ceiling
    ep = _hosted_session(tmp_path, fx, "anthropic", [HOSTED_SUBMIT], editor=editor)
    assert ep.requests[0]["body"]["max_tokens"] == cap


def test_a_local_endpoints_cap_is_unchanged(tmp_path: Path, fx: FixtureBundle) -> None:
    ep = _hosted_session(
        tmp_path,
        fx,
        "anthropic",
        [HOSTED_SUBMIT],
        editor={"locality": "local", "history_mode": "paging", "window_tokens": 200_000},
    )
    assert ep.requests[0]["body"]["max_tokens"] == context.DEFAULT_OUTPUT_CAP


def test_an_effort_on_a_local_endpoint_ends_the_session_by_name() -> None:
    with pytest.raises(proto.ThinkingUnsupportedError):
        proto.check_thinking_supported("medium", "openai_compatible", "generic", "local")
    proto.check_thinking_supported("medium", "anthropic", "generic", "external")
    proto.check_thinking_supported(None, "anthropic", "generic", "external")


def test_the_engine_and_the_backend_and_the_gateway_name_the_same_efforts() -> None:
    from wsp.models import EFFORTS
    from wsp_gateway import allowlist

    assert tuple(proto.EFFORTS) == EFFORTS
    assert set(EFFORTS) == set(allowlist.ANTHROPIC_EFFORTS) == set(allowlist.HOSTED_EFFORTS)
