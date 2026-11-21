"""One action per turn, in both modes (feature 018; FTASKS 3.1, 3.5, 3.6; FR-018.5, FR-018.7).

In the backend's interpreter: ``protocol.py`` and ``tools.py`` import numpy and the worker only.
The session's tool specs come from a REAL ``Session`` built on a fixture bundle (never run), so
what is walked here is exactly what the engine offers. Its request bodies are checked against
the gateway's own ``allowlist.rebuild`` (stdlib-only, in this venv).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import make_bundle
from wsp_gateway import allowlist
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine import tools as tools_mod
from wsp_worker.miwsp_engine.client import GatewayClient
from wsp_worker.miwsp_engine.session import Session

GATEWAY = {"ip": "127.0.0.1", "port": 9, "token": "t"}
OBSERVE = [
    {"name": "practice_metrics", "version": "2", "cost": 1},
    {"name": "stratified_false_alarms", "version": "1", "cost": 1},
    {"name": "direction_cosines", "version": "1", "cost": 1},
    {"name": "firing_positions", "version": "1", "cost": 1},
    {"name": "compare_to_parent", "version": "2", "cost": 2},
]
READ = ["list_tensors", "describe_tensor", "read_tensor"]
EDIT = ["write_tensor", "diff", "checkpoint", "restore", "revert"]


def _session(tmp: Path, kind: str, **over: Any) -> Session:
    fx = make_bundle(tmp / "bundle", rule="mean")
    return Session(line_for(tmp / kind, fx, kind, GATEWAY, observation_tools=OBSERVE, **over))


@pytest.fixture(scope="module")
def sessions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Session]:
    tmp = tmp_path_factory.mktemp("protocol")
    return {
        "inspect": _session(tmp / "i", "inspect"),
        "repair": _session(tmp / "r", "repair"),
        "create": _session(tmp / "c", "create"),
        "program": _session(tmp / "p", "repair", mode="program"),
        "no_python": _session(tmp / "n", "repair", python=False),
    }


# ----------------------------------------------------------------------------- 3.1 tool sets

FORWARD = [f"forward_{t['name']}" for t in OBSERVE]


@pytest.mark.parametrize(
    ("which", "expected"),
    [
        ("inspect", [*READ, "python", "submit", *FORWARD]),
        ("repair", [*READ, "python", *EDIT, "submit", *FORWARD]),
        ("create", [*READ, "python", *EDIT, "submit", *FORWARD]),
        ("program", [*READ, "python", "submit", *FORWARD, "test_program", "submit_program"]),
        ("no_python", [*READ, *EDIT, "submit", *FORWARD]),
    ],
)
def test_each_operation_offers_exactly_its_tools(
    sessions: dict[str, Session], which: str, expected: list[str]
) -> None:
    assert sessions[which].toolbox.names() == expected


def test_every_tool_schema_stays_inside_the_subset_miserver_enforces(
    sessions: dict[str, Session],
) -> None:
    walked = 0
    for session in sessions.values():
        for spec in session.toolbox.specs.values():
            assert tools_mod.keywords_outside_subset(spec.parameters) == [], spec.name
            assert spec.parameters.get("additionalProperties") is False, spec.name
            walked += 1
        schema = proto.one_action_schema(session.toolbox.specs)
        assert tools_mod.keywords_outside_subset(schema) == []
    assert walked > 40  # the walk saw every session's tools


def test_the_subset_walk_names_a_keyword_outside_it() -> None:
    """The walk's own control: a planted ``pattern`` is reported, with its path."""
    planted = {"type": "object", "properties": {"x": {"type": "string", "pattern": "a+"}}}
    assert tools_mod.keywords_outside_subset(planted) == ["$.properties.x.pattern"]


def test_a_forward_tool_is_charged_only_when_it_returns(tmp_path: Path) -> None:
    """3.3: a raising call costs nothing; exhaustion answers weightspace's text, runs nothing."""
    calls: list[str] = []

    def ok() -> dict[str, int]:
        calls.append("ok")
        return {"v": 1}

    def boom() -> dict[str, int]:
        calls.append("boom")
        raise ValueError("bad split")

    budget = tools_mod.ForwardBudget(2)
    box = tools_mod.Toolbox(
        [
            tools_mod.forward_spec("ok", "d", {"type": "object", "properties": {}}, 1, ok),
            tools_mod.forward_spec("boom", "d", {"type": "object", "properties": {}}, 1, boom),
            tools_mod.forward_spec("big", "d", {"type": "object", "properties": {}}, 3, ok),
        ],
        budget,
    )
    assert box.call("forward_boom", {}) == "ValueError: bad split" and budget.used == 0
    assert box.call("forward_ok", {}).endswith("[forward budget: 1 of 2 left]")
    exhausted = box.call("forward_big", {})
    assert exhausted == (
        "forward budget exhausted (1 left, this probe costs 3). Submit your answer."
    )
    assert calls == ["boom", "ok"] and budget.used == 1


def test_arguments_are_validated_before_a_tool_runs() -> None:
    ran: list[Any] = []
    box = tools_mod.Toolbox(
        [tools_mod.submit_spec(lambda answer, confidence=None: ran.append(answer) or "ok")],
        tools_mod.ForwardBudget(0),
    )
    out = box.call("submit", {"answer": "x", "confidence": 3})
    assert out.startswith("invalid arguments for submit") and "above 1" in out and ran == []
    assert box.call("submit", {"answer": "x", "extra": 1}).startswith("invalid arguments")
    assert box.call("nope", {}).startswith("unknown tool 'nope'")


def test_output_is_truncated_at_twelve_thousand_characters() -> None:
    long = "x" * 15_000
    assert tools_mod.truncate(long) == "x" * 12_000 + "\n... [truncated 3,000 chars]"


# ----------------------------------------------------------------------------- 3.5, 3.6


def _specs(session: Session) -> dict[str, tools_mod.ToolSpec]:
    return session.toolbox.specs


def _reply(*calls: tuple[str, Any], text: str | None = None) -> proto.Reply:
    return proto.Reply(text, [proto.ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)])


def test_one_tool_call_is_one_action(sessions: dict[str, Session]) -> None:
    got = proto.one_action(
        _reply(("python", '{"code": "1"}')), proto.TOOL_CALLS, _specs(sessions["repair"])
    )
    assert got == proto.Action("python", {"code": "1"}, "c0")


def test_a_two_action_reply_is_refused_and_names_the_count(sessions: dict[str, Session]) -> None:
    """The negative control of FR-018.7: two calls, nothing returned to run."""
    reply = _reply(("python", {"code": "W['weight'][0] = 1"}), ("submit", {"answer": "a"}))
    got = proto.one_action(reply, proto.TOOL_CALLS, _specs(sessions["repair"]))
    assert got == proto.Refusal("multiple_actions", 2)
    answered = proto.followups(proto.TOOL_CALLS, reply, got, got.text())
    # Every call is answered (or the next request is malformed), and each says nothing ran.
    assert [m["tool_call_id"] for m in answered] == ["c0", "c1"]
    assert all("this reply had 2. Nothing was run." in m["content"] for m in answered)


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        (proto.Reply("just text"), "no_action"),
        (proto.Reply(None, [proto.ToolCall("c", "python", "{not json")]), "unparseable"),
        (proto.Reply(None, [proto.ToolCall("c", "python", "[1, 2]")]), "unparseable"),
        (proto.Reply(None, [proto.ToolCall("c", "fetch_url", "{}")]), "not_offered"),
    ],
    ids=["no-action", "bad-json", "not-an-object", "not-offered"],
)
def test_tool_call_mode_refusals(
    sessions: dict[str, Session], reply: proto.Reply, kind: str
) -> None:
    got = proto.one_action(reply, proto.TOOL_CALLS, _specs(sessions["repair"]))
    assert isinstance(got, proto.Refusal) and got.kind == kind


def test_structured_mode_parses_and_revalidates(sessions: dict[str, Session]) -> None:
    specs = _specs(sessions["repair"])
    good = json.dumps({"tool": "read_tensor", "arguments": {"name": "weight", "index": "0:2"}})
    assert proto.one_action(proto.Reply(good), proto.STRUCTURED, specs) == proto.Action(
        "read_tensor", {"name": "weight", "index": "0:2"}
    )
    cases = {
        "": "no_action",
        "not json": "unparseable",
        json.dumps([{"tool": "diff", "arguments": {}}] * 2): "multiple_actions",
        json.dumps({"tool": "read_tensor", "arguments": {"name": 3}}): "unparseable",
        json.dumps({"tool": "read_tensor", "arguments": {}, "extra": 1}): "unparseable",
        json.dumps({"tool": "fetch_url", "arguments": {}}): "not_offered",
    }
    for text, kind in cases.items():
        got = proto.one_action(proto.Reply(text), proto.STRUCTURED, specs)
        assert isinstance(got, proto.Refusal) and got.kind == kind, text


def test_structured_output_is_accepted_only_for_miserver_openai() -> None:
    proto.check_supported(proto.STRUCTURED, "openai_compatible", "miserver")
    for kind, serving in (("openai_compatible", "generic"), ("anthropic", "miserver")):
        with pytest.raises(proto.ProtocolUnsupportedError):
            proto.check_supported(proto.STRUCTURED, kind, serving)
    with pytest.raises(proto.ProtocolUnsupportedError):
        proto.check_supported("free_text", "openai_compatible", "miserver")


# ----------------------------------------------------------------------------- the gateway


def _body(session: Session, kind: str, serving: str, protocol: str) -> dict[str, Any]:
    specs = session.toolbox.specs
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c0", "name": "python", "arguments": '{"code": "1"}'}],
        },
        {"role": "tool", "tool_call_id": "c0", "name": "python", "content": "1"},
        {"role": "user", "content": "Call `submit` now."},
    ]
    client = GatewayClient(kind, "m", GATEWAY)
    fields = proto.request_fields(protocol, kind, serving, specs)
    return client.body(messages, fields=fields, max_tokens=1024)


@pytest.mark.parametrize(
    ("kind", "serving"), [("openai_compatible", "generic"), ("anthropic", "external")]
)
def test_tool_call_bodies_pass_the_gateway_allowlist_unchanged(
    sessions: dict[str, Session], kind: str, serving: str
) -> None:
    body = _body(sessions["repair"], kind, serving, proto.TOOL_CALLS)
    rebuilt, _headers = allowlist.rebuild(kind, serving, body, {})
    assert rebuilt == body


def test_miserver_bodies_pass_the_gateway_allowlist_unchanged_and_only_for_miserver(
    sessions: dict[str, Session],
) -> None:
    """018-P2 FTASKS 9.3 (FR-018.9, F18-D11; this replaces P1's T-1 pin, which held the refusal
    until the admission landed): for a miServer endpoint the gateway admits the engine's
    ``parallel_tool_calls: false`` (tool-call mode) and ``response_format`` (structured mode) and
    passes each body through unchanged; for a generic endpoint it refuses exactly that field, and
    nothing else: without it, the same body passes."""
    for protocol, field in (
        (proto.TOOL_CALLS, "parallel_tool_calls"),
        (proto.STRUCTURED, "response_format"),
    ):
        body = _body(sessions["repair"], "openai_compatible", "miserver", protocol)
        assert field in body
        rebuilt, _headers = allowlist.rebuild("openai_compatible", "miserver", body, {})
        assert rebuilt == body
        with pytest.raises(allowlist.FieldRefused) as refused:
            allowlist.rebuild("openai_compatible", "generic", body, {})
        assert refused.value.field == field
        without = {k: v for k, v in body.items() if k != field}
        assert allowlist.rebuild("openai_compatible", "generic", without, {})[0] == without
    structured = _body(sessions["repair"], "openai_compatible", "miserver", proto.STRUCTURED)
    assert "tools" not in structured and "stop" not in structured


def _keys(schema: object) -> set[str]:
    """Every keyword in ``schema`` at any depth (property NAMES excluded)."""
    found: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                for sub in value.values():
                    found |= _keys(sub)
            else:
                found.add(key)
                found |= _keys(value)
    elif isinstance(schema, list):
        for item in schema:
            found |= _keys(item)
    return found


def test_the_sent_schema_carries_no_numeric_bounds_and_replies_are_still_checked_against_them(
    sessions: dict[str, Session],
) -> None:
    """2026-11-20: miServer's llama.cpp engine refuses ``minimum``/``maximum`` (its grammar cannot
    enforce them), so the request leaves them out; the reply is validated against the full schema,
    so an out-of-range ``confidence`` is still refused before anything runs."""
    session = sessions["repair"]
    specs = session.toolbox.specs
    assert "submit" in specs
    full = proto.one_action_schema(specs)
    assert {"minimum", "maximum"} <= _keys(full)  # the bounds exist (submit's confidence)
    sent = proto.request_fields(proto.STRUCTURED, proto.OPENAI, "miserver", specs)
    wire = sent["response_format"]["json_schema"]["schema"]
    assert not ({"minimum", "maximum"} & _keys(wire))
    bad = json.dumps({"tool": "submit", "arguments": {"answer": "x", "confidence": 2}})
    decided = proto._one(json.loads(bad), specs)
    assert isinstance(decided, proto.Refusal) and decided.kind == "unparseable"
    ok = proto._one({"tool": "submit", "arguments": {"answer": "x", "confidence": 0.5}}, specs)
    assert isinstance(ok, proto.Action)
