"""Bodies from ``miwsp_engine`` pass the gateway's allow-list unchanged, INSIDE the worker image
(feature 018; FR-018.10's in-image AC; FTASKS 10.1).

The image's own copies of the engine (``/opt/wsp/wsp_worker/miwsp_engine``) and of the gateway's
``allowlist.py`` (the ``test`` target), in one process: every body the engine's client builds, for
each endpoint kind and each action protocol, is rebuilt by ``allowlist.rebuild`` into an identical
dict. It replaced ``test_allowlist_real_bodies.py`` (weightspace's SDK bodies), deleted when
``weightspace`` and its SDKs left the image (018-P4, FTASKS 15.5): it is the one real-body test.
The tool specs are the engine's own builders (``tools.read_specs``, ``python_spec``,
``edit_specs``, ``submit_spec``, ``forward_spec``);
the conversation carries every message shape the engine sends: system, user, an assistant turn with
a tool call, its tool result, a nudge and an assistant text turn.

Run by ``tests/image/test_wsp_worker_image.py::test_in_container_suite``; anywhere else it skips.
Standard library, numpy, pytest and the image's worker package only.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WSP_IN_IMAGE_TESTS") != "1",
    reason="runs only inside the WSP worker image (tests/image/test_wsp_worker_image.py starts it)",
)

GATEWAY = {"ip": "127.0.0.1", "port": 9, "token": "t"}


def _specs() -> dict[str, Any]:
    from wsp_worker.miwsp_engine import tools

    def nothing(**_kw: Any) -> str:
        return ""

    specs = [
        *tools.read_specs(nothing, nothing, nothing),
        tools.python_spec(nothing, can_edit=True),
        *tools.edit_specs(nothing, nothing, nothing, nothing, nothing),
        tools.submit_spec(nothing),
        tools.forward_spec(
            "practice_metrics",
            "Metrics of the probe on the practice rows.",
            {"type": "object", "properties": {}},
            1,
            nothing,
        ),
    ]
    return {s.name: s for s in specs}


MESSAGES = [
    {"role": "system", "content": "You edit a probe's weights."},
    {"role": "user", "content": "Repair the probe."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c0", "name": "python", "arguments": '{"code": "1"}'}],
    },
    {"role": "tool", "tool_call_id": "c0", "name": "python", "content": "1"},
    {"role": "user", "content": "Continue using the tools, or call `submit` when done."},
    {"role": "assistant", "content": "Thinking about it."},
    {"role": "user", "content": "Call `submit` now."},
]


#: ``(kind, serving, protocol, the field that carries the tools or the schema)``. One test, not a
#: parametrize: ``test_in_container_suite`` counts ``def test_`` lines against the passes.
CASES = [
    ("openai_compatible", "generic", "tool_calls", "tools"),
    ("openai_compatible", "miserver", "tool_calls", "parallel_tool_calls"),
    ("openai_compatible", "miserver", "structured", "response_format"),
    ("anthropic", "generic", "tool_calls", "tools"),
]


def test_the_engines_bodies_pass_the_allowlist_unchanged() -> None:
    from wsp_gateway import allowlist
    from wsp_worker.miwsp_engine import protocol as proto
    from wsp_worker.miwsp_engine.client import GatewayClient

    specs = _specs()
    for kind, serving, protocol, carries in CASES:
        fields = proto.request_fields(protocol, kind, serving, specs)
        body = GatewayClient(kind, "editor-model", GATEWAY).body(
            MESSAGES, fields=fields, max_tokens=1024
        )
        case = (kind, serving, protocol)
        assert carries in body, (case, sorted(body))
        rebuilt, headers = allowlist.rebuild(kind, serving, body, {}, structured_admitted=False)
        assert rebuilt == body, case
        assert headers["content-type"] == "application/json", case
    # Feature 019 P2 (S-5, FR-019.13): the engine's real structured body for a GENERIC endpoint
    # passes the real allow-list when the endpoint is admitted (a current protocol-test pass, or
    # its own test session) and is refused, by that one field, when it is not.
    fields = proto.request_fields("structured", "openai_compatible", "generic", specs)
    body = GatewayClient("openai_compatible", "editor-model", GATEWAY).body(
        MESSAGES, fields=fields, max_tokens=1024
    )
    assert "response_format" in body and "parallel_tool_calls" not in body
    rebuilt, _headers = allowlist.rebuild(
        "openai_compatible", "generic", body, {}, structured_admitted=True
    )
    assert rebuilt == body
    try:
        allowlist.rebuild("openai_compatible", "generic", body, {}, structured_admitted=False)
    except allowlist.FieldRefused as refused:
        assert refused.field == "response_format"
    else:
        raise AssertionError("an unadmitted generic endpoint's response_format was let through")


def test_the_engines_hosted_bodies_pass_the_allowlist_unchanged() -> None:
    """Feature 020 (H-12; FR-020.13, FR-020.16): the engine's real hosted bodies, as its client
    builds them for an append-only session, pass the real allow-list unchanged: the Anthropic
    body with both cache markers, adaptive summarized thinking and each effort (and none at
    ``default``); the OpenAI-compatible body with the effort its endpoint declares. The same
    bodies with a ``ttl`` on the cache marker, or a ``temperature``, are refused."""
    from wsp_gateway import allowlist
    from wsp_worker.miwsp_engine import protocol as proto
    from wsp_worker.miwsp_engine.client import GatewayClient

    specs = _specs()
    for effort in ("low", "medium", "high", None):
        fields = proto.request_fields(
            "tool_calls", "anthropic", "generic", specs, thinking=effort, locality="external"
        )
        body = GatewayClient("anthropic", "claude-x", GATEWAY, cache=True).body(
            MESSAGES, fields=fields, max_tokens=32_000
        )
        assert body["cache_control"] == {"type": "ephemeral"}, effort
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}, effort
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}, effort
        assert body.get("output_config") == (None if effort is None else {"effort": effort})
        rebuilt, headers = allowlist.rebuild(
            "anthropic", "generic", body, {}, structured_admitted=False
        )
        assert rebuilt == body, effort
        assert headers["anthropic-version"] == "2026-10-21"
        for bad, field in (
            ({**body, "cache_control": {"type": "ephemeral", "ttl": "1h"}}, "cache_control"),
            ({**body, "temperature": 1.0}, "temperature"),
        ):
            try:
                allowlist.rebuild("anthropic", "generic", bad, {}, structured_admitted=False)
            except allowlist.FieldRefused as refused:
                assert refused.field == field
            else:
                raise AssertionError(f"{field} was let through")
    fields = proto.request_fields(
        "tool_calls", "openai_compatible", "generic", specs, thinking="medium", locality="external"
    )
    body = GatewayClient("openai_compatible", "oa-x", GATEWAY).body(
        MESSAGES, fields=fields, max_tokens=1024
    )
    assert body["reasoning_effort"] == "medium"
    endpoint = {"id": "ep-oa", "serving": "generic", "thinking": "medium"}
    rebuilt, _headers = allowlist.rebuild(
        "openai_compatible", "generic", body, {}, structured_admitted=False, endpoint=endpoint
    )
    assert rebuilt == body
