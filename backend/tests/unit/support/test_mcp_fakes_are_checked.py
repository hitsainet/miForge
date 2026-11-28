"""MCP fakes are checked (021 FR-021.14; MF19, FTID §8.4).

- The recording client's answer is checked against the contract entry of the route each call
  reaches; a fake missing a required field is recorded as a failure naming the route (NC-F's case,
  as a permanent test), and so fails the test that planted it.
- ``call_tool`` checks the tool's answer against the tool's published output schema.
- A call with no planted answer answers the route's example (``answer_examples``), and the example
  of every route validates.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support import answer_validator
from tests.support.answer_examples import example_answer
from tests.support.mcp_harness import build_harness, call_tool


def _taken() -> list[str]:
    """The failures recorded so far, taken: recorded on purpose here, so charged to no test (the
    report hook reads them after the test body, before fixture teardown)."""
    found = [str(f) for f in answer_validator.FAILURES]
    answer_validator.FAILURES.clear()
    return found


def _every_route_status() -> list[tuple[str, str]]:
    contract = answer_validator.answer_check().contract
    return [
        (key, status)
        for key, entry in sorted(contract["responses"].items())
        for status in sorted(entry)
        if status[:1] == "2" and status not in entry.get("x-no-body", ())
    ]


@pytest.mark.parametrize(("key", "status"), _every_route_status())
def test_the_example_of_every_route_validates(key: str, status: str) -> None:
    check = answer_validator.answer_check()
    assert check.check(key, int(status), example_answer(key, status)) == []


def test_a_fake_missing_a_required_field_is_recorded_naming_its_route() -> None:
    good = example_answer("GET /api/v1/runs/{run_id}")
    mcp, _client = build_harness(response=good)
    call_tool(mcp, "forge_get_run", {"run_id": "run-1"})
    assert _taken() == []
    missing = {k: v for k, v in good.items() if k != "status"}
    mcp, _client = build_harness(response=missing)
    call_tool(mcp, "forge_get_run", {"run_id": "run-1"})
    messages = _taken()
    assert any(
        "GET /api/v1/runs/{run_id}" in m and "a fake answer" in m and "'status'" in m
        for m in messages
    ), messages


def test_a_tool_answer_off_its_output_schema_is_recorded() -> None:
    """A tool answer off its output schema is recorded (the fake check is off, so only the tool
    check speaks): a list where the schema wants an object."""
    mcp, _client = build_harness(response=["not", "an", "object"], check=False)
    call_tool(mcp, "forge_get_run", {"run_id": "run-1"})
    found = _taken()
    assert any("tool forge_get_run" in f for f in found), found


def test_an_unplanted_call_answers_the_routes_example() -> None:
    mcp, _client = build_harness()
    answer = call_tool(mcp, "forge_get_run", {"run_id": "run-1"})
    assert answer == example_answer("GET /api/v1/runs/{run_id}")


def _served_tool(answer: Any) -> Any:
    """The production registry over a MockTransport answering ``answer`` (not the harness)."""
    import asyncio

    import httpx

    from mcp_server.config import MCPSettings
    from mcp_server.health_gate import HealthGate
    from mcp_server.server import build_server

    async def open_gate(self: Any, *_: str) -> None:
        return None

    HealthGate.unavailable = open_gate  # type: ignore[method-assign]
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    client._http = httpx.AsyncClient(
        base_url="http://backend/api/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=answer)),
    )
    return asyncio.run(mcp.call_tool("forge_get_run", {"run_id": "run-1"}))


def test_a_mock_transport_fake_and_its_tool_answer_are_checked_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review round 1, finding 1: every MCP client answer is checked (the validator's plugin wraps
    the client's verbs and ``MCPServer.call_tool``), not only the harness's."""
    from mcp_server.health_gate import HealthGate

    monkeypatch.setattr(HealthGate, "unavailable", HealthGate.unavailable)
    _served_tool(example_answer("GET /api/v1/runs/{run_id}"))
    assert _taken() == []
    _served_tool({"run_id": "run-1"})
    found = _taken()
    assert any("GET /api/v1/runs/{run_id}" in f and "a fake answer" in f for f in found), found
    assert any("tool forge_get_run" in f for f in found), found


@pytest.mark.mcp_fake_is_not_a_route_answer(reason="this test proves the marker turns both off")
def test_the_named_exemption_turns_both_checks_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp_server.health_gate import HealthGate

    monkeypatch.setattr(HealthGate, "unavailable", HealthGate.unavailable)
    _served_tool({"run_id": "run-1"})
    assert _taken() == []


def test_after_an_exempt_test_the_checks_are_on_again() -> None:
    """The marker's effect is per test, never carried into the next (review round 2, F3): set up
    a marked item, then an unmarked one, through the plugin's own setup hook (no reliance on file
    order, review round 3)."""
    from types import SimpleNamespace

    mark = SimpleNamespace(args=(), kwargs={"reason": "a reason"})
    marked = SimpleNamespace(nodeid="t::a", get_closest_marker=lambda name: mark)
    plain = SimpleNamespace(nodeid="t::b", get_closest_marker=lambda name: None)
    answer_validator.pytest_runtest_setup(marked)
    assert answer_validator.fakes_unchecked()
    answer_validator.pytest_runtest_setup(plain)
    assert not answer_validator.fakes_unchecked(), "the exemption leaked into the next test"


def test_a_patched_request_is_still_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that replaces ForgeClient.request is checked too: the verbs are wrapped (review
    round 2, F2)."""
    from mcp_server.client import ForgeClient
    from mcp_server.health_gate import HealthGate

    async def answer(self: Any, method: str, path: str, **_kw: Any) -> Any:
        return {"run_id": "run-1", "planted_key": 1}

    monkeypatch.setattr(HealthGate, "unavailable", HealthGate.unavailable)
    monkeypatch.setattr(ForgeClient, "request", answer)
    _served_tool({})
    found = _taken()
    assert any("GET /api/v1/runs/{run_id}" in f and "a fake answer" in f for f in found), found


def test_the_exemption_needs_its_reason() -> None:
    """A marker without a reason (or with an empty one) errors its test (review rounds 2, 3)."""
    from types import SimpleNamespace

    def item(*args: Any, **kwargs: Any) -> Any:
        mark = SimpleNamespace(args=args, kwargs=kwargs)
        return SimpleNamespace(nodeid="t::x", get_closest_marker=lambda name: mark)

    for missing in (item(), item(""), item(reason="  ")):
        with pytest.raises(pytest.UsageError, match="needs its reason"):
            answer_validator.pytest_runtest_setup(missing)
    answer_validator.pytest_runtest_setup(item(reason="a reason"))
    assert answer_validator.fakes_unchecked()
    answer_validator._STATE["fakes_unchecked"] = False


def test_the_root_conftest_loads_without_the_backend(tmp_path: Any) -> None:
    """The wsp-worker-image job runs backend/tests/image with pytest alone: the root conftest's
    configure must not import the app's dependencies there (review round 2, F1)."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys\n"
        # Absent, as in the pytest-only venv: importing them fails, find_spec answers None.
        "for name in ('fastapi', 'jsonschema', 'referencing', 'mcp', 'mcp_server'):\n"
        "    sys.modules[name] = None\n"
        f"sys.path.insert(0, {str(root.parent)!r})\n"
        "import importlib.util\n"
        "import tests.conftest as c\n"
        "class PM:\n"
        "    def is_registered(self, p): return False\n"
        "    def register(self, p, n): raise SystemExit('registered without the backend')\n"
        "class Config: pluginmanager = PM()\n"
        "c.pytest_configure(Config())\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-1500:]


@pytest.mark.parametrize(
    ("verb", "path", "route"),
    [
        ("get", "/runs/run-1", "GET /api/v1/runs/{run_id}"),
        ("post", "/runs/define", "POST /api/v1/runs/define"),
        ("put", "/settings/log_level", "PUT /api/v1/settings/{key}"),
        ("delete", "/pack-drafts/d-1", "DELETE /api/v1/pack-drafts/{draft_id}"),
    ],
)
def test_each_verb_checks_its_answer(
    verb: str, path: str, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every verb's answer is checked, one failure naming the route (review round 3)."""
    import asyncio

    from mcp_server.client import ForgeClient

    async def answer(self: Any, method: str, p: str, **_kw: Any) -> Any:
        return {"planted_key": 1}

    monkeypatch.setattr(ForgeClient, "request", answer)
    client = ForgeClient("http://forge.test", "agent:forge-mcp")
    call = getattr(client, verb)
    args = (path,) if verb in ("get", "delete") else (path, {})
    asyncio.run(call(*args))
    asyncio.run(client.close())
    found = _taken()
    assert len(found) == 1 and route in found[0] and "a fake answer" in found[0], found


def test_the_wrapped_verbs_are_the_clients_verbs() -> None:
    """No verb of ForgeClient goes unchecked: the wrapped set equals the client's public async
    methods that answer a backend call (review round 3)."""
    import inspect

    from mcp_server.client import ForgeClient

    public = {
        name
        for name, fn in inspect.getmembers(ForgeClient, inspect.iscoroutinefunction)
        if not name.startswith("_") and name not in ("request", "close")
    }
    assert public == set(answer_validator.CLIENT_VERBS)
    assert all(getattr(getattr(ForgeClient, v), "__mf_answer_validator__", False) for v in public)
