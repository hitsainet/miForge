# Origin (shape): miDataworks (Onegaishimas/miDataworks) backend/tests/support/mcp_harness.py
# @ a055a38 - RecordingClient and the caller harness, adapted (012 FTID section 8).
"""The caller harness for MCP tools: one recording client, EVERY module in the registry.

The harness iterates ``CATEGORY_MODULES`` itself, never a hand-picked subset. The recording
client's method signatures must EQUAL ``ForgeClient``'s (``tests/unit/mcp/test_client.py``
compares them with ``inspect.signature``): a stand-in more forgiving than the real client lets
a tool calling it wrongly pass every test and raise ``TypeError`` in production.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from mcp_server.config import MCPSettings
from mcp_server.context import ToolContext
from mcp_server.tools import CATEGORY_MODULES


def _pending_body() -> dict[str, Any]:
    """The gate's real pending body (``approval_service.pending_body``) for a gpu_start request:
    what an agent's call to a gated route answers while the operator decides."""
    from datetime import timedelta

    from models.approval import Approval
    from services import approval_service

    now = approval_service.now()
    return approval_service.pending_body(
        Approval(
            id="apr_test",
            action="gpu_start",
            executor_key="api.v1.endpoints.runs_loop.define_run",
            summary={"route": "POST /api/v1/runs/define"},
            payload={"body": {}},
            request_digest="0" * 64,
            requested_by="agent:forge-mcp",
            status="pending",
            created_at=now,
            expires_at=now + timedelta(hours=23),
        )
    )


PENDING_BODY = _pending_body()


def route_of(method: str, path: str) -> str | None:
    """The contract key a tool's call ``(method, path)`` reaches, or None (021 FR-021.14). A
    literal segment wins over a parameter, as the output-schema generator decides."""
    from tests.support import answer_validator
    from tests.support.mcp_ledger import full, template_matches

    contract = answer_validator.answer_check().contract
    concrete = full(path.split("?", 1)[0])
    keys = [
        k
        for k in (*contract["responses"], *contract["excluded"])
        if k.split(" ", 1)[0] == method and template_matches(k.split(" ", 1)[1], concrete)
    ]
    fewest = min((k.count("{") for k in keys), default=0)
    keys = [k for k in keys if k.count("{") == fewest]
    return keys[0] if len(keys) == 1 else None


def _contract() -> dict[str, Any]:
    from tests.support import answer_validator

    contract: dict[str, Any] = answer_validator.answer_check().contract
    return contract


def check_fake_answer(method: str, path: str, answer: Any) -> None:
    """A fake backend answer must be one the route can give (021 FR-021.14, MF19): valid against
    one of the route's declared answers, or the gate's pending approval on a gated route. A miss is
    recorded with the answer validator's failures, so it fails the test that planted it."""
    from tests.support import answer_validator

    if answer_validator.fakes_unchecked():
        return
    key = route_of(method, path)
    if key is None:
        return
    check = answer_validator.answer_check()
    entry = check.contract["responses"].get(key)
    if entry is None:  # an excluded (byte) route: a fake answers bytes or nothing to check
        return
    statuses = [k for k in entry if k[:1] == "2"]
    if answer == {"status_code": 204} and "204" in entry.get("x-no-body", ()):
        return
    found: list[Any] = []
    for status in statuses:
        found = check.check(key, int(status), answer)
        if not found:
            return
    if not statuses:
        return
    failure = answer_validator.AnswerContractError(
        key, "fake", found[0].json_path if found else "", f"a fake answer: {found[0].message}"
    )
    answer_validator.FAILURES.append(failure)
    answer_validator.SESSION["failures"].append(failure)


#: Routes whose answer's status follows the request (the example answer takes that status).
STATUS_BY_REQUEST = {
    # checkpoints.py:continue_run: `submit: true` submits (202), otherwise the plan (200).
    "POST /api/v1/runs/{run_id}/continue": lambda kw: (
        "202" if ((kw.get("json_body") or {}).get("submit")) else "200"
    ),
}


class RecordingClient:
    """Records ``(method, path, kwargs)`` for every call a tool issues. Its answer is checked
    against the contract entry of the route each call reaches (:func:`check_fake_answer`)."""

    def __init__(self, response: Any = None, *, check: bool = True) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        #: None: answer each call with its route's example answer (``answer_examples``).
        self.response = response
        self.check = check

    async def _record(self, method: str, path: str, **kwargs: Any) -> Any:
        self.calls.append((method, path, kwargs))
        answer = self.response
        if answer is None:
            from tests.support.answer_examples import example_answer

            key = route_of(method, path)
            known = key is not None and key in _contract()["responses"]
            status = STATUS_BY_REQUEST.get(key or "", lambda _kw: None)(kwargs)
            answer = example_answer(key, status) if known and key is not None else {"ok": True}
        if self.check:
            check_fake_answer(method, path, answer)
        return answer

    async def get(self, path: str, **params: Any) -> Any:
        return await self._record("GET", path, **params)

    async def post(self, path: str, json_body: Any = None, **params: Any) -> Any:
        return await self._record("POST", path, json_body=json_body, **params)

    async def put(self, path: str, json_body: Any) -> Any:
        return await self._record("PUT", path, json_body=json_body)

    async def delete(self, path: str, **params: Any) -> Any:
        return await self._record("DELETE", path, **params)


class FakeGate:
    """A health gate that is open, or closed on one dependency."""

    def __init__(self, closed: str | None = None) -> None:
        self.closed = closed
        self.invalidated = 0

    async def unavailable(self, *dependencies: str) -> dict[str, str] | None:
        if self.closed is not None and self.closed in dependencies:
            return {"unavailable": self.closed, "reason": f"{self.closed} down: test"}
        return None

    def invalidate(self) -> None:
        self.invalidated += 1


def build_harness(
    gate: FakeGate | None = None, response: Any = None, *, check: bool = True
) -> tuple[MCPServer, RecordingClient]:
    """Register EVERY module of every category against one recording client. ``check=False``
    only for a test whose fake is deliberately no route's answer (the overlay sentinel)."""
    mcp = MCPServer("harness")
    client = RecordingClient(response, check=check)
    ctx = ToolContext(settings=MCPSettings(auth_token="x" * 32), gate=gate or FakeGate())  # type: ignore[arg-type]
    for modules in CATEGORY_MODULES.values():
        for module in modules:
            module.register(mcp, client, ctx)  # type: ignore[arg-type]
    return mcp, client


def check_tool_answer(name: str, answer: Any) -> None:
    """A tool's answer validates against its published output schema (021 FR-021.14), as
    ``shape_results`` sends it (a non-dict under ``result``). A miss is recorded with the answer
    validator's failures."""
    import jsonschema
    from mcp_types import CallToolResult

    from mcp_server.server import load_output_schemas, structured
    from tests.support import answer_validator

    if answer_validator.fakes_unchecked():
        return
    if isinstance(answer, CallToolResult):  # a server built with shape_results answers this
        if answer.structured_content is None:
            return
        answer = answer.structured_content
    schema = load_output_schemas().get(name)
    if schema is None:
        return
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(structured(answer)))
    if errors:
        failure = answer_validator.AnswerContractError(
            f"tool {name}", "answer", "", f"off its output schema: {errors[0].message[:300]}"
        )
        answer_validator.FAILURES.append(failure)
        answer_validator.SESSION["failures"].append(failure)


def call_tool(mcp: MCPServer, name: str, arguments: dict[str, Any], *, check: bool = True) -> Any:
    """Call a tool through MCPServer's own argument validation and return the raw result. The
    answer is checked against the tool's output schema (:func:`check_tool_answer`)."""
    answer = asyncio.run(mcp._tool_manager.call_tool(name, arguments, Context(mcp_server=mcp)))
    if check:
        check_tool_answer(name, answer)
    return answer


def example_response(request: Any, status: int | None = None) -> Any:
    """An ``httpx.Response`` answering ``request`` with its route's example answer
    (``answer_examples``), for a MockTransport whose test is about the request: so the fake is an
    answer the route can give (021 FR-021.14). An unknown route answers ``{}``."""
    import httpx

    from tests.support.answer_examples import example_answer

    path = request.url.path
    path = path[len("/api/v1") :] if path.startswith("/api/v1/") else path
    key = route_of(request.method, path)
    if key is None or key not in _contract()["responses"]:
        return httpx.Response(status or 200, json={})
    entry = _contract()["responses"][key]
    chosen = str(status) if status else sorted(k for k in entry if k[:1] == "2")[0]
    return httpx.Response(int(chosen), json=example_answer(key, chosen))
