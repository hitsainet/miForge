# Origin: miDataworks (Onegaishimas/miDataworks)
#   backend/tests/support/answer_validator.py @ d12c6ad7
# Mode: adapt (docs/REUSE.md). Kept: the ASGI wrapper on FastAPI.build_middleware_stack, the strict
# check, the failure messages, report mode and its shape recorder. Changed: the route is keyed by
# its endpoint's module.qualname through miForge's route table (FT5: FastAPI 0.159's route path is
# router-relative); a non-2xx status the entry declares is checked too (health's 503); the WSP
# worker's internal app is skipped by name, and any other backend route the contract lacks fails.
"""The answer validator (021 FR-021.12; MF11, FT5): every 2xx JSON answer any test produces is
checked against the committed answer contract, ``docs/schemas/miforge-answers-v1.json``, in its
STRICT form.

**How it is attached.** ``backend/tests/conftest.py`` registers this module as a plugin in its
``pytest_configure``. Its own ``pytest_configure`` replaces
``fastapi.FastAPI.build_middleware_stack`` so that every FastAPI app's stack is wrapped, outermost,
by :class:`ValidatingASGI`: every app ``create_app()`` builds, and any app a test builds, with no
edit to the files that build them. FastAPI builds the stack on an app's first request, after
``pytest_configure``. The app objects stay ``FastAPI`` instances, so a test reading ``.routes``,
``.openapi()`` or ``.dependency_overrides`` sees no difference.

**What a check does,** for an answer whose media type is ``application/json`` and whose status is
2xx or declared by its entry, from a route whose endpoint lives in the backend (``api.``) (a 2xx
answer of another media type, or of none, is checked against ``excluded`` and the entry's
``x-media-alternatives`` and ``x-no-body``: ``AnswerCheck.check_media``):

1. the contract key is the one the endpoint (``module.qualname``) serves in ``create_app()``'s route
   table; a backend route the contract does not hold FAILS, except the WSP worker's internal app
   (``api.wsp_internal_app``), which the contract does not describe; a test's own route is skipped;
2. the entry for the answer's status; an undeclared status FAILS, and so does a JSON answer from an
   ``excluded`` route;
3. the body is validated against the entry's strict form (``contract_tools.strict_form``): an object
   with named properties refuses a key it does not declare; a map accepts any key. The published
   file stays permissive (D3).

The wrapper forwards every message as it comes and only copies the body, so it changes no answer. A
failure is recorded and the test that produced it fails (``pytest_runtest_makereport``), naming the
route, the status, the key path and the model.

**Report mode (FTASKS 2.3, branch only).** While ``REPORT_ONLY`` is True, failures are recorded but
fail nothing; with ``MF_ANSWER_REPORT_DIR`` set, each worker writes its violations and the keys each
route sent to ``answers-<worker>.jsonl`` there (the working list for P3). FTASKS 8.1 turns the
validator strict; nothing reaches ``main`` with it True.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

#: FTASKS 2.3, 8.1: off since P6 (FR-021.6: strict is the only mode). True only for a measuring
#: run on a scratch copy; ``test_report_mode_is_off`` holds it False.
REPORT_ONLY = False
REPORT_DIR_VARIABLE = "MF_ANSWER_REPORT_DIR"
JSON = "application/json"
_INSTALLED = "__mf_answer_validator__"
#: Backend modules whose routes the contract does not describe, and why.
NOT_DESCRIBED = {
    "api.wsp_internal_app": "the WSP worker's internal gateway app; no tool or sibling calls it",
}


class AnswerContractError(AssertionError):
    """An answer its contract entry does not describe."""

    def __init__(self, route: str, status: int | str, json_path: str, message: str) -> None:
        self.route, self.status, self.json_path, self.message = (
            route,
            str(status),
            json_path,
            message,
        )
        where = json_path or "(body)"
        super().__init__(f"{route} answered {status} off its contract at {where}: {message}")


#: Failures not yet charged to a test (cleared and read by the report hooks).
FAILURES: list[AnswerContractError] = []
#: Every failure of the session, and the top-level keys each (route, status) sent (report mode).
SESSION: dict[str, Any] = {"failures": [], "keys": {}, "answers": {}}


def _pointer(*parts: str) -> str:
    return "/" + "/".join(quote(p.replace("~", "~0").replace("/", "~1"), safe="~") for p in parts)


class AnswerCheck:
    """The committed contract, compiled lazily: one validator per (route, status), strict form."""

    def __init__(self, contract: dict[str, Any], schemas_dir: Path) -> None:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        from schemas.contract_tools import strict_form

        self.contract = contract
        resources = [
            (
                contract["$id"],
                Resource.from_contents(strict_form(contract), default_specification=DRAFT202012),
            )
        ]
        for sid, row in (contract.get("external") or {}).items():
            doc = json.loads((schemas_dir.parents[1] / row["file"]).read_text(encoding="utf-8"))
            resources.append((sid, Resource.from_contents(doc, default_specification=DRAFT202012)))
        self.registry = Registry().with_resources(resources)
        self._validators: dict[tuple[str, str], Any] = {}

    def knows(self, key: str) -> bool:
        return key in self.contract["responses"] or key in self.contract["excluded"]

    def validator(self, key: str, status: str) -> Any:
        if (key, status) not in self._validators:
            from jsonschema import Draft202012Validator

            uri = self.contract["$id"] + "#" + _pointer("responses", key, status)
            self._validators[(key, status)] = Draft202012Validator(
                {"$ref": uri}, registry=self.registry
            )
        return self._validators[(key, status)]

    def check(self, key: str, status: int, body: Any) -> list[AnswerContractError]:
        """Every way one parsed JSON body of ``key`` with ``status`` is off the contract."""
        entry = self.contract["responses"].get(key)
        excluded = self.contract["excluded"].get(key)
        if entry is None and body is None and excluded and excluded["media_type"] is None:
            return []  # a "no body" route (a 204) whose empty answer carries a JSON media type
        if entry is not None and str(status) in entry.get("x-no-body", ()):
            # A status that answers no body beside a JSON one (a gated route's 204).
            if body is None:
                return []
            return [AnswerContractError(key, status, "", f"status {status} answers no body")]
        if entry is None:
            where = "excluded" if key in self.contract["excluded"] else "absent"
            return [AnswerContractError(key, status, "", f"a JSON answer from a route {where}")]
        if str(status) not in entry:
            declared = sorted(k for k in entry if k[:1].isdigit())
            return [
                AnswerContractError(
                    key, status, "", f"status {status} is not declared (declared: {declared})"
                )
            ]
        found = [
            AnswerContractError(key, status, _json_path(e), _message(e))
            for e in _leaves(self.validator(key, str(status)).iter_errors(body))
        ]
        if key.split(" ", 1)[1].startswith(APPROVAL_PATHS):
            found += [
                AnswerContractError(key, status, where, f"a secret-named key {name!r} (FTDD §8)")
                for where, name in _secret_named(body, "")
            ]
        return found

    def check_media(self, key: str, status: int, media: str) -> list[AnswerContractError]:
        """A 2xx answer that is not JSON: its media type is the one ``excluded`` names for the
        route, or one of the entry's ``x-media-alternatives`` (021 FT4). A 2xx with no body and no
        media type is checked by its status alone."""
        entry = self.contract["responses"].get(key)
        excluded = self.contract["excluded"].get(key)
        if entry is None and excluded is None:
            return [AnswerContractError(key, status, "", f"a {media!r} answer from a route absent")]
        if not media:
            if (entry and str(status) in entry.get("x-no-body", ())) or (
                excluded and excluded["media_type"] is None
            ):
                return []
            return [AnswerContractError(key, status, "", "an answer with no media type")]
        allowed = (
            {excluded["media_type"]}
            if excluded is not None
            else set((entry or {}).get("x-media-alternatives") or {})
        )
        if media in allowed:
            return []
        return [
            AnswerContractError(
                key,
                status,
                "",
                f"media type {media!r} is not declared (declared: {sorted(allowed)})",
            )
        ]


#: The routes whose free-form maps carry stored requests (an approval's payload, summary, result),
#: which the strict form cannot see inside; no key there named like a secret may carry a value
#: other than the redaction marker (021 FTDD §8; ``core.secret_keys``, operator 2026-11-26).
APPROVAL_PATHS = ("/api/v1/approvals",)


def _secret_named(node: Any, where: str) -> list[tuple[str, str]]:
    """Every key at or below ``node`` that ``core.secret_keys`` names a secret, with a value that
    is not ``[redacted]`` (or empty)."""
    from core.secret_keys import REDACTED, is_secret_key

    out: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for name, value in node.items():
            here = f"{where}.{name}" if where else str(name)
            text = str(name)
            if is_secret_key(text) and value not in (None, "", REDACTED):
                out.append((here, text))
            out += _secret_named(value, here)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            out += _secret_named(item, f"{where}[{index}]")
    return out


def _leaves(errors: Any) -> list[Any]:
    """The errors worth naming: inside a union (``anyOf``), the deepest branch errors."""
    out: list[Any] = []
    for error in errors:
        inner = _leaves(error.context or [])
        if not inner:
            out.append(error)
            continue
        relevant = [
            leaf
            for leaf in inner
            if not (leaf.validator == "type" and leaf.validator_value == "null")
        ]
        out.append(max(relevant or inner, key=lambda e: len(list(e.absolute_path))))
    return out


def _json_path(error: Any) -> str:
    out = ""
    for part in error.absolute_path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _message(error: Any) -> str:
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        declared = set((error.schema or {}).get("properties", {}))
        extra = sorted(set(error.instance) - declared)
        model = (error.schema or {}).get("title", "?")
        return f"undeclared key {', '.join(map(repr, extra))} (model {model})"
    return str(error.message)


_STATE: dict[str, Any] = {}


def answer_check() -> AnswerCheck:
    """The session's check, built on first use from the committed contract."""
    if "check" not in _STATE:
        from schemas.answers_contract import CONTRACT_PATH, SCHEMAS_DIR

        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        _STATE["check"] = AnswerCheck(contract, SCHEMAS_DIR)
    check: AnswerCheck = _STATE["check"]
    return check


def endpoint_name(endpoint: Callable[..., Any]) -> str:
    return f"{endpoint.__module__}.{endpoint.__qualname__}"


def endpoint_keys() -> dict[str, dict[str, str]]:
    """``module.qualname`` of each real endpoint -> {METHOD: contract key}, from create_app()."""
    if "endpoints" not in _STATE:
        from fastapi.routing import APIRoute

        from api.app import create_app
        from api.route_table import mounted_api_routes

        out: dict[str, dict[str, str]] = {}
        for ctx in mounted_api_routes(create_app()):
            route = ctx.original_route
            if not isinstance(route, APIRoute) or not route.include_in_schema:
                continue
            for method in ctx.methods or ():
                out.setdefault(endpoint_name(ctx.endpoint), {})[
                    method
                ] = f"{method} {ctx.path_format}"
        _STATE["endpoints"] = out
    found: dict[str, dict[str, str]] = _STATE["endpoints"]
    return found


def route_key(method: str, route: Any) -> str | None:
    """The contract key of a matched backend route, or None for a route the check skips: a test's
    own route, or one in :data:`NOT_DESCRIBED`. A backend route the table does not know is returned
    under a key the contract lacks, so it fails."""
    endpoint = getattr(route, "endpoint", None)
    module = (getattr(endpoint, "__module__", "") or "") if endpoint is not None else ""
    if not module.startswith("api.") or module in NOT_DESCRIBED:
        return None
    assert endpoint is not None
    mapped = endpoint_keys().get(endpoint_name(endpoint), {}).get(method)
    return mapped or f"{method} {getattr(route, 'path', '')} ({endpoint_name(endpoint)})"


def _observe(shape: dict[str, Any], value: Any) -> None:
    """Merge one JSON value into an inferred shape (report mode only): the JSON types seen, for
    objects each key's count and shape, for arrays the items' shape."""
    kind = (
        "null"
        if value is None
        else (
            "boolean"
            if isinstance(value, bool)
            else (
                "integer"
                if isinstance(value, int)
                else (
                    "number"
                    if isinstance(value, float)
                    else (
                        "string"
                        if isinstance(value, str)
                        else "array" if isinstance(value, list) else "object"
                    )
                )
            )
        )
    )
    shape.setdefault("types", {})
    shape["types"][kind] = shape["types"].get(kind, 0) + 1
    if kind == "object":
        props = shape.setdefault("props", {})
        if len(props) > 200:  # a map keyed by data: stop listing keys
            shape["map"] = True
            return
        for name, sub in value.items():
            _observe(props.setdefault(name, {}), sub)
    elif kind == "array":
        for item in value[:50]:
            _observe(shape.setdefault("items", {}), item)
    elif kind == "string" and len(shape.setdefault("strings", [])) < 8:
        if value not in shape["strings"] and len(value) <= 60:
            shape["strings"].append(value)


def record(key: str, status: int, body: Any) -> None:
    """Check one answer and record what it sent and what is off the contract."""
    seen = SESSION["keys"].setdefault(f"{key} {status}", set())
    if isinstance(body, dict):
        seen.update(body)
    if os.environ.get(REPORT_DIR_VARIABLE):
        _observe(SESSION.setdefault("shapes", {}).setdefault(f"{key} {status}", {}), body)
    SESSION["answers"][f"{key} {status}"] = SESSION["answers"].get(f"{key} {status}", 0) + 1
    for failure in answer_check().check(key, status, body):
        FAILURES.append(failure)
        SESSION["failures"].append(failure)


class ValidatingASGI:
    """Wraps an ASGI app; checks each 2xx JSON answer once the body is complete. Every message is
    forwarded unchanged and at once; only the body bytes are copied."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def __getattr__(self, name: str) -> Any:
        return getattr(self.app, name)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        state: dict[str, Any] = {"status": 0, "json": False, "body": bytearray()}

        async def watched(message: Any) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                headers = {k.lower(): v for k, v in message.get("headers") or ()}
                media = headers.get(b"content-type", b"").split(b";")[0].strip()
                state["json"] = media == JSON.encode()
                state["media"] = media.decode("latin-1")
            elif message["type"] == "http.response.body" and state["json"]:
                state["body"] += message.get("body", b"")
                if not message.get("more_body", False):
                    self._check(scope, state)
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                self._check_media(scope, state)
            await send(message)

        await self.app(scope, receive, watched)

    @staticmethod
    def _check_media(scope: Any, state: dict[str, Any]) -> None:
        status = state["status"]
        if not 200 <= status < 300:
            return
        key = route_key(scope.get("method", ""), scope.get("route"))
        if key is None:
            return
        for failure in answer_check().check_media(key, status, state.get("media", "")):
            FAILURES.append(failure)
            SESSION["failures"].append(failure)

    @staticmethod
    def _check(scope: Any, state: dict[str, Any]) -> None:
        status = state["status"]
        key = route_key(scope.get("method", ""), scope.get("route"))
        if key is None:
            return
        declared = answer_check().contract["responses"].get(key) or {}
        if not (200 <= status < 300 or str(status) in declared):
            return
        body = json.loads(bytes(state["body"])) if state["body"] else None
        record(key, status, body)


ORIGINAL_BUILD: Callable[..., Any] | None = None


def install() -> None:
    """Wrap every FastAPI app's middleware stack in :class:`ValidatingASGI` (idempotent)."""
    global ORIGINAL_BUILD
    from fastapi import FastAPI

    if getattr(FastAPI.build_middleware_stack, _INSTALLED, False):
        return
    original = FastAPI.build_middleware_stack
    ORIGINAL_BUILD = original

    def build_middleware_stack(self: Any) -> Any:
        return ValidatingASGI(original(self))

    setattr(build_middleware_stack, _INSTALLED, True)
    FastAPI.build_middleware_stack = build_middleware_stack  # type: ignore[method-assign]


#: The MCP client's verbs; ``test_mcp_fakes_are_checked.py`` holds this equal to ForgeClient's own.
CLIENT_VERBS = ("get", "post", "put", "delete")


def install_mcp_checks() -> None:
    """021 FR-021.14 for every MCP test, not only the harness's: a backend answer the MCP client
    returns (a fake transport's, a test's replacement ``request``, or the real app's) is checked
    against its route's contract entry, and a tool answer ``MCPServer.call_tool`` returns against
    the tool's output schema (``mcp_harness.check_fake_answer``, ``check_tool_answer``). The
    client's verbs are wrapped, not ``request``: they call ``self.request``, so a test that
    replaces ``request`` is checked too (review round 2, F2). Idempotent; signatures kept."""
    import functools

    from mcp.server.mcpserver import MCPServer

    from mcp_server.client import ForgeClient

    if getattr(MCPServer.call_tool, _INSTALLED, False):
        return

    def checked(method: str, verb: Any) -> Any:
        @functools.wraps(verb)
        async def call(self: Any, path: str, *args: Any, **kwargs: Any) -> Any:
            answer = await verb(self, path, *args, **kwargs)
            from tests.support.mcp_harness import check_fake_answer

            check_fake_answer(method, path, answer)
            return answer

        setattr(call, _INSTALLED, True)
        return call

    for name in CLIENT_VERBS:
        setattr(ForgeClient, name, checked(name.upper(), getattr(ForgeClient, name)))

    original_call = MCPServer.call_tool

    @functools.wraps(original_call)
    async def call_tool(self: Any, name: str, arguments: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original_call(self, name, arguments, *args, **kwargs)
        from tests.support.mcp_harness import check_tool_answer

        check_tool_answer(name, result)
        return result

    setattr(call_tool, _INSTALLED, True)
    MCPServer.call_tool = call_tool  # type: ignore[method-assign]


def installed() -> bool:
    from fastapi import FastAPI

    return bool(getattr(FastAPI.build_middleware_stack, _INSTALLED, False))


# -- pytest hooks ---------------------------------------------------------------------------------


#: The marker a test carries when its MCP fake is deliberately no route's answer (a transport or
#: formatting test feeding ``[]`` to every tool): its reason is required, and the fake and tool
#: checks record nothing for it. The answer validator itself still checks every HTTP answer.
UNCHECKED_MARKER = "mcp_fake_is_not_a_route_answer"


def fakes_unchecked() -> bool:
    return bool(_STATE.get("fakes_unchecked"))


def pytest_configure(config: pytest.Config) -> None:
    install()
    install_mcp_checks()
    config.addinivalue_line(
        "markers",
        f"{UNCHECKED_MARKER}(reason): this test's MCP fake answer is deliberately not one a route "
        "gives (021 FR-021.14's checks record nothing for it; the reason is required)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    FAILURES.clear()
    marker = item.get_closest_marker(UNCHECKED_MARKER)
    if marker is not None:
        reason = marker.kwargs.get("reason", marker.args[0] if marker.args else "")
        if not (isinstance(reason, str) and reason.strip()):
            raise pytest.UsageError(f"{item.nodeid}: @{UNCHECKED_MARKER} needs its reason")
    _STATE["fakes_unchecked"] = marker is not None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Any:
    outcome = yield
    if call.when not in ("call", "teardown") or not FAILURES:
        return
    failures = list(FAILURES)
    FAILURES.clear()
    for failure in failures:
        SESSION.setdefault("by_test", []).append((item.nodeid, str(failure)))
    if REPORT_ONLY:
        return
    report = outcome.get_result()
    if report.passed:
        lines = "\n".join(f"  {f}" for f in failures)
        report.outcome = "failed"
        report.longrepr = f"{len(failures)} answer(s) off the contract:\n{lines}"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    directory = os.environ.get(REPORT_DIR_VARIABLE)
    if not directory or not (SESSION["keys"] or SESSION.get("by_test")):
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    path = Path(directory) / f"answers-{worker}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for nodeid, text in SESSION.get("by_test", []):
            out.write(json.dumps({"kind": "violation", "test": nodeid, "text": text}) + "\n")
        for where, keys in sorted(SESSION["keys"].items()):
            count = SESSION["answers"].get(where, 0)
            row = {"kind": "keys", "route": where, "keys": sorted(keys), "answers": count}
            out.write(json.dumps(row) + "\n")
        for where, shape in sorted(SESSION.get("shapes", {}).items()):
            out.write(json.dumps({"kind": "shape", "route": where, "shape": shape}) + "\n")
