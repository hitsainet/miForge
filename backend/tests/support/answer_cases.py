"""The per-router answer cases (021 FR-021.13; MF20, WT8): every contract route answers in a test.

Each module ``tests/integration/answers/test_<router>_answer.py`` registers one test per route of
its router with :func:`case`, drives the real route through the real app, and passes the answer to
:func:`answered`, which checks it against the committed contract (the validator checks it too, on
the way out). The module's set-equality test compares :data:`CASES` for its router with
:func:`router_keys`: a route added without a case, or a case removed, is red, naming the route.

A gated route is driven twice: as an agent (``answered(..., pending=True)``: the gate's pending
``202``) and as the operator (its own answer).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

from tests.support import answer_validator

F = TypeVar("F", bound=Callable[..., Any])

#: router module -> {contract key -> the test functions that drive it}.
CASES: dict[str, dict[str, list[str]]] = {}


def case(router: str, *keys: str) -> Callable[[F], F]:
    """Register the decorated test as driving ``keys`` of ``router`` (``api.v1.endpoints.x``)."""

    def register(fn: F) -> F:
        for key in keys:
            CASES.setdefault(router, {}).setdefault(key, []).append(fn.__name__)
        return fn

    return register


def router_keys(router: str) -> set[str]:
    """The contract keys (``responses`` and ``excluded``) whose endpoint lives in ``router``."""
    contract = answer_validator.answer_check().contract
    known = set(contract["responses"]) | set(contract["excluded"])
    out: set[str] = set()
    for name, methods in answer_validator.endpoint_keys().items():
        if name.rsplit(".", 1)[0] == router:  # module.qualname; endpoints are top-level functions
            out |= {key for key in methods.values() if key in known}
    return out


def answered(key: str, response: Any, *, pending: bool = False) -> Any:
    """Assert ``response`` is a declared, valid answer of ``key``; return its parsed body (or its
    bytes for a non-JSON answer). ``pending``: the answer is the gate's pending approval."""
    check = answer_validator.answer_check()
    status = response.status_code
    assert 200 <= status < 300 or status == 503, f"{key} answered {status}: {response.text[:300]}"
    media = str(response.headers.get("content-type", "")).split(";")[0].strip()
    if media != answer_validator.JSON:
        found = check.check_media(key, status, media)
        assert not found, [str(f) for f in found]
        return response.content
    body = json.loads(response.content) if response.content else None
    found = check.check(key, status, body)
    assert not found, [str(f) for f in found]
    if pending:
        assert status == 202 and isinstance(body, dict) and body.get("status") == "pending", body
    return body


def assert_covered(router: str) -> None:
    """The set-equality test of one module: its cases EQUAL its router's contract routes."""
    cases = set(CASES.get(router, {}))
    routes = router_keys(router)
    assert cases == routes, {
        "routes without a case": sorted(routes - cases),
        "cases of no route": sorted(cases - routes),
    }
