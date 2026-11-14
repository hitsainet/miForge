"""No requirement, decision or feature ids in text the backend serves  (copy review, 2026-10-20).

``test_new_run_form_copy.py`` guards the strings the New run form renders. This file widens
it to **every** piece of prose the backend can send to a person:

1. **Every prose string literal in ``backend/src``.** Refusals are built at hundreds of call
   sites as f-strings, and plenty of served text lives in tables (capability names, recovery
   steps, miLLM refusal translations, column-ignore reasons), so the scan reads the syntax
   tree and checks every string constant that is not a docstring, a log call's text or an
   ``assert``. It reports how many strings it read and fails below a floor, so a layout
   change that left it reading nothing cannot pass silently.
2. **Every parameterless ``GET`` route, served.** Registries, listings, settings
   descriptions and status payloads, read through the mounted app, every string anywhere in
   the JSON - so a new route or field is covered without an edit here.

Ids belong in docstrings and comments, where the people who maintain the code read them.
``R-02.25's first condition.`` means nothing to the operator choosing a stop condition.

**Not scanned:** ``mcp_server/``, whose tool descriptions are read by agents and published as
the generated MCP contract; and a string that is nothing but an id (``"R-01.21"``, ``"F6"``),
which is a data value - a requirement column in a table, a block's writer - not prose.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api import health as health_module
from api.app import create_app
from api.deps import session_dependency
from api.route_table import mounted_routes
from core.config import Settings, reset_settings
from tests.support.jobs import memory_engine, session_for

SRC = Path(__file__).resolve().parents[2] / "src"

#: Agent-facing, with its own generated contract (``docs/mcp-contract.md``).
NOT_SCANNED = ("mcp_server",)

#: Text that names a document, a requirement, a decision or a feature instead of saying
#: something to the operator. A superset of ``test_new_run_form_copy.JARGON``.
IDS = (
    re.compile(r"\bR-\d{2}\.\d+"),
    re.compile(r"\bFR-\d{3}\.\d+"),
    re.compile(r"\b(?:ADR|BRD|PRD|TDD|DEF|AMB|OQ|NFR)-\d+"),
    re.compile(r"\bF(?:TDD|PRD|TID|TASKS)\b"),
    re.compile(r"§\s?\d"),
    # Feature numbers, F1 to F12. Not F16/F32/F64, which are dtypes.
    re.compile(r"\bF(?:[1-9]|1[0-2])\b(?:'s|\u2019s)?"),
    re.compile(r"\bE-\d+\b"),
    re.compile(r"\bacceptance criterion \d"),
    re.compile(r"\bfoundation task\b"),
    # Markdown bold renders as literal asterisks on every screen. Not a `***` mask.
    re.compile(r"\*\*(?=\w)|(?<=\w)\*\*"),
)

#: How many prose strings the source scan must read. ~6,000 on 2026-10-20; the floor
#: sits well below so ordinary churn does not trip it, and far above zero so a scan that
#: reads nothing does.
STRING_FLOOR = 3000

#: How many parameterless GET routes must answer with JSON. 30 on 2026-10-20.
ROUTE_FLOOR = 20

_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def ids_in(text: str) -> list[str]:
    return [match.group(0) for pattern in IDS for match in pattern.finditer(text)]


def _is_only_an_id(text: str) -> bool:
    return any(pattern.fullmatch(text.strip()) for pattern in IDS)


def prose_in(source: str) -> Iterator[tuple[int, str]]:
    """Every string constant in ``source`` that a person could be shown.

    Skipped: docstrings, the arguments of a log call (operator logs, not served text),
    ``assert`` statements, and a constant that is only an id.
    """
    tree = ast.parse(source)
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skipped.add(id(first.value))
        elif isinstance(node, ast.Assert):
            skipped.update(id(inner) for inner in ast.walk(node))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LOG_METHODS
        ):
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                skipped.update(id(inner) for inner in ast.walk(argument))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skipped
            and not _is_only_an_id(node.value)
        ):
            yield node.lineno, node.value


def test_the_scanner_finds_an_id_it_is_shown() -> None:
    """The control: a scan that cannot see an id would pass everything below."""
    planted = (
        '"""A docstring may cite R-02.42."""\n'
        'log.info("logs may cite R-02.42")\n'
        'raise refuse("x", f"Refusing before any allocation (R-02.42) for {name}.")\n'
        'raise ValueError("a run must name **one** source")\n'
        'NOTE = {"note": "see BRD-02 §5.10"}\n'
        'OWNER = "F6"\n'
        'MASK = "***"\n'
    )
    found = sorted(hit for _, text in prose_in(planted) for hit in ids_in(text))
    assert found == sorted(["R-02.42", "**", "**", "BRD-02", "§5"]), found
    assert ids_in("weights are BF16 or F16, never F32") == []


def test_no_prose_in_the_source_carries_an_id() -> None:
    scanned = 0
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC)
        if relative.parts[0] in NOT_SCANNED:
            continue
        for line, text in prose_in(path.read_text(encoding="utf-8")):
            scanned += 1
            hits = ids_in(text)
            if hits:
                found.append(f"{relative}:{line}: {hits} in {text[:100]!r}")
    assert scanned >= STRING_FLOOR, f"read only {scanned} strings - is the scan reading?"
    assert found == [], "\n".join(found)


# ---------------------------------------------------------------------------- served routes


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    checkpoints, parts = tmp_path / "checkpoints", tmp_path / "parts"
    checkpoints.mkdir()
    parts.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(_env_file=None, checkpoint_root=checkpoints, parts_root=parts),  # type: ignore[call-arg]
    )
    engine = memory_engine()

    def one_session() -> Iterator[Session]:
        with session_for(engine) as session:
            yield session

    app = create_app()
    for name in ("postgres", "redis"):
        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, name, lambda: {"status": "ok", "detail": None}
        )
    app.dependency_overrides[session_dependency] = one_session
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    reset_settings()


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_no_parameterless_GET_serves_an_id(client: TestClient) -> None:
    paths = sorted(
        route.path
        for route in mounted_routes(client.app)  # type: ignore[arg-type]
        if isinstance(route.original_route, APIRoute)
        and "GET" in route.methods
        and "{" not in route.path
    )
    answered = 0
    found: list[str] = []
    for path in paths:
        response = client.get(path)
        if "application/json" not in response.headers.get("content-type", ""):
            continue
        answered += 1
        for text in _strings(response.json()):
            hits = ids_in(text)
            if hits:
                found.append(f"GET {path} -> {response.status_code}: {hits} in {text[:120]!r}")
    assert answered >= ROUTE_FLOOR, f"only {answered} of {len(paths)} routes answered JSON"
    assert found == [], "\n".join(found)
