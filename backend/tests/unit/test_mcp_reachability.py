"""G3 (FR-012.19, MR-12): every tool a module declares is in the LIVE registry, and every
category is named in all four layers.

Membership is decided by the live server (``build_server`` + ``list_tools`` - the call an
agent makes); the AST of each module says what it DECLARES. The two must be equal per
category and in total, so a declared tool that never registers, a module missing from
``CATEGORY_MODULES``, or a category missing from any layer each fail here. Every scan asserts
it found something before it asserts anything else.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from mcp_server import tools as tools_package
from mcp_server.config import DEFAULT_CATEGORIES, VALID_CATEGORIES, MCPSettings
from mcp_server.server import build_server, tool_names_declared
from mcp_server.tools import CATEGORY_MODULES

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "gitops" / "pilot" / "mcp.yaml"


def live_tools(categories: str | None = None) -> set[str]:
    settings = MCPSettings(
        auth_token="x" * 32, **({} if categories is None else {"tool_categories": categories})
    )
    mcp, _ = build_server(settings)
    return {tool.name for tool in asyncio.run(mcp.list_tools())}


def declared(category: str) -> set[str]:
    return {name for module in CATEGORY_MODULES[category] for name in tool_names_declared(module)}


def manifest_categories() -> set[str]:
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text()) if doc]
    (deployment,) = [d for d in docs if d["kind"] == "Deployment"]
    (container,) = deployment["spec"]["template"]["spec"]["containers"]
    env: dict[str, Any] = {e["name"]: e for e in container["env"]}
    return {c.strip() for c in env["MCP_TOOL_CATEGORIES"]["value"].split(",") if c.strip()}


def test_the_live_registry_is_every_declared_tool() -> None:
    every = set().union(*(declared(c) for c in CATEGORY_MODULES))
    assert len(every) >= 80, f"the AST scan found only {len(every)} tools - it read the wrong thing"
    assert live_tools() == every, (
        f"declared but not registered: {sorted(every - live_tools())}; registered but not "
        f"declared: {sorted(live_tools() - every)}"
    )


@pytest.mark.parametrize("category", sorted(CATEGORY_MODULES))
def test_each_category_registers_exactly_its_declared_tools(category: str) -> None:
    expected = declared(category)
    assert expected, f"{category} declares no tool"
    assert live_tools(category) == expected


def test_every_tool_module_on_disk_is_in_the_registry() -> None:
    """A module imported nowhere would declare tools no category serves."""
    on_disk = {
        path.stem
        for path in Path(tools_package.__file__).parent.glob("*.py")
        if path.stem != "__init__"
    }
    registered = {
        module.__name__.rsplit(".", 1)[1] for ms in CATEGORY_MODULES.values() for module in ms
    }
    assert on_disk, "no tool module found on disk"
    assert on_disk == registered


def test_the_four_category_layers_agree() -> None:
    registry = set(CATEGORY_MODULES)
    assert len(registry) == 10
    assert registry == set(VALID_CATEGORIES), "config.VALID_CATEGORIES"
    assert registry == set(DEFAULT_CATEGORIES.split(",")), "config.DEFAULT_CATEGORIES"
    assert registry == manifest_categories(), "MCP_TOOL_CATEGORIES in gitops/pilot/mcp.yaml"


def test_an_unknown_category_is_dropped_not_honoured() -> None:
    assert live_tools("core,no-such-category") == declared("core")
