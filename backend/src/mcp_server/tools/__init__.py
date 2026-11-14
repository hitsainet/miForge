"""MCP tool modules, one per category: THE ONLY REGISTRY (012 FTDD section 5.2).

Each module exposes ``register(mcp, client, ctx)``; ``server.build_server`` calls it only when
the category is enabled through ``MCP_TOOL_CATEGORIES``. Every guard reads this one map.

A category is registered in FOUR layers, each with a test that fails when it is removed
(``tests/unit/test_mcp_reachability.py``): this map, ``config.VALID_CATEGORIES``,
``config.DEFAULT_CATEGORIES`` and ``MCP_TOOL_CATEGORIES`` in ``gitops/pilot/mcp.yaml``.
"""

from types import ModuleType

from . import (
    adapters,
    chains,
    checkpoints,
    core,
    drafts,
    hotupdate,
    inputs,
    packs,
    runs,
    settings,
)

# category name -> modules (registration order = tools/list order)
CATEGORY_MODULES: dict[str, list[ModuleType]] = {
    "core": [core],
    "runs": [runs],
    "inputs": [inputs],
    "checkpoints": [checkpoints],
    "hotupdate": [hotupdate],
    "adapters": [adapters],
    "packs": [packs],
    "drafts": [drafts],
    "chains": [chains],
    "settings": [settings],
}
