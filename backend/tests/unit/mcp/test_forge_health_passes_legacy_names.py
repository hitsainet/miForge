"""``forge_health`` passes the health body through unchanged, new keys included (FR-015.6).

The suite's P7 sampler may read ``legacy_names`` and ``legacy_names_meta`` through MCP as well
as over HTTP, so the tool must not rebuild the body from a list of known keys.
"""

from __future__ import annotations

from tests.support.mcp_harness import build_harness, call_tool

BODY = {
    "status": "ok",
    "integrations": {},
    "legacy_names": {
        "in_use": [
            {
                "kind": "env",
                "old": "MILLM_AUDIT_LABEL",
                "new": "MISERVER_AUDIT_LABEL",
                "last_seen": "2026-11-08T12:00:00Z",
            }
        ],
        "also_set": [],
        "conflicts": [],
        "retired_but_set": [],
    },
    "legacy_names_meta": {"error": None, "in_use_window_hours": 24},
}


def test_block_passes_through() -> None:
    mcp, client = build_harness(response=BODY)
    result = call_tool(mcp, "forge_health", {})
    assert client.calls == [("GET", "/api/health", {})]
    assert result == BODY
    assert result["legacy_names_meta"] == {"error": None, "in_use_window_hours": 24}
