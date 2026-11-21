"""``forge_howto``'s ``topic`` accepts exactly the topics ``HOWTO`` holds (FTID 016 §2.3).

Pre-existing defect fixed with feature 016: ``HOWTO`` had an ``archive`` topic that the ``topic``
Literal omitted, so ``forge_howto(topic="archive")`` failed validation while the topic list offered
it. Read from the LIVE server's tool schema, not from the source.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp_server.config import MCPSettings
from mcp_server.server import build_server
from mcp_server.tools.core import HOWTO


def _topic_schema() -> Any:
    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    return tools["forge_howto"].input_schema["properties"]["topic"]


def _enum(schema: Any) -> set[str]:
    found: set[str] = set()
    for option in schema.get("anyOf", [schema]):
        found |= set(option.get("enum", []))
    return found


def test_the_topic_choices_are_exactly_the_howto_topics() -> None:
    choices = _enum(_topic_schema())
    assert choices, "the schema lists no topic at all - the scan read nothing"
    assert choices == set(HOWTO)


def test_the_wsp_topic_names_the_two_wsp_actions() -> None:
    assert "wsp_session" in HOWTO["wsp"] and "wsp_session_external" in HOWTO["wsp"]


def test_the_wsp_topic_names_the_endpoint_proposal_and_its_action() -> None:
    """2026-11-20: the proposal tool, its action and what it may not carry."""
    text = HOWTO["wsp"]
    assert "forge_wsp_register_endpoint" in text and "wsp_endpoint_register" in text
    assert "no API key and no prices" in text and "Editor endpoints screen" in text


def test_the_approvals_topic_names_every_live_approval_action() -> None:
    from api.agent_origin import APPROVAL_ACTIONS

    missing = [action for action in APPROVAL_ACTIONS if action not in HOWTO["approvals"]]
    assert not missing, missing
