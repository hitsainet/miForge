"""Review of 015 P0 (independent): the narrowed ``internal-host`` rule still forbids every
``mcp-*``, ``k8s-*`` and ``.hitsai.local`` name, and admits ONLY a whole ``docs-<app>.hitsai.net``.

The existing planted cases (``mcp.hitsai.net``, ``mcp-docs-x.hitsai.net``, ``k8s-docs.hitsai.net``)
do not distinguish a lookahead widened to ``(?:docs|mcp)-<app>``: none of them is an
``mcp-<app>.hitsai.net`` host, which is the real tunnel shape (``mcp-forge.hitsai.net``).
"""

from __future__ import annotations

import pytest

from tests.unit.test_manual_publish_allowlist import _compiled, scan


def _caught(text: str) -> bool:
    rules = [r for r in _compiled() if r[0] == "internal-host"]
    assert rules, "no internal-host rule"
    return bool(scan([("planted.md", text)], rules))


@pytest.mark.parametrize(
    "planted",
    [
        "the tunnel mcp-forge.hitsai.net",
        "the tunnel mcp-miworkbench.hitsai.net",
        "the LAN name mcp-forge.hitsai.local",
        "the ingress k8s-miforge.hitsai.local",
        "the ingress k8s-miforge.hitsai.net",
        "MCP-FORGE.HITSAI.NET in capitals",
    ],
)
def test_mcp_and_k8s_hosts_stay_forbidden(planted: str) -> None:
    assert _caught(planted), f"internal-host admitted {planted!r}"


@pytest.mark.parametrize(
    "planted",
    [
        "the endpoint mcp.docs-miforge.hitsai.net",
        "the endpoint k8s.docs-miforge.hitsai.net",
    ],
)
def test_a_host_under_a_docs_label_is_not_the_public_manual(planted: str) -> None:
    """Before P0 ``\\b[a-z0-9-]+\\.hitsai\\.net`` caught ``docs-miforge.hitsai.net`` inside these;
    the exception now starts its lookahead at the ``docs-`` label, so any deeper name passes."""
    assert _caught(planted), f"internal-host admitted {planted!r}"
