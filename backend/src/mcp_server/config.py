# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/mcp_server/config.py @ a055a38
# Mode: adapt (docs/REUSE.md). Kept: MCPSettings, warn-and-drop of unknown categories, the
# non-prefixed backend URL, the identity validated against the REST header pattern.
# Changed: miForge's ten categories; MIFORGE_API_URL; identity agent:forge-mcp.
"""MCP server configuration (environment prefix ``MCP_``; 012 FTID section 3.4).

**Four layers name a category, and each is pinned by a test that fails when it is removed**
(``tests/unit/test_mcp_reachability.py``): ``tools.CATEGORY_MODULES``, ``VALID_CATEGORIES``,
``DEFAULT_CATEGORIES`` and ``MCP_TOOL_CATEGORIES`` in ``gitops/pilot/mcp.yaml``.
``VALID_CATEGORIES`` cannot be derived from the registry here (``tools`` imports this module).

The MCP pod has no database URL, Redis URL or volume (FR-012.3), so nothing here may read the
backend's ``Settings`` - this package does not import the backend at all.
"""

from __future__ import annotations

import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

#: Every category this build serves (FTDD section 5.2). Keep in step with
#: ``tools.CATEGORY_MODULES``; the reachability test fails otherwise.
VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "core",
        "runs",
        "inputs",
        "checkpoints",
        "hotupdate",
        "adapters",
        "packs",
        "drafts",
        "chains",
        "settings",
    }
)

#: All ten on by default. The deployment sets the same list explicitly.
DEFAULT_CATEGORIES = ",".join(sorted(VALID_CATEGORIES))

#: The REST layer's header pattern (``api/agent_origin.py``), repeated because this package
#: must not import the backend; ``tests/unit/mcp/test_client.py`` compares the two by AST.
AGENT_IDENTITY_PATTERN = r"^agent:[a-z0-9][a-z0-9-]{0,47}$"


class MCPSettings(BaseSettings):
    """Runtime configuration for the miForge MCP server."""

    #: `hide_input_in_errors`: a rejected value never appears in a validation error, which is
    #: logged at startup - this model holds the bearer token (the backend's own rule, ADR-020,
    #: pinned for every settings class by `test_all_base_urls_refuse_credentials.py`).
    model_config = SettingsConfigDict(env_prefix="MCP_", extra="ignore", hide_input_in_errors=True)

    auth_token: str = Field(default="", description="Bearer token required on the HTTP transport")
    agent_identity: str = Field(
        default="agent:forge-mcp",
        pattern=AGENT_IDENTITY_PATTERN,
        description="Sent as X-Miforge-Agent on every backend request (FR-012.22)",
    )
    allow_anonymous: bool = Field(
        default=False, description="Permit start without a token (stdio transport only)"
    )
    #: Every interface inside the pod; the bearer token gates every request but /health.
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8765)
    tool_categories: str = Field(default=DEFAULT_CATEGORIES)

    @property
    def api_url(self) -> str:
        """Backend base URL. Not ``MCP_``-prefixed, as miDataworks' ``DATAWORKS_API_URL``."""
        return os.environ.get("MIFORGE_API_URL", "http://localhost:8000").rstrip("/")

    def requested_categories(self) -> set[str]:
        """Exactly what ``MCP_TOOL_CATEGORIES`` asked for, recognised or not."""
        return {c.strip() for c in self.tool_categories.split(",") if c.strip()}

    def unknown_categories(self) -> set[str]:
        """Requested names this build does not recognise (reported on ``/health``)."""
        return self.requested_categories() - VALID_CATEGORIES

    def enabled_categories(self) -> set[str]:
        """The recognised categories; unknown names are WARNED AND DROPPED, never honoured.

        Raising here would turn every category addition into a crash-loop window: the manifest
        syncs in minutes and the image takes longer, so a new manifest meets an image whose
        list predates it. A typo in the deployed manifest is caught in CI instead
        (``test_mcp_reachability.py`` reads ``gitops/pilot/mcp.yaml``).
        """
        requested = self.requested_categories()
        unknown = requested - VALID_CATEGORIES
        if unknown:
            logger.warning(
                "Ignoring unknown MCP tool categories: %s (valid: %s). Expected briefly while a "
                "manifest change leads its image; if it persists, those tools are OFF.",
                sorted(unknown),
                sorted(VALID_CATEGORIES),
            )
        return requested & VALID_CATEGORIES
