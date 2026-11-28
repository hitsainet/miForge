"""What every tool module is handed besides the client (012 FTID section 3.4).

``GATED_TOOLS`` names the tools whose REST route waits for the operator when this server calls
it, and the approval action. It is a **statement** for descriptions and the generated contract,
never a control: the gate lives in REST (``api/agent_origin.py``).
``tests/unit/test_mcp_parity.py`` checks every entry against what the LIVE gate decides for the
payload the tool actually sends - so ``forge_plan_continue``, which calls a gated route with
``submit: false``, is correctly absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from .config import MCPSettings
from .health_gate import HealthGate

#: Dependency sets a tool declares. Every route but health reads PostgreSQL; a route that
#: starts work enqueues through Redis.
CORE = ("backend",)
READ = ("backend", "postgres")
JOB = ("backend", "postgres", "redis")


@dataclass(frozen=True)
class ToolContext:
    settings: MCPSettings
    gate: HealthGate


#: tool name -> the approval action an agent call waits for (statement; the gate is REST's).
#: The three run-starting tools wait for ``gpu_start``, or ``hotupdate_attach`` when the
#: request attaches to miLLM (FR-012.15) - the backend decides which, from the payload.
GATED_TOOLS: dict[str, str] = {
    "forge_submit_run": "gpu_start",
    "forge_define_run": "gpu_start",
    "forge_create_chain": "gpu_start",
    "forge_continue_run": "hotupdate_attach",
    "forge_evaluate_now": "gpu_start",
    "forge_merge_adapter": "gpu_start",
    "forge_delete_pack": "delete",
    "forge_delete_draft": "delete",
    "forge_set_setting": "config_write",
    "forge_reset_setting": "config_write",
    # Feature 016: wsp_session_external when the stored endpoint is external (FR-016.37).
    "forge_wsp_start_run": "wsp_session",
    # Feature 018 (FR-018.8): a protocol test is an inspect, so only an external endpoint's waits,
    # and then always as wsp_session_external (018-P2 deviation P2-D6).
    "forge_wsp_test_endpoint_protocol": "wsp_session_external",
    # 2026-11-20: an agent's proposal of a local, keyless, unpriced editor endpoint.
    "forge_wsp_register_endpoint": "wsp_endpoint_register",
}

#: The tools whose action escalates to ``hotupdate_attach`` for an attach-mode request.
ESCALATES_TO_ATTACH: frozenset[str] = frozenset(
    {"forge_submit_run", "forge_define_run", "forge_create_chain"}
)

#: 017 FR-017.29 (suite A17): said wherever an agent's call reaches the Hugging Face Hub.
ANONYMOUS_HUB = (
    " An agent's Hugging Face Hub fetches are anonymous: they never use a credential on the "
    "miForge host, so a private or gated repository is refused (model_revision_not_found or "
    "model_fetch_failed)."
)

WAITS = (
    " Waits for the operator: an agent's call returns a pending approval "
    "({approval_id, status: 'pending', action, summary, expires_at}); poll "
    "forge_get_approval_status."
)

# --- argument shapes, validated by the MCP SDK BEFORE any request --------------------------

#: An identifier that goes into a URL path: no slash, and it starts with a letter or digit, so
#: it can be neither ``..`` nor another path - an argument cannot re-route a request to another
#: endpoint. (No look-ahead: the MCP SDK validates with Rust's regex, which has none.)
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$"
#: An adapter name, ``<base_model>:<run_id>:adapter-<step>``; a Hub base carries one slash, and
#: each segment starts with a letter or digit, so neither can be ``..``.
ADAPTER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,190}(/[A-Za-z0-9][A-Za-z0-9_.:-]{0,190})?$"
#: A SHA-256 hex digest.
SHA256_PATTERN = r"^[0-9a-f]{64}$"
#: A settings key: lower snake case.
KEY_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"

Id = Annotated[str, Field(pattern=ID_PATTERN, description="An identifier (no slash, no '..').")]
RunId = Annotated[str, Field(pattern=ID_PATTERN, description="A run id.")]
Step = Annotated[int, Field(ge=0, description="A training step (the canonical integer).")]
Limit = Annotated[int, Field(ge=1, le=1000, description="Items per page.")]
Offset = Annotated[int, Field(ge=0, description="Items to skip.")]
