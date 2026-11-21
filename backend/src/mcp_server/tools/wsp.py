"""Category ``wsp``: weight-space programming sessions (feature 016; FR-016.38).

Phase 1 registered ``forge_wsp_start_run``, with its gate (R3A-4, PF-58); every other 016 WSP route
carries a temporary ``ROUTE_EXEMPT`` row until FTASKS 21.1 adds its tool in phase 3. Feature 018
(FR-018.19) adds ``forge_wsp_test_endpoint_protocol``.

``forge_wsp_register_endpoint`` (2026-11-20) PROPOSES a local editor endpoint with no key and no
prices: the route files it as ``wsp_endpoint_register`` and registers it once the operator
approves. The tool declares no ``api_key``, price or ``locality`` argument, so the MCP layer refuses
one by name; the route refuses them again for any agent that calls REST directly. The endpoint
PATCH and DELETE have no tool, for good: they are the operator's (``403 wsp_endpoint_operator_only``
for an agent), on miForge's Editor endpoints screen.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import JOB, READ, WAITS, ToolContext
from ..health_gate import gated_call

#: ``wsp.endpoints.ENDPOINT_ID`` as a JSON-schema pattern (no look-ahead: Rust's regex).
ENDPOINT_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"

StartBody = Annotated[
    dict[str, Any],
    Field(
        description=(
            "The start body, as forge_validate_run's WSP twin takes it: bundle {owner, bundle_id, "
            "bundle_sha256}, kind (inspect | create | repair), mode (session | program), task, "
            "endpoint_id, sensitivity (the bundle's own), primary_sets, and optionally budgets, "
            "change_limits, observation_tools, provider ({id}: weightspace or miwsp_engine; "
            "omitted, the default provider, weightspace until the switch), caller_ref. "
            "miwsp_engine also takes thinking (off | on | default; omitted, the endpoint's "
            "setting) and, with thinking on, thinking_tokens (the reasoning allowance per turn; "
            "omitted, the endpoint's, else 4096). The run freezes the thinking in effect: its "
            "request shows thinking and thinking_tokens, and its list row reasoning_returned. "
            "Both providers take output_cap (on | off; omitted, the endpoint's): off names no "
            "per-turn output cap, so a reply may use the room the context window has left. "
            "miwsp_engine also takes paging (on | off; omitted, the endpoint's, on by default): "
            "on pages bulky tool results out of the editor's history after a few turns (the "
            "editor can recall them) and keeps an experiment ledger in every request; the run's "
            "list row shows pages_out, recalls, ledger_rows and peak_prompt_tokens. weightspace "
            "is refused paging."
        )
    ),
]


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool(
        description=(
            "Start a weight-space programming session: an editor LLM reads and edits the bundle's "
            "weights in an isolated worker, within the frozen token, cost and time budget. Two "
            "providers: weightspace (the default until the switch) and miwsp_engine (the suite's "
            "own engine: one action per turn, in the endpoint's action protocol, tool_calls or "
            "structured; run forge_wsp_test_endpoint_protocol on an endpoint first; it thinks "
            "or not as the endpoint's thinking says, unless the body's thinking overrides it). "
            "The run is queued; poll it by run_id. An inspect through a local endpoint runs at "
            "once."
            + WAITS
            + " Action wsp_session, or wsp_session_external when the endpoint is external."
        )
    )
    async def forge_wsp_start_run(body: StartBody) -> Any:
        """Start a WSP session run."""
        return await gated_call(ctx.gate, JOB, lambda: client.post("/wsp/runs", json_body=body))

    @mcp.tool(
        description=(
            "Test an editor endpoint's action protocol (tool_calls or structured, the operator's "
            "setting on the endpoint): a short inspect session of miwsp_engine on miForge's "
            "built-in test bundle, through the normal run path, the gateway and its metering. "
            "Returns {run_id, status, thinking}; the run's row in the WSP run list then shows "
            "refused_turns (0 means the editor kept to one action per turn), handoffs and, when "
            "the endpoint's thinking is on, reasoning_returned (true when the server sent the "
            "editor's reasoning back, so it honours thinking). Needs miwsp_engine "
            "enabled (provider_unavailable otherwise). A local endpoint's test runs at once."
            + WAITS
            + " Action wsp_session_external when the endpoint is external; the operator changes "
            "an endpoint's protocol, never an agent."
        )
    )
    async def forge_wsp_test_endpoint_protocol(
        endpoint_id: Annotated[str, Field(description="The editor endpoint's id.")],
        caller_ref: Annotated[
            str | None,
            Field(description="Your own reference, kept on the run (at most 64 characters)."),
        ] = None,
    ) -> Any:
        """Start an endpoint's protocol test run."""
        body = {} if caller_ref is None else {"caller_ref": caller_ref}
        return await gated_call(
            ctx.gate,
            JOB,
            lambda: client.post(f"/wsp/endpoints/{endpoint_id}/protocol-test", json_body=body),
        )

    @mcp.tool(
        description=(
            "Propose a LOCAL editor endpoint for WSP sessions: an OpenAI-compatible or Anthropic "
            "server on a private, loopback or cluster host (for example miServer in the cluster, "
            "serving: miserver). It carries no API key and no prices: an agent cannot send "
            "either, and an external endpoint (which needs both) is the operator's, on miForge's "
            "Editor endpoints screen. The route refuses a key, a price, or an external base_url "
            "before filing anything (wsp_endpoint_agent_key_refused, "
            "wsp_endpoint_agent_prices_refused, wsp_endpoint_agent_external_refused), and runs the "
            "registration's own checks (the host allowlist, serving: miserver for a miServer "
            "host, structured and thinking on only on miServer). Once the operator approves, "
            "miForge registers it once and the approval's result is the endpoint (without a "
            "key). Then use its id in forge_wsp_start_run, or run "
            "forge_wsp_test_endpoint_protocol on it." + WAITS + " Action wsp_endpoint_register."
        )
    )
    async def forge_wsp_register_endpoint(
        endpoint_id: Annotated[
            str,
            Field(
                pattern=ENDPOINT_ID_PATTERN,
                description="The new endpoint's id: 1-40 characters of a-z, 0-9, _ and -. Ids "
                "are never reused.",
            ),
        ],
        kind: Annotated[
            Literal["openai_compatible", "anthropic"],
            Field(description="The wire protocol the server speaks."),
        ],
        base_url: Annotated[
            str,
            Field(
                max_length=512,
                description="http(s)://host[:port] with an empty path or exactly /v1; the host "
                "must be local and in miForge's WSP_LOCAL_HOST_ALLOWLIST.",
            ),
        ],
        model: Annotated[
            str,
            Field(min_length=1, max_length=256, description="The model name the server serves."),
        ],
        serving: Annotated[
            Literal["generic", "miserver"],
            Field(description="miserver for a miServer host (required there), else generic."),
        ] = "generic",
        action_protocol: Annotated[
            Literal["tool_calls", "structured"] | None,
            Field(
                description="How miwsp_engine asks for one action per turn: tool_calls (the "
                "default) or structured (an openai_compatible miServer endpoint only)."
            ),
        ] = None,
        context_window_tokens: Annotated[
            int | None,
            Field(
                ge=1,
                le=10_000_000,
                description="The editor's context window; omit to use the window miServer "
                "reports, else the engine's default.",
            ),
        ] = None,
        thinking: Annotated[
            Literal["off", "on", "default"] | None,
            Field(
                description="Whether miwsp_engine asks the editor to think: off (the default; "
                "sends enable_thinking false on miServer), on (enable_thinking true and a "
                "reasoning allowance per turn; an openai_compatible miServer endpoint only) or "
                "default (sends nothing; the server decides)."
            ),
        ] = None,
        thinking_tokens: Annotated[
            int | None,
            Field(
                ge=1,
                le=10_000_000,
                description="The reasoning allowance per turn with thinking on, below the "
                "context window; omit for the engine's default (4096).",
            ),
        ] = None,
        output_cap: Annotated[
            Literal["on", "off"] | None,
            Field(
                description="Whether each editor request names the per-turn output cap: on "
                "(the default) or off (no cap; the gateway sizes one to the context window)."
            ),
        ] = None,
        paging: Annotated[
            Literal["on", "off"] | None,
            Field(
                description="Whether miwsp_engine pages bulky tool results out of the editor's "
                "history and keeps its experiment ledger: on (the default) or off."
            ),
        ] = None,
        output_cap_field: Annotated[
            Literal["max_tokens", "max_completion_tokens"] | None,
            Field(
                description="The request field that caps output; omit for the kind's default "
                "(anthropic: max_tokens, else max_completion_tokens)."
            ),
        ] = None,
    ) -> Any:
        """Propose a local, keyless editor endpoint (waits for the operator)."""
        body: dict[str, Any] = {
            "id": endpoint_id,
            "kind": kind,
            "base_url": base_url,
            "model": model,
            "serving": serving,
        }
        optional = {
            "action_protocol": action_protocol,
            "context_window_tokens": context_window_tokens,
            "thinking": thinking,
            "thinking_tokens": thinking_tokens,
            "output_cap": output_cap,
            "paging": paging,
            "output_cap_field": output_cap_field,
        }
        body.update({k: v for k, v in optional.items() if v is not None})
        return await gated_call(
            ctx.gate, READ, lambda: client.post("/wsp/endpoints", json_body=body)
        )
