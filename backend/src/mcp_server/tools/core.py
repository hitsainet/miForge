"""Category ``core``: health, host, registries, providers, selectors, approvals, guidance.

``forge_howto`` makes no request; it is the server's own guidance, with its counts derived
from the registry.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import CORE, ID_PATTERN, READ, ToolContext
from ..health_gate import gated_call

#: A Hub repository id, ``name`` or ``org/name`` - the same rule the REST route applies.
HUB_REPO_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$"

HOWTO: dict[str, str] = {
    "overview": (
        "miForge makes weights: it creates a model from an architecture config, changes an "
        "existing one in place, or trains an adapter, through one loop - step, test against a "
        "held-out set and a retention set, save a checkpoint. Workflow: forge_list_run_types and "
        "forge_list_stop_conditions -> write a run definition (model_source, run_type, "
        "run_configuration, signal_source, pack or parts, stop_conditions) -> forge_validate_run "
        "(free, creates nothing) -> forge_define_run (waits for the operator) -> poll "
        "forge_get_approval_status -> forge_get_run / forge_list_steps / forge_list_scores."
    ),
    "results": (
        "Three result shapes. (1) The resource. (2) A pending approval: {approval_id, status: "
        "'pending', action, summary, request_digest, expires_at}. The operator approves or "
        "rejects it in miForge's Approvals view; poll forge_get_approval_status until it is "
        "executed (result holds the route's answer), failed (error holds the route's refusal), "
        "rejected or expired. Agents cannot approve (403 agent_cannot_decide). An approval "
        "expires after 24 hours. (3) {unavailable: <dependency>, reason}: the backend, PostgreSQL "
        "or Redis is down; report it, do not retry in a loop. A refusal is raised as a tool error "
        "carrying the backend's error code."
    ),
    "approvals": (
        "Four actions wait for the operator when an agent asks: gpu_start (a run, a chain, an "
        "adapter merge, an out-of-cadence evaluation), hotupdate_attach (training on the weights "
        "miLLM is serving: an attach-mode run or chain, or continuing an attached run), delete "
        "(a pack version or a pack draft) and config_write (a runtime setting). Everything "
        "else - reads, validation, budget estimates, dry-runs, plans, authoring packs and "
        "drafts, importing parts, requesting a save, cancelling a run, stopping a chain, "
        "archiving or unarchiving a run or a part - runs at once. The request is validated "
        "before it is stored, so an invalid one is refused immediately with the route's own "
        "error and never reaches the operator."
    ),
    "archive": (
        "Archive hides a finished run (forge_archive_run) or a part (forge_archive_part) from "
        "the default lists; it deletes nothing - checkpoints, trainer state and history stay, "
        "and forge_get_run / forge_get_part still read it. forge_list_runs and forge_list_parts "
        "take include_archived=true to show them. A pending or running run is refused "
        "(run_not_archivable); continuing an archived run unarchives it. Unarchive reverses it."
    ),
    "gpu": (
        "One run per card, by GPU UUID. A pending approval holds no lease: the card is leased "
        "when the operator approves, so a card taken in between ends the approval failed with "
        "gpu_lease_held - ask again. forge_list_gpus and forge_host_resources show the cards."
    ),
    "settings": (
        "forge_get_settings shows every setting with its source (env or override); secrets are "
        "redacted. forge_set_setting / forge_reset_setting change only the runtime allow-list "
        "(checkpoint pruning, miLLM and miDataworks URLs, log_level, hot-update timeouts and "
        "tolerances) and wait for the operator; pass a reason, it is kept. Enabling pruning "
        "deletes checkpoints. forge_get_settings_history lists past changes, newest first, "
        "with who made each one and the approval that let an agent's change run."
    ),
}


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_health() -> Any:
        """miForge's health: PostgreSQL, Redis and each optional integration (miLLM,
        miDataworks, the activation source), each by name with a reason when down."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/api/health"))

    @mcp.tool()
    async def forge_howto(
        topic: Annotated[
            Literal["overview", "results", "approvals", "gpu", "settings"] | None,
            Field(description="One topic; omit for the list of topics with the overview."),
        ] = None,
    ) -> Any:
        """Guidance for agents: the workflow, the three result shapes (resource, pending
        approval, unavailable), the four approval actions, GPUs and settings. Call this first.
        Makes no request to the backend."""
        if topic is None:
            return {"topics": sorted(HOWTO), "overview": HOWTO["overview"]}
        return {"topic": topic, "text": HOWTO[topic]}

    @mcp.tool()
    async def forge_host_resources() -> Any:
        """The host's CPU, memory and each GPU's utilisation and memory, keyed by GPU UUID."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/host/resources"))

    @mcp.tool()
    async def forge_list_gpus() -> Any:
        """The visible GPUs by UUID (never by index), with name and free memory."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/gpus"))

    @mcp.tool()
    async def forge_list_model_sources() -> Any:
        """Every acquisition mode (load, create, attach) and whether each can run here, with
        the reason when not, plus the model roots a load is confined to."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/models/sources"))

    @mcp.tool()
    async def forge_list_local_models(
        limit: Annotated[int | None, Field(ge=1, le=500, description="Page size.")] = None,
        offset: Annotated[int | None, Field(ge=0, description="Items to skip.")] = None,
        has_weights: Annotated[
            bool | None, Field(description="Only directories that hold weights.")
        ] = None,
    ) -> Any:
        """Model directories under the model roots: path, architecture, weights, tokenizer."""
        return await gated_call(
            ctx.gate,
            CORE,
            lambda: client.get(
                "/models/local", limit=limit, offset=offset, has_weights=has_weights
            ),
        )

    @mcp.tool()
    async def forge_list_local_tokenizers(
        limit: Annotated[int | None, Field(ge=1, le=500, description="Page size.")] = None,
        offset: Annotated[int | None, Field(ge=0, description="Items to skip.")] = None,
    ) -> Any:
        """Directories under the model roots with tokenizer files at their root (a tokenizer,
        model or checkpoint directory). A create run names one as its tokenizer_path."""
        return await gated_call(
            ctx.gate,
            CORE,
            lambda: client.get("/tokenizers/local", limit=limit, offset=offset),
        )

    @mcp.tool()
    async def forge_check_hub_revision(
        repo: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                pattern=HUB_REPO_PATTERN,
                description="Hub repository, such as org/name.",
            ),
        ],
        revision: Annotated[
            str | None,
            Field(
                min_length=1, max_length=200, description="Branch, tag or commit; omit for main."
            ),
        ] = None,
    ) -> Any:
        """Resolve a Hugging Face repository's revision to its commit, and say whether that
        commit has a tokenizer, a config and weights at its root. Downloads nothing."""
        return await gated_call(
            ctx.gate,
            CORE,
            lambda: client.get("/hub/revision", repo=repo, revision=revision),
        )

    @mcp.tool()
    async def forge_list_providers() -> Any:
        """The inference providers (in-process, or any OpenAI-compatible endpoint) and each
        one's availability. Every score records the provider that produced it."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/providers"))

    @mcp.tool()
    async def forge_check_provider(
        name: Annotated[str, Field(pattern=ID_PATTERN, description="Provider name.")],
        prompt: Annotated[
            str | None, Field(max_length=2000, description="A probe prompt; omit for default.")
        ] = None,
    ) -> Any:
        """Send one prompt to a provider and return its answer. Writes nothing."""
        body = {} if prompt is None else {"prompt": prompt}
        return await gated_call(
            ctx.gate, CORE, lambda: client.post(f"/providers/{name}/check", json_body=body)
        )

    @mcp.tool()
    async def forge_list_run_types() -> Any:
        """The three run types (change, create, adapter) from the live registry."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/run-types"))

    @mcp.tool()
    async def forge_list_stop_conditions() -> Any:
        """The five stop conditions, their required parameters and their precedence."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/stop-conditions"))

    @mcp.tool()
    async def forge_list_refresh_modes() -> Any:
        """The refresh modes for a self-generation run's prompt set."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/refresh-modes"))

    @mcp.tool()
    async def forge_list_selectors() -> Any:
        """The selector and grader plugins (rule, judge model, reference match, human review,
        composite) from the live registry."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/selectors"))

    @mcp.tool()
    async def forge_evaluate_completions(
        request: Annotated[
            dict[str, Any],
            Field(description="{config, items, catalog?, source?} as POST /selectors/evaluate."),
        ],
    ) -> Any:
        """Run a selector configuration over completions and return each verdict. Persists
        nothing; a judge selector calls its provider."""
        return await gated_call(
            ctx.gate, CORE, lambda: client.post("/selectors/evaluate", json_body=request)
        )

    @mcp.tool()
    async def forge_list_approvals(
        status: Annotated[
            Literal["pending", "executing", "executed", "rejected", "expired", "failed"] | None,
            Field(description="Filter by status; omit for every approval, newest first."),
        ] = None,
    ) -> Any:
        """Approval requests: action, who asked, the summary the operator decides on, status,
        expiry, and the outcome once decided."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/approvals", status=status))

    @mcp.tool()
    async def forge_list_approval_actions() -> Any:
        """The four approval actions and what each means."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/approvals/actions"))

    @mcp.tool()
    async def forge_get_approval_status(
        approval_id: Annotated[str, Field(pattern=ID_PATTERN, description="Approval id, apr_...")],
    ) -> Any:
        """One approval: pending, executing, executed (result holds the route's answer),
        failed (error holds its refusal), rejected (with reason) or expired. Poll this after a
        tool returned a pending approval."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/approvals/{approval_id}"))
