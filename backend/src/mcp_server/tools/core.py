"""Category ``core``: health, host, registries, providers, selectors, approvals, guidance.

``forge_howto`` makes no request; it is the server's own guidance, with its counts derived
from the registry.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import ANONYMOUS_HUB, CORE, ID_PATTERN, READ, ToolContext
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
        "or Redis is down; report it, do not retry in a loop. A pending approval also carries "
        "approval {id, status, decide_in: miforge, then: server_runs, summary}, the suite's "
        "shared shape: miForge runs the request itself once approved, so poll "
        "forge_get_approval_status and never call the tool again. Answers where the workflow "
        "continues carry next_steps [{tool, args, why}]. A failed call is an error whose text "
        "is four parts: line 1 '<code>: <message>' (the code lower case), then 'fix: ...', "
        "then 'next: [...]' (a JSON list of next steps), then the original error text."
    ),
    "approvals": (
        "Seven actions wait for the operator when an agent asks: gpu_start (a run, a chain, an "
        "adapter merge, an out-of-cadence evaluation), hotupdate_attach (training on the weights "
        "miServer is serving: an attach-mode run or chain, or continuing an attached run), delete "
        "(a pack version or a pack draft), config_write (a runtime setting), wsp_session or "
        "wsp_session_external (a weight-space programming session), and wsp_endpoint_register "
        "(a local editor endpoint an agent proposed). Everything "
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
        "gpu_lease_held - ask again. forge_list_gpus and forge_host_resources show the cards. "
        "Every Hugging Face Hub fetch made for an agent - checking a revision, validating or "
        "estimating a definition, and the run's own downloads once approved - is anonymous: "
        "it never uses a credential on the miForge host, so a private or gated repository is "
        "refused (model_revision_not_found or model_fetch_failed), and validating for an agent "
        "reads only the model's metadata, never its weights."
    ),
    "settings": (
        "forge_get_settings shows every setting with its source (env or override); secrets are "
        "redacted. forge_set_setting / forge_reset_setting change only the runtime allow-list "
        "(checkpoint pruning, miServer and miDataworks URLs, log_level, hot-update timeouts and "
        "tolerances) and wait for the operator; pass a reason, it is kept. Enabling pruning "
        "deletes checkpoints. forge_get_settings_history lists past changes, newest first, "
        "with who made each one and the approval that let an agent's change run."
    ),
    "wsp": (
        "Weight-space programming: an editor LLM reads and edits a probe's weights in an isolated "
        "worker. forge_wsp_start_run queues a session run from a start body (an imported bundle, "
        "kind inspect, create or repair, an editor endpoint, the bundle's own sensitivity, the "
        "primary held-out sets). It waits for the operator as wsp_session, or as "
        "wsp_session_external when the endpoint is external (the practice data and weights leave "
        "this network); an inspect through a local endpoint runs at once. A sensitive or "
        "unknown-sensitivity bundle is refused for an external endpoint. The run's result is a "
        "candidate patch, never a change to the bundle itself. Two providers: weightspace (the "
        "default until the switch) and miwsp_engine (the suite's own engine), picked by the start "
        "body's provider {id}. miwsp_engine takes one action per turn in the endpoint's "
        "action_protocol: tool_calls (any endpoint) or structured (a miServer OpenAI-kind "
        "endpoint only); an endpoint may also carry context_window_tokens, else the engine uses "
        "the window miServer reports, and thinking (off, the default; on, a miServer OpenAI-kind "
        "endpoint only, with thinking_tokens as the reasoning allowance per turn, default "
        "4096; or default, sending nothing), output_cap (on, the default; off names no "
        "per-turn output cap) and paging (on, the default: miwsp_engine pages bulky tool "
        "results out of the history after a few turns, offers recall(page) and keeps an "
        "experiment ledger in every request; off). A start body's thinking (and "
        "thinking_tokens), output_cap and paging override the endpoint's for that run, and the "
        "run freezes the values in effect. "
        "Changing any of these on an existing endpoint is the operator's, never an agent's. "
        "forge_wsp_register_endpoint proposes a NEW local endpoint (a private, "
        "loopback or cluster host, such as miServer in the cluster with serving: miserver) with "
        "its protocol and window: it carries no API key and no prices, and waits for the "
        "operator as wsp_endpoint_register; a key, a price or an external base_url is refused "
        "before anything is filed. External endpoints, keys, prices, changes and removals are "
        "the operator's, on miForge's Editor endpoints screen. "
        "forge_wsp_test_endpoint_protocol runs a short miwsp_engine inspect on a "
        "built-in bundle to test an endpoint's protocol: its run row shows refused_turns (0 is "
        "clean) and, with thinking on, reasoning_returned (whether the server sent reasoning); "
        "it runs at once for a local endpoint and waits as wsp_session_external for an "
        "external one. The other WSP tools arrive in a later release."
    ),
}


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_health() -> Any:
        """miForge's health: PostgreSQL, Redis and each optional integration (miServer,
        miDataworks, the activation source), each by name with a reason when down."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/api/health"))

    @mcp.tool()
    async def forge_howto(
        topic: Annotated[
            Literal["overview", "results", "approvals", "archive", "gpu", "settings", "wsp"] | None,
            Field(description="One topic; omit for the list of topics with the overview."),
        ] = None,
    ) -> Any:
        """Guidance for agents: the workflow, the three result shapes (resource, pending
        approval, unavailable), the seven approval actions, archiving, GPUs, settings and WSP
        sessions. Call this first.
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

    @mcp.tool(
        description=(
            "Resolve a Hugging Face repository's revision to its commit, and say whether that "
            "commit has a tokenizer, a config and weights at its root. Downloads nothing."
            + ANONYMOUS_HUB
        )
    )
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
        """The six approval actions and what each means."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/approvals/actions"))

    @mcp.tool()
    async def forge_get_approval_status(
        approval_id: Annotated[str, Field(pattern=ID_PATTERN, description="Approval id, apr_...")],
    ) -> Any:
        """One approval: pending, executing, executed (result holds the route's answer),
        failed (error holds its refusal), rejected (with reason) or expired. Poll this after a
        tool returned a pending approval."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/approvals/{approval_id}"))
