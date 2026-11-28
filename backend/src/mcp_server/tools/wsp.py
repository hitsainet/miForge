"""Category ``wsp``: weight-space programming sessions (feature 016; FR-016.38).

Phase 1 registered ``forge_wsp_start_run``, with its gate (R3A-4, PF-58); every other 016 WSP route
carries a temporary ``ROUTE_EXEMPT`` row until FTASKS 21.1 adds its tool in phase 3. Feature 018
(FR-018.19) adds ``forge_wsp_test_endpoint_protocol``. Feature 019 P3 (TID §3.11) adds the read
``forge_wsp_get_pool``.

Feature 020 (FR-020.25): ``forge_wsp_register_endpoint`` declares ``history_mode`` (a local endpoint
may be ``append_only``), and both descriptions name the hosted path's refusals and history modes. No
tool is added.

``forge_wsp_list_endpoints`` (2026-11-25, the MCP gap of ENH-010's finish plan, track A) reads
the editor endpoints: ``GET /wsp/endpoints``'s fields for each, without the prices and the key's
fingerprint (:data:`WITHHELD`). The route never serves a key or its ciphertext.

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
            "change_limits, observation_tools, provider ({id}: miwsp_engine, the default and the "
            "only one; weightspace was retired and is refused provider_unknown), caller_ref. "
            "miwsp_engine also takes thinking (off | on | default, or on an external endpoint an "
            "effort: low | medium | high; omitted, the endpoint's setting) and, with thinking on, "
            "thinking_tokens (the reasoning allowance per turn; omitted, the endpoint's, else "
            "4096). The run freezes the thinking in effect: its request shows thinking and "
            "thinking_tokens, and its list row reasoning_returned. "
            "It takes output_cap (on | off; omitted, the endpoint's): off names no "
            "per-turn output cap, so a reply may use the room the context window has left. "
            "miwsp_engine also takes paging (on | off; omitted, the endpoint's, on by default): "
            "on pages bulky tool results out of the editor's history after a few turns (the "
            "editor can recall them) and keeps an experiment ledger in every request; the run's "
            "list row shows pages_out, recalls, ledger_rows and peak_prompt_tokens. On an "
            "endpoint whose history_mode is append_only, paging is off and a body naming paging "
            "on is refused wsp_history_mode_conflict. "
            "miwsp_engine also takes python (on | off; omitted, on): off gives the editor no "
            "python tool for this run, and the request freezes python: off."
        )
    ),
]


#: What ``forge_wsp_list_endpoints`` leaves out of each endpoint: the four prices
#: (``wsp.endpoints.PRICE_FIELDS``, a literal here because the MCP server does not import the
#: backend) and the key's fingerprint. ``has_key`` (whether one is sealed) stays.
WITHHELD = (
    "price_in_per_mtok",
    "price_out_per_mtok",
    "price_cache_write_per_mtok",
    "price_cache_read_per_mtok",
    "key_fingerprint",
)


#: What the WSP run reads (2026-11-26) remove wherever it appears in an answer, at any depth: the
#: same keys as :data:`WITHHELD`, and ``api_key`` (no route serves one; a read never relays it).
RUN_READ_WITHHELD = (*WITHHELD, "api_key")

#: A WSP run id as ``wsp.runs.new_run_id`` makes it: ``wsp_`` and 12 characters (no slash).
WSP_RUN_ID_PATTERN = r"^wsp_[A-Za-z0-9]{1,12}$"
WspRunId = Annotated[
    str,
    Field(
        pattern=WSP_RUN_ID_PATTERN,
        description="A WSP run id (wsp_ and 12 characters), as forge_wsp_start_run returned it.",
    ),
]

#: The transcript tool's page bounds: entries per page, and the JSON bytes of a page's entries.
TRANSCRIPT_PAGE_DEFAULT, TRANSCRIPT_PAGE_MAX = 20, 100
TRANSCRIPT_BYTES_DEFAULT, TRANSCRIPT_BYTES_MAX = 64 * 1024, 256 * 1024


def withheld_removed(answer: Any) -> Any:
    """``answer`` with every :data:`RUN_READ_WITHHELD` key removed at any depth (a dict's keys,
    inside lists too); everything else, refusals included, as it came."""
    if isinstance(answer, dict):
        return {k: withheld_removed(v) for k, v in answer.items() if k not in RUN_READ_WITHHELD}
    if isinstance(answer, list):
        return [withheld_removed(item) for item in answer]
    return answer


def without_prices_or_key(listing: Any) -> Any:
    """``GET /wsp/endpoints``'s answer with :data:`WITHHELD` removed from every endpoint; any
    other answer (a refusal, an unexpected shape) is returned as it came."""
    if not isinstance(listing, dict) or not isinstance(listing.get("endpoints"), list):
        return listing
    return {
        **listing,
        "endpoints": [
            ({k: v for k, v in row.items() if k not in WITHHELD} if isinstance(row, dict) else row)
            for row in listing["endpoints"]
        ],
    }


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool(
        description=(
            "Start a weight-space programming session: an editor LLM reads and edits the bundle's "
            "weights in an isolated worker, within the frozen token, cost and time budget. The "
            "provider is miwsp_engine (the suite's own engine: one action per turn, in the "
            "endpoint's action protocol, tool_calls or structured; run "
            "forge_wsp_test_endpoint_protocol on an endpoint first; it thinks or not as the "
            "endpoint's thinking says, unless the body's thinking overrides it). "
            "The run is queued; poll it by run_id. An inspect through a local endpoint runs at "
            "once. Runs share a pool of workers, one session each; forge_wsp_get_pool shows "
            "which are busy and why a queued run waits (one active run per editor endpoint, "
            "and per miServer server). Each endpoint keeps its history one of two ways "
            "(history_mode): paging rewrites older tool results into page stubs; append_only "
            "never changes what was sent, so a hosted model's cache serves it (an external "
            "endpoint's default, and an external Anthropic endpoint's only mode). A start on an "
            "external endpoint with no stated context window is refused endpoint_window_required "
            "(the operator sets the window), and one on an external Anthropic endpoint without "
            "both cache prices wsp_endpoint_cache_prices_required (the operator registers it "
            "again); nothing is queued. When the editor refuses the task, the session ends at "
            "once and the run fails with failure.code editor_refused, the category in its "
            "message, and keeps no candidate."
            + WAITS
            + " Action wsp_session, or wsp_session_external when the endpoint is external."
        )
    )
    async def forge_wsp_start_run(body: StartBody) -> Any:
        """Start a WSP session run."""
        return await gated_call(ctx.gate, JOB, lambda: client.post("/wsp/runs", json_body=body))

    @mcp.tool(
        description=(
            "Dry-run a WSP start body: freeze it exactly as forge_wsp_start_run would, with the "
            "same checks and the same refusals (the bundle, the endpoint and its runs_refused, "
            "the protocol test, the budgets, change limits, observation tools, thinking, paging, "
            "python, sensitivity), and return what the run would store: the frozen request, its "
            "request_sha256, the approval action the start would file (null when it would run at "
            "once), the endpoint's locality, the sensitivity and the budgets. Creates no run, "
            "files no approval and calls no editor. Use it to check a body before filing a start "
            "that waits for the operator."
        )
    )
    async def forge_wsp_validate_run(body: StartBody) -> Any:
        """Validate and freeze a WSP start body without starting it."""
        return await gated_call(
            ctx.gate, READ, lambda: client.post("/wsp/runs/validate", json_body=body)
        )

    @mcp.tool(
        description=(
            "Test an editor endpoint's action protocol (tool_calls or structured, the operator's "
            "setting on the endpoint): a short inspect session of miwsp_engine on miForge's "
            "built-in test bundle, through the normal run path, the gateway and its metering. "
            "Returns {run_id, status, thinking}; the run's row in the WSP run list then shows "
            "refused_turns (0 means the editor kept to one action per turn), handoffs and, when "
            "the endpoint's thinking is on, reasoning_returned (true when the server sent the "
            "editor's reasoning back, so it honours thinking). Needs miwsp_engine "
            "enabled (provider_unavailable otherwise). A local endpoint's test runs at once. "
            "The result is stored on the endpoint when the run ends (GET /api/v1/wsp/endpoints "
            "shows protocol_test: passed or failed, the run, when, and the base URL, model and "
            "protocol tested): a pass is a completed run with zero refused turns and at least one "
            "served call. A GENERIC structured endpoint runs nothing but this test until it "
            "passes (wsp_endpoint_protocol_untested, 409), and a change of its protocol clears "
            "the pass; a miServer endpoint needs no pass."
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
            "Read the WSP worker pool: how many workers are busy, idle, restarting (its "
            "session ended and its pod is being replaced; it takes work again once a fresh pod "
            "answers) or not answering, the "
            "pool size and the provisioned maximum, runs queued per editor endpoint, and active "
            "runs that share one server. Read-only. Use it to see why a run waits."
        )
    )
    async def forge_wsp_get_pool() -> Any:
        """Read the WSP worker pool."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/wsp/pool"))

    @mcp.tool(
        description=(
            "List how many times each held-out set has been scored: one row per owner, set "
            "name and row_ids_sha256, with heldout_uses, last_used_at and last_candidate_id. A "
            "use is counted inside a verdict's transaction, so a failed scoring never counts. "
            "Read it before and after an evaluation to see the uses it spent. Read-only."
        )
    )
    async def forge_wsp_list_heldout_uses() -> Any:
        """List each held-out set's use count."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/wsp/heldout-uses"))

    @mcp.tool(
        description=(
            "List the editor endpoints WSP sessions can run on: each one's id, kind, base URL, "
            "model, locality, serving, action protocol, thinking, output cap, paging, history "
            "mode, output ceiling, whether a key is sealed, the context window (stated, read at "
            "registration, read by the latest run, what the next run expects and what a run "
            "uses when its read fails), runs_refused (why a start would be refused today) and "
            "the latest protocol test. Read-only. No key and no prices: those are the "
            "operator's, on miForge's Editor endpoints screen. Use an id in forge_wsp_start_run "
            "or forge_wsp_test_endpoint_protocol."
        )
    )
    async def forge_wsp_list_endpoints() -> Any:
        """List the WSP editor endpoints, without prices or key."""

        async def call() -> Any:
            return without_prices_or_key(await client.get("/wsp/endpoints"))

        return await gated_call(ctx.gate, READ, call)

    @mcp.tool(
        description=(
            "Propose a LOCAL editor endpoint for WSP sessions: an OpenAI-compatible or Anthropic "
            "server on a private, loopback or cluster host (for example miServer in the cluster, "
            "serving: miserver). Agents propose local endpoints only. It carries no API key and "
            "no prices: an agent cannot send either, and an external endpoint (which needs both, "
            "its window and, for Anthropic, both cache prices) is the operator's, on miForge's "
            "Editor endpoints screen. A local endpoint keeps history_mode paging unless you name "
            "append_only (nothing already sent changes). The route refuses a key, a price, or an "
            "external base_url before filing anything (wsp_endpoint_agent_key_refused, "
            "wsp_endpoint_agent_prices_refused, wsp_endpoint_agent_external_refused), and runs the "
            "registration's own checks (the host allowlist, serving: miserver for a miServer "
            "host; structured on any openai_compatible endpoint, a generic one then running only "
            "its protocol test until that passes; thinking on only on miServer). Once the "
            "operator approves, "
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
                "default) or structured (a response_format JSON schema; any openai_compatible "
                "endpoint; a generic one needs a passing forge_wsp_test_endpoint_protocol before "
                "any other run)."
            ),
        ] = None,
        context_window_tokens: Annotated[
            int | None,
            Field(
                ge=1,
                le=10_000_000,
                description="The editor's context window; omit to use the window the server "
                "reports at each run's push (miServer's /api/health/inference, else the "
                "OpenAI-compatible /v1/models max_model_len of vLLM and SGLang, else llama.cpp's "
                "per-slot /props n_ctx), else 8192, which the run records as assumed.",
            ),
        ] = None,
        thinking: Annotated[
            Literal["off", "on", "default"] | None,
            Field(
                description="Whether miwsp_engine asks the editor to think: off (the default; "
                "sends enable_thinking false on miServer), on (enable_thinking true and a "
                "reasoning allowance per turn; an openai_compatible miServer endpoint only) or "
                "default (sends nothing; the server decides). Local values only: the effort "
                "levels are an external endpoint's, set by the operator."
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
        history_mode: Annotated[
            Literal["paging", "append_only"] | None,
            Field(
                description="How miwsp_engine keeps the session's conversation: paging (the "
                "default for a local endpoint; older tool results become page stubs, recalled on "
                "request) or append_only (nothing already sent changes; a long result is cut once "
                "and recalled by offset)."
            ),
        ] = None,
        output_ceiling_tokens: Annotated[
            int | None,
            Field(
                ge=256,
                le=10_000_000,
                description="Most output tokens one editor request may ask for; empty means "
                "32,000. At most the context window when known (stated here, or read from the "
                "server at registration).",
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
            "history_mode": history_mode,
            "output_ceiling_tokens": output_ceiling_tokens,
            "output_cap_field": output_cap_field,
        }
        body.update({k: v for k, v in optional.items() if v is not None})
        return await gated_call(
            ctx.gate, READ, lambda: client.post("/wsp/endpoints", json_body=body)
        )

    @mcp.tool(
        description=(
            "Read one WSP run (an id starting wsp_; forge_get_run reads training runs only): "
            "its row as the WSP run list shows it (status, stop reason, outcome, failure, the "
            "window and output ceiling the push used and their source, editor_wait_since (when "
            "it began waiting for its editor model; a queued run with it set is waiting now), "
            "refused, repeated and kernel-restart counts, handoffs, thinking, paging counters, "
            "usage, metered cost and "
            "cache tokens, and live progress while it runs) and its miforge.wsp-run/v1 document "
            "(the frozen request, sensitivity, usage, hashes, replay, candidate, verdict and "
            "lineage). Read-only. No key and no prices. Then read the editor's transcript with "
            "forge_wsp_get_transcript and the gateway's rows with forge_wsp_get_run_ledger."
        )
    )
    async def forge_wsp_get_run(run_id: WspRunId) -> Any:
        """Read one WSP run's row and document, without prices or key."""

        async def call() -> Any:
            return withheld_removed(await client.get(f"/wsp/runs/{run_id}/record"))

        return await gated_call(ctx.gate, READ, call)

    @mcp.tool(
        description=(
            "Read a page of a WSP run's transcript: the editor's turns, tool calls and tool "
            "results as the worker recorded them, served as data (never interpreted), oldest "
            "first. A page holds at most limit entries and at most max_bytes of entry JSON, but "
            "always at least one entry, so one larger entry is served alone. The answer gives "
            "offset, total (every entry), next_offset (null after the last page; pass it as "
            "offset for the next page), transcript_sha256 and partial (true when a line was cut, "
            "served as {raw}). A run with no result yet has no entries. Read-only. No key and no "
            "prices."
        )
    )
    async def forge_wsp_get_transcript(
        run_id: WspRunId,
        offset: Annotated[int, Field(ge=0, description="Entries to skip (next_offset).")] = 0,
        limit: Annotated[
            int,
            Field(ge=1, le=TRANSCRIPT_PAGE_MAX, description="Most entries in the page."),
        ] = TRANSCRIPT_PAGE_DEFAULT,
        max_bytes: Annotated[
            int,
            Field(
                ge=1024,
                le=TRANSCRIPT_BYTES_MAX,
                description="Most bytes of entry JSON in the page (one larger entry is served "
                "alone).",
            ),
        ] = TRANSCRIPT_BYTES_DEFAULT,
    ) -> Any:
        """Read one bounded page of a WSP run's transcript."""

        async def call() -> Any:
            return withheld_removed(
                await client.get(
                    f"/wsp/runs/{run_id}/transcript",
                    offset=offset,
                    limit=limit,
                    max_bytes=max_bytes,
                )
            )

        return await gated_call(ctx.gate, READ, call)

    @mcp.tool(
        description=(
            "Read a WSP run's gateway ledger: each reservation (the endpoint, whether external, "
            "the worst-case tokens and USD held before an editor call, when it opened and when "
            "it settled, at what cost) and each usage row (the call's sequence number, endpoint, "
            "time, tokens in and out, cost, cache tokens, outcome, HTTP status and latency), "
            "oldest first, with totals over every row of the run and the endpoint ids that "
            "appear. Use it to check a run's spend and that its rows name only its own "
            "endpoint. Each list pages on its own cursor: pass the last id as "
            "after_reservation_id, the last seq as after_seq, while more_reservations or "
            "more_usage is true. Read-only. Counts and codes only: never a prompt, an answer, a "
            "key or a price."
        )
    )
    async def forge_wsp_get_run_ledger(
        run_id: WspRunId,
        after_seq: Annotated[
            int | None, Field(ge=-1, description="Only usage rows after this seq.")
        ] = None,
        after_reservation_id: Annotated[
            int | None, Field(ge=0, description="Only reservations after this id.")
        ] = None,
        limit: Annotated[
            int | None, Field(ge=1, le=200, description="Most rows of each list (200).")
        ] = None,
    ) -> Any:
        """Read a WSP run's reservations and usage rows."""

        async def call() -> Any:
            return withheld_removed(
                await client.get(
                    f"/wsp/runs/{run_id}/ledger",
                    after_seq=after_seq,
                    after_reservation_id=after_reservation_id,
                    limit=limit,
                )
            )

        return await gated_call(ctx.gate, READ, call)

    @mcp.tool(
        description=(
            "Replay a completed WSP session or program run from its stored transcript: a new "
            "run (style replay) that re-executes the recorded tool calls on the same bundle in a "
            "worker, with no editor LLM, no gateway, no token and no GPU, so it costs no tokens "
            "and no USD, and files no approval: it is queued and starts when a pool worker is "
            "free. It holds that worker (CPU) for its length, pushes the bundle into it, and "
            "counts toward your queued-run limit (wsp_queue_limit_reached). It is "
            "never a candidate and spends no held-out use; when it ends its document's replay "
            "says whether it reproduced the original's weights (content_sha256). Returns "
            "{run_id, request_sha256, status: queued}; read it with forge_wsp_get_run. Refused "
            "validation_failed unless the run is a completed session or program run with a "
            "transcript, and replay_unavailable_provider_retired for a retired provider's run."
        )
    )
    async def forge_wsp_replay_run(run_id: WspRunId) -> Any:
        """Replay a completed WSP run from its transcript (ungated: no LLM, no gateway)."""
        return await gated_call(ctx.gate, JOB, lambda: client.post(f"/wsp/runs/{run_id}/replay"))
