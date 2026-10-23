"""Category ``runs``: the run index, a run's record and logs, validation, and starting work.

``forge_submit_run``, ``forge_define_run`` and ``forge_evaluate_now`` wait for the operator
(``gpu_start``; an attach-mode run waits for ``hotupdate_attach``). ``forge_cancel_run`` does
not: stopping is the safe direction (D7). Nor do ``forge_archive_run`` and
``forge_unarchive_run``: they hide and show a finished run and delete nothing (D9).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import JOB, READ, SHA256_PATTERN, WAITS, RunId, Step, ToolContext
from ..health_gate import gated_call

Definition = Annotated[
    dict[str, Any],
    Field(
        description=(
            "An R-02.53 run definition: model_source {mode: load|create|attach, path|repo|"
            "config}, run_type, run_configuration {learning_rate, warmup_steps, batch_size, "
            "...}, signal_source, pack {pack_id, version} or parts, stop_conditions (at least "
            "one), cadences, seed, gpu_request ('auto' or a GPU UUID). forge_validate_run "
            "lists every problem."
        )
    ),
]


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_list_runs(
        limit: Annotated[int, Field(ge=1, le=200, description="Page size.")] = 25,
        offset: Annotated[int, Field(ge=0, description="Items to skip.")] = 0,
        status: Annotated[str | None, Field(max_length=32, description="Filter.")] = None,
        run_type: Annotated[
            str | None, Field(max_length=32, description="A name from forge_list_run_types.")
        ] = None,
        include_archived: Annotated[
            bool, Field(description="Also list archived runs (hidden by default).")
        ] = False,
    ) -> Any:
        """Every run, newest first, a page at a time, with the filter vocabularies. Archived
        runs are left out unless include_archived."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.get(
                "/runs",
                limit=limit,
                offset=offset,
                status=status,
                run_type=run_type,
                include_archived=include_archived,
            ),
        )

    @mcp.tool()
    async def forge_get_run(run_id: RunId) -> Any:
        """One run's record: status, last step, liveness (heartbeat age against the janitor's
        threshold), stop reason, failure, attachment."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/runs/{run_id}"))

    @mcp.tool()
    async def forge_list_steps(
        run_id: RunId,
        limit: Annotated[int | None, Field(ge=1, le=1000, description="Page size.")] = None,
        before_step: Annotated[int | None, Field(ge=0, description="Page before this.")] = None,
    ) -> Any:
        """A run's step records, newest first: loss, grad norm, learning rate, wall time."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.get(f"/runs/{run_id}/steps", limit=limit, before_step=before_step),
        )

    @mcp.tool()
    async def forge_get_run_definition(run_id: RunId) -> Any:
        """What the run was asked to do (submitted) and what that resolved to."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/definition"))

    @mcp.tool()
    async def forge_get_run_stop_conditions(run_id: RunId) -> Any:
        """Every enabled stop condition with its target, latest value and headroom."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/stop-conditions")
        )

    @mcp.tool()
    async def forge_list_boundaries(
        run_id: RunId,
        limit: Annotated[int | None, Field(ge=1, le=500, description="Page size.")] = None,
    ) -> Any:
        """The run's boundary records (test, retention, save, stop), newest first."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/boundaries", limit=limit)
        )

    @mcp.tool()
    async def forge_list_batches(
        run_id: RunId,
        limit: Annotated[int | None, Field(ge=1, le=1000, description="Page size.")] = None,
        before: Annotated[str | None, Field(max_length=64, description="Page cursor.")] = None,
    ) -> Any:
        """The batch records the run consumed: where each came from, by content hash."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.get(f"/runs/{run_id}/batches", limit=limit, before=before),
        )

    @mcp.tool()
    async def forge_get_batch(
        run_id: RunId,
        batch_sha256: Annotated[str, Field(pattern=SHA256_PATTERN, description="Batch hash.")],
    ) -> Any:
        """One batch record."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/batches/{batch_sha256}")
        )

    @mcp.tool()
    async def forge_list_scores(run_id: RunId) -> Any:
        """The run's test and retention scores, each with its sample size and provider."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/scores"))

    @mcp.tool()
    async def forge_get_scores_at_step(run_id: RunId, step: Step) -> Any:
        """The scores recorded at one step, with their per-item detail."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/scores/{step}"))

    @mcp.tool()
    async def forge_get_run_review(run_id: RunId) -> Any:
        """The run's human-review block: queue counts in miDataworks, or not configured."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/review"))

    @mcp.tool()
    async def forge_validate_run(definition: Definition) -> Any:
        """Validate a run definition with the same validator the create path uses. Allocates
        nothing and creates nothing; answers ok or EVERY failure, with the budget against free
        VRAM and the projected footprint."""
        return await gated_call(
            ctx.gate, READ, lambda: client.post("/runs/validate", json_body=definition)
        )

    @mcp.tool()
    async def forge_estimate_budget(
        request: Annotated[
            dict[str, Any],
            Field(
                description="{run_type, config, model_source? | architecture_config? | "
                "param_count?, adapter?, gpu_request?} as POST /budget/estimate. model_source is "
                "the run definition's model source; the server reads the model's size from it."
            ),
        ],
    ) -> Any:
        """Will this run fit? The memory estimate beside each visible GPU's free memory.
        Submits nothing."""
        return await gated_call(
            ctx.gate, READ, lambda: client.post("/budget/estimate", json_body=request)
        )

    @mcp.tool()
    async def forge_preflight_signal_source(
        spec: Annotated[dict[str, Any], Field(description="A signal source spec.")],
    ) -> Any:
        """Check a signal source can be read (parts present, columns resolved) without a run."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.post("/signal-sources/preflight", json_body={"spec": spec}),
        )

    @mcp.tool(
        description=(
            "Submit F1's acquisition-only run (POST /runs): load, create or attach, with the "
            "R-02.13 configuration. Prefer forge_define_run, which records a full definition."
            + WAITS
            + " Action gpu_start, or hotupdate_attach for mode 'attach'."
        )
    )
    async def forge_submit_run(
        request: Annotated[
            dict[str, Any],
            Field(
                description="{mode, path|repo|config, learning_rate, warmup_steps, "
                "batch_size, gradient_accumulation?, optimizer_state_precision?, seed?, run_id?, "
                "gpu_request?}"
            ),
        ],
    ) -> Any:
        """Submit F1's acquisition-only run (POST /runs)."""
        return await gated_call(ctx.gate, JOB, lambda: client.post("/runs", json_body=request))

    @mcp.tool(
        description=(
            "Define and start a run (R-02.53): validated, recorded, leased, enqueued once approved."
            + WAITS
            + " Action gpu_start, or hotupdate_attach when model_source.mode is 'attach'."
        )
    )
    async def forge_define_run(definition: Definition) -> Any:
        """Define and start a run (R-02.53)."""
        return await gated_call(
            ctx.gate, JOB, lambda: client.post("/runs/define", json_body=definition)
        )

    @mcp.tool(
        description=(
            "Evaluate a run now, out of cadence, against its pinned provider (test and/or "
            "retention). Recorded as out-of-cadence, never joining the step series."
            + WAITS
            + " Action gpu_start."
        )
    )
    async def forge_evaluate_now(
        run_id: RunId,
        kinds: Annotated[
            list[Literal["test", "retention"]] | None,
            Field(description="Which sets; omit for both."),
        ] = None,
    ) -> Any:
        """Evaluate a run now, out of cadence."""
        return await gated_call(
            ctx.gate,
            JOB,
            lambda: client.post(f"/runs/{run_id}/evaluate", json_body={"kinds": kinds}),
        )

    @mcp.tool()
    async def forge_cancel_run(run_id: RunId) -> Any:
        """Request a cooperative stop: the run stops at its next step boundary. Runs at once
        (stopping is never gated); poll forge_get_run for the final state."""
        return await gated_call(ctx.gate, READ, lambda: client.post(f"/runs/{run_id}/cancel"))

    @mcp.tool()
    async def forge_archive_run(run_id: RunId) -> Any:
        """Hide a finished (completed, failed or cancelled) run from the default run lists.
        Deletes nothing - its checkpoints, trainer state and history stay - and forge_get_run
        still reads it. A pending or running run is refused (run_not_archivable). Runs at once
        (a non-deleting write); forge_unarchive_run reverses it."""
        return await gated_call(ctx.gate, READ, lambda: client.post(f"/runs/{run_id}/archive"))

    @mcp.tool()
    async def forge_unarchive_run(run_id: RunId) -> Any:
        """Return an archived run to the default run lists. Runs at once."""
        return await gated_call(ctx.gate, READ, lambda: client.post(f"/runs/{run_id}/unarchive"))
