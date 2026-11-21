<!-- GENERATED FILE: DO NOT EDIT BY HAND.
Regenerate from backend/src:
  python -c "from mcp_server.contract import write_contract; write_contract()"
backend/tests/contract/test_mcp_contract_generated.py fails when this differs from the registry. -->

# miForge MCP contract

Every tool the miForge MCP server registers, its category, its title, its hint
(`read` = `readOnlyHint`, `destructive` = `destructiveHint`, `write` = neither),
whether it publishes an output schema, the backend route it calls and, when an agent calls it,
the approval it waits for. Derived from the live registry by `backend/src/mcp_server/contract.py`.

Agents: call `forge_howto` first; it carries the workflow and the result shapes.

**93 tools in 11 categories; 13 wait for the operator.**

## `core` (18)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_health` | Health | read |  | `GET /api/health` |  | miForge's health: PostgreSQL, Redis and each optional integration (miServer, miDataworks, the activation source), each by name with a reason when down. |
| `forge_howto` | How-to guide | read |  | none (served by the MCP server) |  | Guidance for agents: the workflow, the three result shapes (resource, pending approval, unavailable), the seven approval actions, archiving, GPUs, settings and WSP sessions. |
| `forge_host_resources` | Host resources | read |  | `GET /api/v1/host/resources` |  | The host's CPU, memory and each GPU's utilisation and memory, keyed by GPU UUID. |
| `forge_list_gpus` | List GPUs | read |  | `GET /api/v1/gpus` |  | The visible GPUs by UUID (never by index), with name and free memory. |
| `forge_list_model_sources` | List model sources | read |  | `GET /api/v1/models/sources` |  | Every acquisition mode (load, create, attach) and whether each can run here, with the reason when not, plus the model roots a load is confined to. |
| `forge_list_local_models` | List local models | read |  | `GET /api/v1/models/local` |  | Model directories under the model roots: path, architecture, weights, tokenizer. |
| `forge_list_local_tokenizers` | List local tokenizers | read |  | `GET /api/v1/tokenizers/local` |  | Directories under the model roots with tokenizer files at their root (a tokenizer, model or checkpoint directory). |
| `forge_check_hub_revision` | Check Hub revision | read |  | `GET /api/v1/hub/revision` |  | Resolve a Hugging Face repository's revision to its commit, and say whether that commit has a tokenizer, a config and weights at its root. |
| `forge_list_providers` | List providers | read |  | `GET /api/v1/providers` |  | The inference providers (in-process, or any OpenAI-compatible endpoint) and each one's availability. |
| `forge_check_provider` | Check provider | write |  | `POST /api/v1/providers/{name}/check` |  | Send one prompt to a provider and return its answer. |
| `forge_list_run_types` | List run types | read |  | `GET /api/v1/run-types` |  | The three run types (change, create, adapter) from the live registry. |
| `forge_list_stop_conditions` | List stop conditions | read |  | `GET /api/v1/stop-conditions` |  | The five stop conditions, their required parameters and their precedence. |
| `forge_list_refresh_modes` | List refresh modes | read |  | `GET /api/v1/refresh-modes` |  | The refresh modes for a self-generation run's prompt set. |
| `forge_list_selectors` | List selectors | read | yes | `GET /api/v1/selectors` |  | The selector and grader plugins (rule, judge model, reference match, human review, composite) from the live registry. |
| `forge_evaluate_completions` | Evaluate completions | write | yes | `POST /api/v1/selectors/evaluate` |  | Run a selector configuration over completions and return each verdict. |
| `forge_list_approvals` | List approvals | read | yes | `GET /api/v1/approvals` |  | Approval requests: action, who asked, the summary the operator decides on, status, expiry, and the outcome once decided. |
| `forge_list_approval_actions` | List approval actions | read |  | `GET /api/v1/approvals/actions` |  | The six approval actions and what each means. |
| `forge_get_approval_status` | Get approval status | read | yes | `GET /api/v1/approvals/{approval_id}` |  | One approval: pending, executing, executed (result holds the route's answer), failed (error holds its refusal), rejected (with reason) or expired. |

## `runs` (20)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_list_runs` | List runs | read |  | `GET /api/v1/runs` |  | Every run, newest first, a page at a time, with the filter vocabularies. |
| `forge_get_run` | Get run | read |  | `GET /api/v1/runs/{run_id}` |  | One run's record: status, last step, liveness (heartbeat age against the janitor's threshold), stop reason, failure, attachment. |
| `forge_list_steps` | List steps | read |  | `GET /api/v1/runs/{run_id}/steps` |  | A run's step records, newest first: loss, grad norm, learning rate, wall time. |
| `forge_get_run_definition` | Get run definition | read |  | `GET /api/v1/runs/{run_id}/definition` |  | What the run was asked to do (submitted) and what that resolved to. |
| `forge_get_run_stop_conditions` | Get run stop conditions | read |  | `GET /api/v1/runs/{run_id}/stop-conditions` |  | Every enabled stop condition with its target, latest value and headroom. |
| `forge_list_boundaries` | List boundaries | read |  | `GET /api/v1/runs/{run_id}/boundaries` |  | The run's boundary records (test, retention, save, stop), newest first. |
| `forge_list_batches` | List batches | read |  | `GET /api/v1/runs/{run_id}/batches` |  | The batch records the run consumed: where each came from, by content hash. |
| `forge_get_batch` | Get batch | read |  | `GET /api/v1/runs/{run_id}/batches/{batch_sha256}` |  | One batch record. |
| `forge_list_scores` | List scores | read |  | `GET /api/v1/runs/{run_id}/scores` |  | The run's test and retention scores, each with its sample size and provider. |
| `forge_get_scores_at_step` | Get scores at step | read |  | `GET /api/v1/runs/{run_id}/scores/{step}` |  | The scores recorded at one step, with their per-item detail. |
| `forge_get_run_review` | Get run review | read |  | `GET /api/v1/runs/{run_id}/review` |  | The run's human-review block: queue counts in miDataworks, or not configured. |
| `forge_validate_run` | Validate run | write |  | `POST /api/v1/runs/validate` |  | Validate a run definition with the same validator the create path uses. |
| `forge_estimate_budget` | Estimate budget | write |  | `POST /api/v1/budget/estimate` |  | Will this run fit? The memory estimate beside each visible GPU's free memory. |
| `forge_preflight_signal_source` | Preflight signal source | read |  | `POST /api/v1/signal-sources/preflight` |  | Check a signal source can be read (parts present, columns resolved) without a run. |
| `forge_submit_run` | Submit run | write |  | `POST /api/v1/runs` | `gpu_start / hotupdate_attach` | Record an acquisition-only run: load, create or attach a model with a flat optimizer configuration (mode, path\|repo\|config, learning_rate, warmup_steps, batch_size, ...). |
| `forge_define_run` | Define run | destructive |  | `POST /api/v1/runs/define` | `gpu_start / hotupdate_attach` | Define and start a training run from a full run definition (model source, run type, configuration, signal source, pack or parts, stop conditions, cadences, seed, GPU). |
| `forge_evaluate_now` | Evaluate now | write |  | `POST /api/v1/runs/{run_id}/evaluate` | `gpu_start` | Evaluate a run now, out of cadence, against its pinned provider (test and/or retention). |
| `forge_cancel_run` | Cancel run | destructive |  | `POST /api/v1/runs/{run_id}/cancel` |  | Request a cooperative stop: the run stops at its next step boundary. |
| `forge_archive_run` | Archive run | write |  | `POST /api/v1/runs/{run_id}/archive` |  | Hide a finished (completed, failed or cancelled) run from the default run lists. |
| `forge_unarchive_run` | Unarchive run | write |  | `POST /api/v1/runs/{run_id}/unarchive` |  | Return an archived run to the default run lists. |

## `inputs` (9)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_list_signal_sources` | List signal sources | read |  | `GET /api/v1/signal-sources` |  | The four signal sources (corpus, self-generation, preference pairs, scored completions) and what each needs. |
| `forge_list_parts` | List parts | read |  | `GET /api/v1/parts` |  | The indexed dataset parts (prompt sets, corpora, test and retention sets...). |
| `forge_get_part` | Get part | read |  | `GET /api/v1/parts/{part_id}` |  | One part: its provenance, splits and content hash. |
| `forge_validate_part` | Validate part | read |  | `POST /api/v1/parts/validate` |  | Validate a part manifest against the vendored contract. |
| `forge_import_part` | Import part | write |  | `POST /api/v1/parts/import` |  | Import a dataset part (from a directory or miDataworks). |
| `forge_midataworks_status` | miDataworks status | read |  | `GET /api/v1/midataworks/status` |  | Whether miDataworks is configured and reachable, and what is unavailable without it. |
| `forge_midataworks_available_parts` | miDataworks available parts | read |  | `GET /api/v1/midataworks/available-parts` |  | Dataset versions miDataworks offers as parts; refuses by name when not configured. |
| `forge_archive_part` | Archive part | write |  | `POST /api/v1/parts/{part_id}/archive` |  | Hide a part from the default part lists and pickers. |
| `forge_unarchive_part` | Unarchive part | write |  | `POST /api/v1/parts/{part_id}/unarchive` |  | Return an archived part to the default part lists. |

## `checkpoints` (8)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_list_all_checkpoints` | List all checkpoints | read |  | `GET /api/v1/checkpoints` |  | Every indexed checkpoint across runs. |
| `forge_list_checkpoints` | List checkpoints | read |  | `GET /api/v1/runs/{run_id}/checkpoints` |  | One run's checkpoints, by step. |
| `forge_get_checkpoint` | Get checkpoint | read |  | `GET /api/v1/runs/{run_id}/checkpoints/{step}` |  | One checkpoint's manifest: parent, base model, weights hash, scores. |
| `forge_verify_checkpoint` | Verify checkpoint | read |  | `POST /api/v1/runs/{run_id}/checkpoints/{step}/verify` |  | Recompute a checkpoint's weights hash against its manifest. |
| `forge_get_checkpoint_lineage` | Get checkpoint lineage | read |  | `GET /api/v1/runs/{run_id}/checkpoints/{step}/lineage` |  | The parent chain, walked through manifests to the root. |
| `forge_get_tree` | Get tree | read |  | `GET /api/v1/tree` |  | The artifact forest: checkpoints joined on parent, adapters on their base. |
| `forge_request_save` | Request save | write |  | `POST /api/v1/runs/{run_id}/checkpoints` |  | Ask a running run to save at its next boundary. |
| `forge_plan_branch` | Plan branch | read |  | `POST /api/v1/runs/{run_id}/checkpoints/{step}/branch` |  | Plan a branch from any checkpoint: a new run id, parent = this checkpoint. |

## `hotupdate` (3)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_plan_continue` | Plan continue | read |  | `POST /api/v1/runs/{run_id}/continue` |  | Plan continuing a run from a checkpoint: which checkpoint, whether its trainer state is restored or fresh, and why. |
| `forge_continue_run` | Continue run | destructive |  | `POST /api/v1/runs/{run_id}/continue` | `hotupdate_attach` | Continue an attached run from the checkpoint miServer saved for it: the checkpoint is verified, the trainer state restored, and a new attempt of the same run id starts. |
| `forge_attachment_status` | Attachment status | read |  | `GET /api/v1/attachment` |  | miServer's attachment state now, read live: configured, reachable, attached, holder, resident model, weights state. |

## `adapters` (4)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_list_adapters` | List adapters | read |  | `GET /api/v1/adapters` |  | Every adapter, with its base, rank and scaling. |
| `forge_get_adapter` | Get adapter | read |  | `GET /api/v1/adapters/{name}` |  | One adapter's manifest. |
| `forge_list_merges` | List merges | read |  | `GET /api/v1/merges` |  | The merge record: written, interrupted or refused, each with its reason. |
| `forge_merge_adapter` | Merge adapter | write |  | `POST /api/v1/adapters/{name}/merge` | `gpu_start` | Merge an adapter into its base, writing a merged checkpoint of the named run at the named step. |

## `packs` (10)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_list_packs` | List packs | read | yes | `GET /api/v1/packs` |  | Every pack version, with its domain, validity and the runs that used it. |
| `forge_get_pack` | Get pack | read |  | `GET /api/v1/packs/{pack_id}/versions/{version}` |  | One pack version: its slots and where each part comes from. |
| `forge_get_pack_validity` | Get pack validity | read |  | `GET /api/v1/packs/{pack_id}/versions/{version}/validity` |  | Whether a pack version can define a run, and every missing or unresolved slot. |
| `forge_get_pack_lineage` | Get pack lineage | read |  | `GET /api/v1/packs/{pack_id}/versions/{version}/lineage` |  | Where a pack version came from (cloned from, imported) and what derives from it. |
| `forge_export_pack` | Export pack | read |  | `GET /api/v1/packs/{pack_id}/versions/{version}/export` |  | The portable pack document, for another installation's forge_import_pack. |
| `forge_get_pack_part_ref` | Get pack part reference | read |  | `GET /api/v1/pack-part-refs/{part_id}` |  | An indexed part as the reference a pack slot pins. |
| `forge_create_pack` | Create pack | write |  | `POST /api/v1/packs` |  | Create a pack version from slot documents. |
| `forge_import_pack` | Import pack | write |  | `POST /api/v1/packs/import` |  | Import an exported pack document; an unresolvable one is stored and reported. |
| `forge_clone_pack` | Clone pack | write |  | `POST /api/v1/packs/{pack_id}/versions/{version}/clone` |  | Clone a pack version under a new name; the derivation is recorded. |
| `forge_delete_pack` | Delete pack | destructive |  | `DELETE /api/v1/packs/{pack_id}/versions/{version}` | `delete` | Delete a pack version. |

## `drafts` (10)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_list_drafts` | List drafts | read |  | `GET /api/v1/pack-drafts` |  | Every pack draft in progress, with its current step. |
| `forge_get_draft` | Get draft | read |  | `GET /api/v1/pack-drafts/{draft_id}` |  | One draft: its answers so far and each step's state. |
| `forge_review_draft` | Review draft | read |  | `GET /api/v1/pack-drafts/{draft_id}/validity` |  | Step 7: what is still missing, by name. |
| `forge_create_draft` | Create draft | write |  | `POST /api/v1/pack-drafts` |  | Start a pack draft. |
| `forge_commit_draft_step` | Commit draft step | write |  | `PUT /api/v1/pack-drafts/{draft_id}/steps/{number}` |  | Commit one step's answers. |
| `forge_sample_slot` | Sample slot | read |  | `POST /api/v1/pack-drafts/{draft_id}/samples/{slot}` |  | A seeded sample of a slot's part, for the authoring panel. |
| `forge_score_sample` | Score sample | write |  | `POST /api/v1/pack-drafts/{draft_id}/samples/{slot}/score` |  | Score a slot's sample with a selector or grader through a provider. |
| `forge_dry_run_draft` | Dry run draft | write |  | `POST /api/v1/pack-drafts/{draft_id}/dry-run` |  | Try the draft's selector on sampled completions and show each verdict, before committing the pack. |
| `forge_commit_draft` | Commit draft | write |  | `POST /api/v1/pack-drafts/{draft_id}/commit` |  | Commit a complete draft as a pack version. |
| `forge_delete_draft` | Delete draft | destructive |  | `DELETE /api/v1/pack-drafts/{draft_id}` | `delete` | Delete a pack draft. |

## `chains` (4)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_list_chains` | List chains | read |  | `GET /api/v1/chains` |  | Every chain, newest first, with its position, status and halt code. |
| `forge_get_chain` | Get chain | read |  | `GET /api/v1/chains/{chain_id}` |  | One chain: each stage's outcome and the advance decision for each finished stage. |
| `forge_create_chain` | Create chain | destructive |  | `POST /api/v1/chains` | `gpu_start / hotupdate_attach` | Create a chain: two or more run definitions that run one after another on one GPU, each stage starting when the previous one meets its advance_on conditions. |
| `forge_stop_chain` | Stop chain | destructive |  | `POST /api/v1/chains/{chain_id}/stop` |  | Halt a chain and stop its running stage. |

## `settings` (4)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_get_settings` | Get settings | read |  | `GET /api/v1/settings` |  | Every setting: effective value, source (env or override) and whether it is runtime-writable. |
| `forge_get_settings_history` | Get settings history | read |  | `GET /api/v1/settings/history` |  | The settings change history, newest first: each change's old and new value, who made it (the operator, or an agent with the approval that let it run) and why. |
| `forge_set_setting` | Set setting | destructive |  | `PUT /api/v1/settings/{key}` | `config_write` | Set a runtime override for an allow-listed setting (checkpoint pruning, miServer and miDataworks URLs, log_level, hot-update timeouts and tolerances). |
| `forge_reset_setting` | Reset setting | destructive |  | `DELETE /api/v1/settings/{key}` | `config_write` | Clear a runtime override; the environment value applies again. |

## `wsp` (3)

| Tool | Title | Hint | Output schema | Calls | Waits for | What it does |
|---|---|---|---|---|---|---|
| `forge_wsp_start_run` | WSP start run | write |  | `POST /api/v1/wsp/runs` | `wsp_session` | Start a weight-space programming session: an editor LLM reads and edits the bundle's weights in an isolated worker, within the frozen token, cost and time budget. |
| `forge_wsp_test_endpoint_protocol` | WSP test endpoint protocol | write |  | `POST /api/v1/wsp/endpoints/{endpoint_id}/protocol-test` | `wsp_session_external` | Test an editor endpoint's action protocol (tool_calls or structured, the operator's setting on the endpoint): a short inspect session of miwsp_engine on miForge's built-in test bundle, through the normal run path, the gateway and its metering. |
| `forge_wsp_register_endpoint` | WSP register endpoint | write |  | `POST /api/v1/wsp/endpoints` | `wsp_endpoint_register` | Propose a LOCAL editor endpoint for WSP sessions: an OpenAI-compatible or Anthropic server on a private, loopback or cluster host (for example miServer in the cluster, serving: miserver). |

## REST contract for proxies

What this server - and a future miWorkbench proxy - relies on. Additive changes only.

| Item | Value |
|---|---|
| Base | Backend under `/api/`, REST under `/api/v1` |
| Health | `GET /api/health`: `{"status", "dependencies": {name: {"status", "detail"}}, "integrations"}`; 200, or 503 with the same body when a dependency is degraded |
| Success | Plain JSON resources |
| Error | `{"error": {"code", "message", "details"}}` |
| Header | `X-Miforge-Agent: agent:<name>` (lower case, digits, hyphens; at most 48 characters after `agent:`). This server sends `agent:forge-mcp`; a miWorkbench proxy sends `agent:miworkbench-mcp` (`agent:mistudio-mcp` before miWorkbench was renamed; both are accepted, and stored rows keep the id they were written with). A malformed or empty header is refused on every route with `400 invalid_agent_identity`. A request without it is the operator's |
| Pending | `202` with `{"approval_id", "status": "pending", "action", "summary", "request_digest", "expires_at", "hint"}` |
| Status | `GET /api/v1/approvals/{id}`: `pending`, `executing`, `executed` (with `result`), `failed` (with `error`), `rejected` (with `reason`), `expired` |
| Refusals | An agent approve or reject is `403 agent_cannot_decide`; a second approve is `409 approval_not_pending` |
| Approval actions | `gpu_start`, `hotupdate_attach`, `delete`, `config_write` |
| Class by request | A run, define or chain whose (stage-0) `model_source.mode` is `attach` waits for `hotupdate_attach`; `continue` waits only with `submit: true` |
| Validation first | A gated request is validated with the route's own code before it is stored; an invalid one is refused at once and never queued |
| Digest | An approval binds to the exact stored request; it executes once, in-process, with that payload |
| Expiry | A pending approval expires after 24 hours |
| Transport | Streamable HTTP on port 8765 at `/mcp`; `Authorization: Bearer <MCP_AUTH_TOKEN>` on every request but `/health` |
| OpenAPI | `GET /api/openapi.json`; gated operations carry `x-approval-action` (and `x-approval-escalation`) |
