# Origin (design): miDataworks 022 FTDD section 4 (suite ENH-012 CHAIN_PLAN A13), written for
# miForge (017-D11, 017-D15): miDataworks 022 was not yet built when this file was written.
"""One table: whether each MCP tool reads, writes or destroys (FR-017.12 - .18).

Every registered tool gets a title and both ``readOnlyHint`` and ``destructiveHint`` from
``TOOL_CLASS``. ``idempotentHint`` and ``openWorldHint`` stay unset (017-D8). A registered tool
with no row refuses the server's start (017-D3), so a tool added later cannot ship unclassified.

The rules (suite CHAIN_PLAN A13; 017 FTDD section 4), pinned to the route ledger's approval
classes by ``tests/unit/mcp/test_annotations_follow_the_ledger.py``:

- **read**: the tool changes nothing. It starts no job, GPU or model call; files no approval;
  spends no paid call; writes no row, file or artifact. Ledger class ``read`` or ``plan``,
  unless it calls an inference provider (``PROVIDER_CALLING``, 017-D4) or fills the Hub cache
  (``HUB_FETCHING``, 017-D26).
- **destructive**: deletes, purges, clears or cancels; may change the weights miServer serves in
  place; or changes a setting that enables later deletion (enabling pruning deletes
  checkpoints). Ledger classes ``delete``, ``config``, ``gpu_or_attach``, ``attach``, plus the
  cancelling tools ``forge_cancel_run`` and ``forge_stop_chain``.
- **write**: everything else. ``forge_submit_run`` is ledger class ``gpu_or_attach`` but its run
  cannot train (``runs_loop.py``: a run without a recorded definition is refused by name), so it
  is a write, not destructive (017-D5).
"""

from __future__ import annotations

from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

Kind = Literal["read", "write", "destructive"]
READ: Kind = "read"
WRITE: Kind = "write"
DESTRUCTIVE: Kind = "destructive"

#: Ledger class ``plan``, but the tool sends a prompt to an inference provider, which may be
#: paid or hold a GPU (017-D4, A13: "spends no paid call").
PROVIDER_CALLING: frozenset[str] = frozenset(
    {
        "forge_check_provider",  # sends one probe prompt to the named provider
        "forge_evaluate_completions",  # a judge selector calls its provider
        "forge_score_sample",  # scores a sample through a provider
        "forge_dry_run_draft",  # a judge selector calls its provider
    }
)

#: Ledger class ``plan``, but given a Hugging Face Hub source the route writes into the Hub cache
#: under ``HF_HOME`` (017-D26; A13: "writes no row, file or artifact"): ``forge_validate_run``
#: resolves the revision and fetches the model's metadata (the operator's validate fetches the
#: snapshot), ``forge_estimate_budget`` fetches ``config.json`` and the safetensors header.
HUB_FETCHING: frozenset[str] = frozenset({"forge_validate_run", "forge_estimate_budget"})

#: tool name -> kind. One row per registered tool; keep the category order of ``tools/``.
TOOL_CLASS: dict[str, Kind] = {
    # core
    "forge_health": READ,
    "forge_howto": READ,
    "forge_host_resources": READ,
    "forge_list_gpus": READ,
    "forge_list_model_sources": READ,
    "forge_list_local_models": READ,
    "forge_list_local_tokenizers": READ,
    "forge_check_hub_revision": READ,
    "forge_list_providers": READ,
    "forge_check_provider": WRITE,
    "forge_list_run_types": READ,
    "forge_list_stop_conditions": READ,
    "forge_list_refresh_modes": READ,
    "forge_list_selectors": READ,
    "forge_evaluate_completions": WRITE,
    "forge_list_approvals": READ,
    "forge_list_approval_actions": READ,
    "forge_get_approval_status": READ,
    # runs
    "forge_list_runs": READ,
    "forge_get_run": READ,
    "forge_list_steps": READ,
    "forge_get_run_definition": READ,
    "forge_get_run_stop_conditions": READ,
    "forge_list_boundaries": READ,
    "forge_list_batches": READ,
    "forge_get_batch": READ,
    "forge_list_scores": READ,
    "forge_get_scores_at_step": READ,
    "forge_get_run_review": READ,
    "forge_validate_run": WRITE,
    "forge_estimate_budget": WRITE,
    "forge_preflight_signal_source": READ,
    "forge_submit_run": WRITE,
    "forge_define_run": DESTRUCTIVE,
    "forge_evaluate_now": WRITE,
    "forge_cancel_run": DESTRUCTIVE,
    "forge_archive_run": WRITE,
    "forge_unarchive_run": WRITE,
    # inputs
    "forge_list_signal_sources": READ,
    "forge_list_parts": READ,
    "forge_get_part": READ,
    "forge_validate_part": READ,
    "forge_import_part": WRITE,
    "forge_midataworks_status": READ,
    "forge_midataworks_available_parts": READ,
    "forge_archive_part": WRITE,
    "forge_unarchive_part": WRITE,
    # checkpoints
    "forge_list_all_checkpoints": READ,
    "forge_list_checkpoints": READ,
    "forge_get_checkpoint": READ,
    "forge_verify_checkpoint": READ,
    "forge_get_checkpoint_lineage": READ,
    "forge_get_tree": READ,
    "forge_request_save": WRITE,
    "forge_plan_branch": READ,
    # hotupdate
    "forge_plan_continue": READ,
    "forge_continue_run": DESTRUCTIVE,
    "forge_attachment_status": READ,
    # adapters
    "forge_list_adapters": READ,
    "forge_get_adapter": READ,
    "forge_list_merges": READ,
    "forge_merge_adapter": WRITE,
    # packs
    "forge_list_packs": READ,
    "forge_get_pack": READ,
    "forge_get_pack_validity": READ,
    "forge_get_pack_lineage": READ,
    "forge_export_pack": READ,
    "forge_get_pack_part_ref": READ,
    "forge_create_pack": WRITE,
    "forge_import_pack": WRITE,
    "forge_clone_pack": WRITE,
    "forge_delete_pack": DESTRUCTIVE,
    # drafts
    "forge_list_drafts": READ,
    "forge_get_draft": READ,
    "forge_review_draft": READ,
    "forge_create_draft": WRITE,
    "forge_commit_draft_step": WRITE,
    "forge_sample_slot": READ,
    "forge_score_sample": WRITE,
    "forge_dry_run_draft": WRITE,
    "forge_commit_draft": WRITE,
    "forge_delete_draft": DESTRUCTIVE,
    # chains
    "forge_list_chains": READ,
    "forge_get_chain": READ,
    "forge_create_chain": DESTRUCTIVE,
    "forge_stop_chain": DESTRUCTIVE,
    # settings
    "forge_get_settings": READ,
    "forge_get_settings_history": READ,
    "forge_set_setting": DESTRUCTIVE,
    "forge_reset_setting": DESTRUCTIVE,
    # wsp (feature 016): starts a gated WSP session (files an approval, may hold a GPU or spend a
    # paid call); it deletes, cancels and serves nothing, so a write.
    "forge_wsp_start_run": WRITE,
    # feature 018: queues a gated inspect run (files an approval for an external endpoint); it
    # deletes nothing, so a write.
    "forge_wsp_test_endpoint_protocol": WRITE,
    # 2026-11-20: files an approval that registers a local endpoint; deletes nothing, a write.
    "forge_wsp_register_endpoint": WRITE,
}

#: Words a title spells differently from the tool name's lower-case word (017-D2).
ACRONYMS: dict[str, str] = {
    "gpu": "GPU",
    "gpus": "GPUs",
    "midataworks": "miDataworks",
    "hub": "Hub",
    "howto": "How-to",
    "id": "ID",
    "wsp": "WSP",
}

#: Titles the generated form reads badly for.
TITLE_OVERRIDES: dict[str, str] = {
    "forge_howto": "How-to guide",
    "forge_get_pack_part_ref": "Get pack part reference",
}


def title_for(name: str) -> str:
    """``forge_list_gpus`` -> ``List GPUs``: the name without its prefix, in words."""
    if name in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[name]
    words = name.removeprefix("forge_").split("_")
    out = [ACRONYMS.get(word, word) for word in words]
    first = out[0]
    if first == words[0]:  # not an acronym: capitalise the sentence
        out[0] = first[:1].upper() + first[1:]
    return " ".join(out)


def annotations_for(kind: Kind) -> ToolAnnotations:
    """Both hints set; ``idempotentHint`` and ``openWorldHint`` unset (017-D8)."""
    return ToolAnnotations(
        read_only_hint=kind == READ,
        destructive_hint=kind == DESTRUCTIVE,
    )


def apply_annotations(mcp: MCPServer) -> int:
    """Give EVERY registered tool its title and hints. Returns the count.

    Raises ``SystemExit`` naming every registered tool with no row (017-D3): the server does
    not start with an unclassified tool.
    """
    tools = mcp._tool_manager.list_tools()  # the SDK offers no public hook
    missing = sorted(tool.name for tool in tools if tool.name not in TOOL_CLASS)
    if missing:
        raise SystemExit(
            "MCP tools without a row in mcp_server/annotations.py TOOL_CLASS (read, write or "
            f"destructive, by the rules in that module): {', '.join(missing)}. The server "
            "does not start with an unclassified tool."
        )
    for tool in tools:
        tool.title = title_for(tool.name)
        tool.annotations = annotations_for(TOOL_CLASS[tool.name])
    return len(tools)
