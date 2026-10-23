"""The route ledger as data: what every MCP tool calls, its approval class, and what is exempt.

The design record is 012 FTDD section 5.2 (83 routes, 83 tools). This module is the form the
guards check against the LIVE OpenAPI document and the LIVE tool registry:

- ``EXPECTED_CALLS`` - every registered tool's one documented call: method, path, the tool's
  arguments, the exact payload it must hand the client, and its **approval class**:
  ``read`` / ``plan`` / ``write`` (no approval) or ``gpu`` / ``gpu_or_attach`` / ``attach`` /
  ``delete`` / ``config`` (the operator decides). The class column is what G9 checks against
  what the live gate DECIDES for that payload - never against a hand list inside a test.
- ``CALLER_ASSERTION_EXEMPT`` - a tool that makes no call, with the reason.
- ``ROUTE_EXEMPT`` - a SERVED route no tool calls, with the reason.
- ``NOT_IN_SCHEMA`` - served paths outside the OpenAPI document, with the reason.

The ledger lives under ``tests/`` so the public mirror keeps it (``0xcc/`` is stripped there).
"""

from __future__ import annotations

import re
from typing import Any

#: Paths here are relative to ``/api/v1`` unless they start with ``/api/``.
API = "/api/v1"

#: The approval classes; the last five wait for the operator.
CLASSES = ("read", "plan", "write", "gpu", "gpu_or_attach", "attach", "delete", "config")
GATED_CLASSES = frozenset({"gpu", "gpu_or_attach", "attach", "delete", "config"})
#: class -> the action the gate must decide for the ledger's payload.
CLASS_ACTION = {
    "gpu": "gpu_start",
    "gpu_or_attach": "gpu_start",
    "attach": "hotupdate_attach",
    "delete": "delete",
    "config": "config_write",
}


def full(path: str) -> str:
    return path if path.startswith("/api/") else API + path


def normalise(template: str) -> str:
    """``/api/v1/runs/{run_id}`` -> ``/api/v1/runs/{}``."""
    return re.sub(r"\{[^}]*\}", "{}", template)


def template_matches(template: str, concrete: str) -> bool:
    """Whether ``concrete`` is a path the route ``template`` serves.

    ``{name}`` on the adapter routes is a PATH converter - an adapter trained on a Hub base is
    named ``owner/repo:run:adapter-N`` - so it matches across a slash; every other parameter
    is one segment.
    """
    regex = re.escape(template).replace(re.escape("{name}"), ".+")
    regex = re.sub(r"\\\{[^}]*\\\}", "[^/]+", regex)
    return re.match("^" + regex + "$", concrete) is not None


_DEFINITION: dict[str, Any] = {
    "run_id": "run-1",
    "model_source": {"mode": "load", "path": "/models/base"},
    "run_type": "change",
    "run_configuration": {"learning_rate": 1e-5, "warmup_steps": 0, "batch_size": 1},
    "signal_source": {"source": "corpus"},
    "parts": [{"part_id": "part-1", "kind": "corpus"}],
    "stop_conditions": [{"condition": "step_count", "config": {"total": 40}}],
}
_SUBMIT: dict[str, Any] = {
    "mode": "load",
    "path": "/models/base",
    "learning_rate": 1e-5,
    "warmup_steps": 0,
    "batch_size": 1,
}
_CHAIN: dict[str, Any] = {
    "name": "curriculum",
    "stages": [
        {
            "definition": {k: v for k, v in _DEFINITION.items() if k != "run_id"},
            "advance_on": ["step_count"],
        },
        {
            "definition": {k: v for k, v in _DEFINITION.items() if k != "run_id"},
            "advance_on": ["step_count"],
        },
    ],
}
_ADAPTER = "LiquidAI/LFM2-1.2B:run-1:adapter-40"

#: tool -> (method, path, tool kwargs, recorded kwargs, approval class)
EXPECTED_CALLS: dict[str, tuple[str, str, dict[str, Any], dict[str, Any], str]] = {
    # ---------------------------------------------------------------- core
    "forge_health": ("GET", "/api/health", {}, {}, "read"),
    "forge_host_resources": ("GET", "/host/resources", {}, {}, "read"),
    "forge_list_gpus": ("GET", "/gpus", {}, {}, "read"),
    "forge_list_model_sources": ("GET", "/models/sources", {}, {}, "read"),
    "forge_list_local_models": (
        "GET",
        "/models/local",
        {"limit": 10, "has_weights": True},
        {"limit": 10, "offset": None, "has_weights": True},
        "read",
    ),
    "forge_list_local_tokenizers": (
        "GET",
        "/tokenizers/local",
        {"limit": 10},
        {"limit": 10, "offset": None},
        "read",
    ),
    "forge_check_hub_revision": (
        "GET",
        "/hub/revision",
        {"repo": "org/tok", "revision": "main"},
        {"repo": "org/tok", "revision": "main"},
        "read",
    ),
    "forge_list_providers": ("GET", "/providers", {}, {}, "read"),
    "forge_check_provider": (
        "POST",
        "/providers/judge/check",
        {"name": "judge", "prompt": "Say hello."},
        {"json_body": {"prompt": "Say hello."}},
        "plan",
    ),
    "forge_list_run_types": ("GET", "/run-types", {}, {}, "read"),
    "forge_list_stop_conditions": ("GET", "/stop-conditions", {}, {}, "read"),
    "forge_list_refresh_modes": ("GET", "/refresh-modes", {}, {}, "read"),
    "forge_list_selectors": ("GET", "/selectors", {}, {}, "read"),
    "forge_evaluate_completions": (
        "POST",
        "/selectors/evaluate",
        {"request": {"config": {"kind": "rule"}, "items": [{"completion": "x"}]}},
        {"json_body": {"config": {"kind": "rule"}, "items": [{"completion": "x"}]}},
        "plan",
    ),
    "forge_list_approvals": (
        "GET",
        "/approvals",
        {"status": "pending"},
        {"status": "pending"},
        "read",
    ),
    "forge_list_approval_actions": ("GET", "/approvals/actions", {}, {}, "read"),
    "forge_get_approval_status": (
        "GET",
        "/approvals/apr_1",
        {"approval_id": "apr_1"},
        {},
        "read",
    ),
    # ---------------------------------------------------------------- runs
    "forge_list_runs": (
        "GET",
        "/runs",
        {"status": "running"},
        {
            "limit": 25,
            "offset": 0,
            "status": "running",
            "run_type": None,
            "include_archived": False,
        },
        "read",
    ),
    "forge_get_run": ("GET", "/runs/run-1", {"run_id": "run-1"}, {}, "read"),
    "forge_list_steps": (
        "GET",
        "/runs/run-1/steps",
        {"run_id": "run-1", "limit": 50},
        {"limit": 50, "before_step": None},
        "read",
    ),
    "forge_get_run_definition": ("GET", "/runs/run-1/definition", {"run_id": "run-1"}, {}, "read"),
    "forge_get_run_stop_conditions": (
        "GET",
        "/runs/run-1/stop-conditions",
        {"run_id": "run-1"},
        {},
        "read",
    ),
    "forge_list_boundaries": (
        "GET",
        "/runs/run-1/boundaries",
        {"run_id": "run-1", "limit": 5},
        {"limit": 5},
        "read",
    ),
    "forge_list_batches": (
        "GET",
        "/runs/run-1/batches",
        {"run_id": "run-1"},
        {"limit": None, "before": None},
        "read",
    ),
    "forge_get_batch": (
        "GET",
        "/runs/run-1/batches/" + "a" * 64,
        {"run_id": "run-1", "batch_sha256": "a" * 64},
        {},
        "read",
    ),
    "forge_list_scores": ("GET", "/runs/run-1/scores", {"run_id": "run-1"}, {}, "read"),
    "forge_get_scores_at_step": (
        "GET",
        "/runs/run-1/scores/40",
        {"run_id": "run-1", "step": 40},
        {},
        "read",
    ),
    "forge_get_run_review": ("GET", "/runs/run-1/review", {"run_id": "run-1"}, {}, "read"),
    "forge_validate_run": (
        "POST",
        "/runs/validate",
        {"definition": _DEFINITION},
        {"json_body": _DEFINITION},
        "plan",
    ),
    "forge_estimate_budget": (
        "POST",
        "/budget/estimate",
        {"request": {"run_type": "change", "config": {"learning_rate": 1e-5}}},
        {"json_body": {"run_type": "change", "config": {"learning_rate": 1e-5}}},
        "plan",
    ),
    "forge_preflight_signal_source": (
        "POST",
        "/signal-sources/preflight",
        {"spec": {"source": "corpus"}},
        {"json_body": {"spec": {"source": "corpus"}}},
        "plan",
    ),
    "forge_submit_run": (
        "POST",
        "/runs",
        {"request": _SUBMIT},
        {"json_body": _SUBMIT},
        "gpu_or_attach",
    ),
    "forge_define_run": (
        "POST",
        "/runs/define",
        {"definition": _DEFINITION},
        {"json_body": _DEFINITION},
        "gpu_or_attach",
    ),
    "forge_evaluate_now": (
        "POST",
        "/runs/run-1/evaluate",
        {"run_id": "run-1", "kinds": ["test"]},
        {"json_body": {"kinds": ["test"]}},
        "gpu",
    ),
    "forge_cancel_run": (
        "POST",
        "/runs/run-1/cancel",
        {"run_id": "run-1"},
        {"json_body": None},
        "write",
    ),
    # Archive (operator 2026-10-20): hide/show, deletes nothing, so ungated (D9).
    "forge_archive_run": (
        "POST",
        "/runs/run-1/archive",
        {"run_id": "run-1"},
        {"json_body": None},
        "write",
    ),
    "forge_unarchive_run": (
        "POST",
        "/runs/run-1/unarchive",
        {"run_id": "run-1"},
        {"json_body": None},
        "write",
    ),
    # ---------------------------------------------------------------- inputs
    "forge_list_signal_sources": ("GET", "/signal-sources", {}, {}, "read"),
    "forge_list_parts": (
        "GET",
        "/parts",
        {"part_kind": "corpus", "include_archived": True},
        {"part_kind": "corpus", "include_archived": True},
        "read",
    ),
    "forge_archive_part": (
        "POST",
        "/parts/part-1/archive",
        {"part_id": "part-1"},
        {"json_body": None},
        "write",
    ),
    "forge_unarchive_part": (
        "POST",
        "/parts/part-1/unarchive",
        {"part_id": "part-1"},
        {"json_body": None},
        "write",
    ),
    "forge_get_part": ("GET", "/parts/part-1", {"part_id": "part-1"}, {}, "read"),
    "forge_validate_part": (
        "POST",
        "/parts/validate",
        {"manifest": {"schema": "midataworks.dataset-version/v1"}},
        {"json_body": {"manifest": {"schema": "midataworks.dataset-version/v1"}}},
        "plan",
    ),
    "forge_import_part": (
        "POST",
        "/parts/import",
        {"request": {"source": "directory", "locator": "/data/parts/x"}},
        {"json_body": {"source": "directory", "locator": "/data/parts/x"}},
        "write",
    ),
    "forge_midataworks_status": (
        "GET",
        "/midataworks/status",
        {"probe": True},
        {"probe": True},
        "read",
    ),
    "forge_midataworks_available_parts": (
        "GET",
        "/midataworks/available-parts",
        {},
        {"kind": None},
        "read",
    ),
    # ---------------------------------------------------------------- checkpoints
    "forge_list_all_checkpoints": (
        "GET",
        "/checkpoints",
        {"run_id": "run-1"},
        {"run_id": "run-1", "domain": None, "base_model": None, "limit": None},
        "read",
    ),
    "forge_list_checkpoints": (
        "GET",
        "/runs/run-1/checkpoints",
        {"run_id": "run-1"},
        {},
        "read",
    ),
    "forge_get_checkpoint": (
        "GET",
        "/runs/run-1/checkpoints/40",
        {"run_id": "run-1", "step": 40},
        {},
        "read",
    ),
    "forge_verify_checkpoint": (
        "POST",
        "/runs/run-1/checkpoints/40/verify",
        {"run_id": "run-1", "step": 40},
        {"json_body": None},
        "read",
    ),
    "forge_get_checkpoint_lineage": (
        "GET",
        "/runs/run-1/checkpoints/40/lineage",
        {"run_id": "run-1", "step": 40},
        {"limit": None},
        "read",
    ),
    "forge_get_tree": (
        "GET",
        "/tree",
        {"domain": "legal"},
        {"run_id": None, "domain": "legal", "kind": None, "limit": None, "offset": None},
        "read",
    ),
    "forge_request_save": (
        "POST",
        "/runs/run-1/checkpoints",
        {"run_id": "run-1"},
        {"json_body": None},
        "write",
    ),
    "forge_plan_branch": (
        "POST",
        "/runs/run-1/checkpoints/40/branch",
        {"run_id": "run-1", "step": 40, "new_run_id": "run-2"},
        {"json_body": {"run_id": "run-2", "optimizer_config_hash": None}},
        "plan",
    ),
    # ---------------------------------------------------------------- hotupdate
    "forge_plan_continue": (
        "POST",
        "/runs/run-1/continue",
        {"run_id": "run-1", "step": 100},
        {"json_body": {"step": 100, "optimizer_config_hash": None, "submit": False}},
        "plan",
    ),
    "forge_continue_run": (
        "POST",
        "/runs/run-1/continue",
        {"run_id": "run-1"},
        {"json_body": {"step": None, "optimizer_config_hash": None, "submit": True}},
        "attach",
    ),
    "forge_attachment_status": ("GET", "/attachment", {}, {}, "read"),
    # ---------------------------------------------------------------- adapters
    "forge_list_adapters": ("GET", "/adapters", {}, {"run_id": None}, "read"),
    "forge_get_adapter": ("GET", f"/adapters/{_ADAPTER}", {"name": _ADAPTER}, {}, "read"),
    "forge_list_merges": (
        "GET",
        "/merges",
        {"limit": 10},
        {"adapter_name": None, "limit": 10},
        "read",
    ),
    "forge_merge_adapter": (
        "POST",
        f"/adapters/{_ADAPTER}/merge",
        {"name": _ADAPTER, "run_id": "run-1", "step": 41},
        {"json_body": None, "run_id": "run-1", "step": 41},
        "gpu",
    ),
    # ---------------------------------------------------------------- packs
    "forge_list_packs": ("GET", "/packs", {}, {}, "read"),
    "forge_get_pack": (
        "GET",
        "/packs/legal/versions/1",
        {"pack_id": "legal", "version": "1"},
        {},
        "read",
    ),
    "forge_get_pack_validity": (
        "GET",
        "/packs/legal/versions/1/validity",
        {"pack_id": "legal", "version": "1"},
        {},
        "read",
    ),
    "forge_get_pack_lineage": (
        "GET",
        "/packs/legal/versions/1/lineage",
        {"pack_id": "legal", "version": "1"},
        {},
        "read",
    ),
    "forge_export_pack": (
        "GET",
        "/packs/legal/versions/1/export",
        {"pack_id": "legal", "version": "1"},
        {},
        "read",
    ),
    "forge_get_pack_part_ref": ("GET", "/pack-part-refs/part-1", {"part_id": "part-1"}, {}, "read"),
    "forge_create_pack": (
        "POST",
        "/packs",
        {"request": {"name": "legal", "slots": {}}},
        {"json_body": {"name": "legal", "slots": {}}},
        "write",
    ),
    "forge_import_pack": (
        "POST",
        "/packs/import",
        {"document": {"pack": {"name": "legal"}}},
        {"json_body": {"pack": {"name": "legal"}}},
        "write",
    ),
    "forge_clone_pack": (
        "POST",
        "/packs/legal/versions/1/clone",
        {"pack_id": "legal", "version": "1", "into": "legal-v2"},
        {"json_body": {"into": "legal-v2", "version": "1"}},
        "write",
    ),
    "forge_delete_pack": (
        "DELETE",
        "/packs/legal/versions/1",
        {"pack_id": "legal", "version": "1"},
        {},
        "delete",
    ),
    # ---------------------------------------------------------------- drafts
    "forge_list_drafts": ("GET", "/pack-drafts", {}, {}, "read"),
    "forge_get_draft": ("GET", "/pack-drafts/draft-1", {"draft_id": "draft-1"}, {}, "read"),
    "forge_review_draft": (
        "GET",
        "/pack-drafts/draft-1/validity",
        {"draft_id": "draft-1"},
        {},
        "read",
    ),
    "forge_create_draft": (
        "POST",
        "/pack-drafts",
        {"name": "legal"},
        {"json_body": {"name": "legal", "version": "1", "domain": None}},
        "write",
    ),
    "forge_commit_draft_step": (
        "PUT",
        "/pack-drafts/draft-1/steps/2",
        {"draft_id": "draft-1", "number": 2, "answers": {"domain": "legal"}},
        {"json_body": {"domain": "legal"}},
        "write",
    ),
    "forge_sample_slot": (
        "POST",
        "/pack-drafts/draft-1/samples/test_set",
        {"draft_id": "draft-1", "slot": "test_set"},
        {"json_body": None},
        "plan",
    ),
    "forge_score_sample": (
        "POST",
        "/pack-drafts/draft-1/samples/test_set/score",
        {"draft_id": "draft-1", "slot": "test_set", "request": {"config": {"kind": "rule"}}},
        {"json_body": {"config": {"kind": "rule"}}},
        "plan",
    ),
    "forge_dry_run_draft": (
        "POST",
        "/pack-drafts/draft-1/dry-run",
        {"draft_id": "draft-1", "request": {"config": {"kind": "rule"}}},
        {"json_body": {"config": {"kind": "rule"}}},
        "plan",
    ),
    "forge_commit_draft": (
        "POST",
        "/pack-drafts/draft-1/commit",
        {"draft_id": "draft-1"},
        {"json_body": None},
        "write",
    ),
    "forge_delete_draft": (
        "DELETE",
        "/pack-drafts/draft-1",
        {"draft_id": "draft-1"},
        {},
        "delete",
    ),
    # ---------------------------------------------------------------- chains
    "forge_list_chains": ("GET", "/chains", {}, {"limit": None}, "read"),
    "forge_get_chain": ("GET", "/chains/chain-1", {"chain_id": "chain-1"}, {}, "read"),
    "forge_create_chain": (
        "POST",
        "/chains",
        {"spec": _CHAIN},
        {"json_body": _CHAIN},
        "gpu_or_attach",
    ),
    "forge_stop_chain": (
        "POST",
        "/chains/chain-1/stop",
        {"chain_id": "chain-1", "requested_by": "agent review"},
        {"json_body": {"requested_by": "agent review"}},
        "write",
    ),
    # ---------------------------------------------------------------- settings
    "forge_get_settings": ("GET", "/settings", {}, {}, "read"),
    # 013 G3: the change history, filtered and limited exactly as asked.
    "forge_get_settings_history": (
        "GET",
        "/settings/history",
        {"key": "log_level", "limit": 7},
        {"key": "log_level", "limit": 7},
        "read",
    ),
    "forge_set_setting": (
        "PUT",
        "/settings/log_level",
        {"key": "log_level", "value": "DEBUG", "reason": "chasing a slow save"},
        {"json_body": {"value": "DEBUG", "reason": "chasing a slow save"}},
        "config",
    ),
    "forge_reset_setting": (
        "DELETE",
        "/settings/log_level",
        {"key": "log_level", "reason": "done"},
        {"reason": "done"},
        "config",
    ),
}

#: A registered tool that makes no backend call, with the reason.
CALLER_ASSERTION_EXEMPT: dict[str, str] = {
    "forge_howto": "Guidance served by the MCP server itself; it makes no request to the backend.",
}

#: (METHOD, normalised template) -> why no tool calls this SERVED route.
#: The exemption reason the operator gave for every purge route (2026-10-20).
PURGE_EXEMPT_REASON = "operator-only permanent delete"

ROUTE_EXEMPT: dict[tuple[str, str], str] = {
    ("POST", "/api/v1/approvals/{}/approve"): (
        "The operator decides, never an agent: an agent call is refused 403 agent_cannot_decide "
        "(FR-012.13), so a tool would be a tool that always fails."
    ),
    ("POST", "/api/v1/approvals/{}/reject"): (
        "The operator decides, never an agent: an agent call is refused 403 agent_cannot_decide "
        "(FR-012.13), so a tool would be a tool that always fails."
    ),
    # Purge (operator decision 2026-10-20): never an agent's. The router refuses any request
    # carrying X-Miforge-Agent with 403 purge_operator_only, so a tool would always fail - and
    # the operator ruled it must not exist. `test_purge.py` asserts these stay out of the
    # registry and in this map.
    **dict.fromkeys(
        (
            ("POST", "/api/v1/runs/purge/plan"),
            ("POST", "/api/v1/runs/purge"),
            ("POST", "/api/v1/runs/{}/purge"),
            ("POST", "/api/v1/parts/purge/plan"),
            ("POST", "/api/v1/parts/purge"),
            ("POST", "/api/v1/parts/{}/purge"),
            ("GET", "/api/v1/purges"),
        ),
        PURGE_EXEMPT_REASON,
    ),
}

#: Served paths the OpenAPI document does not list, with the reason no tool calls them.
NOT_IN_SCHEMA: dict[str, str] = {
    "/api/openapi.json": "The schema document itself; served for proxies, not a capability.",
    "/api/docs": "Swagger UI, human-facing.",
    "/docs/oauth2-redirect": "Swagger helper; FastAPI default, no capability.",
    "/socket.io": "Live push; tools poll the run record instead (ADR-007, ADR-013).",
}

#: 83 tools, as FTDD section 5.2 counts them, plus the four archive tools (operator 2026-10-20),
#: plus the settings history (013 G3, 2026-10-20).
LEDGER_TOOL_TOTAL = 90
