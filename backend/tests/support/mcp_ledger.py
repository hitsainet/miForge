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
CLASSES = (
    "read",
    "plan",
    "write",
    "gpu",
    "gpu_or_attach",
    "attach",
    "delete",
    "config",
    "wsp",
    "wsp_external",
    "wsp_endpoint",
)
GATED_CLASSES = frozenset(
    {"gpu", "gpu_or_attach", "attach", "delete", "config", "wsp", "wsp_external", "wsp_endpoint"}
)
#: class -> the action the gate must decide for the ledger's payload.
CLASS_ACTION = {
    "gpu": "gpu_start",
    "gpu_or_attach": "gpu_start",
    "attach": "hotupdate_attach",
    "delete": "delete",
    "config": "config_write",
    # Feature 016 (FR-016.37): a WSP session through a LOCAL endpoint; an external one escalates
    # to wsp_session_external (the probe stores the ledger's endpoint, mcp_gate_probe.py).
    "wsp": "wsp_session",
    # Feature 018 (FR-018.8): a call that waits only for an EXTERNAL endpoint, and then as
    # wsp_session_external (the protocol test is an inspect: a local one runs at once). Its ledger
    # row names the probe's stored external endpoint, mcp_gate_probe.WSP_LEDGER_EXTERNAL_ENDPOINT.
    "wsp_external": "wsp_session_external",
    # 2026-11-20: an agent's proposal of a local, keyless, unpriced editor endpoint.
    "wsp_endpoint": "wsp_endpoint_register",
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
#: A WSP repair through the local endpoint ``ep-local`` (``mcp_gate_probe.WSP_LEDGER_ENDPOINT``).
_WSP_START: dict[str, Any] = {
    "bundle": {"owner": "miworkbench", "bundle_id": "b-1", "bundle_sha256": "0" * 64},
    "kind": "repair",
    "mode": "session",
    "task": "fewer false alarms",
    "endpoint_id": "ep-local",
    "sensitivity": "not_sensitive",
    "primary_sets": ["eval"],
}

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
    # ---------------------------------------------------------------- wsp (feature 016)
    # Phase 1 registers only the start tool, with its gate (R3A-4, PF-58).
    "forge_wsp_start_run": (
        "POST",
        "/wsp/runs",
        {"body": _WSP_START},
        {"json_body": _WSP_START},
        "wsp",
    ),
    # Feature 018 (FR-018.19): the endpoint protocol test, on the probe's external endpoint.
    "forge_wsp_test_endpoint_protocol": (
        "POST",
        "/wsp/endpoints/ep-ext/protocol-test",
        {"endpoint_id": "ep-ext", "caller_ref": "compare-modes"},
        {"json_body": {"caller_ref": "compare-modes"}},
        "wsp_external",
    ),
    # 2026-11-20: an agent PROPOSES a local editor endpoint; it carries no key and no prices,
    # and the optional fields it was not given are not sent.
    "forge_wsp_register_endpoint": (
        "POST",
        "/wsp/endpoints",
        {
            "endpoint_id": "miserver-gemma4-26b",
            "kind": "openai_compatible",
            "base_url": "http://miserver-backend.miserver.svc:8000/v1",
            "model": "gemma-4-26b",
            "serving": "miserver",
            "action_protocol": "structured",
            "output_ceiling_tokens": 4096,
            # Feature 020 (FR-020.25): a local endpoint may be append_only.
            "history_mode": "append_only",
        },
        {
            "json_body": {
                "id": "miserver-gemma4-26b",
                "kind": "openai_compatible",
                "base_url": "http://miserver-backend.miserver.svc:8000/v1",
                "model": "gemma-4-26b",
                "serving": "miserver",
                "action_protocol": "structured",
                "history_mode": "append_only",
                "output_ceiling_tokens": 4096,
            }
        },
        "wsp_endpoint",
    ),
    # Feature 019 P3 (TID §3.11): the worker pool, a read of the database only. Class "read": the
    # TID wrote "wsp", a GATED class (wsp_session), against its own "READ, no GATED_TOOLS row".
    "forge_wsp_get_pool": ("GET", "/wsp/pool", {}, {}, "read"),
    # 2026-11-27: each held-out set's use count (the phase-2 round read uses_before over REST).
    "forge_wsp_list_heldout_uses": ("GET", "/wsp/heldout-uses", {}, {}, "read"),
    # 2026-11-25 (ENH-010 finish plan, track A): the editor endpoints, a read; the tool drops the
    # prices and the key's fingerprint from the answer (``tools/wsp.py::WITHHELD``).
    "forge_wsp_list_endpoints": ("GET", "/wsp/endpoints", {}, {}, "read"),
    # 2026-11-26 (the 019 P4 MCP gaps). One run's row and document: the record route, not
    # GET /wsp/runs/{id} (which serves the document alone and stays exempt below), so one call
    # answers both; the tool drops RUN_READ_WITHHELD at any depth.
    "forge_wsp_get_run": (
        "GET",
        "/wsp/runs/wsp_0123456789ab/record",
        {"run_id": "wsp_0123456789ab"},
        {},
        "read",
    ),
    # A bounded page of the transcript: the tool always sends limit and max_bytes.
    "forge_wsp_get_transcript": (
        "GET",
        "/wsp/runs/wsp_0123456789ab/transcript",
        {"run_id": "wsp_0123456789ab", "offset": 40, "limit": 10, "max_bytes": 4096},
        {"offset": 40, "limit": 10, "max_bytes": 4096},
        "read",
    ),
    # The gateway's reservation and usage rows; absent cursors are not sent.
    "forge_wsp_get_run_ledger": (
        "GET",
        "/wsp/runs/wsp_0123456789ab/ledger",
        {"run_id": "wsp_0123456789ab", "after_seq": 3, "limit": 50},
        {"after_seq": 3, "after_reservation_id": None, "limit": 50},
        "read",
    ),
    # A replay: no LLM, no gateway, no token, no GPU (PF-13), so ungated, class write.
    "forge_wsp_replay_run": (
        "POST",
        "/wsp/runs/wsp_0123456789ab/replay",
        {"run_id": "wsp_0123456789ab"},
        {"json_body": None},
        "write",
    ),
    # 2026-11-26: a dry run of forge_wsp_start_run's body (the same body, sent as it is), which
    # freezes it and files nothing: class plan, ungated; its route is in READ_ONLY_ROUTES.
    "forge_wsp_validate_run": (
        "POST",
        "/wsp/runs/validate",
        {"body": _WSP_START},
        {"json_body": _WSP_START},
        "plan",
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
    ("POST", "/api/v1/approvals/approve-batch"): (
        "The operator approves waiting settings changes together (ENH-006 D79), never an agent: "
        "an agent call is refused 403 agent_cannot_decide, so a tool would always fail."
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
    # Feature 016 (WSP): TEMPORARY rows (R3A-4, PF-58). The routes land in phase 1 and their
    # tools in phase 3; each row is removed when its tool lands.
    **dict.fromkeys(
        (
            ("GET", "/api/v1/wsp/providers"),
            ("GET", "/api/v1/wsp/runs"),
            ("GET", "/api/v1/wsp/runs/{}"),
            ("POST", "/api/v1/wsp/runs/{}/cancel"),
            ("POST", "/api/v1/wsp/bundles/import"),
            ("GET", "/api/v1/wsp/bundles"),
            ("GET", "/api/v1/wsp/bundles/{}"),
            ("POST", "/api/v1/wsp/weight-ops/inspect"),
            ("POST", "/api/v1/wsp/patches/verify"),
            ("POST", "/api/v1/wsp/patches/stack"),
            ("GET", "/api/v1/wsp/programs/{}"),
            ("POST", "/api/v1/wsp/programs/{}/rerun"),
            # Phase 2 (FTASKS 16.4b): seven more, each with its tool in 21.1.
            ("GET", "/api/v1/wsp/candidates"),
            ("GET", "/api/v1/wsp/candidates/{}"),
            ("GET", "/api/v1/wsp/candidates/{}/lineage"),
            ("GET", "/api/v1/wsp/candidates/{}/run-document"),
            ("POST", "/api/v1/wsp/candidates/{}/evaluate"),
            ("GET", "/api/v1/wsp/candidates/{}/verdict"),
        ),
        "Temporary: tool lands in phase 3, FTASKS 21.1.",
    ),
    # Live progress (2026-11-20): the web UI's two live reads of a run. TEMPORARY, like the run
    # reads above: their tools join forge_get_wsp_run's in phase 3 (FTASKS 21.1).
    **dict.fromkeys(
        (
            ("GET", "/api/v1/wsp/runs/{}/progress"),
            ("GET", "/api/v1/wsp/runs/{}/calls"),
        ),
        "Temporary: the web UI's live WSP reads; tools land with the run tools, FTASKS 21.1.",
    ),
    # Feature 016 phase 2: PERMANENT rows (FTDD §5.2). Binary downloads miWorkbench fetches over
    # REST (its get_patch, R1B-X8, and the P23 reload comparison); a tool would carry raw bytes.
    **dict.fromkeys(
        (
            ("GET", "/api/v1/wsp/candidates/{}/tensors"),
            ("GET", "/api/v1/wsp/candidates/{}/patch/{}"),
            ("GET", "/api/v1/wsp/candidates/{}/verdict/scores"),
        ),
        "Binary download (safetensors, a patch file, float32 scores) read by miWorkbench over "
        "REST; no tool carries raw bytes (FTDD §5.2).",
    ),
    # Feature 016: PERMANENT (FTDD §5.2). Removing an endpoint is the operator's alone (403
    # wsp_endpoint_operator_only for an agent), on the Editor endpoints screen. Its POST twin left
    # this map on 2026-11-20: forge_wsp_register_endpoint proposes a local, keyless endpoint.
    ("DELETE", "/api/v1/wsp/endpoints/{}"): (
        "Operator-only: removing an editor endpoint ends every session that would use it; the "
        "operator does it on the Editor endpoints screen (FR-016.16, D15)."
    ),
    # Feature 018: PERMANENT (FR-018.19). The operator's change of an endpoint's action protocol
    # or context window (403 wsp_endpoint_operator_only for an agent): it decides how every later
    # session on that endpoint talks to the editor, so it is the operator's call, like the
    # registration it amends.
    ("PATCH", "/api/v1/wsp/endpoints/{}"): (
        "Operator-only: an endpoint's action protocol and context window are the operator's "
        "setting (FR-018.8, FR-018.19); an agent is refused 403 wsp_endpoint_operator_only."
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
#: plus the two tokenizer reads and the settings history (013 G3, 2026-10-20) = 90; plus
#: forge_wsp_start_run (feature 016, phase 1) = 91; plus forge_wsp_test_endpoint_protocol (feature
#: 018, FR-018.19) = 92; plus forge_wsp_register_endpoint (2026-11-20) = 93; plus forge_wsp_get_pool
#: (feature 019 P3) = 94; plus forge_wsp_list_endpoints (2026-11-25) = 95; plus forge_wsp_get_run,
#: forge_wsp_get_transcript, forge_wsp_get_run_ledger and forge_wsp_replay_run (2026-11-26) = 99;
#: plus forge_wsp_validate_run (2026-11-26) = 100; plus forge_wsp_list_heldout_uses (2026-11-27)
#: = 101. 118 at 016-P3.
LEDGER_TOOL_TOTAL = 101

#: The ``POST`` routes a READ-ONLY tool calls (017 FR-017.14). A read-only tool's recorded calls
#: are ``GET`` or one of these, and each of these is proved to write nothing by
#: ``tests/integration/test_read_only_routes_write_nothing.py`` (row counts of every table, a
#: listing of every artifact root and of ``HF_HOME``, no Celery task sent). ``forge_validate_run``
#: and ``forge_estimate_budget`` are not here: a Hub source fills the Hub cache, so they are
#: writes (017-D26). A route that turns out to write moves its tool to write, recorded.
READ_ONLY_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/v1/signal-sources/preflight"),  # forge_preflight_signal_source
        ("POST", "/api/v1/parts/validate"),  # forge_validate_part
        ("POST", "/api/v1/runs/{}/checkpoints/{}/branch"),  # forge_plan_branch (a plan)
        ("POST", "/api/v1/runs/{}/continue"),  # forge_plan_continue (submit: false)
        ("POST", "/api/v1/pack-drafts/{}/samples/{}"),  # forge_sample_slot
        ("POST", "/api/v1/runs/{}/checkpoints/{}/verify"),  # forge_verify_checkpoint
        ("POST", "/api/v1/wsp/runs/validate"),  # forge_wsp_validate_run (2026-11-26)
    }
)
