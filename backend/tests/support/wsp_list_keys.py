"""The keys every row of miForge's two untyped WSP lists carries (feature 020, FR-020.26).

Plain data, importable with nothing installed: ``tests/integration/wsp/test_list_row_keys.py``
asserts the routes' real answers carry exactly these keys, and miAssist's cross-repo
``test_miforge_wsp_list_overlays_carry_the_keys_miforge_pins`` reads this module from a ``git
archive`` of the commit it pins and requires its FU-10 overlays
(``tests/fixtures/openapi/response_sources/miforge.json``) to list the same keys. Change a row,
change this module, and re-pin miAssist.
"""

from __future__ import annotations

#: Every key of one ``GET /api/v1/wsp/endpoints`` row (``wsp.endpoints.public_document``).
ENDPOINT_ROW_KEYS = {
    "action_protocol",
    "base_url",
    "context_window_tokens",
    "created_at",
    "created_by",
    "has_key",
    "history_mode",
    "id",
    "key_fingerprint",
    "kind",
    "latest_run_window",
    "locality",
    "model",
    "output_cap",
    "output_cap_field",
    "output_ceiling_effective",
    "output_ceiling_tokens",
    "paging",
    "price_cache_read_per_mtok",
    "price_cache_write_per_mtok",
    "price_in_per_mtok",
    "price_out_per_mtok",
    "protocol_test",
    "registration_window",
    "runs_refused",
    "serving",
    "thinking",
    "thinking_tokens",
    "window_if_read_fails",
    "window_now",
    "window_warnings",
}

#: Every key of one ``GET /api/v1/wsp/runs`` row (``api/v1/endpoints/wsp_runs.py`` ``list_runs``).
RUN_ROW_KEYS = {
    "action_protocol",
    "caller",
    "caller_ref",
    "created_at",
    "editor_wait_since",
    "endpoint_id",
    "failure",
    "finished_at",
    "forward_passes_used",
    "handoffs",
    "kernel_restarts",
    "kind",
    "ledger_rows",
    "metered",
    "mode",
    "outcome",
    "outcome_reason",
    "output_cap",
    "output_ceiling_lowered",
    "output_ceiling_tokens",
    "pages_out",
    "paging",
    "peak_prompt_tokens",
    "progress",
    "provider",
    "provider_version",
    "reasoning_returned",
    "reasoning_turns",
    "recalls",
    "refused_turns",
    "repeated_turns",
    "run_id",
    "started_at",
    "status",
    "stop_reason",
    "thinking",
    "thinking_tokens",
    "usage",
    "window_reported_tokens",
    "window_source",
    "window_tokens",
    "window_warnings",
}

#: Route -> the keys of one of its rows: what miAssist compares its overlays with.
ROW_KEYS: dict[str, set[str]] = {
    "GET /api/v1/wsp/endpoints": ENDPOINT_ROW_KEYS,
    "GET /api/v1/wsp/runs": RUN_ROW_KEYS,
}
