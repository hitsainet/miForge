"""FTDD §5.6: every WSP refusal code has the status the design names (FTASKS 7.2's table half).

The expected table is transcribed from FTDD §5.6 here, independently of ``api/app.py``, so a code
dropped from or mis-mapped in ``REFUSAL_STATUS`` turns this red. The two ``*_not_configured`` codes
take the suffix rule (503) without a row.
"""

from __future__ import annotations

import pytest

from api.app import REFUSAL_STATUS, status_for

FTDD_5_6 = {
    "provider_unavailable": 503,
    "provider_unknown": 422,
    "provider_operation_unsupported": 422,
    "wsp_bundle_invalid": 422,
    "wsp_bundle_hash_mismatch": 422,
    "wsp_bundle_format_refused": 422,
    "wsp_bundle_parity_failed": 422,
    "wsp_bundle_not_found": 404,
    "wsp_bundle_fetch_failed": 502,
    "wsp_sensitivity_mismatch": 409,
    "wsp_scorer_busy": 429,
    "wsp_bundle_too_large": 413,
    # Production defect 2026-11-20 (FTDD §5.6 row added with it).
    "wsp_bundle_too_large_for_worker": 413,
    # Fix 2026-11-21 (FTDD §5.6 row added with it).
    "wsp_time_budget_too_small": 422,
    "wsp_patch_malformed": 422,
    "wsp_patch_base_mismatch": 409,
    "wsp_patch_not_chained": 409,
    "wsp_change_limit_exceeded": 422,
    "wsp_feature_renumbered": 422,
    "wsp_phase_not_enabled": 409,
    "wsp_endpoint_unknown": 422,
    "wsp_endpoint_operator_only": 403,
    "wsp_endpoint_pricing_required": 422,
    "wsp_sensitive_data_external_endpoint": 403,
    "wsp_cost_cap_above_maximum": 422,
    "wsp_session_not_active": 409,
    "wsp_reservation_closed": 409,
    "wsp_verdict_not_found": 404,
    "wsp_request_in_flight": 409,
    "wsp_endpoint_changed": 409,
    "wsp_endpoint_url_refused": 422,
    "wsp_queue_limit_reached": 429,
    "wsp_bundle_not_imported": 409,
    "wsp_sensitivity_raised": 409,
    "wsp_request_field_refused": 422,
    "wsp_daily_spend_cap_reached": 429,
    "wsp_endpoint_key_binding_failed": 409,
    "wsp_approval_action_stale": 409,
    "wsp_program_nondeterministic": 422,
    "worker_not_idle": 409,
    "worker_file_mismatch": 422,
    "wsp_heldout_call_cap_reached": 429,
    "wsp_launcher_refused_in_production": 409,
    "wsp_request_sha256_mismatch": 409,
    "wsp_run_not_found": 404,
    "wsp_candidate_not_found": 404,
    "wsp_program_not_found": 404,
    "wsp_run_not_cancellable": 409,
    "wsp_result_invalid": 422,
    "wsp_inspect_changed_weights": 422,
    "wsp_program_invalid": 422,
    "wsp_session_token_invalid": 401,
    "wsp_gateway_token_invalid": 401,
    "wsp_budget_exhausted": 429,
    "wsp_streaming_unsupported": 422,
    "wsp_resource_limit": 409,
    # Feature 018 (FTDD §5.6).
    "wsp_endpoint_protocol_unsupported": 422,
    "wsp_endpoint_thinking_unsupported": 422,
}


@pytest.mark.parametrize(("code", "status"), sorted(FTDD_5_6.items()))
def test_each_code_has_its_status(code: str, status: int) -> None:
    assert REFUSAL_STATUS.get(code) == status, code


@pytest.mark.parametrize("code", ["wsp_not_configured", "miworkbench_not_configured"])
def test_not_configured_codes_take_the_suffix_rule(code: str) -> None:
    assert code not in REFUSAL_STATUS and status_for(code) == 503
