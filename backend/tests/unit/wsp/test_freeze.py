"""FR-016.22: the frozen request and its digest (FTASKS 7.7)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from core.errors import Refusal
from tests.support.wsp_fixtures import make_bundle
from wsp.freeze import approval_action, effective_sensitivity, freeze
from wsp.settings import WSPSettings
from wsp.weight_ops.hashing import jcs_sha256

LOCAL = {"id": "ep-local", "locality": "local", "model": "qwen"}
EXTERNAL = {"id": "ep-claude", "locality": "external", "model": "claude"}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> dict[str, Any]:
    return make_bundle(tmp_path_factory.mktemp("b"), rule="mean").manifest


def body(bundle: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bundle": {
            "owner": "miworkbench",
            "bundle_id": bundle["bundle_id"],
            "bundle_sha256": bundle["bundle_sha256"],
        },
        "kind": "repair",
        "mode": "session",
        "task": "reduce false alarms on everyday text",
        "endpoint_id": "ep-local",
        "budgets": {"tokens": 5000, "time_s": 600, "max_steps": 20, "forward_passes": 100},
        "change_limits": {
            "max_changed_entries": 64,
            "allowed_tensors": None,
            "frozen_tensors": ["mean", "std"],
        },
        "observation_tools": [{"name": "practice_metrics", "version": "2", "cost": 1}],
        "primary_sets": ["eval"],
        "sensitivity": bundle["sensitivity"]["value"],
    }
    out.update(overrides)
    return out


def run(
    bundle: dict[str, Any],
    b: dict[str, Any],
    *,
    endpoint=LOCAL,
    queued: int = 0,
    settings=None,
    **kw,
):
    return freeze(
        b,
        bundle=bundle,
        bundle_imported=kw.get("imported", True),
        endpoint=endpoint,
        provider_version="0.1.0",
        queued_for_caller=queued,
        settings=settings or WSPSettings(),
    )


def _refused(fn) -> Refusal:
    with pytest.raises(Refusal) as err:
        fn()
    return err.value


def test_equal_requests_give_equal_digests(bundle) -> None:
    a, b = run(bundle, body(bundle)), run(bundle, body(bundle))
    assert a.request == b.request and a.request_sha256 == b.request_sha256
    assert a.request_sha256 == jcs_sha256(a.request)  # a stored request re-hashes to its digest


#: (dotted path in the start body, a different value). Every field the frozen request covers.
CHANGES = [
    ("task", "something else"),
    ("mode", "program"),
    ("kind", "create"),
    ("goal", "improve recall"),
    ("budgets.tokens", 5001),
    ("budgets.time_s", 601),
    ("budgets.max_steps", 21),
    ("budgets.forward_passes", 101),
    ("budgets.cost_usd", 1.0),
    ("change_limits.max_changed_entries", 65),
    ("change_limits.frozen_tensors", ["mean"]),
    ("change_limits.allowed_tensors", ["weight"]),
    ("observation_tools", [{"name": "firing_positions", "version": "1", "cost": 1}]),
    ("primary_sets", ["everyday"]),
    # The engine's own value (until 018-P4 an effort level, "high", under the weightspace default).
    ("thinking", "default"),
    ("provider", {"id": "other"}),
]


def _set(b: dict[str, Any], path: str, value: Any) -> None:
    *head, last = path.split(".")
    node = b
    for key in head:
        node = node[key]
    node[last] = value


@pytest.mark.parametrize(("path", "value"), CHANGES, ids=[c[0] for c in CHANGES])
def test_every_frozen_field_moves_the_digest(bundle, path: str, value: Any) -> None:
    base = run(bundle, body(bundle))
    changed = copy.deepcopy(body(bundle))
    _set(changed, path, value)
    assert run(bundle, changed).request_sha256 != base.request_sha256


def test_the_endpoint_moves_the_digest(bundle) -> None:
    other = dict(LOCAL, id="ep-local-2")
    assert (
        run(bundle, body(bundle), endpoint=other).request_sha256
        != run(bundle, body(bundle)).request_sha256
    )


def test_the_provider_version_moves_the_digest(bundle) -> None:
    a = run(bundle, body(bundle))
    b = freeze(
        body(bundle),
        bundle=bundle,
        bundle_imported=True,
        endpoint=LOCAL,
        provider_version="0.1.1",
        queued_for_caller=0,
        settings=WSPSettings(),
    )
    assert a.request_sha256 != b.request_sha256


def test_caller_ref_is_stored_beside_the_request_and_moves_no_digest(bundle) -> None:
    a = run(bundle, body(bundle, caller_ref="exp-1"))
    b = run(bundle, body(bundle, caller_ref="exp-2"))
    assert a.caller_ref == "exp-1" and "caller_ref" not in a.request
    assert a.request_sha256 == b.request_sha256


@pytest.mark.parametrize(
    ("value", "effective"),
    [
        ("not_sensitive", "not_sensitive"),
        ("sensitive", "sensitive"),
        ("unknown", "sensitive"),
        (None, "sensitive"),
    ],
)
def test_p11_unknown_is_sensitive(value, effective) -> None:
    assert effective_sensitivity(value) == effective


@pytest.mark.parametrize("value", ["sensitive", "unknown"])
def test_external_endpoint_with_a_sensitive_or_unknown_bundle_is_refused(
    bundle, value: str
) -> None:
    marked = copy.deepcopy(bundle)
    marked["sensitivity"]["value"] = value
    err = _refused(lambda: run(marked, body(marked), endpoint=EXTERNAL))
    assert err.code == "wsp_sensitive_data_external_endpoint"
    assert run(marked, body(marked)).sensitivity_effective == "sensitive"  # local proceeds


def test_external_with_a_not_sensitive_bundle_passes_and_asks(bundle) -> None:
    frozen = run(bundle, body(bundle), endpoint=EXTERNAL)
    assert frozen.approval_action == "wsp_session_external" and frozen.locality == "external"
    assert frozen.request["budgets"]["cost_usd"] == 10.0  # D16's default, external only


def test_a_cap_above_fifty_dollars_is_refused(bundle) -> None:
    b = body(bundle)
    b["budgets"]["cost_usd"] = 50.01
    assert _refused(lambda: run(bundle, b, endpoint=EXTERNAL)).code == "wsp_cost_cap_above_maximum"
    b["budgets"]["cost_usd"] = 50.0
    run(bundle, b, endpoint=EXTERNAL)


def test_a_declared_sensitivity_unlike_the_bundles_is_refused(bundle) -> None:
    err = _refused(lambda: run(bundle, body(bundle, sensitivity="unknown")))
    assert err.code == "wsp_sensitivity_mismatch"


def test_an_eleventh_queued_run_from_one_caller_is_refused(bundle) -> None:
    """Operator 2026-11-20: the limit is 10 (it was 3, and a fourth approved run failed)."""
    run(bundle, body(bundle), queued=3)  # the fourth: accepted now
    run(bundle, body(bundle), queued=9)  # the tenth
    err = _refused(lambda: run(bundle, body(bundle), queued=10))
    assert err.code == "wsp_queue_limit_reached"
    message = err.envelope()["error"]["message"]
    assert "the limit is 10" in message
    # What to do, including when an APPROVAL is what was refused (it executes the same check).
    assert "file this request again" in message and "not retried" in message


def test_a_bundle_not_yet_imported_is_refused(bundle) -> None:
    assert (
        _refused(lambda: run(bundle, body(bundle), imported=False)).code
        == "wsp_bundle_not_imported"
    )


def _sized(bundle: dict[str, Any], total: int) -> dict[str, Any]:
    """``bundle`` with its manifest's file sizes summing to ``total`` (the rest on file one)."""
    sized = copy.deepcopy(bundle)
    others = sum(int(f["bytes"]) for f in sized["files"][1:])
    sized["files"][0]["bytes"] = total - others
    return sized


def test_a_bundle_the_worker_cannot_hold_is_refused_before_anything_else(bundle) -> None:
    """Production defect 2026-11-20: the size is compared at freeze, so no approval is filed for a
    run that cannot be pushed. It comes before the queue check: no wait makes it fit."""
    big = _sized(bundle, 57 * 1024**3)
    refusal = _refused(lambda: run(big, body(big), queued=3))
    assert refusal.code == "wsp_bundle_too_large_for_worker"
    assert refusal.details is not None and refusal.details["push_bytes"] == 57 * 1024**3


def test_the_phase2_bundle_that_was_evicted_freezes(bundle) -> None:
    fits = _sized(bundle, 34_881_633_698)
    assert run(fits, body(fits)).request["kind"] == "repair"


def test_the_freeze_limit_follows_the_setting(bundle) -> None:
    big = _sized(bundle, 57 * 1024**3)
    assert run(big, body(big), settings=WSPSettings(wsp_worker_work_bytes=128 * 1024**3))


def test_unknown_primary_sets_and_a_foreign_bundle_digest_are_refused(bundle) -> None:
    assert (
        _refused(lambda: run(bundle, body(bundle, primary_sets=["nope"]))).code
        == "validation_failed"
    )
    foreign = body(bundle)
    foreign["bundle"] = dict(foreign["bundle"], bundle_sha256="f" * 64)
    assert _refused(lambda: run(bundle, foreign)).code == "validation_failed"


@pytest.mark.parametrize(
    ("operation", "locality", "action"),
    [
        ("repair", "local", "wsp_session"),
        ("create", "local", "wsp_session"),
        ("inspect", "local", None),
        ("repair", "external", "wsp_session_external"),
        ("inspect", "external", "wsp_session_external"),
    ],
)
def test_the_approval_action(operation: str, locality: str, action: str | None) -> None:
    assert approval_action(operation, locality) == action


@pytest.mark.parametrize(
    "tool",
    [
        {"name": "read_practice_text", "version": "1", "cost": 1},
        {"name": "practice_metrics", "version": "9", "cost": 1},
        {"name": "compare_to_parent", "version": "2", "cost": 1},  # its cost is 2
    ],
    ids=["unknown", "version", "cost"],
)
def test_an_observation_tool_outside_the_catalogue_is_refused(bundle, tool) -> None:
    """FR-016.18: a tool is the worker's, at its own version and cost; never the caller's claim."""
    refused = _refused(lambda: run(bundle, {**body(bundle), "observation_tools": [tool]}))
    assert refused.code == "validation_failed"


def test_observation_tools_are_frozen_sorted_by_name(bundle) -> None:
    """FTID §3.5: a set is sorted, so the order a caller lists it in moves no digest."""
    a = [
        {"name": "practice_metrics", "version": "2", "cost": 1},
        {"name": "compare_to_parent", "version": "2", "cost": 2},
    ]
    one = run(bundle, {**body(bundle), "observation_tools": a})
    two = run(bundle, {**body(bundle), "observation_tools": list(reversed(a))})
    assert one.request_sha256 == two.request_sha256
    assert [t["name"] for t in one.request["observation_tools"]] == [
        "compare_to_parent",
        "practice_metrics",
    ]


# Review 016-P2, R16P2-3: chaining (FR-016.5) is not wired, so a run on a candidate is refused at
# freeze, before its approval and its budget, not failed at intake afterwards.
@pytest.mark.parametrize(
    "change",
    [
        {"parent": {"kind": "candidate", "ref": "wspc_000000000001"}},
        {"chain": True},
        {"parent": {"kind": "candidate", "ref": "wspc_000000000001"}, "chain": True},
    ],
    ids=["candidate-parent", "chain", "both"],
)
def test_a_run_on_a_candidate_is_refused_until_chaining_is_wired(bundle, change) -> None:
    refusal = _refused(lambda: run(bundle, {**body(bundle), **change}))
    assert refusal.code == "wsp_phase_not_enabled"


def test_a_bundle_parent_naming_another_bundle_is_refused(bundle) -> None:
    other = {"parent": {"kind": "bundle", "ref": "0" * 64}}
    refusal = _refused(lambda: run(bundle, {**body(bundle), **other}))
    assert refusal.code == "validation_failed"


def test_the_bundle_parent_given_explicitly_is_the_default(bundle) -> None:
    explicit = {"parent": {"kind": "bundle", "ref": bundle["bundle_sha256"]}}
    assert (
        run(bundle, {**body(bundle), **explicit}).request_sha256
        == run(bundle, body(bundle)).request_sha256
    )


# ------------------------------------------------ an impossible time budget (fix 2026-11-21)


def _practice(bundle: dict[str, Any], tokens: int, d_model: int, dtype: str) -> dict[str, Any]:
    sized = copy.deepcopy(bundle)
    sized["practice"].update(tokens=tokens, d_model=d_model, dtype=dtype)
    return sized


#: Run wsp_cb092cedad97's practice set: 4,257,928 tokens x 4096, fp16 (34.9 GB).
INCIDENT = {"tokens": 4_257_928, "d_model": 4096, "dtype": "float16"}


def _budgets(time_s: float, passes: int) -> dict[str, Any]:
    return {"tokens": 300_000, "time_s": time_s, "max_steps": 20, "forward_passes": passes}


def test_the_incidents_budget_is_not_refused(bundle) -> None:
    """40 passes over 34.9 GB at 4e9 B/s is ~350 s; 2400 s is plenty, so no refusal: the run
    failed on the slow scorer and the deadline, which are fixed, not on its budget."""
    big = _practice(bundle, **INCIDENT)
    assert run(big, body(big, budgets=_budgets(2400, 40))).request["budgets"]["time_s"] == 2400


def test_a_time_budget_under_half_the_forward_passes_is_refused_with_a_fix(bundle) -> None:
    big = _practice(bundle, **INCIDENT)
    refusal = _refused(lambda: run(big, body(big, budgets=_budgets(120, 40))))
    assert refusal.code == "wsp_time_budget_too_small"
    assert refusal.details is not None
    fix = refusal.details["fix"]
    per_pass = 4_257_928 * 4096 * 2 / 4.0e9
    assert refusal.details["forward_pass_s"] == pytest.approx(per_pass, abs=1e-3)
    assert f"budgets.time_s to at least {int(40 * per_pass) + 1}" in fix
    assert f"budgets.forward_passes to at most {int(120 // per_pass)}" in fix
    assert "WSP_FORWARD_PASS_BYTES_PER_S" in refusal.message


def test_the_check_bites_only_under_half_the_estimate(bundle) -> None:
    """Conservative: a false refusal is worse than none. 40 passes of ~8.72 s is ~349 s; 175 s
    (just over half) freezes, 174 s is refused."""
    big = _practice(bundle, **INCIDENT)
    assert run(big, body(big, budgets=_budgets(175, 40)))
    assert _refused(lambda: run(big, body(big, budgets=_budgets(174, 40)))).code == (
        "wsp_time_budget_too_small"
    )


def test_no_forward_passes_is_never_refused_for_time(bundle) -> None:
    big = _practice(bundle, **INCIDENT)
    assert run(big, body(big, budgets=_budgets(60, 0)))


def test_the_time_check_follows_its_setting(bundle) -> None:
    big = _practice(bundle, **INCIDENT)
    faster = WSPSettings(wsp_forward_pass_bytes_per_s=4.0e10)
    assert run(big, body(big, budgets=_budgets(120, 40)), settings=faster)


# ------------------------------------- 2026-11-21: budget defaults by the endpoint's locality


def _no_budgets(bundle: dict[str, Any], **budgets: Any) -> dict[str, Any]:
    b = body(bundle)
    b["budgets"] = dict(budgets)
    return b


def test_a_local_runs_defaults_are_safety_stops(bundle) -> None:
    """Operator 2026-11-21: for a local editor the defaults catch a runaway (5,000,000 tokens,
    100 steps, 2 hours), not limits the editor works against; no cost cap; forward passes 0."""
    budgets = run(bundle, _no_budgets(bundle)).request["budgets"]
    assert budgets == {
        "tokens": 5_000_000,
        "max_steps": 100,
        "time_s": 7200,
        "forward_passes": 0,
        "cost_usd": None,
    }


def test_an_external_runs_defaults_are_the_earlier_ones_and_the_cost_cap(bundle) -> None:
    budgets = run(bundle, _no_budgets(bundle), endpoint=EXTERNAL).request["budgets"]
    assert budgets == {
        "tokens": 5_000_000,  # feature 020 (F20-D17, DEP-1): was 100,000
        "max_steps": 50,
        "time_s": 3600,
        "forward_passes": 0,
        "cost_usd": 10.0,
    }


@pytest.mark.parametrize("endpoint", [LOCAL, EXTERNAL], ids=["local", "external"])
def test_a_bodys_own_budget_wins_over_either_default(bundle, endpoint) -> None:
    named = {"tokens": 1234, "max_steps": 7, "time_s": 900, "forward_passes": 3}
    budgets = run(bundle, _no_budgets(bundle, **named), endpoint=endpoint).request["budgets"]
    assert {k: budgets[k] for k in named} == named
    # One named, the rest from the endpoint's defaults.
    one = run(bundle, _no_budgets(bundle, tokens=42), endpoint=endpoint).request["budgets"]
    assert one["tokens"] == 42
    assert one["max_steps"] == (100 if endpoint is LOCAL else 50)


def test_the_defaults_are_settings_within_the_contracts_maxima(bundle) -> None:
    from pydantic import ValidationError

    custom = WSPSettings(
        wsp_default_tokens_local=77,
        wsp_default_max_steps_local=9,
        wsp_default_time_budget_s_local=1800,
        wsp_default_tokens_external=55,
        wsp_default_max_steps_external=8,
        wsp_default_time_budget_s_external=1200,
    )
    local = run(bundle, _no_budgets(bundle), settings=custom).request["budgets"]
    external = run(bundle, _no_budgets(bundle), endpoint=EXTERNAL, settings=custom)
    assert (local["tokens"], local["max_steps"], local["time_s"]) == (77, 9, 1800)
    assert (
        external.request["budgets"]["tokens"],
        external.request["budgets"]["max_steps"],
        external.request["budgets"]["time_s"],
    ) == (55, 8, 1200)
    for bad in (
        {"wsp_default_max_steps_local": 501},
        {"wsp_default_time_budget_s_local": 86_401},
        {"wsp_default_time_budget_s_external": 59},
        {"wsp_default_tokens_local": 0},
    ):
        with pytest.raises(ValidationError):
            WSPSettings(**bad)


def test_every_tool_version_stays_and_a_versionless_tool_gets_the_newest(bundle) -> None:
    """2026-11-21: practice_metrics and compare_to_parent have a version 2 (the rates at the
    parent's bar). Version 1 is still accepted, so a body (or a replay's request) naming it runs
    it; a tool named without a version, or a cost, gets the newest at its own cost; an unknown
    version is refused, naming each version and cost."""
    v1 = {"name": "practice_metrics", "version": "1", "cost": 1}
    frozen = run(bundle, {**body(bundle), "observation_tools": [v1]}).request
    assert frozen["observation_tools"] == [v1]
    bare = [{"name": "practice_metrics"}, {"name": "compare_to_parent", "cost": 2}]
    frozen = run(bundle, {**body(bundle), "observation_tools": bare}).request
    # 2026-11-22 (D128 round 7): the newest is version 3, each head at its own re-cut bar.
    assert frozen["observation_tools"] == [
        {"name": "compare_to_parent", "version": "4", "cost": 2},
        {"name": "practice_metrics", "version": "4", "cost": 1},
    ]
    v2 = [{"name": "practice_metrics", "version": "2", "cost": 1}]
    assert run(bundle, {**body(bundle), "observation_tools": v2}).request["observation_tools"] == v2
    unknown = {"name": "practice_metrics", "version": "9", "cost": 1}
    refused = _refused(lambda: run(bundle, {**body(bundle), "observation_tools": [unknown]}))
    assert "practice_metrics version '1' cost 1" in refused.message
    assert "practice_metrics version '2' cost 1" in refused.message
    assert "practice_metrics version '3' cost 1" in refused.message
    twice = [v1, {"name": "practice_metrics", "version": "2", "cost": 1}]
    assert _refused(lambda: run(bundle, {**body(bundle), "observation_tools": twice})).code == (
        "validation_failed"
    )


# ------------------------------------------- feature 020: efforts and the history mode (H-5)

HOSTED_ANTHROPIC = {
    "id": "ep-claude",
    "locality": "external",
    "model": "claude",
    "kind": "anthropic",
    "serving": "generic",
    "thinking": "default",
    "history_mode": "append_only",
}


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_an_effort_is_frozen_as_itself_and_the_request_validates(bundle, effort: str) -> None:
    from wsp.schemas import validate_frozen_request

    frozen = run(bundle, body(bundle, thinking=effort), endpoint=HOSTED_ANTHROPIC)
    assert frozen.request["thinking"] == effort
    assert "thinking_tokens" not in frozen.request
    validate_frozen_request(frozen.request)  # the vendored run schema admits it unchanged
    endpoint_effort = {**HOSTED_ANTHROPIC, "thinking": effort}
    assert run(bundle, body(bundle), endpoint=endpoint_effort).request["thinking"] == effort


def test_a_hosted_anthropic_default_freezes_null_and_off_is_refused(bundle) -> None:
    assert run(bundle, body(bundle), endpoint=HOSTED_ANTHROPIC).request["thinking"] is None
    err = _refused(lambda: run(bundle, body(bundle, thinking="off"), endpoint=HOSTED_ANTHROPIC))
    assert err.code == "wsp_endpoint_thinking_unsupported"


def test_an_effort_on_a_local_endpoint_is_refused(bundle) -> None:
    local = {**LOCAL, "kind": "openai_compatible", "serving": "generic"}
    err = _refused(lambda: run(bundle, body(bundle, thinking="medium"), endpoint=local))
    assert err.code == "wsp_endpoint_thinking_unsupported"


@pytest.mark.parametrize("asked", [None, "off"])
def test_append_only_freezes_paging_off(bundle, asked: Any) -> None:
    extra = {} if asked is None else {"paging": asked}
    frozen = run(bundle, body(bundle, **extra), endpoint=HOSTED_ANTHROPIC)
    assert frozen.request["paging"] == "off"
    local_append = {**LOCAL, "history_mode": "append_only"}
    assert run(bundle, body(bundle, **extra), endpoint=local_append).request["paging"] == "off"


def test_paging_on_for_an_append_only_endpoint_is_refused_by_name(bundle) -> None:
    err = _refused(lambda: run(bundle, body(bundle, paging="on"), endpoint=HOSTED_ANTHROPIC))
    assert err.code == "wsp_history_mode_conflict"


def test_a_paging_endpoint_still_freezes_its_own_paging(bundle) -> None:
    paging = {**LOCAL, "history_mode": "paging"}
    assert "paging" not in run(bundle, body(bundle), endpoint=paging).request
    assert run(bundle, body(bundle, paging="off"), endpoint=paging).request["paging"] == "off"


def test_the_hosted_default_token_budget_is_five_million(bundle) -> None:
    """F20-D17 (DEP-1): a hosted start with no token budget freezes 5,000,000; the cost cap still
    applies; a start body's budget still wins."""
    budgets = {"time_s": 600, "max_steps": 20, "forward_passes": 0}
    frozen = run(bundle, body(bundle, budgets=budgets), endpoint=EXTERNAL)
    assert frozen.request["budgets"]["tokens"] == 5_000_000
    assert frozen.request["budgets"]["cost_usd"] == WSPSettings().wsp_cost_cap_default_usd
    named = run(bundle, body(bundle, budgets={**budgets, "tokens": 1234}), endpoint=EXTERNAL)
    assert named.request["budgets"]["tokens"] == 1234
