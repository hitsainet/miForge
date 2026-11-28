"""The shape helper reads the vendored contract and refuses a bad value (017 FR-017.2)."""

from __future__ import annotations

import jsonschema
import pytest

from tests.support.result_language import status_enum, validate

GOOD = {
    "id": "apr_1",
    "status": "pending",
    "decide_in": "miforge",
    "then": "server_runs",
    "summary": "Start GPU work.",
}


def test_a_good_approval_validates() -> None:
    validate("approval", GOOD)


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "status": "approved_ish"},
        {**GOOD, "decide_in": "miForge"},
        {**GOOD, "summary": ""},
        {**GOOD, "call_again_with": {"tool": "forge_x"}},
        {k: v for k, v in GOOD.items() if k != "then"},
        {**GOOD, "next": "x"},
    ],
)
def test_a_bad_approval_is_refused(bad: dict[str, object]) -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate("approval", bad)


def test_the_status_enum_holds_every_status_the_queue_writes() -> None:
    from models.approval import APPROVAL_STATUSES

    assert set(APPROVAL_STATUSES) <= set(status_enum())


def test_the_approval_summary_never_exceeds_the_contracts_500() -> None:
    """The contract caps ``approval.summary`` at 500 characters; removing the truncation was
    green across the suite (R3 review fix OR3-017-3, 2026-11-16; the reviewer's case, N-F09)."""
    from services.result_language import SUMMARY_LIMIT, summary_words

    words = summary_words("gpu_start", {"model": "org/" + "m" * 600, "run_id": "r" * 64})
    assert SUMMARY_LIMIT == 500
    assert 1 <= len(words) <= 500, len(words)
    assert words.endswith("...")
