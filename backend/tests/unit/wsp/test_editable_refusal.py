"""``runs.refuse_non_editable``: the bundle's ``editable`` flags, enforced by the backend (review
OR1-016-2). The intake and the two patch routes call it; their integration tests pin the calls."""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from wsp.runs import refuse_non_editable

BUNDLE = {
    "tensors": [
        {"name": "weight", "editable": True},
        {"name": "bias", "editable": True},
        {"name": "mean", "editable": False},
        {"name": "std", "editable": False},
    ]
}


def _patch(*tensors: dict[str, Any]) -> dict[str, Any]:
    return {"tensors": list(tensors)}


def test_editable_changes_pass() -> None:
    refuse_non_editable(BUNDLE, _patch({"name": "weight", "changed_entries": 3}))
    # An entry that changes nothing is not an edit.
    refuse_non_editable(BUNDLE, _patch({"name": "mean", "changed_entries": 0}))


@pytest.mark.parametrize(
    "tensor",
    [
        {"name": "mean", "changed_entries": 1},
        {"name": "std", "changed_entries": 0, "append": {"rows": 1}},
        {"name": "not_in_the_bundle", "changed_entries": 1},
    ],
    ids=["non-editable", "append-only", "unlisted"],
)
def test_a_change_to_a_tensor_not_marked_editable_is_refused(tensor: dict[str, Any]) -> None:
    with pytest.raises(Refusal) as caught:
        refuse_non_editable(BUNDLE, _patch({"name": "weight", "changed_entries": 1}, tensor))
    assert caught.value.code == "wsp_change_limit_exceeded"
    assert caught.value.details == {"kind": "frozen_tensor", "tensors": [tensor["name"]]}


def test_a_truthy_flag_that_is_not_true_is_not_editable() -> None:
    bundle = {"tensors": [{"name": "weight", "editable": "false"}]}
    with pytest.raises(Refusal):
        refuse_non_editable(bundle, _patch({"name": "weight", "changed_entries": 1}))
