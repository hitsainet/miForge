"""What each README example port must show (feature 016; FTASKS 8.5; R1B-A15; OVN-wsp-forge-68).

One statement of the documented outcomes, asserted by both runners of
``tests/support/wsp_children/readme_examples.py``: the image job's session
(``tests/image/test_wsp_worker_image.py``, whose host has pytest and the standard library only, so
this module imports nothing else) and the local twin under the worker's venv
(``tests/unit/wsp_worker/test_readme_examples.py``). ``outcome`` is the port's JSON line; ``patch``
is the ``patch/patch.json`` the session left in ``/out`` (``None`` when it left none).
"""

from __future__ import annotations

import json
from typing import Any

#: The port's stdout line prefix (``readme_examples.MARK``; the image host cannot import numpy).
MARK = "README-EXAMPLE "
EXAMPLES = ("ex_linear_fix", "ex_read_mlp", "ex_write_xor")


def outcome_line(stdout: str) -> dict[str, Any]:
    """The one outcome line a child printed."""
    lines = [ln[len(MARK) :] for ln in stdout.splitlines() if ln.startswith(MARK)]
    assert len(lines) == 1, f"expected one {MARK!r} line, got {len(lines)}: {stdout[-2000:]}"
    loaded: dict[str, Any] = json.loads(lines[0])
    return loaded


def assert_common(
    name: str, outcome: dict[str, Any], result: dict[str, Any], patch: dict[str, Any] | None
) -> None:
    """Every port: the adapter's session ended by ``submit``, twice alike, and the outcome was
    measured on the patch the session left."""
    assert outcome["example"] == name
    assert outcome["exit"] == 0
    assert outcome["deterministic"] is True, "a second run of the port disagreed with the first"
    assert result["stop_reason"] == outcome["stop_reason"] == "submitted", result
    assert result["answer"] == outcome["answer"]
    assert outcome["content_sha256"] == (None if patch is None else patch["content_sha256"])


def assert_linear_fix(outcome: dict[str, Any], patch: dict[str, Any] | None) -> None:
    """``fix_quadrants``: trained, every class right; after ``bias[3] -= 8``, class 3 is well below
    the others; after the example's edit (in the patch), class 3 is back with them. One entry."""
    trained, before, after = outcome["trained"], outcome["before"], outcome["after"]
    assert min(trained) >= 0.98, f"the trained network is not right on every class: {trained}"
    assert before[3] <= 0.9 and before[3] < min(before[:3]) - 0.1, f"not broken: {before}"
    assert min(after) >= 0.98, f"the patch did not bring class 3 back: {after}"
    assert outcome["mode"] == "fix" and outcome["confidence"] == 0.9, outcome
    answer = "Class 3's output bias was -8; reset it to the others' mean."
    assert outcome["answer"] == answer, outcome["answer"]
    assert patch is not None, "a repair left no patch"
    assert patch["changed_entries_total"] == 1, patch["changed_entries_total"]
    assert outcome["changed"] == {"2.bias": 1}, outcome["changed"]
    assert [t["name"] for t in patch["tensors"]] == ["2.bias"], patch["tensors"]


def assert_read_mlp(outcome: dict[str, Any], patch: dict[str, Any] | None) -> None:
    """``read_which_task``: the network computes its rule on all 256 inputs; the reading of ``R``
    finds inputs 0 and 1 and names that rule; a read leaves no patch."""
    assert outcome["truth"] == "parity_of_first_two"
    assert outcome["accuracy_256"] == 1.0, outcome
    assert outcome["inputs_read"] == [0, 1], f"the reading of R found {outcome['inputs_read']}"
    assert outcome["answer"] == "parity_of_first_two: x0 XOR x1", outcome["answer"]
    assert outcome["mode"] == "read", outcome
    assert patch is None and outcome["changed"] is None, "a read left a patch"


def assert_write_xor(outcome: dict[str, Any], patch: dict[str, Any] | None) -> None:
    """``wsp.write`` (zeroes first): the editor's first reads see zeros where the bundle held
    random weights; the written network computes XOR on all four inputs; ``2.bias``, never
    written, is the zero the session started from."""
    assert all(outcome["init_nonzero"].values()), outcome["init_nonzero"]
    assert outcome["zeros_seen"] == [True, True], "the editor's first reads were not zeros"
    assert outcome["predictions"] == [0, 1, 1, 0], f"not XOR: {outcome['outputs']}"
    assert outcome["outputs"] == [0.0, 1.0, 1.0, 0.0], outcome["outputs"]
    assert outcome["edited_2_bias"] == [0.0], f"2.bias was not zeroed: {outcome['edited_2_bias']}"
    assert outcome["mode"] == "write", outcome
    assert patch is not None, "a write left no patch"
    changed = {"0.bias": 2, "0.weight": 4, "2.bias": 1, "2.weight": 2}
    assert outcome["changed"] == changed, outcome["changed"]


ASSERT = {
    "ex_linear_fix": assert_linear_fix,
    "ex_read_mlp": assert_read_mlp,
    "ex_write_xor": assert_write_xor,
}


def assert_example(
    name: str, outcome: dict[str, Any], result: dict[str, Any], patch: dict[str, Any] | None
) -> None:
    assert_common(name, outcome, result, patch)
    ASSERT[name](outcome, patch)
