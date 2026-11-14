"""A ``gpu`` test with no usable CUDA device SKIPS, by name - and FAILS where a card is promised.

**The defect (found 2026-11-11).** ``pyproject.toml``'s ``addopts`` deselects the ``gpu`` marker
(``-m 'not gpu and not upstream'``), which is why CI - whose torch is the CPU wheel - never ran
these tests. A run that replaces ``addopts`` (``-o addopts=""``, the usual way to drop the
marker filter) selects them again, and four had no guard of their own: three in
``test_boundary_loop_overhead.py`` moved a model ``.to("cuda")`` and one in
``test_signal_source_throughput_gpu.py`` started a CUDA child process, so on a machine whose
torch has no CUDA they RAN and FAILED (``AssertionError: Torch not compiled with CUDA
enabled``; ``the contending load died before READY``). Every other ``gpu`` test already skipped
through its ``device`` fixture. A red that means "this machine has no card" hides the reds that
mean something.

**The rule now (``tests/support/gpu_node.py``, the one place):** no usable device -> skip,
naming the test and the reason. ``MIFORGE_REQUIRE_GPU=1`` -> that skip is a FAILURE, so a run on
the GPU node cannot go quietly green by skipping everything (the form the sibling checks already
take with ``*_REQUIRE_CROSS_REPO_CHECKS``).

The second half of this file runs the real tests in a child pytest with CUDA hidden
(``CUDA_VISIBLE_DEVICES=""``), which makes torch report no device on any machine, a GPU node
included - so the skip path and the require path are both exercised wherever this runs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import gpu_node

BACKEND = Path(__file__).resolve().parents[2]
BOUNDARY = "tests/performance/test_boundary_loop_overhead.py"
SIGNALS = "tests/performance/test_signal_source_throughput_gpu.py"
#: The four that ran and failed (the defect), by node id.
UNGUARDED = (
    f"{BOUNDARY}::TestTheBoundarysOwnOverhead::test_overhead_without_a_database_read",
    f"{BOUNDARY}::TestTheBoundarysOwnOverhead::test_overhead_with_the_per_boundary_database_read",
    f"{BOUNDARY}::TestTheRecordDoesNotYetCarryTheField"
    "::test_loop_overhead_is_enumerated_but_never_written",
    f"{SIGNALS}::TestFiguresTwoAndThreeGenerationThroughput"
    "::test_the_contending_load_is_proved_to_reach_the_card",
)
#: One that always skipped through its ``device`` fixture: the require switch must reach it too.
GUARDED = (
    f"{SIGNALS}::TestFigureOneF2sOwnAssemblyOverhead::test_assembly_overhead_against_step_wall_time"
)


# ------------------------------------------------------------------------- the helper


def test_an_unusable_device_skips_naming_the_test_and_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gpu_node.REQUIRE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception) as caught:
        gpu_node.absent("this torch build has no CUDA support")
    message = str(caught.value)
    assert "test_an_unusable_device_skips_naming_the_test_and_the_reason" in message
    assert "this torch build has no CUDA support" in message
    assert gpu_node.REQUIRE_ENV in message, "the skip must say how to make it a failure"


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_the_exact_switch_value_requires(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(gpu_node.REQUIRE_ENV, value)
    with pytest.raises(pytest.skip.Exception):
        gpu_node.absent("no device")


def test_the_require_switch_turns_the_skip_into_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gpu_node.REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception) as caught:
        gpu_node.absent("no device")
    message = str(caught.value)
    assert "test_the_require_switch_turns_the_skip_into_a_failure" in message
    assert f"{gpu_node.REQUIRE_ENV}=1" in message and "no device" in message


def test_require_cuda_passes_only_when_torch_can_use_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.delenv(gpu_node.REQUIRE_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert gpu_node.cuda_unusable() is None
    gpu_node.require_cuda()  # returns: the test body runs
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reason = gpu_node.cuda_unusable()
    assert reason and "CUDA" in reason
    with pytest.raises(pytest.skip.Exception):
        gpu_node.require_cuda()
    monkeypatch.setenv(gpu_node.REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception):
        gpu_node.require_cuda()


# ------------------------------------------------- the real tests, in a child pytest


def _child(require: bool) -> tuple[int, dict[str, str], str]:
    """Run the four unguarded tests and one guarded one with CUDA hidden; return the exit
    code, ``{node id: outcome}`` and the output."""
    env = {k: v for k, v in os.environ.items() if k != gpu_node.REQUIRE_ENV}
    env["CUDA_VISIBLE_DEVICES"] = ""  # torch.cuda.is_available() is False on any machine
    env.pop("PYTEST_XDIST_WORKER", None)
    if require:
        env[gpu_node.REQUIRE_ENV] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-rA",
            "-q",
            *UNGUARDED,
            GUARDED,
        ],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    outcomes: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(?:\[\d+\]\s+)?(\S+)", line)
        if match and "::" in match.group(2):
            outcomes[match.group(2)] = match.group(1)
    return result.returncode, outcomes, output


def test_without_a_card_the_gpu_tests_skip_by_name() -> None:
    code, _outcomes, output = _child(require=False)
    assert code == 0, f"a gpu test failed on a machine with no usable CUDA device:\n{output}"
    assert re.match(r"^5 skipped in ", output.splitlines()[-1]), output
    for node in (*UNGUARDED, GUARDED):
        name = node.rsplit("::", 1)[1]
        skips = [ln for ln in output.splitlines() if ln.startswith("SKIPPED") and name in ln]
        assert skips, f"{node} did not skip BY NAME:\n{output}"
        assert "GPU:" in skips[0] and gpu_node.REQUIRE_ENV in skips[0], skips[0]


def test_where_a_card_is_required_the_same_tests_fail_instead_of_skipping() -> None:
    """The verification of the non-skip path: with ``MIFORGE_REQUIRE_GPU=1`` not one of them
    may skip. On the GPU node this is what proves they executed."""
    code, outcomes, output = _child(require=True)
    assert code != 0, output
    assert "skipped" not in output.splitlines()[-1], f"a required gpu test skipped:\n{output}"
    for node in (*UNGUARDED, GUARDED):
        assert outcomes.get(node) in ("FAILED", "ERROR"), (node, outcomes.get(node), output)
    assert f"{gpu_node.REQUIRE_ENV}=1" in output
