"""The kernel grandchild cannot reach the token, the network or the practice data (feature 018;
FTASKS 6.3, 6.5, 6.6; FR-018.14; FTDD §8.1).

The probe (``wsp_children/engine_probe.py``) is sent as the editor's own ``python`` code through a
real session started by the real entry point. To make the parent's ``/proc`` a live target, the
session process is given the token in its ENVIRONMENT too (the real child never has it there:
``CHILD_ENV`` is a literal and the token travels on stdin), so an undumpable call that is missing
shows up as a found token rather than as nothing to find.

The negative control is permanent: the same probe run in the session process finds the token.
The two removal controls (the undumpable call, ``GRANDCHILD_RULES``) are mutation controls,
recorded in ``0xcc/reviews/018_P1_controls.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, needles, probe_code, run_case, run_main, tool_outputs
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.support.wsp_scripted_endpoint import TOKEN, ScriptedEndpoint
from wsp_worker.miwsp_engine.session import PRACTICE_FILE_ROLES

CANARY = {"WSP_PROBE_CANARY": TOKEN}


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _practice_paths(line: dict[str, Any]) -> list[str]:
    work = Path(line["paths"]["work"])
    return [str(work / f["name"]) for f in line["files"] if f["role"] in PRACTICE_FILE_ROLES]


def _probe_in_kernel(tmp: Path, fx: FixtureBundle, kind: str) -> tuple[dict[str, str], list[str]]:
    run = tmp / kind
    with ScriptedEndpoint("openai_compatible", []) as ep:
        line = line_for(run, fx, kind, ep.gateway)
        paths = _practice_paths(line)
        assert len(paths) == len(PRACTICE_FILE_ROLES)
        assert all(Path(p).exists() for p in paths)  # present until the session unlinks them
        ep.steps = [
            {"calls": [("python", {"code": probe_code(TOKEN, paths)})]},
            {"calls": [("submit", {"answer": "probed"})]},
        ]
        seen = run_main(run, line, env=CANARY)
    assert seen["exit"] == 0, seen["stderr"][-3000:]
    [output] = [o for o in tool_outputs(run) if "PROBE" in o]
    found: dict[str, str] = json.loads(output.split("PROBE", 1)[1].strip().splitlines()[0])
    return found, paths


@pytest.mark.parametrize("kind", ["repair", "inspect"])
def test_the_probe_in_the_grandchild_finds_and_opens_nothing(
    tmp_path: Path, fx: FixtureBundle, kind: str
) -> None:
    found, paths = _probe_in_kernel(tmp_path, fx, kind)
    # The token in no form, anywhere in the grandchild's own reach.
    for check in ("environ", "cmdline", "objects", "memory"):
        assert found[check] == "absent", (check, found)
    # The parent's /proc is closed by PR_SET_DUMPABLE 0 (the canary sits in its environ).
    assert found["parent_environ"].startswith("refused:"), found
    assert found["parent_mem"].startswith("refused:"), found
    assert found["parent_fd"].startswith("refused:"), found
    assert found["parent_map_files"].startswith("refused:"), found
    assert "parent_fd_token" not in found
    # The practice and activation files are gone from /work.
    assert [found[f"practice_{i}"] for i in range(len(paths))] == ["missing"] * len(paths)
    # No socket of any kind: GRANDCHILD_RULES.
    for name in ("socket_inet", "socket_unix", "socketpair"):
        assert found[name] == "refused:PermissionError", (name, found)


def test_the_negative_control_the_same_probe_in_the_session_process_finds_the_token(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-018.14's AC: in the session process (the token was read from its line) the probe's
    search finds it, so a clean result above is the isolation, not a blind probe."""
    run = tmp_path / "self"
    line = line_for(run, fx, "repair", {"ip": "127.0.0.1", "port": 9, "token": TOKEN})
    seen = run_case(
        run,
        line,
        probe_self={"needles": needles(TOKEN), "practice_paths": _practice_paths(line)},
        _env=CANARY,
    )
    found = seen["probe"]
    assert found["objects"] == "found" and found["memory"] == "found"
    assert found["environ"] == "found"  # the canary
    assert found["practice_0"] == "opened"  # nothing unlinked them in this process
    assert found["socket_inet"] == "opened"  # the child itself may connect (to the gateway)
