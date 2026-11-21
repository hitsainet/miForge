"""The engine's task line (fix 2026-11-21, live round 5 on main ``df8c9c6``).

- **Parity with weightspace 0.3.0:** the engine's mode texts are weightspace's line for line
  (``ws:prompts.py`` ``MODES``, read under the worker's interpreter, where the library is), and
  a session's mode and task follow the adapter's ``session_mode``/``session_task``: a repair with
  a goal IMPROVES and carries its goal. The engine said FIX where weightspace said IMPROVE.
- **Program style** (run ``wsp_554d3932d108``): its first message said "Task type: READ ... You
  cannot change any weights", so the editor changed nothing. It now names the mode, says the
  editor writes a program, names ``test_program`` and ``submit_program`` with the cost, and says
  ``submit`` alone changes nothing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.support import wsp_worker_env
from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_worker.miwsp_engine import prompts
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.transcript import ScriptedClient
from wsp_worker.program_tools import ProgramTools

READ_ONLY = "You cannot change any weights"


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _first_message(tmp_path: Path, fx: FixtureBundle, kind: str, **over: Any) -> str:
    line = line_for(tmp_path / "run", fx, kind, None, python=False, **over)
    if "goal" in over:
        line["spec"]["request"]["goal"] = over["goal"]
    client = ScriptedClient([{"role": "assistant", "content": None, "tool_calls": [_submit()]}])
    s = session_mod.Session(line, client=client)
    s.writer = None
    seen: list[list[dict[str, Any]]] = []
    real = client.complete

    def complete(messages: Any, **kw: Any) -> Any:
        seen.append([dict(m) for m in messages])
        return real(messages, **kw)

    client.complete = complete  # type: ignore[method-assign]
    s._loop()
    return str(seen[0][1]["content"])


def _submit() -> dict[str, Any]:
    return {"id": "c0", "name": "submit", "arguments": {"answer": "a"}}


def test_the_mode_texts_are_weightspaces_line_for_line() -> None:
    python = wsp_worker_env.worker_python(("weightspace",))
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "from weightspace import prompts;"
        "from wsp_worker.adapters import weightspace_adapter as a;"
        "reqs=[{'kind':'inspect'},{'kind':'repair'},{'kind':'repair','goal':'g','task':'t'},"
        "{'kind':'create'}];"
        "print(json.dumps({'modes': prompts.MODES,"
        " 'mode': [a.session_mode(r, 'session') for r in reqs],"
        " 'task': [a.session_task(r) for r in reqs]}))"
    )
    done = subprocess.run(
        [python, "-I", "-c", code, str(wsp_worker_env.SRC)],
        capture_output=True,
        text=True,
        check=True,
    )
    ws = json.loads(done.stdout)
    assert ws["modes"] == prompts.MODES
    requests = [
        {"kind": "inspect"},
        {"kind": "repair"},
        {"kind": "repair", "goal": "g", "task": "t"},
        {"kind": "create"},
    ]
    assert ws["mode"] == [prompts.mode_for(r["kind"], r.get("goal")) for r in requests]
    assert ws["task"] == [
        prompts.task_text(r["kind"], str(r.get("task") or ""), r.get("goal")) for r in requests
    ]


def test_a_repair_with_a_goal_improves_and_carries_its_goal(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    text = _first_message(tmp_path, fx, "repair", goal="fewer false alarms on everyday text")
    assert text.startswith(prompts.MODES["improve"])
    assert "Goal: fewer false alarms on everyday text" in text
    plain = _first_message(tmp_path / "b", fx, "repair")
    assert plain.startswith(prompts.MODES["fix"])


@pytest.mark.parametrize(
    ("kind", "goal", "mode"),
    [("repair", None, "FIX"), ("repair", "a goal", "IMPROVE"), ("create", None, "WRITE")],
)
def test_a_program_session_says_it_writes_a_program(
    tmp_path: Path, fx: FixtureBundle, kind: str, goal: str | None, mode: str
) -> None:
    over: dict[str, Any] = {"mode": "program"}
    if goal is not None:
        over["goal"] = goal
    text = _first_message(tmp_path, fx, kind, **over)
    assert text.startswith(f"Task type: {mode} (program).")
    assert READ_ONLY not in text
    for name in ("test_program", "submit_program", "edit(base, support)"):
        assert name in text
    assert f"costs {ProgramTools.COST} forward pass" in text
    assert "Calling `submit` without a submitted program changes nothing." in text
