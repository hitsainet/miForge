"""The kernel's reserved names stay bound, and an accepted submit counts its delta (2026-11-27).

Run ``wsp_ba149c57f611`` (phase-2 E3): at transcript entry 29 the editor ran
``W = np.array(R['weight'])``. ``kernel_main`` bound ``W`` once, at ``init``, and no ``exec`` ever
bound it back, so ``W`` stayed a plain array and every later ``W['weight'][:] = ...`` raised a
numpy IndexError. The editor fell back to ``write_tensor`` with 4,096 values inline, its replies
were cut off (4 refused turns), and the run's candidate (``weight[0:10] = 0``) is accidental, its
answer saying "No edits applied". Now every ``exec`` ends by binding ``R``, ``W``, ``notes`` and
``np`` back (``kernel_main.RESERVED``) and its output says so; an accepted submit in an editing
session states the delta (``prompts.SUBMITTED_CHANGES``). Both follow the transcript header, so a
run recorded before (E3 among them) replays bit for bit.

The kernel is the REAL grandchild (``test_kernel.py``'s harness: its own interpreter and filter);
the sessions run in process on a scripted client with that real kernel attached.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support import wsp_worker_env
from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_worker.miwsp_engine import kernel_main, prompts
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.kernel import Kernel
from wsp_worker.miwsp_engine.transcript import ScriptedClient, TranscriptWriter, read_script

#: The E3 probe's weight: 4,096 entries, none zero, so zeroing it changes every one.
D = 4_096


class Values:
    def __init__(self) -> None:
        self.v = {
            "weight": np.linspace(0.5, 1.5, D, dtype=np.float32),
            "bias": np.array([0.25], dtype=np.float64),
        }

    def __call__(self) -> dict[str, np.ndarray]:
        return self.v


def _kernel(tmp: Path, *, rebind: bool) -> Kernel:
    wsp_worker_env.require_seccomp()
    kernel = Kernel(tmp / "kernel", Values(), edit=True, rebind=rebind)
    kernel.start()
    return kernel


@pytest.fixture()
def kernel(tmp_path: Path) -> Iterator[Kernel]:
    k = _kernel(tmp_path, rebind=True)
    yield k
    k.close()


# ----------------------------------------------------------------------------- the kernel


def test_a_reassigned_w_is_bound_back_and_the_next_write_lands(kernel: Kernel) -> None:
    """The brief's case: ``W = np.zeros(3)``, then ``W['weight'][:] = 0`` writes all 4,096."""
    first = kernel.exec("W = np.zeros(3)")
    second = kernel.exec("W['weight'][:] = 0")
    # First the write (without the re-bind it is E3's IndexError: control C17), then the note.
    assert "IndexError" not in second.output and "Error" not in second.output, second.output
    assert first.deltas == {}
    assert first.output == kernel_main.RESERVED_NOTE.format(names="W", are="is", them="W")
    idx, vals = second.deltas["weight"]
    assert idx.size == D and idx.tolist() == list(range(D))
    assert np.all(vals == 0.0)
    assert "bias" not in second.deltas


def test_the_e3_copy_then_write_through_w(kernel: Kernel) -> None:
    """Entry 29's code, then the write it was meant to enable."""
    copied = kernel.exec("W = np.array(R['weight'])\nW.shape")
    assert copied.output.startswith("(4096,)\nnote: W is reserved")
    wrote = kernel.exec("W['weight'][:10] = 0")
    assert wrote.deltas["weight"][0].tolist() == list(range(10))


def test_every_reserved_name_is_bound_back_and_named(kernel: Kernel) -> None:
    out = kernel.exec("del R\nnp = None\nW = 1").output
    assert out == kernel_main.RESERVED_NOTE.format(names="R, W, np", are="are", them="R, W, np")
    assert kernel.exec("print(float(R['bias'][0]) + float(np.float64(0)))").output == "0.25\n"
    assert kernel.exec("type(W).__name__").output == "'_Maps'\n"


def test_a_list_assigned_to_notes_becomes_the_notes(kernel: Kernel) -> None:
    out = kernel.exec("notes = ['scale is 0.5']").output
    assert "notes is reserved" in out
    assert kernel.notes == ["scale is 0.5"]
    kernel.exec("notes.append('bias is 0.25')")
    assert kernel.notes == ["scale is 0.5", "bias is 0.25"]
    kernel.exec("notes = 'not a list'")
    assert kernel.notes == ["scale is 0.5", "bias is 0.25"]


def test_an_exec_that_binds_nothing_reserved_says_nothing(kernel: Kernel) -> None:
    out = kernel.exec("w = np.array(R['weight'])\nW['weight'][0] = w[0] * 2\nnotes.append('x')")
    assert out.output == "(no output)" and out.deltas["weight"][0].tolist() == [0]


def test_a_failing_exec_still_binds_back(kernel: Kernel) -> None:
    out = kernel.exec("W = 0\n1 / 0").output
    assert "ZeroDivisionError" in out and out.endswith(
        kernel_main.RESERVED_NOTE.format(names="W", are="is", them="W")
    )
    assert kernel.exec("W['bias'][0] = 1.0").deltas["bias"][0].tolist() == [0]


def test_a_restarted_kernel_keeps_binding_back(kernel: Kernel) -> None:
    kernel.restart()
    assert kernel.rebind
    kernel.exec("W = None")
    assert kernel.exec("W['weight'][1] = 0").deltas["weight"][0].tolist() == [1]


def test_without_rebind_the_old_behaviour_holds(tmp_path: Path) -> None:
    """The replay of a run recorded before: the names stay as the code bound them, as E3 saw."""
    k = _kernel(tmp_path, rebind=False)
    try:
        assert k.exec("W = np.zeros(3)").output == "(no output)"
        broken = k.exec("W['weight'][:] = 0")
        assert "IndexError" in broken.output and broken.deltas == {}
    finally:
        k.close()


# ----------------------------------------------------------------------------- the session


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean", d_model=8, rows=24, seed=0)


def _replies(steps: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"c{i}", "name": n, "arguments": a}],
        }
        for i, (n, a) in enumerate(steps)
    ]


STEPS = [
    ("python", {"code": "W = np.array(R['weight'])"}),
    ("python", {"code": "W['weight'][:] = 0"}),
    ("submit", {"answer": "No edits applied."}),
]


def _session(tmp: Path, fx: FixtureBundle, client: Any, **over: Any) -> session_mod.Session:
    line = line_for(tmp, fx, "repair", None, max_steps=20, python=True, **over)
    return session_mod.Session(line, client=client)


def _run(s: session_mod.Session) -> list[dict[str, Any]]:
    """The loop with a REAL kernel attached (not ``_isolate_and_spawn``, which would make this
    test process undumpable), bound back or not by the session's own flag."""
    wsp_worker_env.require_seccomp()
    s.kernel = Kernel(
        s.tmp / "kernel",
        lambda: s.store.values,
        edit=s.can_edit,
        rebind=s.reserved_names,
    )
    s.kernel.start()
    s.writer = TranscriptWriter(s.out / "transcript.jsonl", s.header())
    try:
        s._loop()
    finally:
        s.kernel.close()
        s.writer.close()
    return [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]


def _tool(lines: list[dict[str, Any]]) -> list[str]:
    return [str(m["content"]) for m in lines if m.get("role") == "tool"]


def test_the_session_binds_w_back_and_the_submit_states_the_delta(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    s = _session(tmp_path / "run", fx, ScriptedClient(_replies(STEPS)))
    lines = _run(s)
    header = lines[0]
    assert header["reserved_names"] == session_mod.RESERVED_NAMES_VERSION
    assert header["submit_changes"] == session_mod.SUBMIT_CHANGES_VERSION
    copied, wrote, submitted = _tool(lines)
    assert "note: W is reserved" in copied
    assert "Error" not in wrote, wrote
    # E3 answered "No edits applied" over a real delta: the submit's own answer now says what
    # the patch carries.
    assert submitted == prompts.SUBMITTED_CHANGES.format(n=8, total=9)
    assert s.store.changes()["weight"] == 8


def test_an_inspect_session_states_no_delta_at_submit(tmp_path: Path, fx: FixtureBundle) -> None:
    line = line_for(tmp_path / "run", fx, "inspect", None, max_steps=5, python=False)
    s = session_mod.Session(line, client=ScriptedClient(_replies([STEPS[-1]])))
    assert not s.submit_changes and "submit_changes" not in s.header()


@pytest.mark.parametrize("recorded_before", [True, False])
def test_a_replay_follows_the_header_it_was_recorded_with(
    tmp_path: Path, fx: FixtureBundle, recorded_before: bool
) -> None:
    """THE replay guarantee. A run recorded before (no ``reserved_names`` or ``submit_changes``
    in its header: E3's) replays with the IndexError and the bare "submitted", bit for bit; one
    recorded after replays the bound-back name and the counted submit."""
    s = _session(tmp_path / "run", fx, ScriptedClient(_replies(STEPS)))
    if recorded_before:
        s.reserved_names = s.submit_changes = False
    lines = _run(s)
    assert ("reserved_names" in lines[0]) is not recorded_before
    assert ("submit_changes" in lines[0]) is not recorded_before
    if recorded_before:
        assert "IndexError" in _tool(lines)[1] and _tool(lines)[2] == "submitted"
    original = s.out / "transcript.jsonl"
    again = line_for(
        tmp_path / "replay",
        fx,
        "repair",
        None,
        max_steps=20,
        python=True,
        mode="replay",
        replay_of_mode="session",
        extra=[("transcript.jsonl", "transcript", original)],
    )
    r = session_mod.Session(again, client=read_script(original))
    assert r.reserved_names is not recorded_before
    assert r.submit_changes is not recorded_before
    replayed = _run(r)
    assert r.replay_divergence is None
    assert _tool(replayed) == _tool(lines)
    assert r.store.sha256() == s.store.sha256()


# ----------------------------------------------------------------------------- the real child


def test_the_real_child_session_binds_w_back(tmp_path: Path, fx: FixtureBundle) -> None:
    """The WIRING, end to end: the child entry point (``engine_cases.py``, the session's own
    ``_isolate_and_spawn`` and kernel), E3's two Python runs, then a submit. The kernel is the
    session's, so a session that stops asking for the re-bind fails here (control C20)."""
    from tests.support.wsp_engine import tool_outputs
    from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import call, session

    steps = [
        call("python", code="W = np.array(R['weight'])"),
        call("python", code="W['weight'][:] = 0"),
        call("submit", answer="No edits applied."),
    ]
    _ep, run, _ = session(tmp_path, fx, "repair", steps)
    copied, wrote, submitted = tool_outputs(run)
    assert "note: W is reserved" in copied
    assert "IndexError" not in wrote, wrote
    header = json.loads((run / "out" / "transcript.jsonl").read_text().splitlines()[0])
    assert header["reserved_names"] == session_mod.RESERVED_NAMES_VERSION
    size = fx.tensors["weight"].values64().size
    assert submitted == prompts.SUBMITTED_CHANGES.format(n=size, total=size + 1)
    assert (run / "out" / "patch" / "patch.json").is_file()
