"""A read too large to show, and the no-action limit (fix 2026-11-22, D128 round 9).

A ``read_tensor`` of the 4,096-entry probe weight was cut at 12,000 characters: the editor got
positions 0 to 1065 as if they were the tensor, without four of its five largest weights. And a
reply with no action ended the session at the third one over the whole run, however much work
was between them (the repeat guard's mistake of the same day).

In-process on a scripted client, as ``test_repeat_guard.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_worker import tensor_read
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine import tools as tools_mod
from wsp_worker.miwsp_engine.transcript import ScriptedClient, TranscriptWriter, read_script


@pytest.fixture()
def big(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean", d_model=4096)


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def act(tool: str, **arguments: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": f"c-{tool}", "name": tool, "arguments": arguments}],
    }


NOTHING = {"role": "assistant", "content": "", "tool_calls": []}
SUBMIT = act("submit", answer="done")


def _session(
    tmp: Path, fx: FixtureBundle, replies: list[dict[str, Any]], **over: Any
) -> session_mod.Session:
    line = line_for(tmp / "run", fx, "repair", None, python=False, max_steps=30, **over)
    s = session_mod.Session(line, client=ScriptedClient(replies))
    return s


def _run(s: session_mod.Session) -> list[dict[str, Any]]:
    s.writer = TranscriptWriter(s.out / "transcript.jsonl", s.header())
    s._loop()
    s.writer.close()
    return [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]


def _outputs(lines: list[dict[str, Any]]) -> list[str]:
    return [str(m["content"]) for m in lines if m.get("role") == "tool"]


def test_a_read_too_large_to_show_is_a_summary_of_the_whole(
    tmp_path: Path, big: FixtureBundle
) -> None:
    text = _outputs(_run(_session(tmp_path, big, [act("read_tensor", name="weight"), SUBMIT])))[0]
    assert "[truncated" not in text and len(text) < 3_000
    assert "4,096 values" in text and "not the values" in text
    w = np.abs(big.tensors["weight"].values64().reshape(-1))
    named = [int(i) for i in re.findall(r"\[(\d+)\]=", text)]
    assert named == np.argsort(-w, kind="stable")[: tensor_read.LARGEST].tolist()
    assert max(named) > 1_065  # it reaches past what the cut used to show


def test_the_range_it_suggests_is_shown_whole(tmp_path: Path, big: FixtureBundle) -> None:
    first = _outputs(_run(_session(tmp_path, big, [act("read_tensor", name="weight"), SUBMIT])))
    found = re.search(r"index='(0:\d+)'", first[0])
    assert found is not None
    read = act("read_tensor", name="weight", index=found.group(1))
    text = _outputs(_run(_session(tmp_path / "again", big, [read, SUBMIT])))[0]
    count = int(found.group(1)[2:])
    assert len(text.split("\n", 1)[1].split()) == count and len(text) <= tools_mod.MAX_TOOL_CHARS


def test_a_small_read_is_still_the_values(tmp_path: Path, big: FixtureBundle) -> None:
    read = act("read_tensor", name="weight", index="0:8")
    text = _outputs(_run(_session(tmp_path, big, [read, SUBMIT])))[0]
    assert text.startswith("weight[0:8] shape=(8,)\n") and len(text.split("\n")[1].split()) == 8


def test_a_slices_positions_are_said_to_be_the_slices() -> None:
    values = np.linspace(-1.0, 1.0, 64)
    text = tensor_read.fit("w", "100:164", values, "x" * 500, 100, 4)
    assert "positions count within [100:164]" in text


def test_a_matrix_names_row_and_column_and_skips_what_is_not_finite() -> None:
    values = np.zeros((8, 8))
    values[3, 5], values[0, 0] = -2.0, np.nan
    text = tensor_read.fit("w", None, values, "x" * 500, 100, 4)
    assert "[3,5]=-2" in text and "[0,0]=" not in text and "index='0:" not in text
    assert "min=-2 max=0" in text and "non-finite values: 1" in text  # one NaN hides no figure


def test_no_action_replies_with_work_between_do_not_end_the_session(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [NOTHING]
    for i in range(3):
        steps += [act("read_tensor", name="weight", index=f"0:{i + 1}"), NOTHING]
    s = _session(tmp_path, fx, [*steps, SUBMIT])
    _run(s)
    assert (s.stop, s.answer) == ("submitted", ("done", None))
    assert s.refused_turns == 4


def test_three_in_a_row_still_end_it(tmp_path: Path, fx: FixtureBundle) -> None:
    s = _session(tmp_path, fx, [NOTHING, NOTHING, NOTHING, SUBMIT])
    _run(s)
    assert (s.stop, s.steps) == ("answered", 3)


def test_a_repeat_is_not_work_between(tmp_path: Path, fx: FixtureBundle) -> None:
    read = act("read_tensor", name="weight")
    s = _session(tmp_path, fx, [read, NOTHING, read, NOTHING, read, NOTHING, SUBMIT])
    _run(s)
    assert (s.stop, s.steps) == ("answered", 6)


def test_a_run_from_before_the_fix_replays_as_it_ran(tmp_path: Path, big: FixtureBundle) -> None:
    steps = [act("read_tensor", name="weight"), NOTHING]
    for i in range(2):
        steps += [act("read_tensor", name="weight", index=f"0:{i + 1}"), NOTHING]
    s = _session(tmp_path, big, [*steps, SUBMIT])
    s.feedback = False
    lines = _run(s)
    assert re.search(r"\n\.\.\. \[truncated [\d,]+ chars\]$", _outputs(lines)[0])
    assert (s.stop, s.steps) == ("answered", 6)  # the third one in all
    original = s.out / "transcript.jsonl"
    line = line_for(
        tmp_path / "replay",
        big,
        "repair",
        None,
        python=False,
        max_steps=30,
        mode="replay",
        replay_of_mode="session",
        extra=[("transcript.jsonl", "transcript", original)],
    )
    r = session_mod.Session(line, client=read_script(original))
    r._loop()
    assert (r.stop, r.steps, r.replay_divergence) == ("answered", 6, None)
