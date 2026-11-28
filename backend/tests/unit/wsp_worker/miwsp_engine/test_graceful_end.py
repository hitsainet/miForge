"""The engine's graceful end before the supervisor's kill (fix 2026-11-21, ``wsp_cb092cedad97``).

The supervisor SIGKILLs the child at the time budget, measured from the session's start, and a
killed session leaves no ``result.json`` and no patch. So the engine (``session.py`` step 4a):

- measures time to that kill (``time_left_s`` in the child line), not to its own start;
- tells the editor once, per budget, that it is nearly used up (checkpoint, then submit);
- stops early enough for one more turn plus the final submit-only turn;
- does not start a forward pass that would not finish before the final turn's reserve.

In-process on a fake clock and a scripted client (no kernel, no gateway): ``Session._loop`` and
``_write_outputs`` are the same code ``run`` calls, without its isolation step, which must never run
in a test process (``PR_SET_DUMPABLE``). The full child is ``test_engine_scripted.py``'s.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_worker.miwsp_engine import prompts
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.protocol import Reply, Usage
from wsp_worker.miwsp_engine.transcript import ScriptedClient

#: FU-3: the end of an edit session's limit instruction (the weights as they stand are kept).
KEPT = prompts.SUBMIT_NOW_EDIT.format(why="").lstrip()


class FakeTime:
    """``session.time`` with a clock the test moves."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class ClockedClient(ScriptedClient):
    """Each request takes ``request_s`` on the fake clock and reports ``tokens`` used."""

    def __init__(
        self, clock: FakeTime, steps: list[tuple[str, dict[str, Any]]], request_s: float
    ) -> None:
        replies = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"c{i}", "name": name, "arguments": args}],
            }
            for i, (name, args) in enumerate(steps)
        ]
        super().__init__(replies)
        self.clock, self.request_s = clock, request_s
        self.seen: list[list[dict[str, Any]]] = []

    def complete(
        self, messages: Sequence[Mapping[str, Any]], *, fields: Mapping[str, Any], max_tokens: int
    ) -> Reply:
        self.seen.append([dict(m) for m in messages])
        self.clock.now += self.request_s
        reply = super().complete(messages, fields=fields, max_tokens=max_tokens)
        reply.usage = Usage(100, 10)
        return reply


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeTime:
    fake = FakeTime()
    monkeypatch.setattr(session_mod, "time", fake)
    return fake


def _session(
    tmp: Path, fx: FixtureBundle, client: ClockedClient, **over: Any
) -> session_mod.Session:
    line = line_for(tmp / "run", fx, "repair", None, python=False, **over)
    s = session_mod.Session(line, client=client)
    s.writer = None
    return s


def _users(messages: list[dict[str, Any]]) -> list[str]:
    return [str(m["content"]) for m in messages if m.get("role") == "user"]


def _tool_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [str(m["content"]) for m in messages if m.get("role") == "tool"]


def test_the_editor_is_told_time_is_nearly_up_and_a_submit_after_it_is_a_candidate(
    tmp_path: Path, fx: FixtureBundle, clock: FakeTime
) -> None:
    """600 s budget, but the supervisor had 100 s left when it wrote the line (a slow start):
    the stop is at 60 s left (the 10% share), and the notice comes when two more turns of the
    longest so far would reach it. The editor checkpoints, then submits: a patch is written."""
    w = fx.tensors["weight"].values64().astype(float).tolist()
    w[0] += 1.0
    steps = [
        ("write_tensor", {"name": "weight", "values": w}),
        ("diff", {}),
        ("checkpoint", {"label": "best"}),
        ("submit", {"answer": "fixed", "confidence": 0.7}),
    ]
    client = ClockedClient(clock, steps, request_s=12.0)
    s = _session(tmp_path, fx, client, time_s=600, time_left_s=100)
    s._loop()
    # Turn 1 ends at 88 s left (12 s turn): 88 > 60 + 24, no notice. Turn 2 ends at 76 s left:
    # 76 <= 60 + 24, so the notice follows it; turns 3 and 4 run (64 > 60 at each start).
    users = _users(client.seen[2])
    notice = [u for u in users if u.startswith("Budget nearly used up")]
    assert len(notice) == 1 and "time: about 76 s of 600 s left" in notice[0]
    assert "Checkpoint your best weights now, then call `submit`" in notice[0]
    assert not any(u.startswith("Budget nearly used up") for u in _users(client.seen[1]))
    assert len(client.seen) == 4 and s.stop == "submitted" and s.answer == ("fixed", 0.7)
    s._write_outputs()
    patch = json.loads((s.out / "patch" / "patch.json").read_text())
    assert patch["changed_entries_total"] == 1
    # Sent once: the later requests carry the same single notice.
    assert sum(u.startswith("Budget nearly used up") for u in _users(client.seen[3])) == 1


def test_the_stop_leaves_room_for_one_more_turn_and_the_final_one(
    tmp_path: Path, fx: FixtureBundle, clock: FakeTime
) -> None:
    """Turns of 40 s against 100 s left: after the first (60 s left) the next turn plus the
    final one (40 + 40 + 10 s) would cross the kill, so the session stops ``time_budget`` and
    offers ``submit`` alone, with the accepted edit kept."""
    w = fx.tensors["weight"].values64().astype(float).tolist()
    w[1] -= 2.0
    steps = [
        ("write_tensor", {"name": "weight", "values": w}),
        ("submit", {"answer": "at the limit"}),
    ]
    client = ClockedClient(clock, steps, request_s=40.0)
    s = _session(tmp_path, fx, client, time_s=600, time_left_s=100)
    s._loop()
    assert s.stop == "time_budget" and s.answer == ("at the limit", None)
    assert _users(client.seen[1])[-1].endswith(KEPT)
    assert clock.now - 1000.0 == 80.0 < 100.0  # the final reply came before the kill
    s._write_outputs()
    assert (s.out / "patch" / "patch.json").exists()


def test_a_forward_pass_that_would_not_finish_is_not_run_or_charged(
    tmp_path: Path, fx: FixtureBundle, clock: FakeTime
) -> None:
    """A practice set whose pass takes ~70 s (the prior from its bytes) with 74 s left when the
    action would run: the pass would end inside the final turn's reserve (the 1 s request plus
    10 s), so it is refused by text and costs nothing; the session goes on to submit."""
    tools = [{"name": "practice_metrics", "version": "2", "cost": 1}]
    steps = [("forward_practice_metrics", {"split": "train"}), ("submit", {"answer": "done"})]
    client = ClockedClient(clock, steps, request_s=1.0)
    s = _session(
        tmp_path, fx, client, time_s=600, time_left_s=125, forward_passes=3, observation_tools=tools
    )
    s._forward_prior_s = 70.0
    clock.now += 50.0  # 75 s left when the first turn starts, 74 s when its action would run
    s._loop()
    refused = _tool_texts(client.seen[1])[0]
    assert refused.startswith("Not run: about 74 s are left and a forward pass has taken about 70")
    assert s.forward.used == 0 and s.answer == ("done", None)


def test_a_forward_pass_that_fits_runs_and_its_time_is_the_next_estimate(
    tmp_path: Path, fx: FixtureBundle, clock: FakeTime
) -> None:
    tools = [{"name": "practice_metrics", "version": "2", "cost": 1}]
    steps = [("forward_practice_metrics", {"split": "train"}), ("submit", {"answer": "done"})]
    client = ClockedClient(clock, steps, request_s=1.0)
    s = _session(tmp_path, fx, client, time_s=600, forward_passes=3, observation_tools=tools)
    s._loop()
    assert json.loads(_tool_texts(client.seen[1])[0].split("\n[forward")[0])["rows"] > 0
    assert s.forward.used == 1 and s._slowest_forward is not None


def test_the_token_notice_shares_the_mechanism(
    tmp_path: Path, fx: FixtureBundle, clock: FakeTime
) -> None:
    """90% of 1,000 tokens is the engine's 900; each request uses 110, so after the seventh
    (770, past 80% of 900 and within two requests of it) the token notice is sent, once."""
    # Seven DIFFERENT reads: the repeat guard (2026-11-21) would answer a repeated one unrun.
    steps = [("list_tensors", {"pattern": f"w{i}*"}) for i in range(7)] + [
        ("submit", {"answer": "t"})
    ]
    client = ClockedClient(clock, steps, request_s=0.5)
    s = _session(tmp_path, fx, client, time_s=600, tokens=1000)
    s._loop()
    notices = [u for u in _users(client.seen[-1]) if u.startswith("Budget nearly used up")]
    assert len(notices) == 1 and "tokens: " in notices[0] and "of 900 used" in notices[0]


def test_a_read_session_is_told_to_submit_its_answer(
    tmp_path: Path, fx: FixtureBundle, clock: FakeTime
) -> None:
    steps = [("list_tensors", {})] * 3 + [("submit", {"answer": "read"})]
    client = ClockedClient(clock, steps, request_s=12.0)
    line = line_for(
        tmp_path / "run", fx, "inspect", None, python=False, time_s=600, time_left_s=100
    )
    s = session_mod.Session(line, client=client)
    s.writer = None
    s._loop()
    notices = [u for u in _users(client.seen[-1]) if u.startswith("Budget nearly used up")]
    assert notices == [prompts.NEARLY_UP_READ.format(details="time: about 76 s of 600 s left")]
