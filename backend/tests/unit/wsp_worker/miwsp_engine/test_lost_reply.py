"""A lost reply does not throw the session away (fix 2026-11-22, D128 round 7, run
``wsp_e571f6971fd2``: the engine's first call timed out behind another run's runaway reply, and
``wsp_gateway_unreachable`` failed the whole run).

``session.py`` step 3c: a call whose reply never came (the connection dropped or given up on, or
the gateway's own upstream failing: ``client.LOST_REPLY_CODES``) is recorded, counted as a
refused turn, and the editor is told once; a second lost reply in a row ends the session as its
time budget does, with the transcript, edits and usage kept, and the run COMPLETES (its stop is a
budget stop). A refusal (any other code) still ends the session ``error`` as before.

The first half runs the real child under the worker's interpreter against a real HTTP endpoint
(``test_engine_scripted.py``'s harness); the second, in process, the final turn's cases.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, result, transcript
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import _patch, _replay, call, session
from wsp import runs
from wsp_worker.miwsp_engine import prompts
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.client import LOST_REPLY_CODES, ClientError
from wsp_worker.miwsp_engine.protocol import Reply
from wsp_worker.miwsp_engine.transcript import ScriptedClient

EDIT = call("python", code="W['weight'][0] += 1.0")
SUBMIT = call("submit", answer="done")
#: The two ways a reply is lost: the connection closes unanswered (``wsp_gateway_unreachable``),
#: or the gateway answers that its upstream call failed (a reply past the call time limit).
LOST = {
    "dropped": {"drop": True},
    "upstream_failed": {"status": 502, "code": "wsp_gateway_upstream_failed"},
}
LOST_CODE = {"dropped": "wsp_gateway_unreachable", "upstream_failed": "wsp_gateway_upstream_failed"}


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _users(body: dict[str, Any]) -> list[str]:
    return [str(m["content"]) for m in body["messages"] if m["role"] == "user"]


def _events(run: Path, name: str) -> list[dict[str, Any]]:
    return [e for e in transcript(run) if e.get("event") == name]


def _completes(res: dict[str, Any]) -> bool:
    """The intake's status rule for a normal exit (``runs.intake_result``)."""
    cause = runs.reported_failure(res, res["stop_reason"])
    stop = runs.labelled_stop(res["stop_reason"] or "answered", cause)
    return cause is None or stop in runs.BUDGET_STOPS or cause["code"] in runs.BUDGET_LIKE_CAUSES


@pytest.mark.parametrize("how", list(LOST))
def test_one_lost_reply_is_told_once_and_the_session_goes_on(
    tmp_path: Path, fx: FixtureBundle, how: str
) -> None:
    ep, run, _ = session(tmp_path, fx, "repair", [EDIT, LOST[how], SUBMIT])
    res = result(run)
    assert res["stop_reason"] == "submitted" and res["failure"] is None
    assert res["answer"] == "done" and res["steps"] == 3
    assert res["engine"]["refused_turns"] == 1
    # Three requests: the edit, the lost one, then the submit; the third carries the notice
    # once, as the last message, and nothing of the lost reply.
    assert len(ep.requests) == 3
    third = ep.requests[2]["body"]
    assert _users(third).count(prompts.LOST_REPLY) == 1
    assert third["messages"][-1] == {"role": "user", "content": prompts.LOST_REPLY}
    assert ep.requests[1]["body"]["messages"] == third["messages"][:-1]
    (event,) = _events(run, "lost_reply")
    assert (event["code"], event["turn"], event["n"]) == (LOST_CODE[how], 2, 1)
    assert _patch(run) is not None and _completes(res)


@pytest.mark.parametrize("how", list(LOST))
def test_two_lost_replies_in_a_row_end_the_session_as_at_its_time_budget(
    tmp_path: Path, fx: FixtureBundle, how: str
) -> None:
    """The graceful end: stop ``time_budget`` with the cause ``editor_reply_lost``, the final
    submit-only turn offered (it submits), the edit and the usage kept, and the run completes."""
    ep, run, _ = session(tmp_path, fx, "repair", [EDIT, LOST[how], LOST[how], SUBMIT])
    res = result(run)
    assert res["stop_reason"] == "time_budget"
    assert res["failure"]["code"] == session_mod.LOST_CAUSE
    assert res["answer"] == "done" and res["engine"]["refused_turns"] == 2
    assert res["tokens"]["total_tokens"] > 0 and res["tokens"]["requests"] == 2
    assert len(ep.requests) == 4
    final = ep.requests[3]["body"]
    assert [t["function"]["name"] for t in final["tools"]] == ["submit"]
    assert _users(final)[-1] == prompts.SUBMIT_NOW_EDIT.format(why=prompts.LOST_TWICE)
    # Told once, after the first loss; the second ended the loop.
    assert _users(final).count(prompts.LOST_REPLY) == 1
    assert [e["n"] for e in _events(run, "lost_reply")] == [1, 2]
    patch = _patch(run)
    assert patch is not None and patch["changed_entries_total"] == 1
    assert _completes(res)


def test_a_replay_takes_the_same_path_through_the_lost_replies(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [EDIT, LOST["upstream_failed"], LOST["dropped"], SUBMIT]
    _ep, first, _ = session(tmp_path, fx, "repair", steps, forward_passes=1)
    again = _replay(tmp_path, fx, first / "out" / "transcript.jsonl", "session")
    original, replayed = result(first), result(again)
    assert replayed["replay_divergence"] is None
    for key in ("stop_reason", "failure", "steps", "answer"):
        assert replayed[key] == original[key], key
    assert replayed["engine"]["refused_turns"] == original["engine"]["refused_turns"] == 2
    a, b = _patch(first), _patch(again)
    assert a is not None and b is not None and a["content_sha256"] == b["content_sha256"]


def test_a_refusal_is_not_a_lost_reply_and_still_ends_the_session(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Only a lost reply is survived: the gateway's in-flight refusal (a 409) ends the session
    ``error`` on its code after one request, as every refusal did."""
    assert "wsp_request_in_flight" not in LOST_REPLY_CODES
    steps = [EDIT, {"status": 409, "code": "wsp_request_in_flight"}, SUBMIT]
    ep, run, _ = session(tmp_path, fx, "repair", steps)
    res = result(run)
    assert res["stop_reason"] == "error" and res["failure"]["code"] == "wsp_request_in_flight"
    assert len(ep.requests) == 2 and _events(run, "lost_reply") == []


# ------------------------------------------------------------- in process: the final turn's cases


class Losing(ScriptedClient):
    """Scripted replies; ``None`` in the script is a lost reply (the gateway unreachable)."""

    def __init__(self, script: list[dict[str, Any] | None]) -> None:
        self.script = script
        super().__init__([m for m in script if m is not None])
        self.n = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> Reply:
        step = self.script[self.n] if self.n < len(self.script) else {}
        self.n += 1
        if step is None:
            raise ClientError(0, "wsp_gateway_unreachable", "ReadTimeout")
        return super().complete(messages, fields=fields, max_tokens=max_tokens)


def _reply(name: str, **args: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": f"c_{name}", "name": name, "arguments": json.dumps(args)}],
    }


def _run(tmp: Path, fx: FixtureBundle, script: list[dict[str, Any] | None], **over: Any) -> Any:
    line = line_for(tmp / "run", fx, "repair", None, python=False, **over)
    s = session_mod.Session(line, client=Losing(script))
    s.writer = None
    s._loop()
    return s


def test_lost_replies_not_in_a_row_never_end_the_session(tmp_path: Path, fx: FixtureBundle) -> None:
    script = [None, _reply("diff"), None, _reply("list_tensors"), _reply("submit", answer="ok")]
    s = _run(tmp_path, fx, script)
    assert s.stop == "submitted" and s.failure is None and s.answer == ("ok", None)
    assert s.refused_turns == 2 and s.lost_replies == 2 and s.steps == 5


def test_a_lost_final_turn_ends_with_no_answer_never_an_error(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The step budget's final submit-only turn, lost once: the session ends at its own stop,
    with no answer and no failure."""
    s = _run(tmp_path, fx, [_reply("diff"), None], max_steps=1)
    assert s.stop == "max_steps" and s.failure is None and s.answer == ("(no answer)", None)
    assert s.lost_replies == 1


def test_a_third_lost_reply_in_the_final_turn_keeps_the_graceful_end(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    s = _run(tmp_path, fx, [_reply("diff"), None, None, None])
    assert s.stop == "time_budget" and s.failure["code"] == session_mod.LOST_CAUSE
    assert s.answer == ("(no answer)", None) and s.lost_replies == 3
