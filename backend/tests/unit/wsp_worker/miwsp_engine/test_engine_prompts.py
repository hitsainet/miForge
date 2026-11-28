"""The engine's task line (fix 2026-11-21, live round 5 on main ``df8c9c6``).

- **A repair with a goal IMPROVES and carries its goal.** The engine said FIX where weightspace
  0.3.0 said IMPROVE. (Until 018-P4 a test here read weightspace's ``MODES`` and its adapter's
  ``session_mode``/``session_task`` under the worker's interpreter and required the engine's
  equal, line for line; the library left the worker with its retirement, FR-018.18.)
- **Program style** (run ``wsp_554d3932d108``): its first message said "Task type: READ ... You
  cannot change any weights", so the editor changed nothing. It now names the mode, says the
  editor writes a program, names ``test_program`` and ``submit_program`` with the cost, and says
  ``submit`` alone changes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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


#: Every text the editor reads, pinned (018 16.2, 2026-11-24). Until 018-P4,
#: ``test_the_mode_texts_are_weightspaces_line_for_line`` held the mode texts to weightspace's;
#: it went with the adapter, and then an edited READ text ("You may change weights.") passed
#: every worker test. These texts are part of what the switch record measured
#: (``0xcc/reviews/018_switch.md``): changing one changes the editor's behaviour, so re-run the
#: comparison (or record why not) in the same change, then update the pin. The names come from
#: the module (every upper-case ``str``, and every upper-case ``dict`` of ``str``), so a new text
#: that nobody pinned fails as well.
EDITOR_TEXT_PINS = {
    # Feature 020 (2026-11-25): four texts that only an append-only session reads; no paging
    # session's text changed, so 018's switch comparison (paging sessions) is not re-run
    # (``0xcc/reviews/020_P1_controls.md`` records why).
    "APPEND_ONLY": "b5e7637da8d40509",
    "FINAL_SUBMIT_ONLY": "9bb3b2985afd4b2e",
    "RECALL_FROM_DESCRIPTION": "853f30b96f086465",
    "RECALL_HINT": "fb405435a025d416",
    "ACTION_CHANGED": "f833bbbf3072f0d2",
    "CUT_OFF": "2a32c48f78b84e08",
    "CUT_OFF_WRITE": "4e4c108b63bb0d39",
    # 2026-11-28 (run wsp_3417a18bc85d, phase-2 E7): new text, appended to the FIRST cut-off
    # answer of a session with the output cap off, whose later requests are then capped (four of
    # E7's replies ran to the 32,000-token ceiling, about 870 s each). The switch comparison runs
    # capped sessions, which never read it; its re-run is pending with FU-3's GPU window.
    "CUT_OFF_CAPPED": "c4e9dc8c034815a3",
    # 2026-11-28 (run wsp_0d4a2f916c9c, phase-2 E1r): new text, ending a python or write_tensor
    # answer only when its edit recreated weights a forward pass measured and a revert or restore
    # then left, after which the previous engine said nothing while the editor spent ~20 of 100
    # steps on one edit. A comparison session that never repeats a reverted edit sees no change;
    # the switch comparison re-run is pending with FU-3's GPU window, as for RESERVED_REBOUND
    # (0xcc/reviews/wsp_window_read_not_resident_2026-11-27.md, fix 4).
    "EDIT_REPEATED": "f13537b1d7d1ced8",
    "FORWARD_EXHAUSTED": "ed869badb620d47e",
    "FORWARD_LEFT": "ad66d81eb5fc4964",
    "HANDOFF": "eab8a00cb510230b",
    "HANDOFF_PAGED": "aa1cfbd398874217",
    "KERNEL_RESTARTED": "624d080df0031957",
    "LOST_REPLY": "77fce978d2410ecb",
    "LOST_TWICE": "a51fb821322bf217",
    "MODES[fix]": "7bf4d3a96c218ae1",
    "MODES[improve]": "1a03177753911f68",
    "MODES[read]": "72b08b17fe90d9d8",
    "MODES[write]": "85fe258b14e9215f",
    "MODE_FOR[create]": "10fd874b68dad080",
    "MODE_FOR[inspect]": "3316348dbadfb7b1",
    "MODE_FOR[repair]": "1c6e6c4c02e55178",
    "MULTIPLE_ACTIONS": "d71c811e922b6732",
    "NEARLY_UP_EDIT": "b0d3a296bf2e0045",
    "NEARLY_UP_READ": "b1bf3ae2273923f8",
    "NOT_OFFERED": "78c57058c35163d9",
    "NO_THINKING": "17fef0689ade0817",
    "NO_TIME_FOR_FORWARD": "6587f29a36244ddd",
    "NUDGE": "514a82c5b58cbe30",
    "ONE_ACTION": "9d26d97b7ec66cd1",
    "PAGING": "d8da46136b3bc501",
    "PROGRAM_AIM[fix]": "aada90cd72282c9b",
    "PROGRAM_AIM[improve]": "831cff95e345dd42",
    "PROGRAM_AIM[read]": "f01615ef0eb0d288",
    "PROGRAM_AIM[write]": "bf0fdfde6130222a",
    "PROGRAM_MODE": "2d87c7a388694437",
    "RECALL_DESCRIPTION": "b80b25e69ec7bfab",
    # 2026-11-27 (run wsp_ba149c57f611): new text, ending a python run's output only when its
    # code assigned or deleted R, W, notes or np, after which the previous engine left that name
    # broken for the rest of the session. A comparison session that never assigns one sees no
    # change; the switch comparison re-run is pending with FU-3's GPU window
    # (0xcc/reviews/wsp_window_read_not_resident_2026-11-27.md).
    "RESERVED_REBOUND": "f24171afd05f2227",
    "REPEATED": "7020c18eac60300b",
    "REPEATED_PAGE": "a95c5ae38a5e065e",
    "REPEAT_LIMIT": "cb2cb7cd7b975a64",
    "REPEAT_LIMIT_IN_A_ROW": "4fed3462c25f3804",
    "RUN_LIMIT": "5ff011d1632d410c",
    "STEP_LIMIT": "00978e5081f9f99e",
    "SUBMIT_CUT": "3e17867abc1450bb",
    "SUBMIT_NOW": "3995d3d4f84860dc",
    # FU-3 (2026-11-25): new text. The switch comparison re-run is pending (a GPU window, the
    # operator's choice); 018 FTASKS FU-3 records it.
    "SUBMIT_NOW_EDIT": "b0af7884549fbe29",
    # FU-2 (2026-11-25): new text, submit's optional figures. The switch comparison re-run is
    # pending, in FU-3's GPU window (both editor-text changes are compared in one run).
    "SUBMIT_FIGURES": "fcdfb411049504b7",
    "SUBMIT_REFUSED": "edede124aaf1015a",
    # 2026-11-27: new text, the answer to an ACCEPTED submit in an editing session. An accepted
    # submit ends the session, so no editor turn ever reads it and no comparison can measure it;
    # it is the transcript's record of the delta beside the editor's answer. Not re-run.
    "SUBMITTED_CHANGES": "f8be9c5a6f79ed4f",
    "SYSTEM": "f78ac1979291e69c",
    "TRUNCATED": "8d103429d41e554e",
    "UNPARSEABLE": "fabdae0874912c29",
    # 2026-11-26 (FU-3's finding, run wsp_4f035a6d8a61): new text, appended to a write's answer
    # when entries it changed moved by less than a reading shows. The switch comparison is not
    # re-run for it: its fixture half is an inspect, which has no write tool, so the text cannot
    # reach any session it measures (the reason FU-3 recorded for SUBMIT_NOW_EDIT's inspect half).
    # 0xcc/reviews/wsp_unnamed_writes_2026-11-26.md records it.
    "WRITE_NOISE": "b369c9fb33dc3445",
}


def _editor_texts() -> dict[str, str]:
    import hashlib

    found = {}
    for name in sorted(vars(prompts)):
        value = getattr(prompts, name)
        if not name.isupper():
            continue
        if isinstance(value, str):
            found[name] = value
        elif isinstance(value, dict) and value and all(isinstance(v, str) for v in value.values()):
            found.update({f"{name}[{key}]": value[key] for key in value})
    return {name: hashlib.sha256(text.encode()).hexdigest()[:16] for name, text in found.items()}


def test_every_editor_text_is_pinned() -> None:
    found = _editor_texts()
    assert len(found) >= 40, "the walk found too few texts: it is not reading the module"
    changed = sorted(
        name
        for name in found.keys() | EDITOR_TEXT_PINS.keys()
        if found.get(name) != EDITOR_TEXT_PINS.get(name)
    )
    assert not changed, (
        "an editor text changed, appeared or went: re-run the switch comparison (018_switch.md) or "
        f"record why not, then update EDITOR_TEXT_PINS. Changed: {changed}"
    )
