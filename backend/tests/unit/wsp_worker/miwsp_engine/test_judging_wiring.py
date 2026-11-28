"""The engine tells the editor how it is judged, and catches an edit that cannot change the
verdict (fix 2026-11-22, D128 round 7). Each session runs for real through the child preamble
against a scripted endpoint that records what it RECEIVED.

- the judging section reaches the endpoint, in the first user message's ``## Context`` block,
  built from the run's own request and bundle;
- the first ``submit`` (or ``submit_program``) of a bias-only edit is not accepted: the editor is
  told once, the transcript records ``affine_equivalent_submit``, and a second submit is accepted;
- a weight turn is accepted at once, and the final submit-only turn never warns;
- a replay follows the original's header: a transcript from before judging (no ``judging`` key)
  replays its bias-only submit as accepted, with no divergence and the same patch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, result, run_main, tool_outputs, transcript
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import call, session
from wsp_worker import judging

PRIMARY = ["eval"]


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _events(run: Path, name: str) -> list[dict[str, Any]]:
    return [e for e in transcript(run) if e.get("event") == name]


def _patch(run: Path) -> dict[str, Any] | None:
    path = run / "out" / "patch" / "patch.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_the_section_reaches_the_endpoint_in_the_context(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [call("submit", answer="nothing to do")]
    ep, run, _ = session(tmp_path, fx, "repair", steps, primary_sets=PRIMARY)
    intro = ep.requests[0]["body"]["messages"][1]["content"]
    request = {"kind": "repair", "primary_sets": PRIMARY}
    expected = judging.section(request, fx.manifest)
    assert expected is not None and expected.startswith(judging.HEADING)
    context = intro.split("## Context\n", 1)[1].split("\n\n## Weights", 1)[0]
    assert context.endswith("\n\n" + expected)
    assert "on every primary set: eval." in context
    assert transcript(run)[0]["judging"] == judging.VERSION


def test_an_inspect_gets_no_section(tmp_path: Path, fx: FixtureBundle) -> None:
    ep, _, _ = session(tmp_path, fx, "inspect", [call("submit", answer="read")])
    assert judging.HEADING not in ep.requests[0]["body"]["messages"][1]["content"]


def test_a_bias_only_submit_is_warned_once_then_accepted(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [
        call("python", code="W['bias'][0] -= 0.5"),
        call("submit", answer="first"),
        call("submit", answer="second"),
    ]
    _ep, run, _ = session(tmp_path, fx, "repair", steps)
    outputs = tool_outputs(run)
    assert outputs[1] == judging.AFFINE_SUBMIT.format(
        what="a uniform shift of the bias", tool="submit"
    )
    assert outputs[2].startswith("submitted")  # 2026-11-27: then the delta it carries
    (event,) = _events(run, "affine_equivalent_submit")
    assert event == {
        "event": "affine_equivalent_submit",
        "turn": 2,
        "tool": "submit",
        "equivalence": "bias_shift",
        "scale": 1.0,
    }
    out = result(run)
    assert out["stop_reason"] == "submitted" and out["answer"] == "second"
    assert set(out["engine"]) == {
        "action_protocol",
        "handoffs",
        "refused_turns",
        "kernel_restarts",
        "reasoning_turns",
        "repeated_turns",
        "pages_out",
        "recalls",
        "ledger_rows",
        "peak_prompt_tokens",
    }  # fmt: skip  (no new engine key: runs.result_counters refuses one)


def test_a_rescale_is_warned_too(tmp_path: Path, fx: FixtureBundle) -> None:
    code = "W['weight'][:] = W['weight'] * 2.0; W['bias'][:] = W['bias'] * 2.0"
    steps = [call("python", code=code), call("submit", answer="a"), call("submit", answer="b")]
    _ep, run, _ = session(tmp_path, fx, "repair", steps)
    (event,) = _events(run, "affine_equivalent_submit")
    assert event["equivalence"] == "positive_rescale" and event["scale"] == pytest.approx(2.0)


def test_a_weight_turn_is_accepted_at_once(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [call("python", code="W['weight'][0] += 1.0"), call("submit", answer="turned")]
    _ep, run, _ = session(tmp_path, fx, "repair", steps)
    assert tool_outputs(run)[1].startswith("submitted")
    assert _events(run, "affine_equivalent_submit") == []


def test_the_final_submit_only_turn_never_warns(tmp_path: Path, fx: FixtureBundle) -> None:
    """At the step limit the editor gets one turn with ``submit`` alone, where it could not act
    on a warning: there the bias-only edit is accepted."""
    steps = [call("python", code="W['bias'][0] -= 0.5"), call("submit", answer="final")]
    _ep, run, _ = session(tmp_path, fx, "repair", steps, max_steps=1)
    out = result(run)
    assert out["answer"] == "final" and tool_outputs(run)[-1].startswith("submitted")
    assert _events(run, "affine_equivalent_submit") == []


BIAS_PROGRAM = "def edit(base, support):\n    return {'bias': np.array(base['bias']) - 0.5}\n"


def test_a_bias_only_program_is_warned_once_then_stored(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [
        call("test_program", code=BIAS_PROGRAM),
        call("submit_program", code=BIAS_PROGRAM),
        call("submit_program", code=BIAS_PROGRAM),
        call("submit", answer="programmed"),
    ]
    _ep, run, _ = session(tmp_path, fx, "repair", steps, mode="program", forward_passes=1)
    outputs = tool_outputs(run)
    tested = json.loads(outputs[0].split("\n[forward budget")[0])
    assert tested["primary_metric"] == "auroc" and "at_own_bar" in tested
    assert tested["affine_equivalent_to_parent"] is True
    assert tested["auroc"] == tested["parent"]["auroc"]
    assert tested["same_verdict_note"] == judging.SAME_VERDICT
    assert outputs[1] == judging.AFFINE_SUBMIT.format(
        what="a uniform shift of the bias", tool="submit_program"
    )
    assert outputs[2].startswith("program stored")
    (event,) = _events(run, "affine_equivalent_submit")
    assert event["tool"] == "submit_program" and event["turn"] == 2
    patch = _patch(run)
    assert patch is not None and [t["name"] for t in patch["tensors"]] == ["bias"]


# ----------------------------------------------------------------------------- replay


def _replay(tmp: Path, fx: FixtureBundle, transcript_path: Path, **over: Any) -> Path:
    run = tmp / "replay"
    line = line_for(
        run,
        fx,
        "repair",
        None,
        mode="replay",
        replay_of_mode="session",
        extra=[("transcript.jsonl", "transcript", transcript_path)],
        **over,
    )
    seen = run_main(run, line, mode="replay")
    assert seen["exit"] == 0, seen["stderr"][-3000:]
    return run


BIAS_STEPS = [
    call("python", code="W['bias'][0] -= 0.5"),
    call("submit", answer="first"),
    call("submit", answer="second"),
]


def test_a_judged_run_replays_its_warning_bit_for_bit(tmp_path: Path, fx: FixtureBundle) -> None:
    _ep, first, _ = session(tmp_path, fx, "repair", BIAS_STEPS)
    run = _replay(tmp_path, fx, first / "out" / "transcript.jsonl")
    assert result(run)["replay_divergence"] is None
    assert len(_events(run, "affine_equivalent_submit")) == 1
    assert result(run)["answer"] == "second"
    original, replayed = _patch(first), _patch(run)
    assert original is not None and replayed is not None
    assert replayed["content_sha256"] == original["content_sha256"]


def _as_before_judging(source: Path, target: Path) -> Path:
    """The same session as an engine before 2026-11-22 recorded it: no ``judging`` in the header,
    no warning (its event, its tool answer and the submit that drew it), the context as it was."""
    lines = [json.loads(x) for x in source.read_text().splitlines()]
    header = {k: v for k, v in lines[0].items() if k != "judging"}
    kept: list[dict[str, Any]] = [header]
    rest = lines[1:]
    for i, obj in enumerate(rest):
        drew_warning = (
            rest[i + 1 : i + 2] and rest[i + 1].get("event") == "affine_equivalent_submit"
        )
        if obj.get("role") == "assistant" and drew_warning:
            continue
        if obj.get("event") == "affine_equivalent_submit":
            continue
        if obj.get("role") == "tool" and str(obj.get("content", "")).startswith("Not submitted"):
            continue
        if obj.get("role") == "user" and judging.HEADING in str(obj.get("content", "")):
            content = str(obj["content"])
            start = content.index("\n\n" + judging.HEADING)
            end = content.index("\n\n## Weights")
            obj = {**obj, "content": content[:start] + content[end:]}
        kept.append(obj)
    target.write_text("".join(json.dumps(o) + "\n" for o in kept))
    return target


def test_a_run_recorded_before_judging_replays_as_it_ran(tmp_path: Path, fx: FixtureBundle) -> None:
    """THE replay guarantee: an old run's bias-only submit was accepted at once, and its replay
    must accept it at once too (no warning, no divergence, the same patch), or no round-7 run
    could be replayed."""
    _ep, first, _ = session(tmp_path, fx, "repair", BIAS_STEPS)
    old = _as_before_judging(first / "out" / "transcript.jsonl", tmp_path / "old.jsonl")
    script = [json.loads(x) for x in old.read_text().splitlines()]
    assert "judging" not in script[0]
    assert [m.get("content") for m in script if m.get("role") == "tool"][-1].startswith("submitted")
    run = _replay(tmp_path, fx, old)
    assert result(run)["replay_divergence"] is None
    assert _events(run, "affine_equivalent_submit") == []
    assert "judging" not in transcript(run)[0]
    intro = next(m for m in transcript(run) if m.get("role") == "user")
    assert judging.HEADING not in intro["content"]
    assert result(run)["answer"] == "second"  # the submit the old engine accepted
    original, replayed = _patch(first), _patch(run)
    assert original is not None and replayed is not None
    assert replayed["content_sha256"] == original["content_sha256"]


TURN_PROGRAM = (
    "def edit(base, support):\n"
    "    w = np.array(base['weight'])\n"
    "    w[0] += np.float32(1.0)\n"
    "    return {'weight': w}\n"
)


def _turned(t: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    w = t["weight"].astype(np.float32)
    w[0] += np.float32(1.0)
    return {**t, "weight": w.astype(np.float64)}


@pytest.mark.parametrize("target", [0.34, 0.0014285714285714286], ids=["0.34", "d128"])
def test_test_program_cuts_at_the_bundles_target(tmp_path: Path, target: float) -> None:
    """Wiring (N3): ``ProgramTools`` reads ``operating_point.target_fpr``; ``test_program``
    reports it and cuts the program's bar (and the parent's) there."""
    from tests.unit.wsp_worker.test_judged_measure import expected_bar

    fx = make_bundle(tmp_path / "bundle", rule="mean", target_fpr=target)
    steps = [call("test_program", code=TURN_PROGRAM), call("submit", answer="x")]
    _ep, run, _ = session(tmp_path, fx, "repair", steps, mode="program", forward_passes=1)
    tested = json.loads(tool_outputs(run)[0].split("\n[forward budget")[0])
    for side, edit in ((tested, _turned), (tested["parent"], None)):
        own = side["at_own_bar"]
        assert own["target_false_alarm_rate"] == target
        assert own["bar"] == expected_bar(fx, target, edit)
    if target == 0.34:
        assert tested["at_own_bar"]["bar"] is not None
