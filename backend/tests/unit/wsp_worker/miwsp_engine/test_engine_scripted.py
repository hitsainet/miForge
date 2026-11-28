"""The engine with scripted endpoints (feature 018; FTASKS 3.2, 3.4, 3.6, 4.2, 4.6, 5.3, 5.5 to
5.7, 6.4, 7.5): the parallel of the retired adapter's ``test_adapter_scripted.py``, plus what only
the engine does.

Each session runs for real under the worker's interpreter, behind the real child preamble
(``engine_cases.py``) or the real entry point (``run_main``), against a real HTTP endpoint that
records what it RECEIVED. What the session leaves in ``/out`` is checked here with the backend's
own code: the patch re-applies through ``wsp.weight_ops.patch.apply_patch`` and equals the
backend's ``build_patch`` of the same edit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_engine import (
    line_for,
    result,
    run_case,
    run_main,
    tool_outputs,
    transcript,
)
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.support.wsp_scripted_endpoint import TOKEN, ScriptedEndpoint
from wsp.weight_ops.patch import apply_patch, build_patch
from wsp.weight_ops.tensor import from_array
from wsp_worker.miwsp_engine import prompts

OPENAI = "openai_compatible"


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def call(name: str, /, **args: Any) -> dict[str, Any]:
    return {"calls": [(name, args)]}


def session(
    tmp: Path,
    fx: FixtureBundle,
    kind: str,
    steps: list[dict[str, Any]],
    *,
    endpoint_kind: str = OPENAI,
    options: dict[str, Any] | None = None,
    **over: Any,
) -> tuple[ScriptedEndpoint, Path, dict[str, Any]]:
    """One engine session through ``engine_cases.py``; the endpoint, the run dir, its output."""
    run = tmp / "run"
    with ScriptedEndpoint(endpoint_kind, steps) as ep:
        line = line_for(run, fx, kind, ep.gateway, editor_kind=endpoint_kind, **over)
        seen = run_case(run, line, **(options or {}))
    assert seen["exit"] == 0
    return ep, run, seen


def _patch(run: Path) -> dict[str, Any] | None:
    path = run / "out" / "patch" / "patch.json"
    return json.loads(path.read_text()) if path.exists() else None


def _w(fx: FixtureBundle) -> np.ndarray:
    return fx.tensors["weight"].values64().astype(np.float32)


# ----------------------------------------------------------------------------- the adapter's cases


def test_a_repair_writes_a_v1_patch_equal_to_the_backends(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [
        call("python", code="W['weight'][0] += 1.0"),
        call("submit", answer="done", confidence=0.7),
    ]
    _ep, run, _ = session(tmp_path, fx, "repair", steps)
    out = run / "out"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == [
        "patch",
        "patch/edit.npz",
        "patch/patch.json",
        "result.json",
        "transcript.jsonl",
    ]
    res = result(run)
    assert (res["answer"], res["confidence"], res["stop_reason"]) == ("done", 0.7, "submitted")
    manifest, rebuilt = apply_patch(out / "patch", fx.tensors)
    expected = _w(fx).copy()
    expected[0] += np.float32(1.0)
    assert np.array_equal(rebuilt["weight"].values64().astype(np.float32), expected)
    ours = build_patch(
        fx.tensors,
        {"weight": from_array(expected)},
        bundle_base_sha256=fx.manifest["bundle_sha256"],
        parent=manifest["parent"],
        parent_patch_sha256=None,
        created_by=manifest["created_by"],
        created_at=manifest["created_at"],
    )
    assert ours is not None and ours.manifest["content_sha256"] == manifest["content_sha256"]
    assert manifest["created_by"] == {
        "provider": "miwsp_engine",
        "provider_version": "0.1.0",
        "run_id": "wsp_000000000001",
    }


def test_a_python_write_past_the_cap_is_refused_at_the_write_and_w_reverts(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-018.4 through Python: refused at the write (not at submit), W back to the accepted
    weights, then a write inside the cap is accepted; confidence stays null (R1B-A14)."""
    steps = [
        call("python", code="W['weight'][:3] += 1.0"),
        call("python", code="float(W['weight'][1])"),
        call("python", code="W['weight'][0] += 1.0"),
        call("submit", answer="second"),
    ]
    _ep, run, _ = session(tmp_path, fx, "repair", steps, max_changed_entries=1)
    refused, reread, accepted, submitted = tool_outputs(run)
    assert "refused: the patch changes 3 entries; the limit is 1" in refused
    assert "W was reset to the last accepted weights" in refused
    assert reread.strip() == repr(float(_w(fx)[1]))
    assert accepted == "(no output)" and submitted.startswith("submitted")
    assert (result(run)["answer"], result(run)["confidence"]) == ("second", None)
    patch = _patch(run)
    assert patch is not None and patch["changed_entries_total"] == 1


def test_a_python_write_to_a_frozen_tensor_is_refused_and_w_reverts(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """6.4: a delta to a frozen tensor is refused; the kernel's W reverts."""
    steps = [
        call("python", code="W['weight'][0] += 1.0"),
        call("python", code="float(W['weight'][0])"),
        call("submit", answer="nothing"),
    ]
    _ep, run, _ = session(tmp_path, fx, "repair", steps, frozen_tensors=["weight"])
    refused, reread, _submitted = tool_outputs(run)
    assert "tensor 'weight' is frozen" in refused
    assert reread.strip() == repr(float(_w(fx)[0]))
    assert _patch(run) is None


def test_create_zeroes_then_writes_and_the_patch_is_against_the_bundle(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [call("python", code="W['weight'][:] = 1.0"), call("submit", answer="made")]
    _ep, run, _ = session(tmp_path, fx, "create", steps)
    _manifest, rebuilt = apply_patch(run / "out" / "patch", fx.tensors)
    assert np.all(rebuilt["weight"].values64() == 1.0)
    assert np.all(rebuilt["bias"].values64() == 0.0)  # zeroed, not written
    assert rebuilt["mean"].sha256 == fx.tensors["mean"].sha256  # not editable: never given


def test_a_create_that_starts_over_the_cap_cannot_submit_until_it_is_under(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-018.4 at submit: a ``create`` starts at zeros, so with ``max_changed_entries=1`` the
    store is already over the cap (D-3 lets writes move it back). Submitting then is refused with
    the engine's text and records no answer; the only gate in the session is ``_submit``'s own
    check (018-P1 review, mutation M3)."""
    # two steps, then the last turn that offers submit alone (5.3): all three are refused
    steps = [call("submit", answer=f"too soon {i}") for i in range(3)]
    _ep, run, _ = session(tmp_path, fx, "create", steps, max_changed_entries=1, max_steps=2)
    outputs = tool_outputs(run)
    zeroed = fx.tensors["weight"].values64().size + 1  # every weight and the bias
    expected = prompts.SUBMIT_REFUSED.format(
        details=f"the patch changes {zeroed} entries; the limit is 1"
    )
    assert outputs == [expected] * 3
    res = result(run)
    assert res["stop_reason"] == "max_steps" and not res["answer"].startswith("too soon")
    # F18-D16 (operator, 2026-11-24): with no accepted submit, an edit over a limit is written
    # nowhere (it used to be written for the intake to refuse), and the transcript says why.
    assert _patch(run) is None
    withheld = [e for e in transcript(run) if e.get("event") == "patch_withheld"]
    assert withheld == [
        {
            "event": "patch_withheld",
            "reason": "no_accepted_submit",
            "violations": [f"the patch changes {zeroed} entries; the limit is 1"],
        }
    ]


def test_a_partial_edit_within_the_limits_is_still_written(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """F18-D16 withholds only an edit that breaks a limit: a run that stops early (here
    ``max_steps``) with an edit inside the limits keeps its patch, which the intake may take as a
    candidate (the 2026-11-21 ruling)."""
    steps = [call("python", code="W['weight'][0] = W['weight'][0] + 0.5")] * 3
    _ep, run, _ = session(tmp_path, fx, "repair", steps, max_steps=2)
    assert result(run)["stop_reason"] == "max_steps"
    partial = _patch(run)
    assert partial is not None and partial["changed_entries_total"] == 1
    assert not [e for e in transcript(run) if e.get("event") == "patch_withheld"]


def test_an_inspect_cannot_write_through_r_and_has_no_w(tmp_path: Path, fx: FixtureBundle) -> None:
    """R is a read-only map (the hole weightspace's R had, its adapter's read_write case until
    018-P4, is closed): setting ``writeable`` raises, and inspect has no W. No patch."""
    steps = [
        call("python", code="a = R['weight']; a.flags.writeable = True; a[0] = 5.0"),
        call("python", code="W['weight'][0] = 5.0"),
        call("submit", answer="looked"),
    ]
    _ep, run, _ = session(tmp_path, fx, "inspect", steps)
    through_r, through_w, _ = tool_outputs(run)
    assert "ValueError" in through_r and "NameError" in through_w
    assert _patch(run) is None


def test_a_run_past_the_token_budget_ends_token_budget(tmp_path: Path, fx: FixtureBundle) -> None:
    """The engine's budget is 90% of the run's (900 of 1000 tokens), as the adapter's."""
    heavy = {"calls": [("python", {"code": "1"})], "usage": (600, 0)}
    _ep, run, _ = session(tmp_path, fx, "repair", [heavy] * 3, tokens=1000)
    res = result(run)
    assert res["stop_reason"] == "token_budget" and res["steps"] == 2
    assert res["answer"] == "(no answer)"
    assert _patch(run) is None


def test_the_client_is_the_gateway_with_the_session_token(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [call("submit", answer="a")]
    for kind, header in ((OPENAI, "authorization"), ("anthropic", "x-api-key")):
        ep, _run, _ = session(tmp_path / kind, fx, "repair", steps, endpoint_kind=kind)
        [request] = ep.requests
        headers = {k.lower(): v for k, v in request["headers"].items()}
        assert headers[header] == (f"Bearer {TOKEN}" if header == "authorization" else TOKEN)
        assert request["body"]["model"] == "m"


def test_observation_tools_draw_their_cost_from_forward_passes(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """3.4: budget 2, two results then a refusal by the exhaustion text; forward_used = 2."""
    tools = [{"name": "practice_metrics", "version": "2", "cost": 1}]
    # Three DIFFERENT passes: the repeat guard (2026-11-21) answers a repeat without charging it.
    steps = [
        call("forward_practice_metrics", split=split) for split in ("train", "validation", "all")
    ] + [call("submit", answer="o")]
    _ep, run, _ = session(tmp_path, fx, "repair", steps, forward_passes=2, observation_tools=tools)
    first, second, third, _ = tool_outputs(run)
    assert json.loads(first.split("\n[forward budget")[0])["tool"] == "practice_metrics"
    assert first.endswith("[forward budget: 1 of 2 left]")
    assert second.endswith("[forward budget: 0 of 2 left]")
    assert third == "forward budget exhausted (0 left, this probe costs 1). Submit your answer."
    assert result(run)["forward_used"] == 2


def test_an_unknown_observation_tool_fails_the_session_before_the_editor_runs(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    run = tmp_path / "run"
    with ScriptedEndpoint(OPENAI, [call("submit", answer="a")]) as ep:
        line = line_for(
            run,
            fx,
            "repair",
            ep.gateway,
            observation_tools=[{"name": "read_text", "version": "1", "cost": 1}],
        )
        seen = run_main(run, line)
    assert seen["exit"] == 78 and "worker_observation_tool_unknown" in seen["stderr"]
    assert not (run / "out" / "transcript.jsonl").exists() and ep.requests == []


PROGRAM = (
    "def edit(base, support):\n"
    "    w = np.array(base['weight'])\n"
    "    return {'weight': (w + np.float32(0.5) * np.sign(w)).astype(w.dtype)}\n"
)


def _program_steps() -> list[dict[str, Any]]:
    return [
        call("python", code="W['bias'][0] = 9.0"),  # program style reads: no W
        call("test_program", code=PROGRAM),
        call("submit_program", code="x = 1\n"),  # no edit: refused
        call("submit_program", code=PROGRAM),
        call("submit_program", code=PROGRAM.replace("0.5", "0.25")),
        call("submit", answer="programmed"),
    ]


def test_a_program_session_reads_stores_one_program_and_its_patch(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    _ep, run, _ = session(
        tmp_path, fx, "repair", _program_steps(), mode="program", forward_passes=1
    )
    wrote, tested, *outputs = tool_outputs(run)
    assert "NameError" in wrote
    assert json.loads(tested.split("\n[forward budget")[0])["split"] == "validation"
    assert outputs[0].startswith("refused: wsp_program_invalid")
    assert outputs[1].startswith("program stored")
    assert outputs[2].startswith("refused: wsp_program_already_submitted")
    out = run / "out"
    assert (out / "program.py").read_text() == PROGRAM
    manifest, rebuilt = apply_patch(out / "patch", fx.tensors)
    w = _w(fx)
    assert np.array_equal(
        rebuilt["weight"].values64().astype(np.float32), w + np.float32(0.5) * np.sign(w)
    )
    assert rebuilt["bias"].sha256 == fx.tensors["bias"].sha256
    assert manifest["created_by"]["provider"] == "miwsp_engine"  # 1.2: never "weightspace"
    assert result(run)["forward_used"] == 1


# ----------------------------------------------------------------------------- replay (5.5)


def _replay(tmp: Path, fx: FixtureBundle, transcript_file: Path, style: str) -> Path:
    run = tmp / "replay"
    line = line_for(
        run,
        fx,
        "repair",
        None,
        mode="replay",
        replay_of_mode=style,
        forward_passes=1,
        extra=[("transcript.jsonl", "transcript", transcript_file)],
    )
    seen = run_main(run, line, mode="replay")
    assert seen["exit"] == 0, seen["stderr"][-3000:]
    return run


@pytest.mark.parametrize("style", ["session", "program"])
def test_a_replay_of_the_transcript_reproduces_the_patch_content(
    tmp_path: Path, fx: FixtureBundle, style: str
) -> None:
    steps = (
        _program_steps()
        if style == "program"
        else [call("python", code="W['weight'][0] += 1.0"), call("submit", answer="done")]
    )
    _ep, first, _ = session(tmp_path, fx, "repair", steps, mode=style, forward_passes=1)
    original = _patch(first)
    assert original is not None
    again = _replay(tmp_path, fx, first / "out" / "transcript.jsonl", style)
    replayed = _patch(again)
    assert replayed is not None and replayed["content_sha256"] == original["content_sha256"]
    assert result(again)["replay_divergence"] is None
    assert result(again)["stop_reason"] == "submitted"


def test_a_replay_of_an_edited_transcript_does_not_reproduce(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [call("python", code="W['weight'][0] += 1.0"), call("submit", answer="done")]
    _ep, first, _ = session(tmp_path, fx, "repair", steps)
    lines = (first / "out" / "transcript.jsonl").read_text().splitlines()
    edited = [x.replace("W['weight'][0] += 1.0", "W['weight'][0] += 2.0") for x in lines]
    assert edited != lines
    copy = tmp_path / "edited.jsonl"
    copy.write_text("\n".join(edited) + "\n")
    again = _replay(tmp_path, fx, copy, "session")
    original, replayed = _patch(first), _patch(again)
    assert original is not None and replayed is not None
    assert replayed["content_sha256"] != original["content_sha256"]


def test_a_replay_that_computes_a_different_tool_output_names_the_turn(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [call("python", code="print(1)"), call("submit", answer="done")]
    _ep, first, _ = session(tmp_path, fx, "repair", steps)
    lines = (first / "out" / "transcript.jsonl").read_text().splitlines()
    edited = [x.replace("print(1)", "print(2)") for x in lines]
    copy = tmp_path / "edited.jsonl"
    copy.write_text("\n".join(edited) + "\n")
    assert result(_replay(tmp_path, fx, copy, "session"))["replay_divergence"] == 1


# ----------------------------------------------------------------------------- 3.2 tools offered


@pytest.mark.parametrize(
    ("kind", "over", "edit", "python"),
    [
        ("inspect", {}, False, True),
        ("repair", {}, True, True),
        ("create", {}, True, True),
        ("repair", {"mode": "program"}, False, True),
        ("repair", {"python": False}, True, False),
    ],
    ids=["inspect", "repair", "create", "program", "repair-no-python"],
)
def test_the_tools_the_endpoint_received_are_the_operations_set(
    tmp_path: Path, fx: FixtureBundle, kind: str, over: dict[str, Any], edit: bool, python: bool
) -> None:
    ep, _run, _ = session(tmp_path, fx, kind, [call("submit", answer="a")], **over)
    expected = ["list_tensors", "describe_tensor", "read_tensor"]
    expected += ["python"] if python else []
    expected += ["write_tensor", "diff", "checkpoint", "restore", "revert"] if edit else []
    expected += ["submit"]
    expected += ["test_program", "submit_program"] if over.get("mode") == "program" else []
    assert ep.tool_names(0) == expected


# ----------------------------------------------------------------------------- 3.6, 4.x, 5.x


def test_a_two_action_reply_is_refused_and_nothing_in_it_runs(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-018.7's negative control, end to end: the write and the submit in one reply both stay
    unrun; the next reply's submit is the answer; the turn is counted refused."""
    both = {
        "calls": [("python", {"code": "W['weight'][0] += 1.0"}), ("submit", {"answer": "first"})]
    }
    _ep, run, _ = session(tmp_path, fx, "repair", [both, call("submit", answer="second")])
    res = result(run)
    assert res["answer"] == "second" and res["engine"]["refused_turns"] == 1
    assert _patch(run) is None
    refused = [m for m in transcript(run) if m.get("event") == "refused"]
    assert refused == [{"event": "refused", "kind": "multiple_actions", "n": 2}]


def test_two_nudges_then_the_text_is_the_answer(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [{"text": "thinking"}, {"text": "still"}, {"text": "my answer"}]
    _ep, run, _ = session(tmp_path, fx, "repair", steps)
    res = result(run)
    assert (res["stop_reason"], res["answer"]) == ("answered", "my answer")
    assert res["engine"]["refused_turns"] == 3
    nudges = [m for m in transcript(run) if m.get("content") == prompts.NUDGE]
    assert len(nudges) == 2


def test_structured_mode_end_to_end(tmp_path: Path, fx: FixtureBundle) -> None:
    """FR-018.7 in structured mode: no tools sent, the reply constrained by ``response_format``,
    parsed and validated again; a two-action reply is refused and runs nothing."""
    steps = [
        {"json": {"tool": "python", "arguments": {"code": "W['weight'][0] += 1.0"}}},
        {
            "json": [
                {"tool": "python", "arguments": {"code": "W['weight'][1] += 1.0"}},
                {"tool": "submit", "arguments": {"answer": "x"}},
            ]
        },
        {"json": {"tool": "submit", "arguments": {"answer": "done", "confidence": 0.4}}},
    ]
    ep, run, _ = session(tmp_path, fx, "repair", steps, editor={"action_protocol": "structured"})
    body = ep.requests[0]["body"]
    assert "tools" not in body and "stop" not in body
    assert body["response_format"]["type"] == "json_schema"
    assert "Reply with ONE JSON object" in body["messages"][0]["content"]
    res = result(run)
    peak = res["engine"].pop("peak_prompt_tokens")
    assert isinstance(peak, int) and peak > 0
    assert res["engine"] == {
        "action_protocol": "structured",
        "handoffs": 0,
        "refused_turns": 1,
        "kernel_restarts": 0,
        "reasoning_turns": 0,
        "repeated_turns": 0,
        # Paging (2026-11-21): ``line_for`` freezes it off for the tests written before it.
        "pages_out": 0,
        "recalls": 0,
        "ledger_rows": 0,
    }
    patch = _patch(run)
    assert patch is not None and patch["changed_entries_total"] == 1  # only the first write ran


def test_structured_mode_runs_on_a_generic_endpoint(tmp_path: Path, fx: FixtureBundle) -> None:
    """Feature 019 P2 (FR-019.11): the engine runs structured mode on a GENERIC OpenAI-compatible
    endpoint (the backend pushes one only with a current protocol-test pass); its body carries
    ``response_format`` and none of miServer's other fields."""
    steps = [{"json": {"tool": "submit", "arguments": {"answer": "done", "confidence": 0.4}}}]
    ep, run, _ = session(
        tmp_path,
        fx,
        "repair",
        steps,
        serving="generic",
        editor={"action_protocol": "structured"},
    )
    body = ep.requests[0]["body"]
    assert body["response_format"]["type"] == "json_schema"
    assert not {"tools", "parallel_tool_calls", "chat_template_kwargs"} & set(body)
    res = result(run)
    assert res["stop_reason"] == "submitted" and res["engine"]["action_protocol"] == "structured"


def test_structured_mode_on_an_anthropic_endpoint_is_refused_by_name(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    ep, run, _ = session(
        tmp_path,
        fx,
        "repair",
        [call("submit", answer="a")],
        endpoint_kind="anthropic",
        editor={"action_protocol": "structured"},
    )
    res = result(run)
    assert res["stop_reason"] == "error"
    assert res["failure"]["code"] == "wsp_endpoint_protocol_unsupported"
    assert ep.requests == []


@pytest.mark.parametrize(
    ("over", "first_usage", "sleep", "reason"),
    [
        ({"max_steps": 1}, (10, 1), 0, "max_steps"),
        ({"tokens": 1000}, (950, 0), 0, "token_budget"),
        (
            {
                "prices": {
                    "input": 1000.0,
                    "output": 1000.0,
                    "cache_write": None,
                    "cache_read": None,
                },
                "cost_usd": 1.0,
            },
            (1000, 0),
            0,
            "cost_budget",
        ),
        ({"time_s": 2}, (10, 1), 2.0, "time_budget"),
    ],
    ids=["max_steps", "token_budget", "cost_budget", "time_budget"],
)
def test_each_budget_stops_with_its_reason_and_keeps_the_partial_result(
    tmp_path: Path,
    fx: FixtureBundle,
    over: dict[str, Any],
    first_usage: tuple[int, int],
    sleep: float,
    reason: str,
) -> None:
    """5.3: one accepted edit, then the budget; the last turn offers submit alone (the endpoint
    RECEIVED only ``submit``); the result and the partial patch are saved."""
    code = f"import time; time.sleep({sleep}); W['weight'][0] += 1.0"
    steps = [
        {"calls": [("python", {"code": code})], "usage": first_usage},
        {"text": "out of budget", "usage": (1, 1)},
    ]
    ep, run, _ = session(tmp_path, fx, "repair", steps, **over)
    res = result(run)
    assert res["stop_reason"] == reason and res["answer"] == "out of budget"
    assert ep.tool_names(1) == ["submit"]
    last_user = ep.requests[1]["body"]["messages"][-1]["content"]
    # FU-3: an edit session is told that the weights as they stand, measured or not, are kept.
    assert last_user.endswith(prompts.SUBMIT_NOW_EDIT.format(why="")[1:])
    assert "including any edit you have not measured" in last_user
    patch = _patch(run)
    assert patch is not None and patch["changed_entries_total"] == 1


@pytest.mark.parametrize(
    ("kind", "over", "edits"),
    [
        ("repair", {}, True),
        ("create", {}, True),
        ("inspect", {}, False),
        ("repair", {"mode": "program"}, False),
    ],
    ids=["repair", "create", "inspect", "program"],
)
def test_the_limits_instruction_says_the_current_weights_are_kept_only_where_weights_are_edited(
    tmp_path: Path, fx: FixtureBundle, kind: str, over: dict[str, Any], edits: bool
) -> None:
    """FU-3 (switch record, ``wsp_7bd0a26682dd``): at a step limit an edit session's last turn
    says the weights as they are now, unmeasured edits included, are what is kept; a read and a
    program session (whose submit keeps a program, not the weights) get the plain text."""
    steps = [call("python", code="print(1)"), {"text": "done", "usage": (1, 1)}]
    ep, _run, _ = session(tmp_path, fx, kind, steps, max_steps=1, **over)
    assert ep.tool_names(1) == ["submit"]
    last_user = ep.requests[1]["body"]["messages"][-1]["content"]
    want = prompts.SUBMIT_NOW_EDIT if edits else prompts.SUBMIT_NOW
    assert last_user == want.format(why=prompts.STEP_LIMIT)


@pytest.mark.parametrize(
    ("serving", "status", "code"),
    [("miserver", 409, "editor_context_exceeded"), ("generic", 400, "invalid_request_error")],
)
def test_an_endpoint_error_ends_the_run_with_its_code_and_no_retry(
    tmp_path: Path, fx: FixtureBundle, serving: str, status: int, code: str
) -> None:
    """4.2 (a gateway 409) and 4.6 (a generic endpoint's non-2xx): ``error``, the code in
    ``failure``, one request each, and the accepted edit kept."""
    steps = [call("python", code="W['weight'][0] += 1.0"), {"status": status, "code": code}]
    ep, run, _ = session(tmp_path, fx, "repair", steps, serving=serving)
    res = result(run)
    assert res["stop_reason"] == "error" and res["failure"]["code"] == code
    assert len(ep.requests) == 2
    assert _patch(run) is not None


def test_the_log_has_one_line_per_turn_and_never_the_token(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """5.7: one log line per turn, from named fields only; the token (every form) appears in
    no log and no output file."""
    steps = [call("python", code="print('hello')"), {"text": "hm"}, call("submit", answer="a")]
    _ep, run, seen = session(tmp_path, fx, "repair", steps)
    logs = [json.loads(x) for x in seen["stdout"].strip().splitlines()[:-1]]
    assert [(x["turn"], x["action"], x["refused"]) for x in logs] == [
        (1, "python", None),
        (2, None, "no_action"),
        (3, "submit", None),
    ]
    assert all(set(x) == {"turn", "action", "refused", "tokens", "forward_left"} for x in logs)
    import base64

    forms = [TOKEN, base64.b64encode(TOKEN.encode()).decode(), json.dumps(TOKEN)[1:-1]]
    texts = [seen["stdout"], seen["stderr"]]
    texts += [p.read_text(errors="replace") for p in (run / "out").rglob("*") if p.is_file()]
    for form in forms:
        assert not any(form in t for t in texts), form


def test_an_anthropic_repair_runs_end_to_end(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [call("python", code="W['weight'][0] += 1.0"), call("submit", answer="done")]
    ep, run, _ = session(tmp_path, fx, "repair", steps, endpoint_kind="anthropic")
    assert result(run)["stop_reason"] == "submitted" and _patch(run) is not None
    second = ep.requests[1]["body"]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"


def test_a_kernel_timeout_restarts_it_and_the_editor_is_told(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """6.4: the timeout kills the kernel; a new one holds the accepted weights; variables are
    gone; the run goes on and counts the restart."""
    steps = [
        call("python", code="W['weight'][0] += 1.0\nkept = 1"),
        call("python", code="while True:\n    pass"),
        call("python", code="kept"),
        call("python", code="float(W['weight'][0])"),
        call("submit", answer="done"),
    ]
    _ep, run, _ = session(tmp_path, fx, "repair", steps, options={"kernel_timeout_s": 2.0})
    _first, hung, gone, weight, _ = tool_outputs(run)
    assert hung == prompts.KERNEL_RESTARTED
    assert "NameError" in gone
    assert weight.strip() == repr(float(_w(fx)[0] + np.float32(1.0)))
    assert result(run)["engine"]["kernel_restarts"] == 1


def test_the_kernel_is_dead_when_the_session_returns(tmp_path: Path, fx: FixtureBundle) -> None:
    """The session's ``finally`` kills and reaps its grandchild: checked in the child itself,
    before it exits (so neither the supervisor's teardown nor the kernel's EOF exit can stand in
    for it)."""
    record = "import os; open('../kernel.pid', 'w').write(str(os.getpid()))"
    steps = [call("python", code=record), call("submit", answer="done")]
    _ep, _run, seen = session(tmp_path, fx, "repair", steps)
    assert seen["kernel_alive"] is False


# ------------------- 2026-11-21: a run recorded with the version-1 practice tools still replays


V1_TOOLS = [
    {"name": "practice_metrics", "version": "1", "cost": 1},
    {"name": "compare_to_parent", "version": "1", "cost": 2},
]


def test_a_run_recorded_with_the_v1_practice_tools_replays_bit_for_bit(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Every run recorded before 2026-11-21 named practice_metrics and compare_to_parent at
    version 1. The worker keeps version 1 beside version 2, so the replay recomputes the same
    tool outputs (no ``at_parent_bar``), the same patch, and no divergence."""
    steps = [
        call("forward_practice_metrics", split="train"),
        # A weight edit (2026-11-22): a bias-only edit now draws the judging warning at submit.
        call("python", code="W['weight'][0] += 0.5"),
        call("forward_compare_to_parent", split="train"),
        call("submit", answer="done"),
    ]
    _ep, first, _ = session(
        tmp_path, fx, "repair", steps, forward_passes=3, observation_tools=V1_TOOLS
    )
    outputs = tool_outputs(first)
    recorded = json.loads(outputs[0].split("\n[forward budget")[0])
    assert recorded["tool"] == "practice_metrics" and "at_parent_bar" not in recorded
    compared = json.loads(outputs[2].split("\n[forward budget")[0])
    assert "at_parent_bar" not in compared["now"] and "at_parent_bar" not in compared["parent"]
    run = tmp_path / "replay"
    line = line_for(
        run,
        fx,
        "repair",
        None,
        mode="replay",
        replay_of_mode="session",
        forward_passes=3,
        observation_tools=V1_TOOLS,
        extra=[("transcript.jsonl", "transcript", first / "out" / "transcript.jsonl")],
    )
    seen = run_main(run, line, mode="replay")
    assert seen["exit"] == 0, seen["stderr"][-3000:]
    assert result(run)["replay_divergence"] is None
    original, replayed = _patch(first), _patch(run)
    assert original is not None and replayed is not None
    assert replayed["content_sha256"] == original["content_sha256"]
    assert result(run)["forward_used"] == result(first)["forward_used"] == 3


# ------------------------------------------- feature 019 P2 (12.5): a replay sizes as its original


def _replay_session(tmp: Path, fx: FixtureBundle, transcript_file: Path) -> Any:
    from wsp_worker.miwsp_engine import session as session_mod

    line = line_for(
        tmp / "replay-session",
        fx,
        "repair",
        None,
        mode="replay",
        replay_of_mode="session",
        forward_passes=1,
        extra=[("transcript.jsonl", "transcript", transcript_file)],
    )
    return session_mod.Session(line)


def test_a_replay_sizes_to_the_window_its_original_recorded(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Operator decision 2026-11-24 (019 P2, 12.5): a replay has no endpoint, so it takes the
    window and the ceiling from its original's ``{"event": "window"}`` line, sizes its cap from
    them, and still reproduces; a transcript from before 019 (no event) replays at 8,192."""
    from wsp_worker.miwsp_engine import context

    steps = [call("python", code="W['weight'][0] += 1.0"), call("submit", answer="done")]
    editor = {"window_tokens": 32_768, "output_ceiling_tokens": 4_096, "window_source": "reported"}
    _ep, first, _ = session(tmp_path, fx, "repair", steps, forward_passes=1, editor=editor)
    transcript = first / "out" / "transcript.jsonl"
    event = json.loads(transcript.read_text().splitlines()[1])
    assert (event["event"], event["tokens"], event["ceiling"]) == ("window", 32_768, 4_096)
    replay = _replay_session(tmp_path, fx, transcript)
    assert (replay.window, replay.ceiling) == (32_768, 4_096)
    assert replay.guard.window == 32_768
    assert replay.cap == context.editor_output_cap(editor) != context.editor_output_cap({})
    again = _replay(tmp_path, fx, transcript, "session")
    assert result(again)["replay_divergence"] is None
    assert _patch(again)["content_sha256"] == _patch(first)["content_sha256"]  # type: ignore[index]
    # A pre-019 transcript: the same lines without the event.
    lines = transcript.read_text().splitlines()
    old = tmp_path / "pre019.jsonl"
    old.write_text("\n".join(x for x in lines if '"event": "window"' not in x) + "\n")
    before = _replay_session(tmp_path / "old", fx, old)
    assert (before.window, before.ceiling) == (8_192, 32_000)


# -------------------------- program refusals carry no program text (fix 2026-11-28, E7 §5.3 gap 2)

LEAKY_PROGRAM = (
    "def edit(base, support):\n"
    "    t = support.tokens(int(support.rows[0]))\n"
    "    raise ValueError('leak 0.123456789 ' + str(float(t[0, 0])))\n"
)


def test_a_program_sessions_test_program_hides_the_programs_message_and_replays(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Through the real child: ``test_program`` answers the editor the data-free refusal, the
    header records ``program_errors: 1``, the child's stderr keeps the message, and the run
    replays without divergence. A transcript from before (no ``program_errors``, the old text)
    replays with the old text, also without divergence."""
    from wsp_worker import program as program_mod

    steps = [
        call("test_program", code=LEAKY_PROGRAM),
        call("submit_program", code=LEAKY_PROGRAM),
        call("submit_program", code=PROGRAM),
        call("submit", answer="programmed"),
    ]
    _ep, first, seen = session(tmp_path, fx, "repair", steps, mode="program", forward_passes=1)
    tested, submitted, *_ = tool_outputs(first)
    raised = program_mod.PROGRAM_RAISED.format(kind="ValueError")
    assert json.loads(tested.split("\n[forward budget")[0]) == {
        "refused": "wsp_program_invalid",
        "message": raised,
    }
    assert submitted == f"refused: wsp_program_invalid: {raised}"
    from wsp_worker import bundle as bundle_mod

    files = [{k: e[k] for k in ("name", "role", "sha256")} for e in fx.manifest["files"]]
    support = program_mod.SupportSet("practice_train", bundle_mod.load(fx.root, fx.manifest, files))
    value = str(float(support.tokens(int(support.rows[0]))[0, 0]))
    # Neither the canary nor the activation reaches any tool output; the child's stderr has both.
    for output in tool_outputs(first):
        assert "0.123456789" not in output and value not in output
    assert "0.123456789" in seen["stderr"] and value in seen["stderr"]
    lines = (first / "out" / "transcript.jsonl").read_text().splitlines()
    header = json.loads(lines[0])
    assert header["program_errors"] == 1
    again = _replay(tmp_path, fx, first / "out" / "transcript.jsonl", "program")
    assert result(again)["replay_divergence"] is None

    # The same run as recorded before the fix: no header flag, and the old message-carrying
    # texts. It replays without divergence, so a replay reads the flag from the header.
    old_text = f"the program raised ValueError: leak 0.123456789 {value}"[:500]
    del header["program_errors"]

    def rewrite(texts: bool) -> Path:
        out = [json.dumps(header)]
        for raw in lines[1:]:
            entry = json.loads(raw)
            if texts and isinstance(entry.get("content"), str):
                entry["content"] = (
                    entry["content"]
                    .replace(json.dumps(raw_message := raised)[1:-1], json.dumps(old_text)[1:-1])
                    .replace(raw_message, old_text)
                )
            out.append(json.dumps(entry))
        path = tmp_path / ("before.jsonl" if texts else "flag-only.jsonl")
        path.write_text("\n".join(out) + "\n")
        return path

    before = _replay(tmp_path / "before", fx, rewrite(texts=True), "program")
    assert result(before)["replay_divergence"] is None
    # The control: the flag gone but the new texts kept diverges at the first test_program turn.
    flag_only = _replay(tmp_path / "flag-only", fx, rewrite(texts=False), "program")
    assert result(flag_only)["replay_divergence"] is not None
