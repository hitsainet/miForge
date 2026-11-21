"""The weightspace adapter with scripted LLMs (feature 016; FTASKS 8.8; FR-016.17's AC; R1B-A3,
R1B-A4, R1B-A6, R1B-A14, R3A-14).

Each case runs ``wsp_worker.adapters.weightspace_adapter.run`` for real, under the worker's
interpreter (``MIFORGE_WSP_WORKER_PYTHON``; skips by name without it, fails under
``MIFORGE_REQUIRE_WSP_WORKER=1``), on a fixture bundle and a child line shaped as the supervisor
writes it. What the session leaves in ``/out`` is then checked HERE with the backend's own code:
the patch re-applies through ``wsp.weight_ops.patch.apply_patch`` and equals the backend's
``build_patch`` of the same edit (v1 patch equality).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.support.wsp_worker_env import SRC, worker_python
from wsp.weight_ops.hashing import file_sha256
from wsp.weight_ops.patch import apply_patch, build_patch
from wsp.weight_ops.tensor import from_array

CASES = Path(__file__).resolve().parents[2] / "support" / "wsp_children" / "adapter_cases.py"


def _line(tmp: Path, fx: FixtureBundle, kind: str, **over: Any) -> Path:
    work, out, scratch = tmp / "work", tmp / "out", tmp / "tmp"
    for d in (work, out, scratch):
        d.mkdir()
    files = []
    for entry in fx.manifest["files"]:
        shutil.copyfile(fx.root / entry["name"], work / entry["name"])
        files.append(
            {
                "name": entry["name"],
                "role": entry["role"],
                "sha256": file_sha256(work / entry["name"]),
                "bytes": entry["bytes"],
            }
        )
    request = {
        "kind": kind,
        "task": "Repair the probe.",
        "parent": {"kind": "bundle", "ref": fx.manifest["bundle_sha256"]},
        "budgets": {
            "tokens": over.get("tokens", 100_000),
            "time_s": 600,
            "max_steps": 10,
            "forward_passes": over.get("forward_passes", 0),
            # A cap with an unpriced endpoint: the library's cost_budget must then stay unset
            # (it needs pricing=; R1B-A6, control C17d).
            "cost_usd": 1.0,
        },
        "change_limits": {
            "max_changed_entries": over.get("max_changed_entries"),
            "allowed_tensors": None,
            "frozen_tensors": [],
        },
        "thinking": None,
        "observation_tools": over.get("observation_tools", []),
        "goal": over.get("goal"),
        "keep_weights": over.get("keep_weights", False),
    }
    if "output_cap" in over:
        request["output_cap"] = over["output_cap"]
    for name, role, source in over.get("extra", []):
        shutil.copyfile(source, work / name)
        files.append(
            {
                "name": name,
                "role": role,
                "sha256": file_sha256(work / name),
                "bytes": (work / name).stat().st_size,
            }
        )
    line = {
        "mode": over.get("mode", "session"),
        "run_id": "wsp_000000000001",
        "spec": {
            "time_budget_s": 600,
            "grace_s": 60,
            "pull_timeout_s": 60,
            "request": request,
            "manifest": fx.manifest,
            **({"replay_of_mode": over["replay_of_mode"]} if "replay_of_mode" in over else {}),
            "editor": {
                "kind": "openai_compatible",
                "model": "m",
                "serving": "miserver",
                "prices": {"input": None, "output": None, "cache_write": None, "cache_read": None},
                **over.get("editor", {}),
            },
        },
        "files": files,
        "gateway": None,
        "paths": {"work": str(work), "out": str(out), "tmp": str(scratch)},
        "time_left_s": 600.0,
    }
    path = tmp / "line.json"
    path.write_text(json.dumps(line))
    return path


def _run(case: str, line: Path | None = None) -> dict[str, Any]:
    python = worker_python()
    args = [python, str(CASES), case] + ([str(line)] if line else [])
    done = subprocess.run(
        args,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    loaded: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return loaded


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


def _result(tmp: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((tmp / "out" / "result.json").read_text())
    return loaded


def test_a_repair_writes_a_v1_patch_equal_to_the_backends(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    seen = _run("repair", _line(tmp_path, fx, "repair"))
    assert seen["exit"] == 0
    out = tmp_path / "out"
    # R3A-14: exactly what the intake knows; res.save's ws/ files are never written.
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == [
        "patch",
        "patch/edit.npz",
        "patch/patch.json",
        "result.json",
        "transcript.jsonl",
    ]
    result = _result(tmp_path)
    assert (result["answer"], result["confidence"], result["stop_reason"]) == (
        "done",
        0.7,
        "submitted",
    )
    manifest, rebuilt = apply_patch(out / "patch", fx.tensors)
    base = fx.tensors["weight"].values64().astype(np.float32)
    expected = base.copy()
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
    assert manifest["changed_entries_total"] == 1


def test_a_limit_refusal_then_success_keeps_confidence_null(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """R1B-A14: the refused submit is not an answer; the loop goes on; confidence stays null."""
    seen = _run("limits", _line(tmp_path, fx, "repair", max_changed_entries=1))
    refusals = [o for o in seen["tool_outputs"] if isinstance(o, str) and o.startswith("refused:")]
    assert refusals and "3 entries" in refusals[0]
    result = _result(tmp_path)
    assert (result["answer"], result["confidence"]) == ("second", None)
    patch = json.loads((tmp_path / "out" / "patch" / "patch.json").read_text())
    assert patch["changed_entries_total"] == 1


def test_create_zeroes_then_writes_and_the_patch_is_against_the_bundle(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """R1B-A3: create runs through the adapter's session, from zeroed editable tensors."""
    _run("create", _line(tmp_path, fx, "create"))
    _manifest, rebuilt = apply_patch(tmp_path / "out" / "patch", fx.tensors)
    assert np.all(rebuilt["weight"].values64() == 1.0)
    assert np.all(rebuilt["bias"].values64() == 0.0)  # zeroed, not written
    assert rebuilt["mean"].sha256 == fx.tensors["mean"].sha256  # not editable: never given


def test_create_with_keep_weights_starts_from_the_bundles_weights(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-016.31 (16.2): ``keep_weights`` keeps the editable tensors; only the write changes."""
    _run("create", _line(tmp_path, fx, "create", keep_weights=True))
    _manifest, rebuilt = apply_patch(tmp_path / "out" / "patch", fx.tensors)
    assert np.all(rebuilt["weight"].values64() == 1.0)
    assert rebuilt["bias"].sha256 == fx.tensors["bias"].sha256  # kept, not zeroed


@pytest.mark.parametrize(("goal", "mode"), [(None, "fix"), ("raise recall at 1% FPR", "improve")])
def test_repair_fixes_or_improves_with_a_goal(
    tmp_path: Path, fx: FixtureBundle, goal: str | None, mode: str
) -> None:
    """FR-016.31 (16.2): repair maps to fix, or to improve when the request carries a goal."""
    seen = _run("repair", _line(tmp_path, fx, "repair", goal=goal))
    assert seen["exit"] == 0
    assert _result(tmp_path)["mode"] == mode
    if goal is not None:
        assert goal in seen["first_user"]


def test_a_write_through_r_is_caught_by_the_live_diff(tmp_path: Path, fx: FixtureBundle) -> None:
    """R1B-A4: an inspect that changed weights leaves a patch, which the intake refuses."""
    _run("read_write", _line(tmp_path, fx, "inspect"))
    patch = json.loads((tmp_path / "out" / "patch" / "patch.json").read_text())
    assert [t["name"] for t in patch["tensors"]] == ["weight"]


def test_a_run_past_the_token_budget_ends_token_budget(tmp_path: Path, fx: FixtureBundle) -> None:
    """R1B-A6: the library's budget is 90% of the run's (900 of 1000 tokens)."""
    _run("token_budget", _line(tmp_path, fx, "repair", tokens=1000))
    assert _result(tmp_path)["stop_reason"] == "token_budget"
    assert not (tmp_path / "out" / "patch").exists()


def test_a_late_start_stops_the_library_before_the_supervisors_kill(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Fix 2026-11-21: ``time_left_s`` (5 s of a 600 s budget) bounds the library's own time
    budget, so it goes straight to its final submit-only turn instead of running into the kill."""
    line = _line(tmp_path, fx, "repair")
    doc = json.loads(line.read_text())
    doc["time_left_s"] = 5.0
    line.write_text(json.dumps(doc))
    _run("late_start", line)
    res = _result(tmp_path)
    assert res["stop_reason"] == "time_budget" and res["steps"] == 0


def test_the_client_is_the_gateway_with_the_session_token_and_no_retries() -> None:
    seen = _run("client")
    assert seen["anthropic"]["base_url"].rstrip("/") == "http://10.152.183.20:8080"
    assert seen["openai_compatible"]["base_url"].rstrip("/") == "http://10.152.183.20:8080/v1"
    for kind in ("anthropic", "openai_compatible"):
        assert seen[kind]["api_key"] == "tok-SENTINEL" and seen[kind]["max_retries"] == 0
        assert seen[kind]["model"] == "ed"
    assert seen["anthropic"]["stream"] is False


# ------------------------------------------------- the output cap (fix 2026-11-20, defect 1a)


def test_every_request_carries_the_engines_cap_for_the_editors_window() -> None:
    """Production 2026-11-20: ``OpenAIChat`` sent no cap, the gateway filled in 32,000, and a
    32,768-token miServer window ended the session ``editor_context_exceeded`` at the first call.
    Both kinds now send ``max_tokens`` = the engine's own rule (``output_cap``: 2,048, at most an
    eighth of the window) over the engine's own window (endpoint, else served, else 8,192).
    What is asserted is what reached a recording server, one request per call."""
    from wsp_worker.miwsp_engine import context

    seen = _run("cap")
    expected = {"no_window": 1_024, "served_32768": 2_048, "endpoint_4096": 512}
    routes = (("anthropic", "/v1/messages"), ("openai_compatible", "/v1/chat/completions"))
    for kind, path in routes:
        for label, cap in expected.items():
            assert seen[f"{kind}/{label}"] == [{"path": path, "max_tokens": cap}], (kind, label)
    # The same numbers the engine computes (one rule, not a copy of it).
    assert context.editor_output_cap({}) == 1_024
    assert context.editor_output_cap({"served_context_tokens": 32_768}) == 2_048
    assert (
        context.editor_output_cap({"context_window_tokens": 4_096, "served_context_tokens": 32_768})
        == 512
    )


def test_the_production_run_path_sends_the_cap_on_every_request(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """``run`` with no LLM injected builds it from the pushed ``spec.editor`` and gateway: each
    request it sends to the (recording) gateway carries the served window's cap."""
    line = _line(tmp_path, fx, "repair", editor={"served_context_tokens": 32_768})
    seen = _run("run_cap", line)
    assert seen["exit"] == 0
    assert seen["sent"], "the session sent no request"
    assert {s["path"] for s in seen["sent"]} == {"/v1/chat/completions"}
    assert [s["max_tokens"] for s in seen["sent"]] == [2_048] * len(seen["sent"])


def test_the_production_run_path_names_no_cap_when_the_runs_output_cap_is_off(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """2026-11-21: the frozen ``output_cap: off`` reaches ``build_llm`` from ``run`` itself, so no
    request the adapter sends names a cap (the gateway's window-aware default applies)."""
    line = _line(tmp_path, fx, "repair", editor={"served_context_tokens": 32_768}, output_cap="off")
    seen = _run("run_cap", line)
    assert seen["exit"] == 0 and seen["sent"], seen
    assert [s["max_tokens"] for s in seen["sent"]] == [None] * len(seen["sent"])


# ----------------------------------------------------------------------- observation tools (8.11)


def test_observation_tools_are_callbacks_that_draw_their_cost_from_forward_passes(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-016.18: each named tool is a ``forward_<name>`` callback at its declared cost; a call
    past ``forward_passes`` is refused by the library's own budget, and nothing runs."""
    tools = [
        {"name": "compare_to_parent", "version": "2", "cost": 2},
        {"name": "practice_metrics", "version": "2", "cost": 1},
    ]
    seen = _run("observe", _line(tmp_path, fx, "repair", forward_passes=2, observation_tools=tools))
    first, second = seen["tool_outputs"][0], seen["tool_outputs"][1]
    answer = json.loads(first.split("\n[forward budget")[0])
    assert answer["tool"] == "practice_metrics"
    # 2026-11-21: the adapter's view carries the bundle's shipped bar, held fixed.
    assert answer["at_parent_bar"]["source"] == "shipped"
    assert answer["at_parent_bar"]["bar"] == fx.manifest["operating_point"]["threshold"]
    assert "[forward budget: 1 of 2 left]" in first
    assert second.startswith("forward budget exhausted (1 left, this probe costs 2)")


def test_an_unknown_observation_tool_fails_the_session_before_the_editor_runs(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    line = _line(
        tmp_path,
        fx,
        "repair",
        observation_tools=[{"name": "read_text", "version": "1", "cost": 1}],
    )
    python = worker_python()
    done = subprocess.run(
        [python, str(CASES), "observe", str(line)],
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode != 0 and "not in this worker's catalogue" in done.stderr
    assert not (tmp_path / "out" / "transcript.jsonl").exists()


# ----------------------------------------------------------------------- program mode (9.4)


def test_a_program_session_reads_stores_one_program_and_its_patch(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """FR-016.23: a program session READS; an invalid program is refused and does not count; the
    first valid one is stored; a second ``submit_program`` is refused; the candidate is the
    stored program's edit."""
    seen = _run("program", _line(tmp_path, fx, "repair", mode="program", forward_passes=1))
    wrote, *outputs = seen["tool_outputs"]
    assert "NameError" in wrote  # the session reads: no writable W (P19)
    tested = json.loads(outputs[0].split("\n[forward budget")[0])
    assert tested["split"] == "validation" and tested["rows"] > 0
    assert outputs[1].startswith("refused: wsp_program_invalid")
    assert outputs[2].startswith("program stored")
    assert outputs[3].startswith("refused: wsp_program_already_submitted")
    out = tmp_path / "out"
    assert (out / "program.py").read_text() == seen["program"]
    _manifest, rebuilt = apply_patch(out / "patch", fx.tensors)
    w = fx.tensors["weight"].values64().astype(np.float32)
    assert np.array_equal(
        rebuilt["weight"].values64().astype(np.float32), w + np.float32(0.5) * np.sign(w)
    )
    # The session itself could not write: only the program's edit is in the patch.
    assert rebuilt["bias"].sha256 == fx.tensors["bias"].sha256


# ----------------------------------------------------------------------- replay (9.1)


def _content(out: Path) -> str | None:
    path = out / "patch" / "patch.json"
    return json.loads(path.read_text())["content_sha256"] if path.exists() else None


@pytest.mark.parametrize("style", ["session", "program"])
def test_a_replay_of_the_transcript_reproduces_the_patch_content(
    tmp_path: Path, fx: FixtureBundle, style: str
) -> None:
    """FR-016.21: ``ScriptedLLM.from_transcript`` over the same bundle gives the same
    ``content_sha256``; its ``patch_sha256`` differs (``created_at``), by construction."""
    case = "repair" if style == "session" else "program"
    first = tmp_path / "first"
    first.mkdir()
    _run(case, _line(first, fx, "repair", mode=style, forward_passes=1))
    original = _content(first / "out")
    assert original is not None
    again = tmp_path / "again"
    again.mkdir()
    extra = [("transcript.jsonl", "transcript", first / "out" / "transcript.jsonl")]
    seen = _run(
        "replay",
        _line(
            again,
            fx,
            "repair",
            mode="replay",
            replay_of_mode=style,
            forward_passes=1,
            extra=extra,
        ),
    )
    assert seen["exit"] == 0
    assert _content(again / "out") == original


def test_a_replay_of_an_edited_transcript_does_not_reproduce(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    _run("repair", _line(first, fx, "repair"))
    lines = (first / "out" / "transcript.jsonl").read_text().splitlines()
    edited = [line.replace("W['weight'][0] += 1.0", "W['weight'][0] += 2.0") for line in lines]
    assert edited != lines
    copy = tmp_path / "edited.jsonl"
    copy.write_text("\n".join(edited) + "\n")
    again = tmp_path / "again"
    again.mkdir()
    _run(
        "replay",
        _line(again, fx, "repair", mode="replay", extra=[("transcript.jsonl", "transcript", copy)]),
    )
    assert _content(again / "out") not in (None, _content(first / "out"))


@pytest.mark.parametrize(
    ("case", "kind", "mode"),
    [("repair", "inspect", "read"), ("create", "create", "write"), ("repair", "repair", "fix")],
)
def test_each_operation_runs_in_its_own_mode(
    tmp_path: Path, fx: FixtureBundle, case: str, kind: str, mode: str
) -> None:
    """FR-016.31/.32 (16.2; review 016-P2 controls A1, A3): inspect reads, create writes, repair
    fixes, as the session the adapter ran reports it. The inspect runs the repair script, whose
    ``W['weight'][0] += 1.0`` has no ``W`` to write through in read mode, so it leaves no patch;
    in write mode it would (the intake's refusal, C15, is the second layer, not the first)."""
    seen = _run(case, _line(tmp_path, fx, kind))
    assert seen["exit"] == 0
    assert _result(tmp_path)["mode"] == mode
    if kind == "inspect":
        assert not (tmp_path / "out" / "patch").exists()
