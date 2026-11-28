"""The worker's provider dispatch (feature 018; FTASKS 7.1, 7.2; FR-018.2).

Two layers:

- **in this process**, ``wsp_worker.__main__.child_main`` itself with the preamble, the stdin read
  and ``os._exit`` replaced, and the engine's ``run`` replaced by a recorder: the payload (the
  very line pushed) and the call count;
- **the real entry point**, ``python -I -c <boot> run`` as the supervisor starts it, against a
  scripted endpoint: the engine's session runs once, with the pushed run id.

Since 018-P4 (FR-018.18) ``weightspace`` is one more id the worker refuses
``worker_provider_unknown``: its adapter and branch are gone.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import wsp_worker.__main__ as main_mod
from tests.support.wsp_engine import line_for, result, run_main, transcript
from tests.support.wsp_fixtures import make_bundle
from tests.support.wsp_scripted_endpoint import ScriptedEndpoint
from wsp_worker import miwsp_engine


class ExitedError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code


class RefusedError(Exception):
    def __init__(self, code: str, exit_code: int) -> None:
        self.code, self.exit_code = code, exit_code


@pytest.fixture()
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"miwsp_engine": []}

    def engine_run(line: Any, **_: Any) -> int:
        calls["miwsp_engine"].append(line)
        return 0

    monkeypatch.setattr(miwsp_engine, "run", engine_run)
    monkeypatch.setattr(main_mod, "child_preamble", lambda: None)

    def refuse(code: str, message: str, exit_code: int) -> None:
        raise RefusedError(code, exit_code)

    def exit_(code: int) -> None:
        raise ExitedError(code)

    monkeypatch.setattr(main_mod, "_refuse", refuse)
    monkeypatch.setattr(main_mod.os, "_exit", exit_)
    return calls


def _line(provider: str | None) -> dict[str, Any]:
    request: dict[str, Any] = {"kind": "repair"}
    if provider is not None:
        request["provider"] = {"id": provider, "version": "x"}
    return {
        "mode": "session",
        "run_id": "wsp_1",
        "spec": {"request": request, "manifest": {}, "editor": {}},
        "files": [],
        "gateway": {"ip": "127.0.0.1", "port": 1, "token": "t"},
        "paths": {"work": "/w", "out": "/o", "tmp": "/t"},
        "time_left_s": 600.0,
    }


def test_a_run_goes_to_the_engine_once_with_the_pushed_line(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, list[Any]]
) -> None:
    line = _line("miwsp_engine")
    monkeypatch.setattr(main_mod, "read_child_line", lambda: line)
    with pytest.raises(ExitedError) as exited:
        main_mod.child_main("run")
    assert exited.value.code == 0
    assert len(recorded["miwsp_engine"]) == 1 and recorded["miwsp_engine"][0] is line


@pytest.mark.parametrize("provider", ["weightspace", "wsplab", None])
def test_any_other_provider_is_refused_and_nothing_runs(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, list[Any]], provider: str | None
) -> None:
    monkeypatch.setattr(main_mod, "read_child_line", lambda: _line(provider))
    with pytest.raises(RefusedError) as refused:
        main_mod.child_main("run")
    assert (refused.value.code, refused.value.exit_code) == ("worker_provider_unknown", 78)
    assert recorded == {"miwsp_engine": []}


def test_the_retired_adapter_is_gone_from_the_worker() -> None:
    """FR-018.18: no adapter package is left to import (the registry test and the dependency
    scan pin the other two removal points)."""
    import importlib.util

    assert importlib.util.find_spec("wsp_worker.adapters") is None


def test_a_replay_is_routed_the_same_way(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, list[Any]]
) -> None:
    line = {**_line("miwsp_engine"), "mode": "replay", "gateway": None}
    monkeypatch.setattr(main_mod, "read_child_line", lambda: line)
    with pytest.raises(ExitedError):
        main_mod.child_main("replay")
    assert recorded["miwsp_engine"] == [line]


# ----------------------------------------------------------------------------- end to end


def test_the_real_entry_point_runs_an_engine_session_once(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path / "bundle", rule="mean")
    steps = [
        {"calls": [("python", {"code": "W['weight'][0] += 1.0"})]},
        {"calls": [("submit", {"answer": "done", "confidence": 0.5})]},
    ]
    with ScriptedEndpoint("openai_compatible", steps) as ep:
        line = line_for(tmp_path / "s", fx, "repair", ep.gateway)
        seen = run_main(tmp_path / "s", line)
    assert seen["exit"] == 0, seen["stderr"][-3000:]
    out = result(tmp_path / "s")
    assert (out["provider"], out["run_id"], out["stop_reason"]) == (
        "miwsp_engine",
        "wsp_000000000001",
        "submitted",
    )
    assert transcript(tmp_path / "s")[0]["provider"] == "miwsp_engine"
    assert len(ep.requests) == 2  # one session: two turns, nothing more
    assert (tmp_path / "s" / "out" / "patch" / "patch.json").exists()


# --------------------------------------------- R1B-A9: a rerun loads no model client (018-P4)


def _rerun_line() -> dict[str, Any]:
    return {**_line("miwsp_engine"), "mode": "program_rerun", "gateway": None}


def test_a_rerun_that_loaded_a_model_client_fails(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, list[Any]]
) -> None:
    """The rerun child's guard, on its own: a rerun during which the engine's model client (or
    ``httpx2`` under it) was loaded fails, though the program returned 0. It replaced the check for
    ``weightspace.llm``, which can no longer be installed (FR-018.18)."""
    from wsp_worker import program

    monkeypatch.setattr(main_mod, "read_child_line", _rerun_line)
    for name in main_mod.MODEL_CLIENT_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(program, "rerun", lambda line, now: 0)
    with pytest.raises(ExitedError) as clean:
        main_mod.child_main("rerun")
    assert clean.value.code == 0  # the control: nothing loaded, the rerun's own code

    def loading(line: Any, now: str) -> int:
        monkeypatch.setitem(sys.modules, "httpx2", types.ModuleType("httpx2"))
        return 0

    monkeypatch.setattr(program, "rerun", loading)
    with pytest.raises(SystemExit) as failed:
        main_mod.child_main("rerun")
    assert "httpx2 was imported during a rerun" in str(failed.value.code)
    assert recorded == {"miwsp_engine": []}  # a rerun never reaches the engine


def test_the_guard_names_the_engines_client() -> None:
    assert main_mod.MODEL_CLIENT_MODULES == ("wsp_worker.miwsp_engine.client", "httpx2")
