"""The worker's provider dispatch (feature 018; FTASKS 7.1, 7.2; FR-018.2).

Two layers:

- **in this process**, ``wsp_worker.__main__.child_main`` itself with the preamble, the stdin read
  and ``os._exit`` replaced, and each provider's ``run`` replaced by a recorder: the payload (the
  very line pushed) and the call count of each branch;
- **the real entry point**, ``python -I -c <boot> run`` as the supervisor starts it, against a
  scripted endpoint: the engine's session runs once, with the pushed run id, and never imports
  ``weightspace`` (``run_provider`` exits non-zero if anything did).
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
    calls: dict[str, list[Any]] = {"miwsp_engine": [], "weightspace": []}
    fake_adapter = types.ModuleType("wsp_worker.adapters.weightspace_adapter")

    def adapter_run(line: Any) -> int:
        calls["weightspace"].append(line)
        return 0

    def engine_run(line: Any, **_: Any) -> int:
        calls["miwsp_engine"].append(line)
        return 0

    fake_adapter.run = adapter_run  # type: ignore[attr-defined]
    import wsp_worker.adapters as adapters

    monkeypatch.setitem(sys.modules, "wsp_worker.adapters.weightspace_adapter", fake_adapter)
    monkeypatch.setattr(adapters, "weightspace_adapter", fake_adapter, raising=False)
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


@pytest.mark.parametrize("provider", ["miwsp_engine", "weightspace"])
def test_a_run_goes_to_its_own_provider_once_with_the_pushed_line(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, list[Any]], provider: str
) -> None:
    line = _line(provider)
    monkeypatch.setattr(main_mod, "read_child_line", lambda: line)
    with pytest.raises(ExitedError) as exited:
        main_mod.child_main("run")
    assert exited.value.code == 0
    other = "weightspace" if provider == "miwsp_engine" else "miwsp_engine"
    assert len(recorded[provider]) == 1 and recorded[provider][0] is line
    assert recorded[other] == []


@pytest.mark.parametrize("provider", ["wsplab", None])
def test_any_other_provider_is_refused_and_nothing_runs(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, list[Any]], provider: str | None
) -> None:
    monkeypatch.setattr(main_mod, "read_child_line", lambda: _line(provider))
    with pytest.raises(RefusedError) as refused:
        main_mod.child_main("run")
    assert (refused.value.code, refused.value.exit_code) == ("worker_provider_unknown", 78)
    assert recorded == {"miwsp_engine": [], "weightspace": []}


def test_a_replay_is_routed_the_same_way(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, list[Any]]
) -> None:
    line = {**_line("miwsp_engine"), "mode": "replay", "gateway": None}
    monkeypatch.setattr(main_mod, "read_child_line", lambda: line)
    with pytest.raises(ExitedError):
        main_mod.child_main("replay")
    assert recorded["miwsp_engine"] == [line] and recorded["weightspace"] == []


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


def test_an_engine_run_that_loaded_weightspace_fails(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, list[Any]]
) -> None:
    """FR-018.2's runtime guard, on its own: an engine run during which ``weightspace`` was
    imported (a fake module stands in) fails, even though the engine returned 0. The real
    entry-point test only shows the guard's absence of effect on a clean run; this one fails
    when the guard is removed (018-P1 review, mutation M2)."""
    monkeypatch.delitem(sys.modules, "weightspace", raising=False)

    def engine_run(line: Any, **_: Any) -> int:
        recorded["miwsp_engine"].append(line)
        monkeypatch.setitem(sys.modules, "weightspace", types.ModuleType("weightspace"))
        return 0

    monkeypatch.setattr(miwsp_engine, "run", engine_run)
    line = _line("miwsp_engine")
    with pytest.raises(SystemExit) as failed:
        main_mod.run_provider(line)
    assert "weightspace was imported" in str(failed.value.code)
    assert recorded == {"miwsp_engine": [line], "weightspace": []}
