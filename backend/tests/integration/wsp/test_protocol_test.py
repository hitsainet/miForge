"""The endpoint protocol test (feature 018; FR-018.8, F18-D7; FTASKS 9.4-9.6).

Two halves:

* **the route** (``POST /api/v1/wsp/endpoints/{id}/protocol-test``) through ``create_app``, the gate
  decorator and the approval service, on the start gate's world (``test_wsp_approvals.py``): the
  run it queues, the built-in bundle it stores (once), its refusals, and its approval (none for a
  local endpoint, ``wsp_session_external`` for an external one);
* **end to end**: the queued run is pushed by the REAL driver, its pushed document becomes the
  engine's child line, the REAL engine (the worker's interpreter, the child preamble) runs it
  against a scripted endpoint, and its ``/out`` goes back through the REAL intake. A compliant
  editor gives ``refused_turns`` 0 in the run's list row; one that sends two actions in a reply
  gives more (FR-018.8's AC).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from tests.integration.wsp.test_dispatch import FakeWorker, World, archive
from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from tests.integration.wsp.test_wsp_approvals import AGENT, _pending
from tests.integration.wsp.test_wsp_approvals import World as GateWorld
from tests.integration.wsp.test_wsp_approvals import world as gate_world  # noqa: F401
from tests.support.wsp_drive import drive_inline, resident_window
from wsp import bundles, protocol_test, runs, tasks
from wsp.settings import get_wsp_settings

#: The engine is the one provider since 018-P4 (FR-018.18).
BOTH = "miwsp_engine"


@pytest.fixture()
def gw(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> GateWorld:
    world: GateWorld = request.getfixturevalue("gate_world")
    monkeypatch.setenv("WSP_PROVIDERS_AVAILABLE", BOTH)
    monkeypatch.setenv("WSP_BUNDLE_ROOT", str(tmp_path / "wsp-root"))
    get_wsp_settings.cache_clear()
    return world


def _test(world: GateWorld, endpoint_id: str, body: Any = None, *, agent: bool = False) -> Any:
    return world.client.post(
        f"/api/v1/wsp/endpoints/{endpoint_id}/protocol-test",
        json={} if body is None else body,
        headers=AGENT if agent else {},
    )


def _run(world: GateWorld) -> dict[str, Any]:
    return dict(
        world.one(
            "SELECT run_id, operation, style, status, provider_id, caller, caller_ref, "
            "endpoint_id, approval_id, request FROM mf_wsp_runs"
        )
    )


# ----------------------------------------------------------------------------- the route


def test_the_operators_test_queues_a_short_engine_inspect_on_the_built_in_bundle(
    gw: GateWorld,
) -> None:
    r = _test(gw, "ep-local", {"caller_ref": "compare-modes"})
    assert r.status_code == 201, r.text
    assert set(r.json()) == {"run_id", "status", "thinking"} and r.json()["status"] == "queued"
    assert r.json()["thinking"] == "off"  # the endpoint's default, frozen in the request
    run = _run(gw)
    assert (run["operation"], run["style"], run["provider_id"], run["caller"]) == (
        "inspect",
        "session",
        "miwsp_engine",
        "operator",
    )
    assert run["caller_ref"] == "protocol_test:ep-local:compare-modes"
    request = json.loads(run["request"]) if isinstance(run["request"], str) else run["request"]
    assert request["task"] == protocol_test.TASK
    assert {k: request["budgets"][k] for k in protocol_test.BUDGETS} == protocol_test.BUDGETS
    assert request["observation_tools"] == protocol_test.OBSERVATION_TOOLS
    assert request["bundle"]["owner"] == "miforge"
    assert request["bundle"]["bundle_id"] == protocol_test.BUNDLE_ID
    bundle = dict(
        gw.one(
            "SELECT owner, bundle_sha256, import_status, sensitivity_value FROM mf_wsp_bundles "
            "WHERE owner = 'miforge'"
        )
    )
    assert (bundle["import_status"], bundle["sensitivity_value"]) == ("imported", "not_sensitive")
    stored = bundles.store_dir(get_wsp_settings(), bundle["bundle_sha256"])
    with gw.engine.connect() as conn:
        manifest = conn.execute(
            sa.text("SELECT manifest FROM mf_wsp_bundles WHERE owner = 'miforge'")
        ).scalar_one()
    manifest = json.loads(manifest) if isinstance(manifest, str) else manifest
    # The stored files pass the import's own checks again (hashes, row ids, parity).
    bundles.check_files(manifest, stored)
    assert sorted(p.name for p in stored.iterdir()) == sorted(f["name"] for f in manifest["files"])


def test_the_test_names_the_engine_whatever_the_default_provider(
    gw: GateWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The protocol test is the engine's (P2-D5), so its start body names ``miwsp_engine`` itself
    (018 16.2, control P2-C9.5e). Since the switch the default is the engine too, so dropping the
    body's provider passed every test; with the default pointing elsewhere it must still run on
    the engine, or be refused, never run on the default."""
    monkeypatch.setenv("WSP_DEFAULT_PROVIDER", "not_the_engine")
    get_wsp_settings.cache_clear()
    r = _test(gw, "ep-local")
    assert r.status_code == 201, r.text
    assert _run(gw)["provider_id"] == "miwsp_engine"


def test_a_second_test_reuses_the_bundle(gw: GateWorld) -> None:
    assert _test(gw, "ep-local").status_code == 201
    assert gw.count("mf_wsp_bundles") == 2  # the world's own bundle and the built-in one
    sha = gw.one("SELECT bundle_sha256 FROM mf_wsp_bundles WHERE owner = 'miforge'")[
        "bundle_sha256"
    ]
    gw.execute("UPDATE mf_wsp_runs SET status = 'cancelled'")
    assert _test(gw, "ep-local").status_code == 201
    assert gw.count("mf_wsp_bundles") == 2
    assert (
        gw.one("SELECT bundle_sha256 FROM mf_wsp_bundles WHERE owner = 'miforge'")["bundle_sha256"]
        == sha
    )


def test_an_agents_test_on_a_local_endpoint_runs_at_once(gw: GateWorld) -> None:
    r = _test(gw, "ep-local", agent=True)
    assert r.status_code == 201, r.text
    run = _run(gw)
    assert (run["caller"], run["approval_id"]) == ("agent:forge-mcp", None)
    assert gw.count("mf_approvals") == 0


def test_an_agents_test_on_an_external_endpoint_waits_and_runs_once_approved(
    gw: GateWorld,
) -> None:
    """FR-018.8 / 9.6: the approval-pending answer, wsp_session_external; nothing queued until the
    operator approves, and then exactly one run carrying the approval."""
    pending = _pending(_test(gw, "ep-ext", agent=True), "wsp_session_external")
    assert gw.count("mf_wsp_runs") == 0
    summary = gw.one("SELECT summary FROM mf_approvals")["summary"]
    summary = json.loads(summary) if isinstance(summary, str) else summary
    assert summary["protocol_test"] is True and summary["kind"] == "inspect"
    assert summary["provider"]["id"] == "miwsp_engine"
    approved = gw.approve(pending["approval_id"])
    assert approved.status_code == 200, approved.text
    run = _run(gw)
    assert (run["provider_id"], run["approval_id"]) == ("miwsp_engine", pending["approval_id"])
    assert gw.count("mf_wsp_runs") == 1


def test_an_approved_test_whose_endpoint_changed_is_refused(gw: GateWorld) -> None:
    """The start's own re-check, shared: a PATCH between filing and approval fails it."""
    pending = _pending(_test(gw, "ep-ext", agent=True), "wsp_session_external")
    gw.execute("UPDATE mf_wsp_endpoints SET context_window_tokens = 1000 WHERE id = 'ep-ext'")
    gw.approve(pending["approval_id"])
    assert gw.count("mf_wsp_runs") == 0
    decided = dict(gw.one("SELECT status, result, error FROM mf_approvals"))
    assert "wsp_endpoint_changed" in json.dumps(decided, default=str), decided


@pytest.mark.parametrize(
    "endpoint_id, body, code",
    [
        ("ep-nope", {}, "wsp_endpoint_unknown"),
        ("ep-local", {"kind": "repair"}, "validation_failed"),
        ("ep-local", {"caller_ref": "x" * 65}, "validation_failed"),
        ("ep-local", {"caller_ref": ""}, "validation_failed"),
    ],
)
def test_a_refused_test_stores_nothing(
    gw: GateWorld, endpoint_id: str, body: dict[str, Any], code: str
) -> None:
    r = _test(gw, endpoint_id, body)
    assert r.json()["error"]["code"] == code, r.text
    assert gw.count("mf_wsp_runs") == 0
    assert gw.count("mf_wsp_bundles") == 1  # no built-in bundle was generated


def test_the_test_is_refused_until_the_engine_is_enabled(
    gw: GateWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WSP_PROVIDERS_AVAILABLE", "")
    get_wsp_settings.cache_clear()
    r = _test(gw, "ep-local")
    assert r.status_code == 503 and r.json()["error"]["code"] == "provider_unavailable", r.text
    assert gw.count("mf_wsp_runs") == 0 and gw.count("mf_wsp_bundles") == 1


# ----------------------------------------------------------------------------- end to end


def _queue_test(world: World, endpoint_id: str) -> str:
    with world.factory() as session:
        prepared = protocol_test.prepare(session, endpoint_id, {}, "operator", get_wsp_settings())
        run = runs.create_run(session, prepared, approval_id=None)
        session.commit()
        return run.run_id


@pytest.fixture()
def dw(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    world: World = request.getfixturevalue("dispatch_world")
    monkeypatch.setenv("WSP_PROVIDERS_AVAILABLE", BOTH)
    get_wsp_settings.cache_clear()
    yield world


def test_an_external_test_is_pushed_without_asking_miworkbench(dw: World) -> None:
    """The built-in bundle holds no real data and miWorkbench does not know it: its stored
    sensitivity stands, so an external test is pushed (a miWorkbench bundle on the same endpoint
    is still re-read: ``test_dispatch.py``'s sensitivity tests)."""
    run_id = _queue_test(dw, "ep-ext")
    dw.sensitivity = "sensitive"  # what a miWorkbench read would answer, were it asked
    worker = FakeWorker()
    assert dw.drive(worker) == [f"pushed:{run_id}"]
    assert dw.reads == []


def test_a_built_in_row_not_marked_not_sensitive_refuses_an_external_test(dw: World) -> None:
    """The shortcut above trusts the built-in row's STORED value, so that value must be checked:
    a row that is not ``not_sensitive`` refuses an external push, and miWorkbench is still not
    asked."""
    run_id = _queue_test(dw, "ep-ext")
    with dw.factory() as session:
        session.execute(
            sa.text(
                "UPDATE mf_wsp_bundles SET sensitivity_value = 'sensitive' WHERE owner = 'miforge'"
            )
        )
        session.commit()
    worker = FakeWorker()
    assert dw.drive(worker) == [f"refused:{run_id}:wsp_sensitivity_raised"]
    assert dw.reads == []


def _engine(
    dw: World,
    tmp_path: Path,
    steps: list[dict[str, Any]],
    endpoint_id: str = "ep-local",
    received: list[dict[str, Any]] | None = None,
    read_window: Any = resident_window,
) -> tuple[str, dict[str, Any]]:
    """Queue a protocol test, push it with the REAL driver and file list, run the pushed document
    in the REAL engine against ``steps``, and take its ``/out`` back through the REAL intake."""
    from tests.support.wsp_engine import run_case
    from tests.support.wsp_scripted_endpoint import ScriptedEndpoint
    from wsp.result_archive import RESULT_MEMBERS

    run_id = _queue_test(dw, endpoint_id)
    worker = FakeWorker()
    actions = drive_inline(
        dw.factory,
        worker,
        dw.settings,
        read_sensitivity=dw.read,
        files_for=tasks.run_files,
        read_window=read_window,
    )
    assert actions == [f"pushed:{run_id}"], actions
    assert worker.pushed is not None
    document = worker.pushed
    run_dir = tmp_path / "child"
    paths = {name: run_dir / name for name in ("work", "out", "tmp")}
    for path in paths.values():
        path.mkdir(parents=True)
    with dw.factory() as session:
        row = runs.get_run(session, run_id)
        files = tasks.run_files(row, session.get(runs.WspBundle, row.bundle_sha256))
    for name, source in files:
        shutil.copyfile(source, paths["work"] / name)
    with ScriptedEndpoint("openai_compatible", steps) as ep:
        line = {
            "mode": document["mode"],
            "run_id": run_id,
            "spec": document["spec"],
            "files": document["files"],
            "gateway": ep.gateway,
            "paths": {k: str(v) for k, v in paths.items()},
            "time_left_s": float(document["spec"]["time_budget_s"]),
        }
        seen = run_case(run_dir, line)
    if received is not None:
        received.extend(ep.requests)
    assert seen["exit"] == 0, seen.get("stderr", "")[-2000:]
    members = {
        str(p.relative_to(paths["out"])): p.read_bytes()
        for p in paths["out"].rglob("*")
        if p.is_file() and str(p.relative_to(paths["out"])) in RESULT_MEMBERS
    }
    worker.state, worker.data = "finished", archive(worker.boot, run_id, **members)
    collected = dw.drive(worker)
    assert collected[0] == f"collected:{run_id}:completed", collected
    from api.v1.endpoints.wsp_runs import list_runs

    with dw.factory() as session:
        (listed,) = [r for r in list_runs(session, None, 50)["runs"] if r["run_id"] == run_id]
    return run_id, listed


SUBMIT = ("submit", {"answer": "A mean-pooled linear probe.", "confidence": 0.5})


def test_a_compliant_editor_gives_zero_refused_turns(dw: World, tmp_path: Path) -> None:
    steps = [
        {"calls": [("read_tensor", {"name": "weight"})]},
        {"calls": [("forward_practice_metrics", {})]},
        {"calls": [SUBMIT]},
    ]
    _run_id, listed = _engine(dw, tmp_path, steps)
    assert listed["provider"] == "miwsp_engine"
    assert (listed["action_protocol"], listed["refused_turns"], listed["handoffs"]) == (
        "tool_calls",
        0,
        0,
    )
    assert listed["forward_passes_used"] == 1
    assert listed["stop_reason"] == "submitted"
    assert listed["caller_ref"] == "protocol_test:ep-local"


def test_an_editor_that_sends_two_actions_gives_refused_turns(dw: World, tmp_path: Path) -> None:
    steps = [
        {"calls": [("read_tensor", {"name": "weight"}), SUBMIT]},
        {"calls": [SUBMIT]},
    ]
    _run_id, listed = _engine(dw, tmp_path, steps)
    assert listed["refused_turns"] > 0
    assert listed["stop_reason"] == "submitted"


def test_a_structured_endpoint_is_tested_in_structured_mode(dw: World, tmp_path: Path) -> None:
    """The test runs the endpoint's CURRENT protocol: after the operator's PATCH to structured,
    the engine sends ``response_format`` and the list row records ``structured``."""
    from wsp import endpoints

    with dw.factory() as session:
        endpoints.update_endpoint(session, "ep-local", {"action_protocol": "structured"})
        session.commit()
    name, args = SUBMIT
    steps = [
        {"json": {"tool": "read_tensor", "arguments": {"name": "weight"}}},
        {"json": [{"tool": "read_tensor", "arguments": {"name": "bias"}}, {"tool": name}]},
        {"json": {"tool": name, "arguments": args}},
    ]
    _run_id, listed = _engine(dw, tmp_path, steps)
    assert (listed["action_protocol"], listed["refused_turns"]) == ("structured", 1)
    assert listed["stop_reason"] == "submitted"


def test_a_generated_bundle_that_fails_the_imports_checks_is_never_stored(
    gw: GateWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The built-in bundle passes the same checks as an import before it is stored: a generator
    that wrote a damaged file (one byte of the heads file changed after its hash was taken) is
    refused, and neither a row nor a store directory is left behind."""
    from wsp import fixture_bundle

    real = fixture_bundle.make_bundle

    def damaged(root: Path, **kw: Any) -> Any:
        fx = real(root, **kw)
        heads = root / "heads.safetensors"
        data = bytearray(heads.read_bytes())
        data[-1] ^= 0x01
        heads.write_bytes(bytes(data))
        return fx

    monkeypatch.setattr(fixture_bundle, "make_bundle", damaged)
    r = _test(gw, "ep-local")
    assert r.json()["error"]["code"] == "wsp_bundle_hash_mismatch", r.text
    assert gw.count("mf_wsp_bundles") == 1 and gw.count("mf_wsp_runs") == 0
    root = Path(get_wsp_settings().wsp_bundle_root) / "bundles"
    assert not root.exists() or [p.name for p in root.iterdir()] == []


# ----------------------------------------------------------------------------- thinking


def _think(dw: World, change: dict[str, Any]) -> None:
    from wsp import endpoints

    with dw.factory() as session:
        endpoints.update_endpoint(session, "ep-local", change)
        session.commit()


def test_a_thinking_endpoints_test_reports_that_reasoning_came_back(
    dw: World, tmp_path: Path
) -> None:
    """2026-11-21: after the operator's PATCH to thinking on, the test runs with it (frozen in
    its request), the SERVER receives ``enable_thinking: true`` and the allowance, and the list
    row says the reasoning came back."""
    _think(dw, {"thinking": "on", "thinking_tokens": 2048})
    received: list[dict[str, Any]] = []
    steps = [
        {"calls": [("read_tensor", {"name": "weight"})], "reasoning": "Read the weight first."},
        {"calls": [SUBMIT], "reasoning": "It is a mean probe."},
    ]
    # miServer reports 8,192 (until 2026-11-27 the figure an empty read assumed): the action's
    # cap below is 1,024, an eighth of it.
    run_id, listed = _engine(
        dw, tmp_path, steps, received=received, read_window=lambda endpoint: 8192
    )
    assert (listed["thinking"], listed["thinking_tokens"]) == ("on", 2048)
    assert (listed["reasoning_turns"], listed["reasoning_returned"]) == (2, True)
    assert all(r["body"]["chat_template_kwargs"] == {"enable_thinking": True} for r in received)
    # The test's protocol is tool_calls: thinking inside the total cap, no allowance field.
    assert all("max_reasoning_tokens" not in r["body"] for r in received)
    assert all(r["body"]["max_tokens"] == 1024 + 2048 for r in received)
    with dw.factory() as session:
        request = runs.get_run(session, run_id).request
    budgets = protocol_test.BUDGETS
    assert request["budgets"]["tokens"] == budgets["tokens"] + budgets["max_steps"] * 2048


def test_a_server_that_ignores_thinking_is_reported(dw: World, tmp_path: Path) -> None:
    _think(dw, {"thinking": "on"})
    steps = [{"calls": [SUBMIT]}]
    _run_id, listed = _engine(dw, tmp_path, steps)
    assert (listed["thinking"], listed["reasoning_turns"], listed["reasoning_returned"]) == (
        "on",
        0,
        False,
    )
    assert listed["thinking_tokens"] == protocol_test.DEFAULT_THINKING_TOKENS


def test_a_test_with_thinking_off_sends_enable_thinking_false(dw: World, tmp_path: Path) -> None:
    received: list[dict[str, Any]] = []
    _run_id, listed = _engine(dw, tmp_path, [{"calls": [SUBMIT]}], received=received)
    assert (listed["thinking"], listed["reasoning_returned"]) == ("off", None)
    assert received[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "max_reasoning_tokens" not in received[0]["body"]


# ----------------------------------------------------------------------------- paging


def test_paging_follows_the_endpoint_through_the_real_push(dw: World, tmp_path: Path) -> None:
    """2026-11-21: the endpoint's ``paging`` reaches the frozen request (``prepare_start`` ->
    ``freeze``), the real push carries it to the real engine, and the engine reads it. On (the
    default, absent from the request): ``recall`` is offered and the ledger rides the request.
    Off (the operator's PATCH): frozen ``"off"``, no recall, no ledger, the list says off."""
    steps = [{"calls": [("read_tensor", {"name": "weight"})]}, {"calls": [SUBMIT]}]
    # A window that pages (under 16,384 paging switches itself off: window_small).
    _think(dw, {"context_window_tokens": 32_768})
    received: list[dict[str, Any]] = []
    run_id, listed = _engine(dw, tmp_path / "on", steps, received=received)
    with dw.factory() as session:
        assert "paging" not in runs.get_run(session, run_id).request
    assert listed["paging"] is None and listed["ledger_rows"] == 0
    assert "recall" in [t["function"]["name"] for t in received[0]["body"]["tools"]]
    _think(dw, {"paging": "off"})
    received_off: list[dict[str, Any]] = []
    run_off, listed_off = _engine(dw, tmp_path / "off", steps, received=received_off)
    with dw.factory() as session:
        assert runs.get_run(session, run_off).request["paging"] == "off"
    assert listed_off["paging"] == "off"
    assert (listed_off["pages_out"], listed_off["recalls"], listed_off["ledger_rows"]) == (0, 0, 0)
    assert "recall" not in [t["function"]["name"] for t in received_off[0]["body"]["tools"]]
