"""The WSP start gate (feature 016; FR-016.37; FTASKS 20.1, 20.2a, 5.5).

Everything here is the production path: ``create_app``, the gate decorator, the approval service,
``wsp.runs``, and the predicates' OWN read sessions, which reach this test's migrated database
because it is bound as the process engine (``db.session``) - exactly how production binds it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from tests.integration.wsp.wsp_db import NOW, engine_at_head, insert
from tests.support.jobs import session_for
from tests.support.wsp_fixtures import enable_wsp, make_bundle
from wsp.settings import get_wsp_settings

AGENT = {"X-Miforge-Agent": "agent:forge-mcp"}
LOCAL = "http://miserver-backend.miserver.svc:8000/v1"
EXTERNAL = "https://api.anthropic.com"


def _j(value: Any) -> Any:
    """A JSON column as a value: SQLite returns its text, PostgreSQL the decoded object."""
    return json.loads(value) if isinstance(value, str | bytes) else value


@dataclass
class World:
    engine: sa.Engine
    client: TestClient
    body: dict[str, Any]

    def count(self, table: str) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar_one())

    def one(self, sql: str, **params: Any) -> Any:
        with self.engine.connect() as conn:
            return conn.execute(sa.text(sql), params).mappings().one()

    def execute(self, sql: str, **params: Any) -> None:
        with self.engine.begin() as conn:
            conn.execute(sa.text(sql), params)

    def start(
        self, endpoint_id: str, *, kind: str = "repair", agent: bool = True, **extra: Any
    ) -> Any:
        body = {**self.body, "endpoint_id": endpoint_id, "kind": kind, **extra}
        return self.client.post("/api/v1/wsp/runs", json=body, headers=AGENT if agent else {})

    def approve(self, approval_id: str) -> Any:
        return self.client.post(f"/api/v1/approvals/{approval_id}/approve")


def _endpoint(conn: sa.Connection, endpoint_id: str, base_url: str, locality: str) -> None:
    external = locality == "external"
    insert(
        conn,
        "mf_wsp_endpoints",
        id=endpoint_id,
        kind="anthropic" if external else "openai_compatible",
        base_url=base_url,
        model="claude" if external else "qwen",
        locality=locality,
        price_in_per_mtok=3 if external else None,
        price_out_per_mtok=15 if external else None,
        output_cap_field="max_tokens",
        serving="generic" if external else "miserver",
        key_ciphertext=b"sealed-key" if external else None,
        created_at=NOW,
        created_by="operator",
    )


@pytest.fixture()
def world(tmp_path, monkeypatch) -> Iterator[World]:
    from db import session as db_session

    enable_wsp(monkeypatch)
    get_wsp_settings.cache_clear()
    fx = make_bundle(tmp_path / "bundle", rule="mean")
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_bundles",
            bundle_sha256=fx.manifest["bundle_sha256"],
            owner="miworkbench",
            bundle_id=fx.manifest["bundle_id"],
            bundle_kind="linear_probe",
            artifact_kind="probe",
            artifact_id="p",
            detector_digest=None,
            model="m",
            layer=12,
            sensitivity_value="not_sensitive",
            manifest=json.dumps(fx.manifest),
            bytes=1,
            import_status="imported",
            received_at=NOW,
            received_by="operator",
        )
        _endpoint(conn, "ep-local", LOCAL, "local")
        _endpoint(conn, "ep-ext", EXTERNAL, "external")
    # The predicates open their own session through the process engine (P52): bind it.
    monkeypatch.setattr(db_session, "_engine", engine)
    monkeypatch.setattr(db_session, "_session_factory", None)
    app = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    body = {
        "bundle": {
            "owner": "miworkbench",
            "bundle_id": fx.manifest["bundle_id"],
            "bundle_sha256": fx.manifest["bundle_sha256"],
        },
        "mode": "session",
        "task": "fewer false alarms",
        "budgets": {"tokens": 1000},
        "change_limits": {},
        "observation_tools": [],
        "primary_sets": ["eval"],
        "sensitivity": "not_sensitive",
    }
    yield World(engine=engine, client=TestClient(app, raise_server_exceptions=False), body=body)
    engine.dispose()
    get_wsp_settings.cache_clear()


def _pending(response: Any, action: str) -> dict[str, Any]:
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending" and body["action"] == action, body
    return dict(body)


# ----------------------------------------------------------------------- the classes


def test_an_agent_local_repair_waits_for_wsp_session(world: World) -> None:
    _pending(world.start("ep-local"), "wsp_session")
    assert world.count("mf_wsp_runs") == 0
    assert world.count("mf_approvals") == 1


def test_an_agent_external_repair_waits_for_wsp_session_external(world: World) -> None:
    _pending(world.start("ep-ext"), "wsp_session_external")
    assert world.count("mf_wsp_runs") == 0


def test_an_agent_local_inspect_runs_at_once_with_no_approval(world: World) -> None:
    response = world.start("ep-local", kind="inspect")
    assert response.status_code == 201, response.text
    run = world.one("SELECT status, approval_id, caller FROM mf_wsp_runs")
    assert dict(run) == {"status": "queued", "approval_id": None, "caller": "agent:forge-mcp"}
    assert world.count("mf_approvals") == 0


def test_an_agent_external_inspect_waits_for_wsp_session_external(world: World) -> None:
    _pending(world.start("ep-ext", kind="inspect"), "wsp_session_external")
    assert world.count("mf_wsp_runs") == 0


def test_a_body_locality_is_never_trusted(world: World) -> None:
    """The external endpoint stays external whatever the body claims (C37's body-locality case)."""
    _pending(
        world.start("ep-ext", kind="inspect", locality="local", endpoint={"locality": "local"}),
        "wsp_session_external",
    )


def test_an_endpoint_whose_url_derives_external_escalates_even_if_its_column_says_local(
    world: World,
) -> None:
    world.execute("UPDATE mf_wsp_endpoints SET locality = 'local' WHERE id = 'ep-ext'")
    _pending(world.start("ep-ext", kind="inspect"), "wsp_session_external")


def test_an_unknown_endpoint_is_refused_and_nothing_is_stored(world: World) -> None:
    response = world.start("ep-nope")
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "wsp_endpoint_unknown"
    assert world.count("mf_approvals") == 0 and world.count("mf_wsp_runs") == 0


def _resize_bundle(world: World, total: int) -> None:
    """Give the stored bundle's manifest files summing to ``total`` bytes (as an imported phase-2
    bundle's would; the stored manifest is what the start freezes against)."""
    row = world.one("SELECT bundle_sha256, manifest FROM mf_wsp_bundles")
    manifest = _j(row["manifest"])
    others = sum(int(f["bytes"]) for f in manifest["files"][1:])
    manifest["files"][0]["bytes"] = total - others
    world.execute(
        "UPDATE mf_wsp_bundles SET manifest = :m, bytes = :b WHERE bundle_sha256 = :s",
        m=json.dumps(manifest),
        b=total,
        s=row["bundle_sha256"],
    )


def test_a_bundle_the_worker_cannot_hold_is_refused_before_an_approval_is_filed(
    world: World,
) -> None:
    """Production defect 2026-11-20: a 35 GB bundle was approved, then evicted the worker mid-push.
    A bundle larger than the worker's /work is refused by the gate's validate hook: no approval
    for the operator to decide, no run."""
    _resize_bundle(world, 57 * 1024**3)
    response = world.start("ep-local")
    assert response.status_code == 413, response.text
    error = response.json()["error"]
    assert error["code"] == "wsp_bundle_too_large_for_worker"
    assert "WSP_WORKER_WORK_BYTES" in error["message"]
    assert world.count("mf_approvals") == 0 and world.count("mf_wsp_runs") == 0
    # The operator's own start is refused the same way.
    assert world.start("ep-local", agent=False).status_code == 413
    assert world.count("mf_wsp_runs") == 0


def test_a_time_budget_too_short_for_its_passes_is_refused_before_an_approval_is_filed(
    world: World,
) -> None:
    """Fix 2026-11-21: 40 forward passes over run wsp_cb092cedad97's 34.9 GB practice set need
    ~350 s at WSP_FORWARD_PASS_BYTES_PER_S; 120 s is under half, so the gate's validate hook
    refuses it with a fix: no approval for the operator to decide, no run."""
    row = world.one("SELECT bundle_sha256, manifest FROM mf_wsp_bundles")
    manifest = _j(row["manifest"])
    manifest["practice"].update(tokens=4_257_928, d_model=4096, dtype="float16")
    world.execute(
        "UPDATE mf_wsp_bundles SET manifest = :m WHERE bundle_sha256 = :s",
        m=json.dumps(manifest),
        s=row["bundle_sha256"],
    )
    budgets = {"tokens": 300_000, "time_s": 120, "max_steps": 20, "forward_passes": 40}
    response = world.start("ep-local", budgets=budgets)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "wsp_time_budget_too_small"
    assert "raise budgets.time_s to at least" in error["details"]["fix"]
    assert world.count("mf_approvals") == 0 and world.count("mf_wsp_runs") == 0
    assert world.start("ep-local", agent=False, budgets=budgets).status_code == 422
    assert world.count("mf_wsp_runs") == 0
    # The same practice set with the incident's own budget waits for its approval.
    _pending(world.start("ep-local", budgets={**budgets, "time_s": 2400}), "wsp_session")


def test_the_phase2_bundle_that_was_evicted_now_waits_for_its_approval(world: World) -> None:
    _resize_bundle(world, 34_881_633_698)
    _pending(world.start("ep-local"), "wsp_session")
    assert world.count("mf_approvals") == 1


def test_the_operator_starts_at_once(world: World) -> None:
    response = world.start("ep-local", agent=False)
    assert response.status_code == 201, response.text
    run = world.one("SELECT status, approval_id, caller, endpoint_locality FROM mf_wsp_runs")
    assert dict(run) == {
        "status": "queued",
        "approval_id": None,
        "caller": "operator",
        "endpoint_locality": "local",
    }


# ----------------------------------------------------------------------- the decision


def test_an_agent_cannot_approve(world: World) -> None:
    pending = _pending(world.start("ep-local"), "wsp_session")
    response = world.client.post(
        f"/api/v1/approvals/{pending['approval_id']}/approve", headers=AGENT
    )
    assert response.status_code == 403, response.text
    assert response.json()["error"]["code"] == "agent_cannot_decide"
    assert world.count("mf_wsp_runs") == 0


def test_the_summary_names_what_the_operator_decides(world: World) -> None:
    pending = _pending(world.start("ep-ext"), "wsp_session_external")
    summary = pending["summary"]
    assert summary["kind"] == "repair" and summary["mode"] == "session"
    assert summary["endpoint"] == {
        "id": "ep-ext",
        "model": "claude",
        "base_url": EXTERNAL,
        "locality": "external",
    }
    assert summary["sensitivity"] == "not_sensitive"
    assert summary["budgets"]["cost_usd"] == 10
    assert len(summary["endpoint_fingerprint"]) == 64
    assert len(summary["request_sha256"]) == 64


def test_the_summary_shows_a_sensitive_bundle_as_sensitive(world: World) -> None:
    """Review OR2-016-4: every other fixture bundle is ``not_sensitive``, so a summary that always
    said so passed. An agent's repair of a SENSITIVE bundle through a local endpoint is gated
    ``wsp_session``, and the card must say what the operator decides on (FR-016.37). Control
    ``S11-ap-summary-sens`` (the summary's sensitivity a constant) turns this red."""
    with world.engine.begin() as conn:
        manifest = _j(conn.execute(sa.text("SELECT manifest FROM mf_wsp_bundles")).scalar_one())
        manifest["sensitivity"]["value"] = "sensitive"
        conn.execute(
            sa.text("UPDATE mf_wsp_bundles SET sensitivity_value = 'sensitive', manifest = :m"),
            {"m": json.dumps(manifest)},
        )
    world.body = {**world.body, "sensitivity": "sensitive"}
    pending = _pending(world.start("ep-local"), "wsp_session")
    assert pending["summary"]["sensitivity"] == "sensitive"
    assert pending["summary"]["endpoint"]["locality"] == "local"


def test_the_executor_stores_the_run_with_its_approval(world: World) -> None:
    pending = _pending(world.start("ep-local"), "wsp_session")
    decided = world.approve(pending["approval_id"])
    assert decided.status_code == 200, decided.text
    approval = world.one("SELECT status, result FROM mf_approvals")
    assert approval["status"] == "executed", dict(approval)
    result = _j(approval["result"])
    run = world.one("SELECT run_id, status, approval_id, caller, request_sha256 FROM mf_wsp_runs")
    assert result == {
        "run_id": run["run_id"],
        "request_sha256": run["request_sha256"],
        "status": "queued",
    }
    assert run["approval_id"] == pending["approval_id"]
    assert run["caller"] == "agent:forge-mcp"
    assert run["request_sha256"] == pending["summary"]["request_sha256"]


def test_an_endpoint_re_registered_external_after_filing_is_stale_at_execution(
    world: World,
) -> None:
    """R1B-M12: the predicates read the stored row; execution re-derives the action."""
    pending = _pending(world.start("ep-local"), "wsp_session")
    world.execute(
        "UPDATE mf_wsp_endpoints SET base_url = :url, locality = 'external', "
        "price_in_per_mtok = 3, price_out_per_mtok = 15 WHERE id = 'ep-local'",
        url=EXTERNAL,
    )
    world.approve(pending["approval_id"])
    approval = world.one("SELECT status, error FROM mf_approvals")
    assert approval["status"] == "failed"
    assert _j(approval["error"])["code"] == "wsp_approval_action_stale"
    assert world.count("mf_wsp_runs") == 0


def test_an_endpoint_changed_after_filing_is_refused_at_execution(world: World) -> None:
    """P63, R2A-22: same locality, a different price: the fingerprint moved."""
    pending = _pending(world.start("ep-ext"), "wsp_session_external")
    world.execute("UPDATE mf_wsp_endpoints SET price_out_per_mtok = 75 WHERE id = 'ep-ext'")
    world.approve(pending["approval_id"])
    approval = world.one("SELECT status, error FROM mf_approvals")
    assert approval["status"] == "failed"
    assert _j(approval["error"])["code"] == "wsp_endpoint_changed"
    assert world.count("mf_wsp_runs") == 0


def test_a_new_key_ciphertext_moves_the_fingerprint(world: World) -> None:
    """R3A-24: the fingerprint covers the ciphertext's sha256, not the 8-hex display value."""
    pending = _pending(world.start("ep-ext"), "wsp_session_external")
    world.execute(
        "UPDATE mf_wsp_endpoints SET key_ciphertext = :c WHERE id = 'ep-ext'", c=b"other-key"
    )
    world.approve(pending["approval_id"])
    approval = world.one("SELECT status, error FROM mf_approvals")
    assert _j(approval["error"])["code"] == "wsp_endpoint_changed"


def test_the_fingerprint_is_recomputed_independently(world: World) -> None:
    """The summary's fingerprint equals a recomputation written out here, field by field."""
    import hashlib

    import rfc8785

    pending = _pending(world.start("ep-ext"), "wsp_session_external")
    expected = hashlib.sha256(
        rfc8785.dumps(
            {
                "id": "ep-ext",
                "kind": "anthropic",
                "base_url": EXTERNAL,
                "model": "claude",
                "locality": "external",
                "serving": "generic",
                "output_cap_field": "max_tokens",
                "price_in_per_mtok": "3.000000",
                "price_out_per_mtok": "15.000000",
                "price_cache_write_per_mtok": None,
                "price_cache_read_per_mtok": None,
                "key_ciphertext_sha256": hashlib.sha256(b"sealed-key").hexdigest(),
            }
        )
    ).hexdigest()
    assert pending["summary"]["endpoint_fingerprint"] == expected


# ----------------------------------------------------------------------- the run routes


def test_a_started_run_reads_back_as_a_valid_run_document(world: World) -> None:
    from wsp.schemas import validate_run

    started = world.start("ep-local", agent=False).json()
    doc = world.client.get(f"/api/v1/wsp/runs/{started['run_id']}").json()
    validate_run(doc)
    assert doc["status"] == "queued" and doc["request_sha256"] == started["request_sha256"]
    assert doc["approval"] is None and doc["boot_id"] is None
    listed = world.client.get("/api/v1/wsp/runs").json()["runs"]
    assert [r["run_id"] for r in listed] == [started["run_id"]]
    assert world.client.get("/api/v1/wsp/runs", params={"status": "running"}).json() == {"runs": []}


def test_a_queued_run_cancels_once(world: World) -> None:
    run_id = world.start("ep-local", agent=False).json()["run_id"]
    first = world.client.post(f"/api/v1/wsp/runs/{run_id}/cancel", headers=AGENT)
    assert first.status_code == 200, first.text
    assert first.json() == {"run_id": run_id, "status": "cancelled", "stop_reason": "cancelled"}
    again = world.client.post(f"/api/v1/wsp/runs/{run_id}/cancel")
    assert again.status_code == 409 and again.json()["error"]["code"] == "wsp_run_not_cancellable"
    assert world.one("SELECT finished_at FROM mf_wsp_runs")["finished_at"] is not None


def test_an_unknown_run_is_404(world: World) -> None:
    for path in ("/api/v1/wsp/runs/wsp_000000000000", "/api/v1/wsp/runs/x/transcript"):
        response = world.client.get(path)
        assert response.status_code == 404, (path, response.text)
        assert response.json()["error"]["code"] == "wsp_run_not_found"


def test_a_run_without_a_result_has_an_empty_transcript(world: World) -> None:
    run_id = world.start("ep-local", agent=False).json()["run_id"]
    body = world.client.get(f"/api/v1/wsp/runs/{run_id}/transcript").json()
    assert body == {"run_id": run_id, "transcript_sha256": None, "entries": [], "partial": False}


# ------------------------------------------------------------------- the queue limit, at execution


def test_ten_approved_runs_queue_and_the_eleventh_approval_fails_saying_what_to_do(
    world: World,
) -> None:
    """Production 2026-11-20: four approvals filed by one agent, the fourth failed at EXECUTION
    (``wsp_queue_limit_reached``, the limit was 3). The limit is 10 now; the approval gate re-runs
    the same check when an approval executes (the route replays ``prepare_start``), and its
    refusal now says what to do: re-file once a run starts."""
    filed = [_pending(world.start("ep-local"), "wsp_session") for _ in range(11)]
    for pending in filed:
        world.approve(pending["approval_id"])
    statuses = world.client.get("/api/v1/approvals").json()["approvals"]
    by_id = {a["approval_id"]: a for a in statuses}
    done = [by_id[p["approval_id"]]["status"] for p in filed]
    assert done[:10] == ["executed"] * 10, done  # the fourth (and up to the tenth) is accepted
    eleventh = by_id[filed[10]["approval_id"]]
    assert eleventh["status"] == "failed"
    assert eleventh["error"]["code"] == "wsp_queue_limit_reached"
    assert "the limit is 10" in eleventh["error"]["message"]
    assert "file this request again" in eleventh["error"]["message"]
    queued = world.one(
        "SELECT COUNT(*) AS n FROM mf_wsp_runs WHERE status = 'queued' AND caller = :c",
        c=AGENT["X-Miforge-Agent"],
    )
    assert queued["n"] == 10


def test_the_operator_queues_ten_and_the_eleventh_is_refused(world: World) -> None:
    for n in range(10):
        response = world.start("ep-local", agent=False)
        assert response.status_code == 201, (n, response.text)
    refused = world.start("ep-local", agent=False)
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "wsp_queue_limit_reached"
    assert world.count("mf_wsp_runs") == 10
