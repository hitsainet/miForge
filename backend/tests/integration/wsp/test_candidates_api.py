"""Candidates and usage are append-only (FR-016.6; FTASKS 3.3), on SQLite and on PostgreSQL.

Both the ORM path and raw SQL are refused, because the guard is a database trigger on each engine
(OVN-wsp-forge-9), not a convention in one code path. The candidate routes' own tests join this file
in phase 2 (16.4) and phase 3 (24.1).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from tests.integration.wsp.wsp_db import (
    bundle_row,
    candidate_row,
    engine_at_head,
    run_row,
    usage_row,
)
from wsp.models import WspCandidate, WspUsage

REFUSED = (
    sa.exc.IntegrityError,
    sa.exc.InternalError,
    sa.exc.ProgrammingError,
    sa.exc.OperationalError,
)


@pytest.fixture()
def seeded(tmp_path: Path) -> Iterator[sa.Engine]:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        bundle_row(conn)
        run_id = run_row(conn)
        candidate_row(conn, run_id)
        usage_row(conn, run_id)
    yield engine
    # Its pooled connection is otherwise held until garbage collection (12.2 round 3, R3-2).
    engine.dispose()


def _refused(engine: sa.Engine, sql: str) -> None:
    with pytest.raises(REFUSED) as err, engine.begin() as conn:
        conn.execute(sa.text(sql))
    assert "append-only" in str(err.value)


def test_candidates_are_append_only(seeded: sa.Engine) -> None:
    engine = seeded
    _refused(engine, "UPDATE mf_wsp_candidates SET changed_entries = 2")
    with pytest.raises(REFUSED), Session(engine) as session:
        row = session.get(WspCandidate, "wspc_000000000001")
        assert row is not None
        row.parent_ref = "tampered"
        session.commit()
    with engine.connect() as conn:  # nothing changed
        assert conn.execute(
            sa.text("SELECT changed_entries, parent_ref FROM mf_wsp_candidates")
        ).one() == (
            1,
            "a" * 64,
        )


def test_usage_is_append_only(seeded: sa.Engine) -> None:
    engine = seeded
    _refused(engine, "UPDATE mf_wsp_usage SET cost_usd = 0.5")
    _refused(engine, "DELETE FROM mf_wsp_usage")
    with pytest.raises(REFUSED), Session(engine) as session:
        row = session.scalars(sa.select(WspUsage)).one()
        row.output_tokens = 99
        session.commit()
    with engine.connect() as conn:
        assert conn.execute(
            sa.text("SELECT COUNT(*), MAX(output_tokens) FROM mf_wsp_usage")
        ).one() == (1, 1)


def test_usage_appends_are_accepted(seeded: sa.Engine) -> None:
    """The control: the trigger refuses changes, not inserts."""
    engine = seeded
    with engine.begin() as conn:
        usage_row(conn, "wsp_000000000001", seq=1)
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_usage")).scalar_one() == 2


# ============================================================ phase 2: the candidate routes (16.4)

import hashlib  # noqa: E402
import json  # noqa: E402
from typing import Any  # noqa: E402

from tests.integration.wsp.test_evaluate import Ev, _client, scorer  # noqa: E402
from tests.integration.wsp.test_evaluate import ev as ev  # noqa: E402
from tests.integration.wsp.wsp_db import NOW, insert  # noqa: E402


def _validate_run(document: dict[str, Any]) -> None:
    from wsp.schemas import errors

    found = errors("miforge-wsp-run-v1.json", document)
    assert not found, found


class TestExport:
    """FR-016.33: the patch files, the rebuilt tensors and the run document, each checkable."""

    def test_the_patch_files_hash_to_their_headers_and_the_candidate(self, ev: Ev) -> None:
        from wsp.weight_ops.patch import patch_sha256

        cid, _ = ev.candidate()
        client, _ = _client(ev)
        got = client.get(f"/api/v1/wsp/candidates/{cid}/patch/patch.json")
        arrays = client.get(f"/api/v1/wsp/candidates/{cid}/patch/edit.npz")
        assert got.status_code == arrays.status_code == 200
        for answer in (got, arrays):
            assert answer.headers["X-Content-SHA256"] == hashlib.sha256(answer.content).hexdigest()
            assert answer.headers["X-Content-Type-Options"] == "nosniff"
        manifest = json.loads(got.content)
        candidate = client.get(f"/api/v1/wsp/candidates/{cid}").json()
        assert patch_sha256(manifest) == candidate["patch_sha256"]
        assert hashlib.sha256(arrays.content).hexdigest() == manifest["arrays_sha256"]
        assert client.get(f"/api/v1/wsp/candidates/{cid}/patch/other").status_code == 400

    def test_a_tampered_stored_patch_is_not_served(self, ev: Ev) -> None:
        from wsp import lineage

        cid, _ = ev.candidate()
        client, _ = _client(ev)
        run_id = client.get(f"/api/v1/wsp/candidates/{cid}").json()["run_id"]
        path = lineage.patch_dir(ev.world.settings, run_id) / lineage.STORED_PATCH_ARRAYS
        path.write_bytes(path.read_bytes() + b"\0")
        answer = client.get(f"/api/v1/wsp/candidates/{cid}/patch/edit.npz")
        assert answer.status_code == 422 and answer.json()["error"]["code"] == "wsp_result_invalid"

    def test_the_tensors_hash_to_the_patchs_edited_sha256(self, ev: Ev) -> None:
        from wsp.weight_ops.st_io import parse_safetensors

        cid, _ = ev.candidate()
        client, _ = _client(ev)
        served = client.get(f"/api/v1/wsp/candidates/{cid}/tensors")
        assert served.status_code == 200
        assert served.headers["X-Content-SHA256"] == hashlib.sha256(served.content).hexdigest()
        tensors = parse_safetensors(served.content)
        manifest = json.loads(client.get(f"/api/v1/wsp/candidates/{cid}/patch/patch.json").content)
        for entry in manifest["tensors"]:
            assert tensors[entry["name"]].sha256 == entry["edited_sha256"]
        from wsp.weight_ops.st_io import read_safetensors

        parent = read_safetensors(ev.tmp_path / "bundle" / "heads.safetensors")
        assert tensors["bias"].sha256 != parent["bias"].sha256  # not the parent's (C33)
        assert tensors["weight"].sha256 == parent["weight"].sha256

    def test_the_run_document_validates_and_names_no_provider_module(self, ev: Ev) -> None:
        cid, sha = ev.candidate()
        client, _ = _client(ev)
        before = client.get(f"/api/v1/wsp/candidates/{cid}/run-document").json()
        _validate_run(before)
        assert before["candidate_id"] == cid and before["verdict"] is None
        assert before["lineage"]["chain"] == [before["patch_sha256"]]
        ev.evaluate(cid, sha, scorer(ev))
        after = client.get(f"/api/v1/wsp/candidates/{cid}/run-document").json()
        _validate_run(after)
        assert after["verdict"]["value"] == "improved" and after["run_sha256"]
        text = json.dumps(after)
        assert "weightspace." not in text and "wsp_worker" not in text

    def test_the_run_route_serves_the_candidate_and_its_verdict(self, ev: Ev) -> None:
        """Fix 2026-11-20: ``GET /wsp/runs/{id}`` built the document without the candidate, so a
        ``candidate`` run read ``candidate_id: null`` and never showed its verdict. It now
        equals the candidate's live run document, before and after a verdict."""
        cid, sha = ev.candidate()
        client, _ = _client(ev)
        run_id = client.get(f"/api/v1/wsp/candidates/{cid}").json()["run_id"]
        served = client.get(f"/api/v1/wsp/runs/{run_id}").json()
        _validate_run(served)
        assert served["outcome"] == "candidate" and served["candidate_id"] == cid
        assert served == client.get(f"/api/v1/wsp/candidates/{cid}/run-document").json()
        ev.evaluate(cid, sha, scorer(ev))
        after = client.get(f"/api/v1/wsp/runs/{run_id}").json()
        assert after["verdict"]["value"] == "improved" and after["run_sha256"]

    def test_the_list_and_get_routes(self, ev: Ev) -> None:
        first, _ = ev.candidate(delta=0.25)
        second, sha = ev.candidate(delta=0.5)
        ev.evaluate(second, sha, scorer(ev))
        client, _ = _client(ev)
        listed = client.get("/api/v1/wsp/candidates").json()["candidates"]
        assert {c["candidate_id"] for c in listed} == {first, second}
        one = client.get(f"/api/v1/wsp/candidates/{second}").json()
        assert one["verdict"]["value"] == "improved"
        assert client.get("/api/v1/wsp/candidates/wspc_000000000000").status_code == 404


class TestLineage:
    def test_a_chain_of_three_returns_three_steps_in_order_ending_at_the_bundle(
        self, tmp_path
    ) -> None:
        from tests.integration.wsp.wsp_db import HEX, bundle_row, run_row

        engine = engine_at_head(tmp_path)
        shas = ["1" * 64, "2" * 64, "3" * 64]
        with engine.begin() as conn:
            bundle_row(conn)
            parent_kind, parent_ref = "bundle", HEX
            for i, sha in enumerate(shas):
                run_id = run_row(conn, run_id=f"wsp_00000000000{i}")
                insert(
                    conn,
                    "mf_wsp_candidates",
                    candidate_id=f"wspc_00000000000{i}",
                    run_id=run_id,
                    parent_kind=parent_kind,
                    parent_ref=parent_ref,
                    bundle_base_sha256=HEX,
                    patch_sha256=sha,
                    changed_entries=1,
                    tensors="{}",
                    created_at=NOW,
                )
                parent_kind, parent_ref = "candidate", f"wspc_00000000000{i}"
        from fastapi.testclient import TestClient

        from api.app import create_app
        from api.deps import session_dependency
        from tests.support.jobs import session_for

        app = create_app()

        def one() -> Any:
            with session_for(engine) as session:
                yield session

        app.dependency_overrides[session_dependency] = one
        client = TestClient(app)
        answer = client.get("/api/v1/wsp/candidates/wspc_000000000002/lineage").json()
        assert [s["patch_sha256"] for s in answer["steps"]] == shas
        assert answer["steps"][0]["parent"] == {"kind": "bundle", "ref": HEX}
        assert answer["bundle_sha256"] == HEX
        assert all(s["provider"] == "weightspace" for s in answer["steps"])
        engine.dispose()


class TestVerdictSnapshot:
    """FR-016.33a (P65, P71, R3A-16, R3B-10)."""

    def test_the_snapshot_is_stored_bytes_whose_sha256_is_run_sha256(self, ev: Ev) -> None:
        cid, sha = ev.candidate()
        verdict = ev.evaluate(cid, sha, scorer(ev))
        client, _ = _client(ev)
        served = client.get(
            f"/api/v1/wsp/candidates/{cid}/run-document",
            params={"verdict_id": verdict["verdict_id"]},
        )
        assert served.status_code == 200
        live = client.get(f"/api/v1/wsp/candidates/{cid}/run-document").json()
        digest = hashlib.sha256(served.content).hexdigest()
        assert served.headers["X-Content-SHA256"] == digest == live["run_sha256"]
        # Independently: RFC 8785 of the live document without the two mutable fields.
        import rfc8785

        without = {k: v for k, v in live.items() if k not in ("run_sha256", "program_reruns")}
        assert hashlib.sha256(rfc8785.dumps(without)).hexdigest() == digest
        assert json.loads(served.content)["verdict"]["verdict_id"] == verdict["verdict_id"]

    def test_a_later_rerun_changes_neither_the_bytes_nor_run_sha256(self, ev: Ev) -> None:
        """A program run's verdict, then a rerun of its program: ``program_reruns`` grows, the
        snapshot and ``run_sha256`` stay, and the live document still reproduces it."""
        import rfc8785

        from tests.integration.wsp.wsp_db import run_row

        cid, sha = ev.candidate()
        client, _ = _client(ev)
        run_id = client.get(f"/api/v1/wsp/candidates/{cid}").json()["run_id"]
        program = "9" * 64
        with ev.world.engine.begin() as conn:  # as the intake leaves a program run
            conn.execute(
                sa.text("UPDATE mf_wsp_runs SET program_sha256 = :p WHERE run_id = :r"),
                {"p": program, "r": run_id},
            )
        verdict = ev.evaluate(cid, sha, scorer(ev))
        path = f"/api/v1/wsp/candidates/{cid}/run-document"
        before = client.get(path, params={"verdict_id": verdict["verdict_id"]}).content
        with ev.world.engine.begin() as conn:
            run_row(
                conn,
                run_id="wsp_0000000000ff",
                style="program_rerun",
                status="completed",
                outcome="no_change",
                program_sha256=program,
                bundle_sha256=ev.manifest["bundle_sha256"],
            )
        live = client.get(path).json()
        assert live["program_reruns"] == ["wsp_0000000000ff"]
        after = client.get(path, params={"verdict_id": verdict["verdict_id"]}).content
        assert after == before
        assert live["run_sha256"] == hashlib.sha256(before).hexdigest()
        without = {k: v for k, v in live.items() if k not in ("run_sha256", "program_reruns")}
        assert hashlib.sha256(rfc8785.dumps(without)).hexdigest() == live["run_sha256"]

    def test_the_snapshot_is_the_stored_bytes_not_a_rerendering(
        self, ev: Ev, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """C33a: a later release that renders the document differently cannot move a stored
        snapshot: the bytes served under a verdict_id are the bytes stored with it."""
        from wsp import runs

        cid, sha = ev.candidate()
        verdict = ev.evaluate(cid, sha, scorer(ev))
        client, _ = _client(ev)
        path = f"/api/v1/wsp/candidates/{cid}/run-document"
        before = client.get(path, params={"verdict_id": verdict["verdict_id"]}).content
        real = runs.run_document

        def rerendered(*args: Any, **kwargs: Any) -> dict[str, Any]:
            document = real(*args, **kwargs)
            return {**document, "caller_ref": "rendered by a later release"}

        monkeypatch.setattr(runs, "run_document", rerendered)
        after = client.get(path, params={"verdict_id": verdict["verdict_id"]})
        assert after.content == before
        assert after.headers["X-Content-SHA256"] == hashlib.sha256(before).hexdigest()

    def test_another_candidates_verdict_id_is_not_found(self, ev: Ev) -> None:
        first, sha1 = ev.candidate(delta=0.25)
        second, _ = ev.candidate(delta=0.5)
        verdict = ev.evaluate(first, sha1, scorer(ev))
        client, _ = _client(ev)
        for cid, vid in ((second, verdict["verdict_id"]), (first, "wspv_000000000000")):
            answer = client.get(
                f"/api/v1/wsp/candidates/{cid}/run-document", params={"verdict_id": vid}
            )
            assert answer.status_code == 404
            assert answer.json()["error"]["code"] == "wsp_verdict_not_found"

    def test_evaluate_and_get_verdict_carry_the_same_verdict_id(self, ev: Ev) -> None:
        """P71 (C33c's red): the id ``evaluate`` answers names the stored row and the snapshot."""
        cid, sha = ev.candidate()
        verdict = ev.evaluate(cid, sha, scorer(ev))
        client, _ = _client(ev)
        assert verdict["verdict_id"].startswith("wspv_")
        assert client.get(f"/api/v1/wsp/candidates/{cid}/verdict").json() == verdict
        with ev.world.engine.connect() as conn:
            assert conn.execute(sa.text("SELECT verdict_id FROM mf_wsp_verdicts")).scalar_one() == (
                verdict["verdict_id"]
            )
        snapshot = client.get(
            f"/api/v1/wsp/candidates/{cid}/run-document",
            params={"verdict_id": verdict["verdict_id"]},
        )
        assert snapshot.status_code == 200


class TestTamperedStores:
    """A stored snapshot or score file changed under the database or WSP_BUNDLE_ROOT is refused,
    never served (``wsp_result_invalid``)."""

    def test_a_tampered_snapshot_is_not_served(self, ev: Ev) -> None:
        cid, sha = ev.candidate()
        verdict = ev.evaluate(cid, sha, scorer(ev))
        with ev.world.engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE mf_wsp_verdicts SET snapshot = :b"), {"b": b'{"tampered":true}'}
            )
        client, _ = _client(ev)
        answer = client.get(
            f"/api/v1/wsp/candidates/{cid}/run-document",
            params={"verdict_id": verdict["verdict_id"]},
        )
        assert answer.status_code == 422 and answer.json()["error"]["code"] == "wsp_result_invalid"

    def test_tampered_scores_are_not_served(self, ev: Ev) -> None:
        from wsp import evaluate as evaluate_mod

        cid, sha = ev.candidate()
        verdict = ev.evaluate(cid, sha, scorer(ev))
        client, _ = _client(ev)
        run_id = client.get(f"/api/v1/wsp/candidates/{cid}").json()["run_id"]
        path = evaluate_mod.scores_dir(ev.world.settings, run_id, verdict["verdict_id"]) / "0.f32"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(bytes(data))
        answer = client.get(f"/api/v1/wsp/candidates/{cid}/verdict/scores", params={"set": "eval"})
        assert answer.status_code == 422 and answer.json()["error"]["code"] == "wsp_result_invalid"


class TestReviewPins:
    """Review 016-P2 (18.4, r016p2): routes whose load-bearing line the whole tree let pass."""

    def test_a_deleted_runs_candidate_is_neither_listed_nor_served(self, ev: Ev) -> None:
        """Controls C5, C6: a run tombstoned by delete (``deleted_at``) takes its candidate off
        the list and out of every candidate route."""
        kept, _ = ev.candidate(delta=0.25)
        gone, _ = ev.candidate(delta=0.5)
        client, _ = _client(ev)
        run_id = client.get(f"/api/v1/wsp/candidates/{gone}").json()["run_id"]
        with ev.world.engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE mf_wsp_runs SET deleted_at = :t WHERE run_id = :r"),
                {"t": NOW, "r": run_id},
            )
        listed = client.get("/api/v1/wsp/candidates").json()["candidates"]
        assert [c["candidate_id"] for c in listed] == [kept]
        for path in ("", "/lineage", "/tensors", "/patch/patch.json", "/run-document"):
            answer = client.get(f"/api/v1/wsp/candidates/{gone}{path}")
            assert answer.status_code == 404, (path, answer.status_code)

    def test_each_lineage_step_carries_its_verdict(self, ev: Ev) -> None:
        """Control C11 (FR-016.35): a step's verdict is its stored verdict's id and value."""
        cid, sha = ev.candidate()
        client, _ = _client(ev)
        before = client.get(f"/api/v1/wsp/candidates/{cid}/lineage").json()
        assert [s["verdict"] for s in before["steps"]] == [None]
        verdict = ev.evaluate(cid, sha, scorer(ev))
        after = client.get(f"/api/v1/wsp/candidates/{cid}/lineage").json()
        assert after["steps"][-1]["verdict"] == {
            "verdict_id": verdict["verdict_id"],
            "value": verdict["value"],
        }
