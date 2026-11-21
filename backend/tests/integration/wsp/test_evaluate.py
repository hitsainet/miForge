"""``evaluate`` (feature 016 phase 2; FTASKS 15.1, 15.2; FR-016.26, .29, .34; P29, P46, P64, P65,
P70, PF-12, PF-35, PF-46).

Every candidate here is made by the REAL driver and intake (``test_dispatch``'s world, a fake
supervisor answering a real result archive), so its patch is stored exactly as production stores
it. The scorer is ``tests.support.fake_miworkbench.FakeScorer``: it validates every request and
every answer against the VENDORED contract, and its metrics come from an independent pure-Python
reference. The route tests run it as a real HTTP server on ``127.0.0.1``, so ``evaluate`` builds
its production client from ``MIWORKBENCH_API_URL`` with nothing injected.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rfc8785
import sqlalchemy as sa
from fastapi.testclient import TestClient

from core.errors import Refusal
from tests.integration.wsp.test_dispatch import FakeWorker, World, _j, archive, build_world
from tests.integration.wsp.test_runs_lifecycle import _store
from tests.support.fake_miworkbench import (
    EVAL_ROWS,
    FakeScorer,
    evaluate_heldout_sets,
    known_truth,
)
from wsp import evaluate
from wsp.settings import get_wsp_settings
from wsp.weight_ops.patch import build_patch, manifest_text
from wsp.weight_ops.st_io import read_safetensors
from wsp.weight_ops.tensor import from_values

BUNDLE_ID = "wsb_0123456789ab"


def _bias_members(manifest: dict[str, Any], tmp_path: Path, delta: float) -> dict[str, bytes]:
    """A patch moving the bias by ``delta``: a monotone shift, so the practice AUROC is the
    parent's and the run qualifies (FR-016.30)."""
    tensors = read_safetensors(tmp_path / "bundle" / "heads.safetensors")
    bias = tensors["bias"].values64().copy()
    bias[0] += delta
    built = build_patch(
        tensors,
        {"bias": from_values(tensors["bias"].dtype, bias)},
        bundle_base_sha256=manifest["bundle_sha256"],
        parent={"kind": "bundle", "ref": manifest["bundle_sha256"], "patch_sha256": None},
        parent_patch_sha256=None,
        created_by={"provider": "weightspace", "provider_version": "0.3.0", "run_id": "r"},
        created_at="2026-11-19T05:00:00Z",
    )
    assert built is not None
    return {
        "patch/patch.json": manifest_text(dict(built.manifest)).encode(),
        "patch/edit.npz": built.arrays,
        "result.json": json.dumps({"stop_reason": "submitted"}).encode(),
    }


@dataclass
class Ev:
    world: World
    manifest: dict[str, Any]
    tmp_path: Path

    def candidate(self, delta: float = 0.25, **body: Any) -> tuple[str, str]:
        """Run one session through the real driver and intake; ``(candidate_id, request_sha)``."""
        if body:
            self.world.body = {**self.world.body, **body}
        run_id = self.world.queue()
        worker = FakeWorker()
        self.world.drive(worker)
        worker.state = "finished"
        worker.data = archive(
            worker.boot, run_id, **_bias_members(self.manifest, self.tmp_path, delta)
        )
        self.world.drive(worker)
        run = self.world.run(run_id)
        assert run["outcome"] == "candidate", (run["status"], run["failure"])
        with self.world.engine.connect() as conn:
            cid = conn.execute(
                sa.text("SELECT candidate_id FROM mf_wsp_candidates WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        return str(cid), str(run["request_sha256"])

    def evaluate(self, cid: str, sha: str, fake: FakeScorer | None, **settings: Any) -> Any:
        values = {"miworkbench_api_url": "http://miworkbench.test" if fake else "", **settings}
        s = get_wsp_settings().model_copy(update=values)
        with self.world.factory() as session:
            out = evaluate.evaluate(
                session, cid, sha, settings=s, transport=None if fake is None else fake.transport()
            )
            session.commit()
            return out

    def uses(self) -> dict[str, int]:
        with self.world.engine.connect() as conn:
            rows = conn.execute(sa.text("SELECT set_name, heldout_uses FROM mf_wsp_heldout_uses"))
            return {str(r[0]): int(r[1]) for r in rows}

    def verdicts(self) -> int:
        with self.world.engine.connect() as conn:
            return int(conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_verdicts")).scalar_one())


@pytest.fixture()
def ev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Ev]:
    monkeypatch.setenv("WSP_BOOTSTRAP_B", "200")
    world = build_world(
        tmp_path,
        monkeypatch,
        bundle_id=BUNDLE_ID,
        heldout_sets=evaluate_heldout_sets(),
        labels_from_scores=True,
    )
    manifest = _store(world, tmp_path)
    yield Ev(world, manifest, tmp_path)
    world.engine.dispose()
    get_wsp_settings.cache_clear()


def scorer(ev: Ev, gap: float = 2.5, **kw: Any) -> FakeScorer:
    return FakeScorer(ev.manifest["heldout_ref"], known_truth(gap), **kw)


def _validate_verdict(document: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator, FormatChecker

    from wsp.schemas import RUN, schema_dir

    root = json.loads((schema_dir() / RUN).read_text())
    validator = Draft202012Validator(
        {"$defs": root["$defs"], "$ref": "#/$defs/Verdict"}, format_checker=FormatChecker()
    )
    errors = sorted(e.message for e in validator.iter_errors(document))
    assert not errors, errors


# ----------------------------------------------------------------------------- the verdict


@pytest.mark.parametrize(
    ("gap", "value"), [(2.5, "improved"), (0.2, "regressed"), (1.0, "not_significant")]
)
def test_known_truth_score_sets_give_their_verdict(ev: Ev, gap: float, value: str) -> None:
    cid, sha = ev.candidate()
    fake = scorer(ev, gap)
    verdict = ev.evaluate(cid, sha, fake)
    assert verdict["value"] == value, verdict["reasons"]
    _validate_verdict(verdict)
    assert fake.calls == 1 and fake.invalid_requests == []
    assert verdict["scorer"]["metrics_agree"] is True
    assert verdict["completeness"] == {"ok": True, "missing": []}
    assert [s["name"] for s in verdict["per_set"]] == ["eval", "everyday", "nonsense"]


def test_the_metric_guard_refuses_a_recall_off_by_one_row(ev: Ev) -> None:
    """P29: miWorkbench's returned recall is one row away from its own scores."""
    cid, sha = ev.candidate()

    def off_by_one(answer: dict[str, Any]) -> None:
        metrics = answer["sets"][0]["metrics"]["candidate"]
        metrics["recall_at_bar"] = metrics["recall_at_bar"] + 1 / (EVAL_ROWS // 2)

    verdict = ev.evaluate(cid, sha, scorer(ev, tamper=off_by_one))
    assert verdict["value"] == "unmeasured"
    assert verdict["reasons"][0] == "scorer_metric_disagreement"
    assert verdict["scorer"]["metrics_agree"] is False
    assert ev.uses() == {"eval": 1, "everyday": 1, "nonsense": 1, "calibration": 1}


def test_unset_miworkbench_is_unmeasured_with_no_call(ev: Ev) -> None:
    cid, sha = ev.candidate()
    fake = scorer(ev)
    s = get_wsp_settings().model_copy(update={"miworkbench_api_url": ""})
    with ev.world.factory() as session:
        verdict = evaluate.evaluate(session, cid, sha, settings=s, transport=fake.transport())
        session.commit()
    assert verdict["value"] == "unmeasured"
    assert verdict["reasons"] == ["miworkbench_not_configured"]
    assert fake.calls == 0 and ev.uses() == {}


def test_the_request_is_exactly_the_contracts_fields(ev: Ev) -> None:
    """R1B-X5: ``run_ref`` with the run's limit, tensors only on candidate and control, the
    bundle's five-field held-out sets, one call, no agent header and no URL."""
    cid, sha = ev.candidate(change_limits={"max_changed_entries": 5})
    fake = scorer(ev)
    ev.evaluate(cid, sha, fake)
    assert fake.calls == 1 and len(fake.requests) == 1
    request = fake.requests[0]
    assert set(request) == {
        "kind",
        "bundle_id",
        "bundle_sha256",
        "run_ref",
        "heldout_ref",
        "heads",
        "primary_sets",
    }
    with ev.world.engine.connect() as conn:
        run_id, patch_sha = conn.execute(
            sa.text("SELECT run_id, patch_sha256 FROM mf_wsp_candidates WHERE candidate_id = :c"),
            {"c": cid},
        ).one()
    assert request["run_ref"] == {
        "miforge_run_id": run_id,
        "patch_sha256": patch_sha,
        "max_changed_entries": 5,
    }
    assert request["heldout_ref"] == ev.manifest["heldout_ref"]
    assert all(
        set(s) == {"name", "view_id", "role", "rows", "row_ids_sha256"}
        for s in request["heldout_ref"]["sets"]
    )
    carrying = {h["label"] for h in request["heads"] if h.get("tensors")}
    assert carrying == {"candidate", "control"}
    assert [h["label"] for h in request["heads"]] == [
        "parent",
        "candidate",
        "control",
        "baseline:logistic_regression",
        "baseline:lda",
        "baseline:diff_of_means",
        "baseline:threshold_recut",
        "baseline:affine_recalibration",
    ]
    assert not any(k.startswith("x-miforge") for k in fake.headers[0])
    assert "url" not in json.dumps(request).lower().replace("view-", "")


def test_the_candidate_head_is_the_patch_applied_and_the_control_is_not(ev: Ev) -> None:
    """P46 (C26b): the candidate's tensors are the REBUILT edit (its bias hashes to the patch's
    ``edited_sha256``), the control's bias differs from both, frozen tensors equal the parent's."""
    cid, sha = ev.candidate()
    fake = scorer(ev)
    ev.evaluate(cid, sha, fake)
    heads = {h["label"]: h for h in fake.requests[0]["heads"]}
    parent = read_safetensors(ev.tmp_path / "bundle" / "heads.safetensors")
    with ev.world.engine.connect() as conn:
        tensors = _j(
            conn.execute(
                sa.text("SELECT tensors FROM mf_wsp_candidates WHERE candidate_id = :c"),
                {"c": cid},
            ).scalar_one()
        )["tensors"]
    edited = {t["name"]: t["edited_sha256"] for t in tensors}
    cand = {t["name"]: t["sha256"] for t in heads["candidate"]["tensors"]}
    ctrl = {t["name"]: t["sha256"] for t in heads["control"]["tensors"]}
    assert cand["bias"] == edited["bias"] != parent["bias"].sha256
    assert ctrl["bias"] not in (cand["bias"], parent["bias"].sha256)
    for name in ("weight", "mean", "std"):
        assert cand[name] == ctrl[name] == parent[name].sha256


def test_a_patch_outside_its_runs_limits_is_refused_with_nothing_sent(ev: Ev) -> None:
    """P46 (C26c): the limits are checked again at evaluate, against the stored run."""
    cid, sha = ev.candidate()
    with ev.world.engine.begin() as conn:
        row = conn.execute(
            sa.text(
                "SELECT r.run_id, r.request FROM mf_wsp_runs r JOIN mf_wsp_candidates c "
                "ON c.run_id = r.run_id WHERE c.candidate_id = :c"
            ),
            {"c": cid},
        ).one()
        request = _j(row[1])
        request["change_limits"]["frozen_tensors"] = ["bias"]
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET request = :q WHERE run_id = :r"),
            {"q": json.dumps(request), "r": row[0]},
        )
    fake = scorer(ev)
    with pytest.raises(Refusal) as refused:
        ev.evaluate(cid, sha, fake)
    assert refused.value.code == "wsp_change_limit_exceeded"
    assert fake.calls == 0 and ev.verdicts() == 0 and ev.uses() == {}


def test_a_calibration_set_without_scores_passes_and_a_ratio_fails(ev: Ev) -> None:
    """R1B-X3, R1B-X5c: the calibration set carries bars only, and passes; a false-alarm figure
    returned as a ratio is refused by the contract, and an integral float by the guard."""
    cid, sha = ev.candidate()
    assert ev.evaluate(cid, sha, scorer(ev))["value"] == "improved"

    def ratio(answer: dict[str, Any]) -> None:
        m = answer["sets"][1]["metrics"]["candidate"]
        m["false_alarms_at_bar"] = m["false_alarms_at_bar"] / m["negatives"]

    cid2, sha2 = ev.candidate(delta=0.5)
    verdict = ev.evaluate(cid2, sha2, scorer(ev, tamper=ratio, validate_answer=False))
    assert verdict["value"] == "unmeasured" and verdict["reasons"] == ["scorer_contract_invalid"]

    def integral_float(answer: dict[str, Any]) -> None:
        m = answer["sets"][1]["metrics"]["candidate"]
        m["false_alarms_at_bar"] = float(m["false_alarms_at_bar"])

    cid3, sha3 = ev.candidate(delta=0.75)
    verdict = ev.evaluate(cid3, sha3, scorer(ev, tamper=integral_float, validate_answer=False))
    assert verdict["value"] == "unmeasured"
    assert verdict["reasons"][0] == "scorer_metric_disagreement"


def _budget(status: int | None = None) -> Any:
    return (
        "heldout_set_head_budget_exhausted",
        {"set": "eval", "row_ids_sha256": "c" * 64, "rows_exhausted": 7},
    )


@pytest.mark.parametrize("status", [409, 429])
def test_budget_exhaustion_is_unmeasured_by_the_bodys_code(
    ev: Ev, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """P64, P70, R3B-9: the 409's body names the set; the same body under 429 is still
    exhaustion. Nothing is counted (no scores came back)."""
    cid, sha = ev.candidate()
    fake = scorer(ev, refusal=_budget())
    monkeypatch.setattr(FakeScorer, "refusal_status", lambda self, code: status)
    verdict = ev.evaluate(cid, sha, fake)
    assert verdict["value"] == "unmeasured"
    assert verdict["reasons"] == ["heldout_set_head_budget_exhausted:eval"]
    assert fake.calls == 1 and ev.uses() == {}


@pytest.mark.parametrize("status", [429, 409])
def test_scorer_busy_refuses_with_nothing_stored_or_counted(
    ev: Ev, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """PF-35, R3B-9: ``scorer_busy`` (even under 409) refuses ``wsp_scorer_busy``."""
    cid, sha = ev.candidate()
    fake = scorer(ev, refusal=("scorer_busy", {}))
    monkeypatch.setattr(FakeScorer, "refusal_status", lambda self, code: status)
    with pytest.raises(Refusal) as refused:
        ev.evaluate(cid, sha, fake)
    assert refused.value.code == "wsp_scorer_busy"
    assert ev.verdicts() == 0 and ev.uses() == {}


def test_a_heldout_digest_other_than_rfc_8785_of_the_sent_ref_is_unmeasured(ev: Ev) -> None:
    """P65, R3B-13: checked against an independent ``hashlib.sha256(rfc8785.dumps(...))``."""
    cid, sha = ev.candidate()
    sent = ev.manifest["heldout_ref"]
    independent = hashlib.sha256(rfc8785.dumps(sent)).hexdigest()
    fake = scorer(ev)
    ev.evaluate(cid, sha, fake)
    assert (
        evaluate.contract_problem(
            {
                "bundle_sha256": fake.requests[0]["bundle_sha256"],
                "heldout_ref_sha256": independent,
                "sets": [],
            },
            fake.requests[0],
        )
        is None
    )

    def other(answer: dict[str, Any]) -> None:
        answer["heldout_ref_sha256"] = hashlib.sha256(
            json.dumps(sent, sort_keys=True).encode()
        ).hexdigest()

    cid2, sha2 = ev.candidate(delta=0.5)
    verdict = ev.evaluate(cid2, sha2, scorer(ev, tamper=other))
    assert verdict["value"] == "unmeasured" and verdict["reasons"][0] == "scorer_contract_invalid"


def test_the_heldout_digest_is_rfc_8785_where_the_two_canonicalisers_differ() -> None:
    """C26h's target: a ``heldout_ref`` holding a float, where ``canonical.sha256_of`` (``1e-05``)
    and RFC 8785 (``0.00001``) disagree. Only the RFC 8785 digest is accepted."""
    from core.canonical import sha256_of

    ref = {"owner": "miworkbench", "sets": [{"name": "eval", "weight": 1e-05}]}
    request = {"bundle_sha256": "b" * 64, "heldout_ref": ref, "heads": []}
    jcs = hashlib.sha256(rfc8785.dumps(ref)).hexdigest()
    assert jcs != sha256_of(ref)
    answer = {"bundle_sha256": "b" * 64, "sets": []}
    assert evaluate.contract_problem({**answer, "heldout_ref_sha256": jcs}, request) is None
    assert evaluate.contract_problem({**answer, "heldout_ref_sha256": sha256_of(ref)}, request)


def test_a_wrong_request_digest_is_refused(ev: Ev) -> None:
    cid, _ = ev.candidate()
    with pytest.raises(Refusal) as refused:
        ev.evaluate(cid, "0" * 64, scorer(ev))
    assert refused.value.code == "wsp_request_sha256_mismatch"


def test_the_second_call_is_idempotent(ev: Ev) -> None:
    """PF-12: the stored verdict comes back; no second call, no second count."""
    cid, sha = ev.candidate()
    fake = scorer(ev)
    first = ev.evaluate(cid, sha, fake)
    second = ev.evaluate(cid, sha, fake)
    assert second == first and fake.calls == 1
    assert ev.uses()["eval"] == 1 and ev.verdicts() == 1


def test_an_outage_verdict_is_replaced_by_the_next_call(ev: Ev) -> None:
    """A verdict from no answer is stored (addressable) but never final."""
    cid, sha = ev.candidate()
    first = ev.evaluate(cid, sha, None)
    assert first["value"] == "unmeasured" and ev.verdicts() == 1
    fake = scorer(ev)
    second = ev.evaluate(cid, sha, fake)
    assert second["value"] == "improved" and second["verdict_id"] != first["verdict_id"]
    assert fake.calls == 1 and ev.verdicts() == 1


def test_the_call_past_the_cap_is_refused_with_nothing_sent(ev: Ev) -> None:
    """PF-46: ``WSP_SCORER_MAX_CALLS_PER_HELDOUT`` uses of a set, then nothing is sent."""
    first, sha1 = ev.candidate()
    ev.evaluate(first, sha1, scorer(ev), wsp_scorer_max_calls_per_heldout=1)
    second, sha2 = ev.candidate(delta=0.5)
    fake = scorer(ev)
    with pytest.raises(Refusal) as refused:
        ev.evaluate(second, sha2, fake, wsp_scorer_max_calls_per_heldout=1)
    assert refused.value.code == "wsp_heldout_call_cap_reached"
    assert fake.calls == 0 and ev.uses()["eval"] == 1


def test_no_route_accepts_caller_heads(ev: Ev) -> None:
    """P46: the evaluate body is ``{request_sha256}`` alone."""
    client, _ = _client(ev)
    cid, sha = ev.candidate()
    for body in (
        {"request_sha256": sha, "heads": [{"label": "candidate", "tensors": []}]},
        {"request_sha256": sha, "url": "http://elsewhere"},
        {},
    ):
        answer = client.post(f"/api/v1/wsp/candidates/{cid}/evaluate", json=body)
        assert answer.status_code == 400 and answer.json()["error"]["code"] == "validation_failed"


# ----------------------------------------------------------------------------- use counts


class TestUseCounts:
    def test_three_candidates_count_one_two_three(self, ev: Ev) -> None:
        seen = []
        for delta in (0.25, 0.5, 0.75):
            cid, sha = ev.candidate(delta=delta)
            verdict = ev.evaluate(cid, sha, scorer(ev))
            seen.append({s["name"]: s["heldout_uses"] for s in verdict["per_set"]})
        assert [s["eval"] for s in seen] == [1, 2, 3]
        assert ev.uses() == {"eval": 3, "everyday": 3, "nonsense": 3, "calibration": 3}

    def test_an_idempotent_recall_and_a_failed_call_do_not_count(self, ev: Ev) -> None:
        cid, sha = ev.candidate()
        ev.evaluate(cid, sha, scorer(ev, refusal=("score_timeout", {})))
        assert ev.uses() == {}
        ev.evaluate(cid, sha, scorer(ev))
        ev.evaluate(cid, sha, scorer(ev))
        assert ev.uses()["eval"] == 1

    def test_the_route_lists_them(self, ev: Ev) -> None:
        cid, sha = ev.candidate()
        ev.evaluate(cid, sha, scorer(ev))
        client, _ = _client(ev)
        rows = client.get("/api/v1/wsp/heldout-uses").json()["heldout_uses"]
        assert {(r["set_name"], r["heldout_uses"], r["last_candidate_id"]) for r in rows} == {
            (n, 1, cid) for n in ("eval", "everyday", "nonsense", "calibration")
        }


# ----------------------------------------------------------------------------- the digest


class TestScoresDigest:
    def test_the_stored_digest_is_the_served_arrays(self, ev: Ev) -> None:
        cid, sha = ev.candidate()
        fake = scorer(ev)
        verdict = ev.evaluate(cid, sha, fake)
        client, _ = _client(ev)
        plans = known_truth(2.5)
        for entry in verdict["per_set"]:
            served = client.get(
                f"/api/v1/wsp/candidates/{cid}/verdict/scores", params={"set": entry["name"]}
            )
            assert served.status_code == 200
            digest = next(h["scores_sha256"] for h in entry["heads"] if h["label"] == "candidate")
            assert hashlib.sha256(served.content).hexdigest() == digest
            assert served.headers["X-Content-SHA256"] == digest
            expected = np.asarray(plans[entry["name"]].scores["candidate"], dtype="<f4")
            assert served.content == expected.tobytes()

    def test_one_ulp_moves_the_digest(self) -> None:
        values = np.linspace(-1, 1, 50, dtype=np.float32)
        moved = values.copy()
        moved[17] = np.nextafter(moved[17], np.float32(2))
        assert evaluate.scores_sha256(values) != evaluate.scores_sha256(moved)
        assert evaluate.scores_sha256(values) == hashlib.sha256(values.tobytes()).hexdigest()


# ----------------------------------------------------------------------------- the route


def _client(ev: Ev) -> tuple[TestClient, Any]:
    from api.app import create_app
    from api.deps import session_dependency
    from tests.support.jobs import session_for

    app = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(ev.world.engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    return TestClient(app, raise_server_exceptions=False), app


def test_the_route_scores_through_the_real_client_and_answers_the_verdict_id(
    ev: Ev, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring: the route builds the production client from ``MIWORKBENCH_API_URL`` (a real
    HTTP fake), and ``evaluate`` and ``GET …/verdict`` carry the same ``verdict_id`` (P71)."""
    cid, sha = ev.candidate()
    fake = scorer(ev)
    with fake.serve() as url:
        monkeypatch.setenv("MIWORKBENCH_API_URL", url)
        get_wsp_settings.cache_clear()
        client, _ = _client(ev)
        answer = client.post(f"/api/v1/wsp/candidates/{cid}/evaluate", json={"request_sha256": sha})
    assert answer.status_code == 200, answer.text
    verdict = answer.json()
    assert verdict["value"] == "improved" and fake.calls == 1
    assert not any(k.startswith("x-miforge") for k in fake.headers[0])
    got = client.get(f"/api/v1/wsp/candidates/{cid}/verdict").json()
    assert got["verdict_id"] == verdict["verdict_id"]
    with ev.world.engine.connect() as conn:
        stored = conn.execute(sa.text("SELECT verdict_id FROM mf_wsp_verdicts")).scalar_one()
    assert stored == verdict["verdict_id"]


def test_the_verdict_is_committed_before_evaluate_answers(
    ev: Ev, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over a REAL server, the snapshot named by the verdict_id ``evaluate`` just answered is
    there at once (the session's commit is slowed so the order is deterministic)."""
    import contextlib
    import time

    import httpx

    from api.app import create_app
    from db import session as db_session
    from tests.support.served import served

    if ev.world.engine.dialect.name == "sqlite":
        pytest.skip(
            "commit order needs PostgreSQL: SQLite serialises the two writers (or shares the "
            "test's one connection), so the late commit cannot show; the PostgreSQL job runs it"
        )
    cid, sha = ev.candidate()
    real_scope = db_session.session_scope

    @contextlib.contextmanager
    def slow_commit() -> Iterator[Any]:
        with real_scope() as session:
            yield session
            time.sleep(0.3)

    monkeypatch.setattr(db_session, "_engine", ev.world.engine)
    monkeypatch.setattr(db_session, "_session_factory", None)
    monkeypatch.setattr(db_session, "session_scope", slow_commit)
    fake = scorer(ev)
    with fake.serve() as scorer_url:
        monkeypatch.setenv("MIWORKBENCH_API_URL", scorer_url)
        get_wsp_settings.cache_clear()
        with served(create_app()) as url, httpx.Client(base_url=url, timeout=60) as http:
            verdict = http.post(
                f"/api/v1/wsp/candidates/{cid}/evaluate", json={"request_sha256": sha}
            ).json()
            snapshot = http.get(
                f"/api/v1/wsp/candidates/{cid}/run-document",
                params={"verdict_id": verdict["verdict_id"]},
            )
    assert snapshot.status_code == 200, snapshot.text


def test_the_control_is_seeded_from_the_patch_content(ev: Ev) -> None:
    """FR-016.27 (C27b): ``control_seed`` is ``int(content_sha256[:8], 16)``, and the control head
    is exactly the seeded permutation of the stored patch, recomputed independently here."""
    import json as _json

    from wsp.weight_ops.control import random_control
    from wsp.weight_ops.st_io import read_safetensors as _read

    cid, sha = ev.candidate()
    fake = scorer(ev)
    verdict = ev.evaluate(cid, sha, fake)
    client, _ = _client(ev)
    manifest = _json.loads(client.get(f"/api/v1/wsp/candidates/{cid}/patch/patch.json").content)
    assert verdict["control_seed"] == int(manifest["content_sha256"][:8], 16)
    assert manifest["content_sha256"][:8] != manifest["patch_sha256"][:8]
    base = _read(ev.tmp_path / "bundle" / "heads.safetensors")
    edited = {**base}
    from wsp.weight_ops.st_io import parse_safetensors

    edited.update(parse_safetensors(client.get(f"/api/v1/wsp/candidates/{cid}/tensors").content))
    expected = random_control(base, edited, ["bias"], manifest["content_sha256"], 0)
    sent = {t["name"]: t["sha256"] for t in fake.requests[0]["heads"][2]["tensors"]}
    assert fake.requests[0]["heads"][2]["label"] == "control"
    assert sent["bias"] == expected.tensors["bias"].sha256


@pytest.mark.parametrize(
    "break_",
    [
        "bar",
        "row_ids",
        "set_name",
        "head_missing",
        "bundle",
        "fired_head_missing",
        "fired_negative",
        "fired_over_truncated",
        "fired_float",
    ],
)
def test_an_answer_that_is_not_for_this_request_is_unmeasured(ev: Ev, break_: str) -> None:
    """P65: the sets and row ids are the bundle's, each head's bar is the same in every set,
    every head is answered, and the answer is for this bundle. F-2: ``truncated_fired`` holds
    one count per head sent, each from 0 to the set's truncated rows (the scoring contract
    types the counts as integers and nothing more, so a negative one, or 2.0, passes its shape
    check)."""
    cid, sha = ev.candidate()

    def tamper(answer: dict[str, Any]) -> None:
        excluded = answer["sets"][0]["excluded"]
        if break_ == "fired_head_missing":
            excluded["truncated_fired"].pop("control")
        elif break_ == "fired_negative":
            excluded["truncated_fired"]["parent"] = -1
        elif break_ == "fired_over_truncated":
            excluded["truncated"] = 2
            excluded["truncated_fired"]["parent"] = 3
        elif break_ == "fired_float":  # 2.0 is an "integer" to JSON Schema, not a count
            excluded["truncated"] = 5
            excluded["truncated_fired"]["parent"] = 2.0
        elif break_ == "bar":
            answer["sets"][1]["bars"]["candidate"]["threshold"] = 1.5
        elif break_ == "row_ids":
            answer["sets"][0]["row_ids_sha256"] = "d" * 64
        elif break_ == "set_name":
            answer["sets"][0]["name"] = "elsewhere"
        elif break_ == "head_missing":
            answer["sets"][0]["scores"].pop("control")
            answer["sets"][0]["metrics"].pop("control")
        else:
            answer["bundle_sha256"] = "e" * 64

    verdict = ev.evaluate(cid, sha, scorer(ev, tamper=tamper))
    assert verdict["value"] == "unmeasured"
    assert verdict["reasons"][0] == "scorer_contract_invalid", verdict["reasons"]


def _distinct_fired(answer: dict[str, Any]) -> None:
    """Every set: 9 truncated rows, and each head's count different (1, 2, 3, ... by label)."""
    for one in answer["sets"]:
        one["excluded"]["truncated"] = 9
        one["excluded"]["truncated_fired"] = {
            label: index + 1 for index, label in enumerate(sorted(one["bars"]))
        }


def test_the_verdict_records_every_heads_truncated_fired_count(ev: Ev) -> None:
    """F-2 (2026-11-19): ``truncated_fired`` is a MAP from each head label the scorer answered to
    its count. Until then the run schema allowed one literal key, ``label``, and ``evaluate`` kept
    only the candidate's count. The map validates against the PUBLISHED schema, is stored by the
    real ``store`` (which validates the whole run document) and reads back unchanged from the
    run-document route."""
    from wsp.schemas import validate_run

    cid, sha = ev.candidate()
    fake = scorer(ev, tamper=_distinct_fired)
    verdict = ev.evaluate(cid, sha, fake)
    assert verdict["value"] == "improved", verdict["reasons"]
    _validate_verdict(verdict)
    sent = sorted(h["label"] for h in fake.requests[0]["heads"])
    assert len(sent) >= 4 and {"parent", "candidate", "control"} <= set(sent)
    expected = {label: index + 1 for index, label in enumerate(sent)}
    assert len(set(expected.values())) == len(sent)  # distinct, so a swapped key shows
    for entry in verdict["per_set"]:
        assert entry["truncated_fired"] == expected, entry["name"]
        assert entry["excluded_truncated"] == 9
    client, _ = _client(ev)
    document = client.get(f"/api/v1/wsp/candidates/{cid}/run-document").json()
    validate_run(document)
    assert [s["truncated_fired"] for s in document["verdict"]["per_set"]] == [expected] * 3


def _sent_values(head: dict[str, Any]) -> dict[str, np.ndarray]:
    """A head's tensors as the scorer received them, decoded here (not by miForge's code)."""
    from wsp.weight_ops.tensor import BITS, Tensor

    out = {}
    for t in head["tensors"]:
        wire = np.dtype(BITS[t["dtype"]]).newbyteorder("<")
        bits = np.frombuffer(base64.b64decode(t["data_b64"]), dtype=wire)
        native = bits.astype(wire.newbyteorder("=")).reshape(t["shape"])
        out[t["name"]] = Tensor(t["dtype"], native).values64()
    return out


def _delta_figures(after: dict[str, np.ndarray], before: dict[str, np.ndarray]) -> dict[str, Any]:
    deltas = [(after[n] - before[n]).reshape(-1) for n in sorted(after)]
    return {
        "changed_entries": sum(int(np.count_nonzero(d)) for d in deltas),
        "l2_delta": math.sqrt(math.fsum(float(x) ** 2 for d in deltas for x in d)),
    }


def test_the_verdict_records_the_candidates_and_the_cast_controls_figures(ev: Ev) -> None:
    """FR-016.27, FTDD 6.4 (R1B-A8; review 016-P2, R16P2-4): the verdict records the candidate's
    changed count and delta norm and the CAST control's actual ones, which rounding to the base
    dtype can move. Both are recomputed from the tensors the scorer was sent and the bundle's own
    file."""
    from wsp.schemas import validate_run

    cid, sha = ev.candidate(delta=0.1)
    fake = scorer(ev)
    verdict = ev.evaluate(cid, sha, fake)
    _validate_verdict(verdict)
    heads = {h["label"]: _sent_values(h) for h in fake.requests[0]["heads"] if "tensors" in h}
    stored = read_safetensors(ev.tmp_path / "bundle" / "heads.safetensors")
    base = {name: tensor.values64() for name, tensor in stored.items()}
    want_candidate = _delta_figures(heads["candidate"], base)
    want_control = _delta_figures(heads["control"], base)
    got = verdict["control_figures"]
    assert set(got) == {"candidate", "control_post_cast"}
    assert got["candidate"]["changed_entries"] == want_candidate["changed_entries"] == 1
    assert got["candidate"]["l2_delta"] == pytest.approx(want_candidate["l2_delta"], rel=1e-12)
    assert got["control_post_cast"]["changed_entries"] == want_control["changed_entries"]
    assert got["control_post_cast"]["l2_delta"] == pytest.approx(
        want_control["l2_delta"], rel=1e-12
    )
    client, _ = _client(ev)
    document = client.get(f"/api/v1/wsp/candidates/{cid}/run-document").json()
    validate_run(document)
    assert document["verdict"]["control_figures"] == got


def test_the_verdict_records_the_cast_controls_own_figures_not_the_candidates(
    ev: Ev, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring half: this fixture's one-entry bias rarely lets the cast move the control's
    figures, so the control is replaced by one whose post-cast figures differ from the candidate's;
    the verdict must carry the control's, in its own field."""
    import dataclasses

    from wsp.weight_ops import control as control_module

    real = control_module.random_control
    moved = control_module.Figures(changed_entries=7, l2_delta=0.75)

    def with_moved_figures(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.replace(real(*args, **kwargs), post_cast=moved)

    monkeypatch.setattr(control_module, "random_control", with_moved_figures)
    cid, sha = ev.candidate(delta=0.1)
    verdict = ev.evaluate(cid, sha, scorer(ev))
    _validate_verdict(verdict)
    figures = verdict["control_figures"]
    assert figures["control_post_cast"] == {"changed_entries": 7, "l2_delta": 0.75}
    assert figures["candidate"]["changed_entries"] == 1
    assert figures["candidate"]["l2_delta"] == pytest.approx(0.1, rel=1e-6)


def test_the_evaluate_route_ends_its_transaction_before_answering() -> None:
    """The wiring, read from the LIVE app: evaluate's session dependency is function-scoped."""
    from api.app import create_app
    from api.deps import session_dependency
    from api.route_table import declared_route, mounted_api_routes

    route = next(
        r
        for r in mounted_api_routes(create_app())
        if r.path == "/api/v1/wsp/candidates/{candidate_id}/evaluate"
    )
    scopes = [
        d.scope
        for d in declared_route(route).dependant.dependencies
        if d.call is session_dependency
    ]
    assert scopes == ["function"]


# ------------------------------------------------------------------ review 016-P2 (18.4, r016p2)
#
# Each test below was written for a mutation of ``evaluate.py`` that the whole WSP tree let pass
# (0xcc/reviews/016_P2_review.md): a contract check or a metric-guard comparison removed, or the
# per-head digest computed from the candidate's scores for every head.


@pytest.mark.parametrize(
    "break_",
    ["bar_missing", "role", "set_twice", "short_scores"],
)
def test_more_answers_that_are_not_for_this_request_are_unmeasured(ev: Ev, break_: str) -> None:
    """Controls E7 (a head without a bar), E8 (a set's role is not the bundle's), E9 (one set
    answered twice) and E10 (a head with fewer scores than labels): each is
    ``scorer_contract_invalid``, never a verdict and never a crash."""
    import copy

    cid, sha = ev.candidate()

    def tamper(answer: dict[str, Any]) -> None:
        first = answer["sets"][0]
        if break_ == "bar_missing":
            first["bars"].pop("control")
        elif break_ == "role":
            first["role"] = "everyday_text"
        elif break_ == "set_twice":
            answer["sets"].append(copy.deepcopy(first))
        else:
            first["scores"]["control"] = first["scores"]["control"][:-1]

    verdict = ev.evaluate(cid, sha, scorer(ev, tamper=tamper, validate_answer=False))
    assert verdict["value"] == "unmeasured"
    assert verdict["reasons"][0] == "scorer_contract_invalid", verdict["reasons"]


@pytest.mark.parametrize("field_", ["positives", "negatives"])
def test_the_metric_guard_checks_the_class_counts(ev: Ev, field_: str) -> None:
    """Controls E11, E12: a head whose returned ``positives`` or ``negatives`` is one off is a
    metric disagreement, like an off-by-one recall."""
    cid, sha = ev.candidate()

    def tamper(answer: dict[str, Any]) -> None:
        answer["sets"][0]["metrics"]["control"][field_] += 1

    verdict = ev.evaluate(cid, sha, scorer(ev, tamper=tamper, validate_answer=False))
    assert verdict["value"] == "unmeasured"
    assert verdict["reasons"][0] == "scorer_metric_disagreement", verdict["reasons"]


def test_every_heads_scores_sha256_is_its_own_scores(ev: Ev) -> None:
    """P23 / FR-016.34 (control E5): ``scores_sha256`` is per head AND per set, each the sha256
    of that head's own float32 little-endian scores as the scorer returned them; and every head
    sent is reported, the control and all five baselines with them (18.3)."""

    cid, sha = ev.candidate()
    answers: list[dict[str, Any]] = []
    verdict = ev.evaluate(cid, sha, scorer(ev, tamper=answers.append))
    assert verdict["value"] == "improved", verdict["reasons"]
    by_name = {s["name"]: s for s in answers[0]["sets"]}
    for entry in verdict["per_set"]:
        returned = by_name[entry["name"]]["scores"]
        assert [h["label"] for h in entry["heads"]] == sorted(returned)
        for head in entry["heads"]:
            raw = np.asarray(returned[head["label"]], dtype="<f4").tobytes()
            assert head["scores_sha256"] == hashlib.sha256(raw).hexdigest(), head["label"]
    # On ``eval`` the fixture gives the candidate, the parent and the control their own draws
    # (the operating-point baselines share the parent's; the false-alarm sets share one draw).
    eval_heads = {h["label"]: h["scores_sha256"] for h in verdict["per_set"][0]["heads"]}
    assert len({eval_heads[k] for k in ("candidate", "parent", "control")}) == 3
    labels = {h["label"] for h in verdict["per_set"][0]["heads"]}
    assert labels == {
        "parent",
        "candidate",
        "control",
        "baseline:logistic_regression",
        "baseline:lda",
        "baseline:diff_of_means",
        "baseline:threshold_recut",
        "baseline:affine_recalibration",
    }
