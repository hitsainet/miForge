"""``api.v1.endpoints.wsp_candidates``'s routes answer as the contract says (021 FR-021.13).

Every candidate is made by the REAL driver and intake (``test_evaluate``'s ``ev`` world), and its
verdict by ``wsp.evaluate`` against the contract-checking fake scorer; the evaluate route itself
builds its production client from ``MIWORKBENCH_API_URL`` pointed at that scorer served over HTTP
on ``127.0.0.1``. The two byte routes the contract excludes (``tensors``, ``verdict/scores``) are
checked by their declared media type.
"""

from __future__ import annotations

import pytest

from tests.integration.wsp.test_evaluate import Ev, _client, scorer
from tests.integration.wsp.test_evaluate import ev as ev
from tests.support.answer_cases import answered, assert_covered, case
from wsp.settings import get_wsp_settings

ROUTER = "api.v1.endpoints.wsp_candidates"


def _scored(ev: Ev) -> tuple[str, dict]:
    """A candidate with a stored verdict (``improved``), and that verdict."""
    cid, sha = ev.candidate()
    return cid, ev.evaluate(cid, sha, scorer(ev))


@case(ROUTER, "GET /api/v1/wsp/candidates")
def test_list_candidates(ev: Ev) -> None:
    """One candidate with a verdict and one without."""
    _scored(ev)
    ev.candidate(delta=0.5)
    client, _ = _client(ev)
    listed = answered("GET /api/v1/wsp/candidates", client.get("/api/v1/wsp/candidates"))
    assert len(listed["candidates"]) == 2


@case(ROUTER, "GET /api/v1/wsp/candidates/{candidate_id}")
def test_get_candidate(ev: Ev) -> None:
    cid, _ = _scored(ev)
    client, _ = _client(ev)
    got = answered(
        "GET /api/v1/wsp/candidates/{candidate_id}", client.get(f"/api/v1/wsp/candidates/{cid}")
    )
    assert got["verdict"] is not None


@case(ROUTER, "GET /api/v1/wsp/candidates/{candidate_id}/lineage")
def test_get_lineage(ev: Ev) -> None:
    cid, _ = _scored(ev)
    client, _ = _client(ev)
    lineage = answered(
        "GET /api/v1/wsp/candidates/{candidate_id}/lineage",
        client.get(f"/api/v1/wsp/candidates/{cid}/lineage"),
    )
    assert lineage["steps"][-1]["verdict"] is not None


@case(ROUTER, "GET /api/v1/wsp/candidates/{candidate_id}/patch/{name}")
def test_get_patch_file(ev: Ev) -> None:
    """Both files: ``patch.json`` (the JSON answer) and ``edit.npz`` (its declared byte
    alternative)."""
    cid, _ = ev.candidate()
    client, _ = _client(ev)
    key = "GET /api/v1/wsp/candidates/{candidate_id}/patch/{name}"
    answered(key, client.get(f"/api/v1/wsp/candidates/{cid}/patch/patch.json"))
    answered(key, client.get(f"/api/v1/wsp/candidates/{cid}/patch/edit.npz"))


@case(ROUTER, "GET /api/v1/wsp/candidates/{candidate_id}/tensors")
def test_get_tensors(ev: Ev) -> None:
    cid, _ = ev.candidate()
    client, _ = _client(ev)
    answered(
        "GET /api/v1/wsp/candidates/{candidate_id}/tensors",
        client.get(f"/api/v1/wsp/candidates/{cid}/tensors"),
    )


@case(ROUTER, "GET /api/v1/wsp/candidates/{candidate_id}/run-document")
def test_get_run_document(ev: Ev) -> None:
    """The live document before a verdict, after one, and the stored snapshot under its
    ``verdict_id``."""
    cid, sha = ev.candidate()
    client, _ = _client(ev)
    key = "GET /api/v1/wsp/candidates/{candidate_id}/run-document"
    path = f"/api/v1/wsp/candidates/{cid}/run-document"
    assert answered(key, client.get(path))["verdict"] is None
    verdict = ev.evaluate(cid, sha, scorer(ev))
    assert answered(key, client.get(path))["verdict"] is not None
    answered(key, client.get(path, params={"verdict_id": verdict["verdict_id"]}))


@case(ROUTER, "GET /api/v1/wsp/candidates/{candidate_id}/verdict")
def test_get_verdict(ev: Ev) -> None:
    cid, _ = _scored(ev)
    client, _ = _client(ev)
    answered(
        "GET /api/v1/wsp/candidates/{candidate_id}/verdict",
        client.get(f"/api/v1/wsp/candidates/{cid}/verdict"),
    )


@case(ROUTER, "GET /api/v1/wsp/candidates/{candidate_id}/verdict/scores")
def test_get_verdict_scores(ev: Ev) -> None:
    cid, _ = _scored(ev)
    client, _ = _client(ev)
    answered(
        "GET /api/v1/wsp/candidates/{candidate_id}/verdict/scores",
        client.get(f"/api/v1/wsp/candidates/{cid}/verdict/scores", params={"set": "eval"}),
    )


@case(ROUTER, "GET /api/v1/wsp/heldout-uses")
def test_list_heldout_uses(ev: Ev) -> None:
    _scored(ev)
    client, _ = _client(ev)
    uses = answered("GET /api/v1/wsp/heldout-uses", client.get("/api/v1/wsp/heldout-uses"))
    assert uses["heldout_uses"]


@case(ROUTER, "POST /api/v1/wsp/candidates/{candidate_id}/evaluate")
def test_evaluate_candidate(ev: Ev, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the route's own production client, against the scorer served over HTTP."""
    cid, sha = ev.candidate()
    fake = scorer(ev)
    with fake.serve() as url:
        monkeypatch.setenv("MIWORKBENCH_API_URL", url)
        get_wsp_settings.cache_clear()
        client, _ = _client(ev)
        verdict = answered(
            "POST /api/v1/wsp/candidates/{candidate_id}/evaluate",
            client.post(f"/api/v1/wsp/candidates/{cid}/evaluate", json={"request_sha256": sha}),
        )
    assert verdict["value"] == "improved" and fake.calls == 1


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
