"""miWorkbench's REAL scorer code against miForge's metric guard and client (feature 016 phase 2;
FTASKS 13.4, 16.4a; FR-016.26, FR-016.33a; P29, P64, P65).

The fake scorer agrees with miForge's guard only if both are right; these tests take the other
side from miWorkbench's own modules (``tests/support/wsp_children/miworkbench_real.py`` loads
them from the sibling checkout, untouched, in a child process):

- ``services.wsp_score.metrics_of`` scores a fixture's float32 scores at their bars; miForge's
  guard accepts exactly those metrics, and a one-row label flip makes it refuse;
- ``services.wsp_patch.jcs_sha256`` is the ``heldout_ref_sha256`` miForge expects;
- the refusal bodies miWorkbench's route writes are what miForge's client dispatches on (found
  2026-11-19: the client read the wrong envelope, so ``scorer_busy`` read as unreachable);
- ``services.wsp_import.run_sha256_of`` reproduces the ``run_sha256`` of a snapshot miForge
  served under a ``verdict_id`` taken from ``GET …/verdict``, and the document validates.

Skips loudly without the sibling; fails under ``MIFORGE_REQUIRE_CROSS_REPO_CHECKS=1``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from tests.support import siblings

CHILD = Path(__file__).resolve().parents[1] / "support" / "wsp_children" / "miworkbench_real.py"


def _backend() -> Path:
    located = siblings.locate(siblings.MIWORKBENCH)
    backend = located.path / "backend"
    if not (backend / "src" / "services" / "wsp_score.py").is_file():
        siblings.absent(
            f"miWorkbench (with 044's scorer) is not checked out at {located.path} "
            f"(tried {', '.join(located.tried)}); cannot run its real scoring code (P29)"
        )
    return backend


def real(case: str, payload: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """One case of miWorkbench's real code, in a child process of this interpreter."""
    data = tmp_path / "miworkbench-data"
    env = {
        "PATH": "/usr/bin:/bin",
        # miWorkbench's settings object is built at import; none of these is ever connected to.
        "DATABASE_URL": "postgresql://unused:unused@127.0.0.1:9/unused",
        "REDIS_URL": "redis://127.0.0.1:9/0",
        "CELERY_BROKER_URL": "redis://127.0.0.1:9/0",
        "CELERY_RESULT_BACKEND": "redis://127.0.0.1:9/0",
        "SECRET_KEY": "cross-repo-test-only-not-a-secret-0000",
        "DATA_DIR": str(data),
        "HF_CACHE_DIR": str(data / "hf"),
        "HF_HOME": str(data / "hf"),
    }
    done = subprocess.run(
        [sys.executable, "-I", str(CHILD), str(_backend()), case],
        input=json.dumps(payload),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    loaded: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return loaded


def _fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """A heldout_ref, a request naming its heads, and per-set scores with ties and a None bar."""
    from tests.support.wsp_fixtures import row_ids_sha256

    rng = np.random.default_rng(3)
    heldout_ref = {
        "owner": "miworkbench",
        "sets": [
            {
                "name": n,
                "view_id": f"view-{n}",
                "role": r,
                "rows": rows,
                "row_ids_sha256": row_ids_sha256(f"view-{n}", list(range(rows))),
            }
            for n, r, rows in (("eval", "eval", 60), ("everyday", "everyday_text", 80))
        ],
    }
    labels = {"eval": [1] * 30 + [0] * 30, "everyday": [0] * 80}
    heads = ("parent", "candidate", "control")
    sets: dict[str, Any] = {}
    for name, ys in labels.items():
        y = np.asarray(ys)
        scores = {}
        for i, head in enumerate(heads):
            s = (rng.normal(size=y.size) + i * y).astype(np.float32)
            s[::7] = np.float32(0.5)  # ties, some exactly at the bar
            scores[head] = [float(v) for v in s]
        thresholds = {"parent": 0.5, "candidate": 0.5, "control": None}
        sets[name] = {"labels": ys, "scores": scores, "thresholds": thresholds}
    request = {
        "bundle_sha256": "b" * 64,
        "heldout_ref": heldout_ref,
        "heads": [{"label": h} for h in heads],
    }
    return request, {"heldout_ref": heldout_ref, "sets": sets}


def _answer(request: dict[str, Any], plan: dict[str, Any], real_out: dict[str, Any]) -> dict:
    sets = []
    for entry in request["heldout_ref"]["sets"]:
        one = plan["sets"][entry["name"]]
        sets.append(
            {
                "name": entry["name"],
                "role": entry["role"],
                "row_ids_sha256": entry["row_ids_sha256"],
                "labels": list(one["labels"]),
                "scores": one["scores"],
                "metrics": real_out["metrics"][entry["name"]],
                # As miWorkbench's wsp_score writes it for a set with no truncated row: one
                # count per head answered (F-2: contract_problem checks the map).
                "excluded": {
                    "truncated": 0,
                    "truncated_fired": dict.fromkeys(one["thresholds"], 0),
                },
                "bars": {
                    label: {"threshold": t, "source": "calibration"}
                    for label, t in one["thresholds"].items()
                },
            }
        )
    return {
        "bundle_sha256": request["bundle_sha256"],
        "heldout_ref_sha256": real_out["heldout_ref_sha256"],
        "sets": sets,
    }


def test_the_guard_accepts_miworkbenchs_real_metrics_and_refuses_a_flipped_label(
    tmp_path: Path,
) -> None:
    from wsp.evaluate import contract_problem, metrics_disagree

    request, plan = _fixture()
    out = real("metrics", plan, tmp_path)
    answer = _answer(request, plan, out)
    assert contract_problem(answer, request) is None  # its heldout digest is miForge's
    assert metrics_disagree(answer) is None
    flipped = json.loads(json.dumps(answer))
    flipped["sets"][0]["labels"][0] = 1 - flipped["sets"][0]["labels"][0]
    found = metrics_disagree(flipped)
    assert found is not None and found.startswith("eval:"), found


def test_miworkbenchs_real_refusal_bodies_dispatch_on_their_code(tmp_path: Path) -> None:
    """P64, R3B-9: the 409 budget body names its set; scorer_busy reads as busy at any status."""
    from clients.miworkbench import ScoreRefused, refusal_of, score, scorer_client
    from tests.unit.wsp.test_scorer_client import REQUEST

    bodies = real(
        "refusals",
        {
            "refusals": {
                "heldout_set_head_budget_exhausted": {
                    "set": "eval",
                    "row_ids_sha256": "c" * 64,
                    "rows_exhausted": 4,
                },
                "scorer_busy": {},
                "capture_not_ready": {},
            }
        },
        tmp_path,
    )
    for code, served in bodies.items():
        for status in (served["status"], 409, 429):

            def handler(request: httpx.Request, body: Any = served["body"], s: int = status) -> Any:
                return httpx.Response(s, json=body)

            client = scorer_client("http://miworkbench.test", httpx.MockTransport(handler))
            with pytest.raises(ScoreRefused) as caught:
                score(client, REQUEST)
            assert caught.value.code == code, (code, status, served["body"])
    budget = refusal_of(bodies["heldout_set_head_budget_exhausted"]["body"])
    assert budget is not None and budget["set"] == "eval" and budget["rows_exhausted"] == 4
    assert bodies["heldout_set_head_budget_exhausted"]["status"] == 409
    assert bodies["scorer_busy"]["status"] == 429


def test_miworkbench_reproduces_run_sha256_from_a_served_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-016.33a (R3B-10): the id from ``GET …/verdict``, the snapshot fetched with it, and
    miWorkbench 044's own recomputation and schema check."""
    _backend()
    from tests.integration.wsp.test_dispatch import build_world
    from tests.integration.wsp.test_evaluate import BUNDLE_ID, Ev, _client, scorer
    from tests.integration.wsp.test_runs_lifecycle import _store
    from tests.support.fake_miworkbench import evaluate_heldout_sets
    from wsp.settings import get_wsp_settings

    monkeypatch.setenv("WSP_BOOTSTRAP_B", "200")
    world = build_world(
        tmp_path,
        monkeypatch,
        bundle_id=BUNDLE_ID,
        heldout_sets=evaluate_heldout_sets(),
        labels_from_scores=True,
    )
    try:
        ev = Ev(world, _store(world, tmp_path), tmp_path)
        cid, sha = ev.candidate()
        ev.evaluate(cid, sha, scorer(ev))
        client, _ = _client(ev)
        verdict_id = client.get(f"/api/v1/wsp/candidates/{cid}/verdict").json()["verdict_id"]
        served = client.get(
            f"/api/v1/wsp/candidates/{cid}/run-document", params={"verdict_id": verdict_id}
        )
        assert served.status_code == 200
        snapshot = json.loads(served.content)
        header = served.headers["X-Content-SHA256"]
        assert real("run_sha256", {"document": snapshot}, tmp_path)["run_sha256"] == header
        # miWorkbench's importer validates a whole document: the two fields the snapshot leaves
        # out (run_sha256 itself, and program_reruns) are put back, and the digest still holds.
        whole = {**snapshot, "run_sha256": header, "program_reruns": []}
        checked = real("run_sha256", {"document": whole}, tmp_path)
        assert checked == {"run_sha256": header, "invalid": None}
        assert snapshot["verdict"]["verdict_id"] == verdict_id
    finally:
        world.engine.dispose()
        get_wsp_settings.cache_clear()
