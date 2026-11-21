"""``repair`` and ``create`` on a linear-probe bundle, END TO END (feature 016 phase 2; FTASKS 16.1;
FR-016.31).

Per operation (create, repair) and per style (session, program): the REAL driver pushes to the
REAL supervisor (``LocalProcessLauncher`` under the worker's interpreter, ``MIFORGE_WSP_WORKER_
PYTHON``), whose child runs the REAL weightspace adapter against a scripted OpenAI-compatible
editor standing where the gateway stands; the real intake qualifies the patch; ``evaluate`` scores
the candidate through the fake scorer to a verdict. Nothing between the start body and the
verdict is a test double except the editor's replies and the scorer.

Skips by name without the worker interpreter; fails under ``MIFORGE_REQUIRE_WSP_WORKER=1``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from tests.integration.wsp.test_dispatch import World, _j, build_world
from tests.integration.wsp.test_runs_lifecycle import _store
from tests.support import wsp_worker_env
from tests.support.fake_miworkbench import FakeScorer, evaluate_heldout_sets, known_truth
from tests.support.wsp_scripted_endpoint import OPENAI, ScriptedEndpoint
from wsp import dispatch, evaluate, tasks
from wsp.settings import get_wsp_settings

#: The program both program-style runs submit: the weight at twice its scale, which keeps every
#: ranking, so the practice AUROC is the parent's and the run qualifies (FR-016.30).
PROGRAM = (
    "def edit(base, support):\n"
    "    w = np.array(base['weight'])\n"
    "    return {'weight': (w * np.float32(2.0)).astype(w.dtype)}\n"
)


def steps(kind: str, mode: str) -> list[dict[str, Any]]:
    if mode == "program":
        return [
            {"calls": [("test_program", {"code": PROGRAM})]},
            {"calls": [("submit_program", {"code": PROGRAM})]},
            {"calls": [("submit", {"answer": "programmed", "confidence": 0.6})]},
        ]
    if kind == "create":
        # From zeroed tensors: write the probe back at twice its scale (the same ranking).
        code = "W['weight'][:] = R['weight'] * 2.0; W['bias'][:] = R['bias'] * 2.0"
    else:
        code = "W['bias'][0] += 0.25"
    return [
        {"calls": [("python", {"code": code})]},
        {"calls": [("submit", {"answer": "done", "confidence": 0.7})]},
    ]


@pytest.fixture()
def e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    wsp_worker_env.require_seccomp()
    monkeypatch.setenv("WSP_BOOTSTRAP_B", "200")
    world = build_world(
        tmp_path,
        monkeypatch,
        bundle_id="wsb_0123456789ab",
        heldout_sets=evaluate_heldout_sets(),
        labels_from_scores=True,
    )
    _store(world, tmp_path)
    yield world
    world.engine.dispose()
    get_wsp_settings.cache_clear()


def _drive_real(world: World) -> list[str]:
    from wsp.launcher import LocalProcessLauncher
    from wsp.worker_client import HttpWorker

    launcher = LocalProcessLauncher(environment="ci", python=wsp_worker_env.worker_python())
    with launcher.session() as local:
        client = HttpWorker(local.url)
        try:
            get_wsp_settings.cache_clear()
            world.settings = get_wsp_settings()

            def once() -> list[str]:
                return dispatch.drive(
                    world.factory,
                    client,
                    world.settings,
                    read_sensitivity=world.read,
                    files_for=tasks.run_files,
                ).actions

            actions = once()
            assert actions and actions[-1].startswith("pushed:"), actions
            deadline = time.monotonic() + 120
            while client.status()["state"] != "finished":
                assert time.monotonic() < deadline, "the session never finished"
                time.sleep(0.2)
            actions += once()
        finally:
            client.close()
        assert local.process.wait(timeout=30) == 0
    return actions


@pytest.mark.parametrize("mode", ["session", "program"])
@pytest.mark.parametrize("kind", ["repair", "create"])
def test_a_run_goes_from_its_start_body_to_a_verdict(
    e2e: World, monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    with ScriptedEndpoint(OPENAI, steps(kind, mode)) as editor:
        monkeypatch.setenv("WSP_GATEWAY_URL", f"http://127.0.0.1:{editor.gateway['port']}")
        get_wsp_settings.cache_clear()
        e2e.settings = get_wsp_settings()
        e2e.body = {**e2e.body, "mode": mode, "budgets": {"tokens": 100_000, "time_s": 120}}
        run_id = e2e.queue(kind=kind)
        actions = _drive_real(e2e)
    assert f"collected:{run_id}:completed" in actions, actions
    run = e2e.run(run_id)
    assert (run["status"], run["outcome"]) == ("completed", "candidate"), run["failure"]
    assert editor.requests, "the editor was never called"
    with e2e.engine.connect() as conn:
        cand = conn.execute(
            sa.text("SELECT candidate_id, parent_kind FROM mf_wsp_candidates WHERE run_id = :r"),
            {"r": run_id},
        ).one()
    expected_parent = "bundle_zeroed" if (kind, mode) == ("create", "session") else "bundle"
    assert cand[1] == expected_parent
    with e2e.engine.connect() as conn:
        bundle_manifest = _j(
            conn.execute(sa.text("SELECT manifest FROM mf_wsp_bundles")).scalar_one()
        )
    fake = FakeScorer(bundle_manifest["heldout_ref"], known_truth(2.5))
    settings = get_wsp_settings().model_copy(
        update={"miworkbench_api_url": "http://miworkbench.test"}
    )
    with e2e.factory() as session:
        verdict = evaluate.evaluate(
            session, cand[0], run["request_sha256"], settings=settings, transport=fake.transport()
        )
        session.commit()
    assert verdict["value"] == "improved", verdict["reasons"]
    assert fake.calls == 1 and fake.invalid_requests == []
    if mode == "program":
        assert run["program_sha256"] is not None
