"""018 FU-2 through the real intake and ``run_document`` (operator, 2026-11-25; the run-contract
revision ``0xcc/reviews/run_contract_r2_2026-11-25.md``).

The transcript is a REAL engine session's (``test_engine_scripted.session``, the child preamble
against a scripted endpoint), pulled from a fake worker by the real driver and taken in by the
real intake, so the worker's line, the intake's read and the document's field are one path:

- a session whose submit stated false figures is stored with ``submitted_check.differs`` true, and
  its run document carries the check and validates against the contract;
- a transcript with no ``claims`` (a run from before FU-2) stores nothing and its document has no
  ``submitted_check``, so every earlier document reads, and hashes, as it did;
- a malformed check refuses the result ``wsp_result_invalid`` with nothing stored.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.integration.wsp.test_dispatch import FakeWorker as _FakeWorker
from tests.integration.wsp.test_dispatch import World, _j, _running, archive, build_world
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import call, session
from wsp import runs, schemas
from wsp.models import WspBundle
from wsp.settings import get_wsp_settings

V4 = [
    {"name": "practice_metrics", "version": "4", "cost": 1},
    {"name": "compare_to_parent", "version": "4", "cost": 2},
]
FALSE = {"split": "validation", "auroc": 0.9999, "recall": 0.999}


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "engine-bundle", rule="mean")


def _engine_transcript(tmp: Path, fx: FixtureBundle, **submit: Any) -> bytes:
    w = fx.tensors["weight"].values64().astype(float).tolist()
    w[0] += 1.0
    steps = [
        {"calls": [("write_tensor", {"name": "weight", "values": w})]},
        call("submit", answer="better", **submit),
    ]
    _, run, _ = session(tmp, fx, "repair", steps, observation_tools=V4)
    return (run / "out" / "transcript.jsonl").read_bytes()


def _take_in(world: World, transcript: bytes) -> tuple[str, list[str]]:
    run_id, worker = _running(world)
    worker.state = "finished"
    result = {"stop_reason": "submitted"}
    worker.data = archive(
        worker.boot,
        run_id,
        **{"result.json": json.dumps(result).encode(), "transcript.jsonl": transcript},
    )
    return run_id, world.drive(worker)


def _document(world: World, run_id: str) -> dict[str, Any]:
    with world.factory() as session_:
        row = runs.get_run(session_, run_id)
        bundle = session_.get(WspBundle, row.bundle_sha256)
        return runs.run_document(row, None if bundle is None else bundle.manifest)


def test_false_figures_reach_the_run_document_flagged(
    world: World, fx: FixtureBundle, tmp_path: Path
) -> None:
    transcript = _engine_transcript(tmp_path / "s", fx, figures=FALSE)
    line = [json.loads(x) for x in transcript.decode().splitlines()][-1]
    assert line["event"] == "submitted_measured" and line["claims"]["differs"] is True
    run_id, actions = _take_in(world, transcript)
    assert actions[0] == f"collected:{run_id}:completed", actions
    stored = _j(world.run(run_id)["submitted_check"])
    assert stored == {**line["claims"], "measured": True, "split": "validation"}
    document = _document(world, run_id)
    assert document["submitted_check"] == stored
    assert document["submitted_check"]["differs"] is True
    schemas.validate_run(document)


def test_measured_figures_stated_back_read_as_matching(
    world: World, fx: FixtureBundle, tmp_path: Path
) -> None:
    first = _engine_transcript(tmp_path / "a", fx)
    now = [json.loads(x) for x in first.decode().splitlines()][-1]["now"]
    claimed = {"split": "validation", "auroc": now["auroc"], "false_alarms": now["false_alarms"]}
    run_id, _ = _take_in(world, _engine_transcript(tmp_path / "b", fx, figures=claimed))
    check = _document(world, run_id)["submitted_check"]
    assert check["compared"] is True and check["differs"] is False
    assert [c["figure"] for c in check["checks"]] == ["auroc", "false_alarms"]


def test_a_transcript_without_claims_stores_nothing_and_the_document_is_unchanged(
    world: World, fx: FixtureBundle, tmp_path: Path
) -> None:
    lines = [json.loads(x) for x in _engine_transcript(tmp_path / "s", fx).decode().splitlines()]
    old = [{k: v for k, v in o.items() if k != "claims"} for o in lines]
    run_id, actions = _take_in(world, "".join(json.dumps(o) + "\n" for o in old).encode())
    assert actions[0] == f"collected:{run_id}:completed", actions
    assert world.run(run_id)["submitted_check"] is None
    document = _document(world, run_id)
    assert "submitted_check" not in document
    schemas.validate_run(document)


@pytest.mark.parametrize(
    "claims",
    [
        {"stated": "yes", "stated_split": None, "compared": False, "differs": None, "checks": []},
        {"stated": False, "stated_split": None, "compared": False, "differs": None},
        {
            "stated": True,
            "stated_split": "validation",
            "compared": True,
            "differs": False,
            "checks": [{"figure": "f1", "claimed": 1, "measured": 1, "matches": True}],
        },
        [],
    ],
    ids=["bool-as-text", "no-checks", "unknown-figure", "not-an-object"],
)
def test_a_malformed_check_refuses_the_result(world: World, claims: Any) -> None:
    line = {"event": "submitted_measured", "measured": False, "reason": "no_time", "claims": claims}
    run_id, actions = _take_in(world, (json.dumps(line) + "\n").encode())
    assert actions[0] == f"intake_refused:{run_id}", actions
    run = world.run(run_id)
    assert run["status"] == "failed" and _j(run["failure"])["code"] == "wsp_result_invalid"
    assert run["submitted_check"] is None


def test_the_last_line_is_the_one_kept(world: World) -> None:
    def line(differs: bool | None) -> dict[str, Any]:
        return {
            "event": "submitted_measured",
            "measured": False,
            "reason": "no_time",
            "claims": {
                "stated": False,
                "stated_split": None,
                "compared": False,
                "differs": differs,
                "checks": [],
            },
        }

    text = json.dumps(line(True)) + "\n" + json.dumps(line(None)) + "\n"
    run_id, _ = _take_in(world, text.encode())
    assert _j(world.run(run_id)["submitted_check"])["differs"] is None


# ------------------------------------------ an inspect that submitted before observing (2026-11-27)


def _inspect_output(
    tmp: Path, fx: FixtureBundle, steps: list[dict[str, Any]], **over: Any
) -> dict[str, bytes]:
    """A REAL inspect session's ``result.json`` and transcript (the engine against a scripted
    endpoint), as the worker would archive them."""
    _, run, _ = session(tmp, fx, "inspect", steps, observation_tools=V4, **over)
    out = run / "out"
    return {
        "result.json": (out / "result.json").read_bytes(),
        "transcript.jsonl": (out / "transcript.jsonl").read_bytes(),
    }


def _take_in_inspect(world: World, members: dict[str, bytes]) -> tuple[str, list[str]]:
    run_id = world.queue(kind="inspect")
    worker = _FakeWorker()
    world.drive(worker)
    worker.state = "finished"
    worker.data = archive(worker.boot, run_id, **members)
    return run_id, world.drive(worker)


def test_an_inspect_that_submits_on_its_first_turn_reads_as_not_measured(
    world: World, fx: FixtureBundle, tmp_path: Path
) -> None:
    """Run ``wsp_0971152b6843``: the 1.2B CPU editor submitted a confident report on its first
    turn, having called no observation tool. Its document now carries ``submitted_check`` with
    ``measured`` false (nothing stated, nothing compared), and still validates."""
    members = _inspect_output(
        tmp_path / "s",
        fx,
        [call("submit", answer="The network is ready; no changes are needed.", confidence=0.95)],
    )
    result = json.loads(members["result.json"])
    assert (result["stop_reason"], result["forward_used"]) == ("submitted", 0)
    run_id, actions = _take_in_inspect(world, members)
    assert actions[0] == f"collected:{run_id}:completed", actions
    row = world.run(run_id)
    assert (row["outcome"], row["stop_reason"], row["forward_passes_used"]) == (
        "report",
        "submitted",
        0,
    )
    assert _j(row["submitted_check"]) == runs.UNOBSERVED_CHECK
    document = _document(world, run_id)
    assert document["submitted_check"]["measured"] is False
    assert document["submitted_check"] == {
        "checks": [],
        "compared": False,
        "differs": None,
        "measured": False,
        "split": None,
        "stated": False,
        "stated_split": None,
    }
    schemas.validate_run(document)


def test_an_inspect_that_observed_first_has_no_check(
    world: World, fx: FixtureBundle, tmp_path: Path
) -> None:
    """One forward pass before the submit: the answer rests on a measurement, so nothing is
    added and its document reads as before."""
    members = _inspect_output(
        tmp_path / "s",
        fx,
        [call("forward_practice_metrics"), call("submit", answer="auroc read", confidence=0.6)],
        forward_passes=2,
    )
    assert json.loads(members["result.json"])["forward_used"] >= 1
    run_id, actions = _take_in_inspect(world, members)
    assert actions[0] == f"collected:{run_id}:completed", actions
    assert world.run(run_id)["submitted_check"] is None
    assert "submitted_check" not in _document(world, run_id)


@pytest.mark.parametrize(
    ("kind", "result"),
    [
        ("inspect", {"stop_reason": "submitted"}),  # no count: unknown is not zero
        ("inspect", {"stop_reason": "max_steps", "forward_used": 0}),  # never submitted
        ("repair", {"stop_reason": "submitted", "forward_used": 0}),  # an edit run: FU-2's own
    ],
    ids=["no-count", "not-submitted", "repair"],
)
def test_only_a_submitted_inspect_with_zero_passes_is_marked(
    world: World, kind: str, result: dict[str, Any]
) -> None:
    run_id = world.queue(kind=kind)
    worker = _FakeWorker()
    world.drive(worker)
    worker.state = "finished"
    worker.data = archive(
        worker.boot,
        run_id,
        **{"result.json": json.dumps(result).encode(), "transcript.jsonl": b"\n"},
    )
    actions = world.drive(worker)
    assert actions[0].startswith(f"collected:{run_id}:"), actions
    assert world.run(run_id)["submitted_check"] is None
