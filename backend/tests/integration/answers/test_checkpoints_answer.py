"""``api.v1.endpoints.checkpoints``'s routes answer as the contract says (021 FR-021.13).

Checkpoints are written through the real writer (``tests.support.checkpoints``) under the
approval world's checkpoint root and indexed by the real ``reindex``. ``POST
/runs/{run_id}/continue`` is gated (``hotupdate_attach``, when it submits) and answers three
ways: the plan (200) and an agent's pending submit (202) on a load-mode run here, and the
submitted continue (202) on the attached run ``test_hotupdate_resume.py`` builds - the real
launcher, the real validator and the real attach pre-flight against the fake miServer surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.integration.test_hotupdate_resume import RUN as ATTACHED_RUN
from tests.integration.test_hotupdate_resume import _submit
from tests.integration.test_hotupdate_resume import world as attached_world  # noqa: F401
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World
from tests.support.checkpoints import write_chain

ROUTER = "api.v1.endpoints.checkpoints"
RUN = "run-a"


def _chain(world: World, steps: tuple[int, ...] = (100, 200), *, status: str = "failed") -> Path:
    """A run row and its saves, each the previous one's child, indexed."""
    from services.reindex import reindex
    from tests.support.jobs import insert_run

    root = world.root / "checkpoints"
    root.mkdir(exist_ok=True)
    with world.session() as session:
        insert_run(session, RUN, status=status, last_step=max(steps))
        session.commit()
    write_chain(root, run_id=RUN, steps=steps)
    with world.session() as session:
        reindex(session, root=root)
        session.commit()
    return root


@case(ROUTER, "POST /api/v1/runs/{run_id}/checkpoints")
def test_request_checkpoint(world: World) -> None:
    from tests.support.jobs import insert_run

    with world.session() as session:
        insert_run(session, RUN, status="running", last_step=12)
        session.commit()
    answered(
        "POST /api/v1/runs/{run_id}/checkpoints",
        world.client.post(f"/api/v1/runs/{RUN}/checkpoints"),
    )


@case(ROUTER, "GET /api/v1/checkpoints")
def test_list_all_checkpoints(world: World) -> None:
    _chain(world)
    body = answered("GET /api/v1/checkpoints", world.client.get("/api/v1/checkpoints"))
    assert body["count"] == 2


@case(ROUTER, "GET /api/v1/runs/{run_id}/checkpoints")
def test_list_run_checkpoints(world: World) -> None:
    _chain(world)
    body = answered(
        "GET /api/v1/runs/{run_id}/checkpoints",
        world.client.get(f"/api/v1/runs/{RUN}/checkpoints"),
    )
    assert body["latest_step"] == 200 and len(body["checkpoints"]) == 2


@case(ROUTER, "GET /api/v1/runs/{run_id}/checkpoints/{step}")
def test_get_checkpoint(world: World) -> None:
    _chain(world)
    answered(
        "GET /api/v1/runs/{run_id}/checkpoints/{step}",
        world.client.get(f"/api/v1/runs/{RUN}/checkpoints/200"),
    )


@case(ROUTER, "GET /api/v1/runs/{run_id}/checkpoints/{step}/lineage")
def test_get_lineage(world: World) -> None:
    _chain(world)
    body = answered(
        "GET /api/v1/runs/{run_id}/checkpoints/{step}/lineage",
        world.client.get(f"/api/v1/runs/{RUN}/checkpoints/200/lineage"),
    )
    assert len(body["nodes"]) == 2


@case(ROUTER, "POST /api/v1/runs/{run_id}/checkpoints/{step}/verify")
def test_verify_checkpoint(world: World) -> None:
    _chain(world)
    answered(
        "POST /api/v1/runs/{run_id}/checkpoints/{step}/verify",
        world.client.post(f"/api/v1/runs/{RUN}/checkpoints/200/verify"),
    )


@case(ROUTER, "POST /api/v1/runs/{run_id}/checkpoints/{step}/branch")
def test_branch_run(world: World) -> None:
    _chain(world)
    answered(
        "POST /api/v1/runs/{run_id}/checkpoints/{step}/branch",
        world.client.post(f"/api/v1/runs/{RUN}/checkpoints/100/branch", json={"run_id": "run-b"}),
    )


@case(ROUTER, "POST /api/v1/runs/{run_id}/continue")
def test_continue_plan_and_an_agents_pending_submit(world: World) -> None:
    """The operator's plan (200) and an agent's submit (the gate's pending 202), both from the
    real ``continue_plan`` over the real chain."""
    _chain(world)
    key = "POST /api/v1/runs/{run_id}/continue"
    url = f"/api/v1/runs/{RUN}/continue"
    plan = answered(key, world.client.post(url, json={}))
    assert plan["plan"]["step"] == 200
    answered(key, world.client.post(url, json={"submit": True}, headers=AGENT), pending=True)


@case(ROUTER, "POST /api/v1/runs/{run_id}/continue")
def test_continue_submitted(attached_world: dict[str, Any]) -> None:  # noqa: F811
    """The operator's submit of an attached run's continue: the real launcher's 202."""
    body = answered("POST /api/v1/runs/{run_id}/continue", _submit(attached_world))
    assert body["submitted"] is True and attached_world["enqueued"] == [ATTACHED_RUN]


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
