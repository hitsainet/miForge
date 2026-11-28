"""``api.v1.endpoints.runs_review``'s routes answer as the contract says (021 FR-021.13)."""

from __future__ import annotations

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.runs_review"


@case(ROUTER, "GET /api/v1/runs/{run_id}/review")
def test_get_run_review(world: World) -> None:
    """A run bound to a miDataworks queue with two posted candidates, through F10's binding
    service (the calls the review loop makes once miDataworks has acknowledged them)."""
    from services.review import binding as binding_service
    from tests.support.jobs import insert_run

    with world.session() as session:
        insert_run(session, "run-reviewed", status="running")
        binding = binding_service.record_binding(
            session,
            run_id="run-reviewed",
            queue_id="queue-1",
            question="Is this completion acceptable?",
            label_set=["accept", "reject"],
        )
        binding_service.record_posted(
            session, binding, step=3, accepted=[("ext-1", "item-1"), ("ext-2", "item-2")]
        )
    answered(
        "GET /api/v1/runs/{run_id}/review", world.client.get("/api/v1/runs/run-reviewed/review")
    )


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
