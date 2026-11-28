"""``api.v1.endpoints.tree``'s route answers as the contract says (021 FR-021.13).

The forest ``tree/test_tree_api.py`` seeds, through the real writers
(``tests.support.tree``): a root, a branch, an adapter, a merged checkpoint, a scratch root
with no domain, and a chain whose parent cannot be resolved, indexed with two run rows.
"""

from __future__ import annotations

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.tree"


@case(ROUTER, "GET /api/v1/tree")
def test_get_tree(world: World) -> None:
    from tests.support import tree as forest

    root = world.root / "checkpoints"
    root.mkdir(exist_ok=True)
    base = forest.root_checkpoint(root, run_id="run-base", step=40, domain="support")
    forest.branch(root, parent=base.name, run_id="run-base", step=80)
    adapter = forest.adapter(root, base=base, run_id="run-adapter", step=50)
    forest.merged_checkpoint(root, base=base, adapter_document=adapter, run_id="run-base", step=120)
    forest.root_checkpoint(root, run_id="run-scratch", step=10, domain=None)
    forest.unresolvable_parent_checkpoint(root, run_id="run-orphan", step=700)
    forest.indexed(world.engine, root, run_ids=("run-base", "run-adapter"))
    body = answered("GET /api/v1/tree", world.client.get("/api/v1/tree"))
    assert len(body["nodes"]) == 6


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
