"""Chaining's wiring: the four routes, the one task, the one schedule  (FR-006.45, .46).

ADR-016 #1: *a capability is not shipped until a test FAILS when its wiring is removed.*
Chaining is the sharpest case of that rule in this feature, because **the sweep is the only
advance path a chain has**: with the task module unregistered or the beat entry missing,
``POST /chains`` still returns 202, stage one still runs, and the curriculum sits at stage
one forever with nothing anywhere saying why. Every test below therefore asserts against the
**mounted** application and the **live** Celery registry, never against an imported router
object or task function.

**The route half is RED until the orchestrator adds the mount line**, and that red is the
negative control: it is the same red that would appear if the line were ever removed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.route_table import mounted_routes

#: The four routes of FR-006.39/.45/.46 and the FTDD's section 5 table. **There is no
#: advance route**, and the absence is asserted below rather than described here.
CHAIN_ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", "/api/v1/chains"),
    ("GET", "/api/v1/chains"),
    ("GET", "/api/v1/chains/{chain_id}"),
    ("POST", "/api/v1/chains/{chain_id}/stop"),
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _mounted(client: TestClient) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in mounted_routes(client.app)  # type: ignore[arg-type]
        for method in getattr(route, "methods", set()) or set()
    }


class TestTheFourRoutesAreReachableFromTheMountedApp:
    def test_each_chain_route_is_in_the_mounted_route_table(self, client: TestClient) -> None:
        """Deleting ``include_router(chains.router)`` must make this red.

        Without it the four handlers would still be implemented, unit-tested and
        documented, and would exist for no caller - the sixteen-unregistered-MCP-tools
        failure applied to the one surface that creates a curriculum.
        """
        mounted = _mounted(client)
        missing = [pair for pair in CHAIN_ROUTES if pair not in mounted]
        assert not missing, f"not mounted: {missing}"

    def test_each_chain_route_is_in_the_generated_OPENAPI_document(
        self, client: TestClient
    ) -> None:
        document = client.app.openapi()  # type: ignore[attr-defined]
        for method, path in CHAIN_ROUTES:
            assert path in document["paths"], f"{path} is not in the OpenAPI document"
            assert method.lower() in document["paths"][path], f"{method} {path}"

    def test_there_is_no_advance_route(self, client: TestClient) -> None:
        """FR-006.45: **one** advance path, and the absence of a second is the requirement.

        An "advance now for responsiveness" route would be a second path and a double-start
        race - racing with the sweep that exists precisely because a dying task's enqueue
        stalls a chain forever. Asserted on the mounted table rather than by reading the
        endpoint module, because a route added anywhere would appear here.
        """
        offending = [
            (method, path)
            for method, path in _mounted(client)
            if path.startswith("/api/v1/chains")
            and any(word in path for word in ("advance", "next", "sweep", "start"))
        ]
        assert not offending, f"a second advance path is mounted: {offending}"


class TestTheSweepIsTheOneAdvancePathAndItIsWired:
    def test_the_task_module_is_in_the_configuration_the_worker_reads(self) -> None:
        """A worker runs ``celery -A workers.celery_app:celery_app``, which imports
        ``TASK_MODULES`` and nothing else. **Deleting the entry must make this red.**"""
        from workers.celery_app import TASK_MODULES

        assert "workers.chain_tasks" in TASK_MODULES

    def test_the_task_is_in_the_live_registry(self) -> None:
        from workers.celery_app import TASK_CHAIN_SWEEP, celery_app

        celery_app.loader.import_default_modules()
        assert TASK_CHAIN_SWEEP in celery_app.tasks, (
            "an enqueued or scheduled sweep would be received by a worker that does not "
            "have the task"
        )

    def test_the_task_routes_to_maintenance(self) -> None:
        """``TASK_ROUTES`` is keyed by **area prefix**, and the name is the whole defect.

        ``maintenance`` rather than ``default`` because a chain advance only ever has work
        to do while ``train`` is saturated by the run it is waiting on - the same reason the
        janitor lives there.
        """
        from workers.celery_app import TASK_CHAIN_SWEEP, queue_for

        assert queue_for(TASK_CHAIN_SWEEP) == "maintenance"

    def test_the_name_the_documents_use_still_routes_nowhere(self) -> None:
        """F6's four documents name this sweeper ``miforge.chain.sweep``.

        There is no ``miforge.chain.*`` entry and there must not be one: a sweep is
        maintenance. This mirrors ``test_run_loop_reachability.py``'s tripwire from the
        other side - that one says the documents' name does not route, this one says the
        shipped name does - so adding a ``miforge.chain.*`` route to "fix" the documents
        turns one of the two red.
        """
        from workers.celery_app import TASK_ROUTES, queue_for

        assert queue_for("miforge.chain.sweep") is None
        assert not any(key.startswith("miforge.chain.") for key in TASK_ROUTES)

    def test_the_sweep_is_on_the_beat_schedule(self) -> None:
        """**A schedule with no entry fires never.**

        The task being registered is not enough: nothing in the product calls it. The beat
        entry *is* the production caller, which is why removing it has to be red rather than
        merely quiet.
        """
        from workers.celery_app import TASK_CHAIN_SWEEP, celery_app

        schedule = celery_app.conf.beat_schedule
        entries = [entry for entry in schedule.values() if entry["task"] == TASK_CHAIN_SWEEP]
        assert len(entries) == 1, f"beat schedule: {sorted(schedule)}"
        assert entries[0]["options"]["queue"] == "maintenance"
        assert float(entries[0]["schedule"]) > 0

    def test_the_task_body_calls_the_sweep_with_a_session_FACTORY(self) -> None:
        """The payload, not "was called" - and the argument is the load-bearing part.

        :func:`services.chains.service.sweep` takes a factory so that one broken chain's
        transaction rolls back alone. Handing it a *session* would type-check, run, and
        remove the isolation: one chain's failure would take the whole sweep with it, which
        is ``services/reindex.py``'s recorded defect.
        """
        from unittest.mock import patch

        from workers import chain_tasks

        with patch("services.chains.service.sweep") as spy:
            spy.return_value = {"looked_at": 0}
            chain_tasks.chain_sweep()
        assert spy.call_count == 1
        (passed,), _ = spy.call_args
        assert callable(passed), "a session was passed where a factory is required"
        # And it must be **the project's own** scope factory, by identity. `callable(...)`
        # alone passes against `lambda: nullcontext(one_session)`, which type-checks, runs,
        # and silently removes the isolation - one chain's failure would then take the whole
        # sweep with it. Asserting the shape is not asserting the property; identity is
        # cheap here and needs no database.
        from workers.run_record import default_session_scope

        assert passed is default_session_scope, (
            "the sweep was handed something other than the one scope factory: a wrapper "
            "around a single session would destroy per-chain isolation while passing every "
            "shape check"
        )
