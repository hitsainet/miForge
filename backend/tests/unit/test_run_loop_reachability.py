"""Reachability for F6  (FR-006.52, FR-006.65; ADR-016 #1; AC-006.34, .43, .45, .46).

**The rule this project's entire review discipline is built on:** *a capability is not
shipped until a test FAILS when its wiring is removed.* It was written after a sibling
shipped sixteen fully-implemented, unit-tested, documented MCP tools that were never
registered with its server - every one of those tests passed by importing the module
directly, so the suite was green and the documentation said done while the feature did not
exist for any caller.

So nothing here imports an endpoint function or a task body. Every assertion is against:

* the **mounted application's** route table - a router that exists but is not included is
  red;
* the **live Celery registry** after ``import_default_modules()`` - a task defined in a
  module nobody imports is a task a worker does not have;
* the **live** stop and refresh registries;
* the **live** service interface for F7's gate.

**Five wirings were owed to F6 and each one had no production caller at all.** They are
listed below with the control that now holds them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app

#: F6's routes. Declared here as well as in ``test_reachability.py``'s manifest, because
#: that file asserts the *whole* surface and this one asserts F6's share answers.
F6_ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", "/api/v1/runs/validate"),
    ("POST", "/api/v1/runs/define"),
    ("GET", "/api/v1/runs/{run_id}/definition"),
    ("GET", "/api/v1/runs/{run_id}/stop-conditions"),
    ("GET", "/api/v1/runs/{run_id}/boundaries"),
    ("GET", "/api/v1/stop-conditions"),
    ("GET", "/api/v1/refresh-modes"),
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


class TestEveryF6RouteIsReachableFromTheMountedApp:
    def test_each_route_is_in_the_mounted_route_table(self, client: TestClient) -> None:
        """Deleting ``include_router(runs_loop.router)`` must make this red. Without the
        control, R-02.53's **one** validator would still be implemented, unit-tested and
        documented, and would exist for no caller - the product's primary action."""
        mounted = {
            (method, route.path)
            for route in client.app.routes  # type: ignore[attr-defined]
            for method in getattr(route, "methods", set()) or set()
        }
        missing = [pair for pair in F6_ROUTES if pair not in mounted]
        assert not missing, f"not mounted: {missing}"

    def test_each_route_is_in_the_generated_OPENAPI_document(self, client: TestClient) -> None:
        """A route that is mounted but not described is a route no client can find.

        The *transport* half - an actual request through each route - is
        ``test_reachability.py``'s, which overrides the session dependency for the whole
        surface. Repeating it here against the configured database would add a Postgres
        dependency to a unit test and would report a connection failure as a reachability
        failure, which is a measurement that did not happen reported as a result.
        """
        document = client.app.openapi()  # type: ignore[attr-defined]
        for method, path in F6_ROUTES:
            assert path in document["paths"], f"{path} is not in the OpenAPI document"
            assert method.lower() in document["paths"][path], f"{method} {path}"

    def test_the_validate_route_declares_a_200_not_a_422(self, client: TestClient) -> None:
        """Asking whether a definition is valid is a **successful question** (E-25). The
        refusals fire where an action is attempted, which is the shape F7 uses for pack
        validity and what lets F9's form put each failure beside its field.

        Asserted on the declared response, because the behaviour itself is exercised
        against a real session in ``test_run_definition.py`` - here the subject is the
        *contract the route publishes*, which is what F9 builds against.
        """
        document = client.app.openapi()  # type: ignore[attr-defined]
        responses = document["paths"]["/api/v1/runs/validate"]["post"]["responses"]
        assert "200" in responses
        accepted = document["paths"]["/api/v1/runs/define"]["post"]["responses"]
        assert "202" in accepted, (
            "define returns 202: the run has been ACCEPTED, not performed - the parameters "
            "are acquired in the worker, under the lease"
        )

    def test_there_is_NO_second_stop_route(self, client: TestClient) -> None:
        """FR-006.24, AMB-3: F6 adds **no second route, no second flag and no second
        mechanism**. F1's ``cancel`` is the stop; F6 owns the condition, the precedence and
        the terminal transition."""
        paths = {
            route.path
            for route in client.app.routes  # type: ignore[attr-defined]
            if hasattr(route, "path")
        }
        assert "/api/v1/runs/{run_id}/stop" not in paths


class TestTheLiveCeleryRegistry:
    def test_the_run_loop_task_is_registered_and_routes_to_train(self) -> None:
        """A task defined in a module nobody imports is a fully-tested capability that does
        not exist for any caller."""
        from training.tasks import TASK_RUN_STEP_LOOP
        from workers.celery_app import celery_app, queue_for

        # Through Celery's OWN loader, not by importing the module here. Importing it
        # directly is precisely the test that passed for sixteen unregistered tools.
        celery_app.loader.import_default_modules()

        assert TASK_RUN_STEP_LOOP in celery_app.tasks
        assert queue_for(TASK_RUN_STEP_LOOP) == "train"

    def test_the_merge_task_is_registered_and_routes_to_train(self) -> None:
        """**Owed wiring 3.** ``mf_merges`` existed with no writer and the merge route
        enqueued nothing: F8 built the operation, the three outcomes and the table and
        reported, accurately, that the enqueue was F6's.

        **Deleting ``workers.merge_tasks`` from ``TASK_MODULES`` must make this red** - the
        route would still accept the request and a worker started with
        ``-A workers.celery_app`` would not have the task.
        """
        from workers.celery_app import TASK_MODULES, celery_app, queue_for
        from workers.merge_tasks import TASK_MERGE_ADAPTER

        celery_app.loader.import_default_modules()
        assert "workers.merge_tasks" in TASK_MODULES, (
            "the module is not named in the configuration the worker reads, so a worker "
            "started with `-A workers.celery_app:celery_app` would not have the task"
        )
        assert TASK_MERGE_ADAPTER in celery_app.tasks, (
            "the merge task is not in the live registry; an enqueued merge would be "
            "received by a worker that does not have it"
        )
        assert queue_for(TASK_MERGE_ADAPTER) == "train", (
            "a task matching no route entry lands on `default`, where a worker with NO "
            "CARD picks it up and fails - the failure F1 recorded for its own task name"
        )

    def test_every_F6_task_name_begins_with_a_routed_area_prefix(self) -> None:
        """``TASK_ROUTES`` is an **explicit** table keyed by area. F1 recorded that all four
        of its documents spelled its task ``miforge.training.…``, which routes **nowhere**;
        F6's documents name a chain sweeper ``miforge.chain.sweep`` with the same defect.
        Reported rather than shipped under a name that does not route."""
        from workers.celery_app import TASK_ROUTES, queue_for
        from workers.merge_tasks import TASK_MERGE_ADAPTER

        assert queue_for(TASK_MERGE_ADAPTER) is not None
        assert queue_for("miforge.chain.sweep") is None, (
            "if this ever routes, F6's chain sweeper may use the documents' name; until "
            "then the name in the documents is a defect and is reported"
        )
        assert "miforge.train.*" in TASK_ROUTES


class TestTheLiveStopAndRefreshRegistries:
    def test_the_five_conditions_are_in_the_live_registry(self) -> None:
        from training import stop

        assert len(stop.names()) == 5
        for name in stop.names():
            assert stop.resolve(name) is not None

    def test_the_two_refresh_modes_are_in_the_live_registry(self) -> None:
        from training import refresh

        assert len(refresh.names()) == 2
        for name in refresh.names():
            assert refresh.resolve(name) is not None

    def test_the_run_types_are_resolvable_and_F6_is_the_FIRST_production_caller(
        self,
    ) -> None:
        """**Owed wiring 1.** Nothing called ``runtypes.resolve()`` on the submit path, so
        ``RunType.preflight`` / ``prepare`` / ``save`` had no production caller at all.

        The validator's budget step now resolves the run type by name and calls its
        ``preflight``; asserted on the source, because a resolve that exists and is never
        reached is the defect.
        """
        import ast
        from pathlib import Path

        from training import runtypes

        assert set(runtypes.known()) == {"change", "create", "adapter"}

        source = (
            Path(__file__).resolve().parents[3]
            / "backend"
            / "src"
            / "services"
            / "run_definition.py"
        ).read_text(encoding="utf-8")
        calls = {
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "resolve" in calls and "preflight" in calls, (
            "the submit path must resolve the run type and call its preflight; without it "
            "F8's three run types are implemented, tested, and reachable by no caller"
        )


class TestF7sGateOnTheLiveServiceInterface:
    def test_require_valid_pack_is_on_the_live_service(self) -> None:
        """F7's own reachability test asserts it is present **on the service**, so F6 calls
        it the same way - the two halves of the seam then check the same object."""
        from packs.service import pack_service

        # `pack_service` is the singleton instance, not a factory - F7's own spelling.
        assert callable(pack_service.require_valid_pack)

    def test_the_validators_live_binding_goes_through_that_service(self) -> None:
        """**Owed wiring 2**, and F7's task 13.2. The cross-feature half: that F6's
        run-create path *actually calls it*. Asserted on the binding, with the payload and
        the call count asserted in ``test_run_definition.py``."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3]
            / "backend"
            / "src"
            / "services"
            / "run_definition.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        live = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_live_gate"
        )
        names = {
            node.func.attr
            for node in ast.walk(live)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "require_valid_pack" in names, (
            "F6's live gate binding must call F7's own gate; F7's task 13.2 is this "
            "assertion's cross-feature half"
        )
        assert "pack_service" in source, "the gate must be reached through F7's service"


class TestTheHeartbeatBlockStaysOnTheRunResponse:
    def test_the_run_response_carries_the_heartbeat_block_with_every_subkey(
        self, client: TestClient
    ) -> None:
        """**M-18.** AC-006.46. Without it the run screen must trust the socket - which
        shows a **stalled** run as live - or derive a threshold from the 60 s throttle,
        which shows a **healthy** run as stalled through every test pass and every save.
        *Both readings have already been tried and withdrawn in F9.*"""
        from api.v1.endpoints.runs import run_payload

        payload = run_payload(
            {
                "id": "r",
                "status": "running",
                "run_type": "change",
                "last_step": 4,
                "cancel_requested_at": None,
                "failure_code": None,
                "heartbeat_at": None,
                "longest_heartbeat_gap_seconds": None,
                "definition_resolved": None,
            }
        )
        assert set(payload["heartbeat"]) == {
            "heartbeat_at",
            "age_seconds",
            "longest_gap_seconds",
            "lapse_threshold_seconds",
            "lapsed",
        }

    def test_the_age_is_computed_SERVER_SIDE_and_the_threshold_is_published(self) -> None:
        """Not a bare timestamp for the client to subtract, and the client does not choose a
        threshold: *two thresholds are two answers to "is this run alive", and the one the
        operator sees would disagree with the one that reaps.*"""
        from datetime import timedelta

        from api.v1.endpoints.runs import run_payload
        from core.config import get_settings
        from workers.run_record import utcnow

        payload = run_payload(
            {
                "id": "r",
                "status": "running",
                "run_type": "change",
                "last_step": 4,
                "cancel_requested_at": None,
                "failure_code": None,
                "heartbeat_at": utcnow() - timedelta(seconds=30),
                "longest_heartbeat_gap_seconds": 95,
                "definition_resolved": None,
            }
        )
        assert payload["heartbeat"]["age_seconds"] >= 29
        assert (
            payload["heartbeat"]["lapse_threshold_seconds"] == get_settings().janitor_lapse_seconds
        )
        assert payload["heartbeat"]["longest_gap_seconds"] == 95

    def test_F6s_own_fields_are_on_the_run_response_present_and_null(self) -> None:
        """A **missing key is indistinguishable from a healthy one**, so each of F6's fields
        is present and null rather than absent when it has no value."""
        from api.v1.endpoints.runs import run_payload

        payload = run_payload(
            {
                "id": "r",
                "status": "pending",
                "run_type": "change",
                "last_step": None,
                "cancel_requested_at": None,
                "failure_code": None,
                "heartbeat_at": None,
                "longest_heartbeat_gap_seconds": None,
                "definition_resolved": None,
            }
        )
        for key in (
            "stop_reason",
            "round",
            "stop_conditions",
            "cadences",
            "retention_exposure_steps",
            "footprint",
            "seed",
            "domain",
        ):
            assert key in payload, f"{key} is absent rather than null"


class TestTheSaveWiringsTwoMissingValuesAreSupplied:
    def test_the_driver_passes_domain_and_signal_source_off_the_definition(self) -> None:
        """**Owed wiring 4.** ``training/tasks.py`` wrote trainer state and **skipped the
        checkpoint**, logging ``checkpoint_save_skipped_no_provenance``, whenever it lacked
        ``signal_source`` (F2's) or ``domain`` (F6's, off R-02.53's definition).

        Both come off the **recorded definition** now. ``ManifestFields`` still gives
        neither a default, so the one thing that cannot happen is a fabricated provenance
        claim in a published artifact.
        """
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "run_driver.py"
        ).read_text(encoding="utf-8")
        drive = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "drive_recorded_run"
        )
        call = next(
            node
            for node in ast.walk(drive)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "drive_run"
        )
        passed = {kw.arg for kw in call.keywords}
        assert {"domain", "signal_source"} <= passed, (
            "without both, the weights save is skipped at every cadence point and the run "
            "produces trainer state and no checkpoint"
        )


class TestTheRegistriesAreREADRatherThanTranscribed:
    """**F9's finding, and the reason these routes exist.**

    F9 built its stop-condition field from F6's *documented precedence order*, marked in
    its own code as "documentation, not a registry", swapping in ``details.available`` once
    a ``run_has_no_stop_condition`` refusal arrived. Defensive and correct, and still a
    **transcription of prose** - which is the drift that produced ``DB_RUN_TYPE`` and the
    stale ``RUN_TYPES`` constant, both in this codebase, both within a fortnight.

    So the assertions here are that the route serves **the registry's own answer**, not a
    list that happens to match it today: the payload is compared against
    ``training.stop.names()`` at call time, and a registration deleted from
    ``_REGISTRATIONS`` is red in the form's data rather than in a comment.
    """

    def test_the_stop_condition_route_serves_the_LIVE_registry(self, client: TestClient) -> None:
        from training import stop

        body = client.get("/api/v1/stop-conditions").json()
        assert [entry["condition"] for entry in body["conditions"]] == list(stop.names()), (
            "the route must serve the registry's own order and contents; a hard-coded list "
            "is the transcription this route exists to remove"
        )
        assert body["count"] == len(stop.names()) == 5
        assert body["closed"] is True, "R-02.25's set is closed and the payload says so"
        assert body["precedence"] == list(stop.precedence.PRECEDENCE)
        assert body["minimum_enabled"] == 1, (
            "FR-006.17: the form must be able to say 'at least one' BEFORE the operator "
            "presses the button, not only after the refusal"
        )

    def test_the_payload_says_which_conditions_take_NO_configuration(
        self, client: TestClient
    ) -> None:
        """Otherwise a form renders a parameter field for the retention floor, and a floor
        typed there would be a **third** layer - F4 FR-004.23 already needed a dedicated
        field to tell the two that exist apart."""
        body = client.get("/api/v1/stop-conditions").json()
        by_name = {entry["condition"]: entry for entry in body["conditions"]}
        assert by_name["retention_floor"]["configured"] is False
        assert by_name["retention_floor"]["required_parameters"] == []
        assert by_name["operator_stop"]["configured"] is False

    def test_the_plateau_entry_names_ALL_FOUR_required_parameters(self, client: TestClient) -> None:
        """DEF-4: **R-02.25 defines none of it**, so all four are required and none is
        defaulted. A form that asked for three would produce a definition the validator
        refuses, which is the refusal loop these routes exist to prevent."""
        from training.stop.loss_plateau import METRICS, REQUIRED_PARAMETERS

        body = client.get("/api/v1/stop-conditions").json()
        plateau = next(e for e in body["conditions"] if e["condition"] == "loss_plateau")
        assert plateau["required_parameters"] == list(REQUIRED_PARAMETERS)
        assert len(plateau["required_parameters"]) == 4
        assert plateau["metrics"] == list(METRICS)

    def test_the_refresh_route_serves_the_LIVE_registry_and_its_applicability(
        self, client: TestClient
    ) -> None:
        """``applies_to`` is the one thing a form gets wrong otherwise: a refresh applies to
        **self-generation runs only**, and is **refused** rather than ignored elsewhere."""
        from training import refresh

        body = client.get("/api/v1/refresh-modes").json()
        assert [entry["mode"] for entry in body["modes"]] == list(refresh.names())
        assert body["count"] == 2
        assert body["closed"] is True
        assert body["applies_to"] == refresh.SELF_GENERATION == "self-generation"
        assert body["requires_cadence"] == "refresh_every"
        assert all(entry["applies_to"] == "self-generation" for entry in body["modes"])

    def test_deleting_a_registration_changes_what_the_ROUTE_serves(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The wiring assertion, as a live control rather than a note.**

        Removing ``retention_floor`` from the registry must change the route's answer -
        which is what makes the form's list a read and not a copy. A route serving a
        literal would be identical before and after, and the form would offer a condition
        the loop cannot resolve.
        """
        from training import stop

        before = client.get("/api/v1/stop-conditions").json()
        assert "retention_floor" in [e["condition"] for e in before["conditions"]]

        remaining = {
            name: condition
            for name, condition in stop._CONDITIONS.items()
            if name != "retention_floor"
        }
        monkeypatch.setattr(stop, "_CONDITIONS", remaining)
        monkeypatch.setattr(
            stop,
            "_REGISTRATIONS",
            tuple(pair for pair in stop._REGISTRATIONS if pair[0] != "retention_floor"),
        )
        after = client.get("/api/v1/stop-conditions").json()
        assert "retention_floor" not in [e["condition"] for e in after["conditions"]]
        assert after["count"] == 4
