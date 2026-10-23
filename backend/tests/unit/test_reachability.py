"""Reachability: every route answers from the mounted app  (foundation 5.3; ADR-016 #1).

**The rule this project's entire review discipline is built on:** *a capability is not
shipped until a test FAILS when its wiring is removed.* It was written after a sibling
shipped **sixteen fully-implemented, unit-tested, documented MCP tools that were never
registered with the server.** Every one of those tests passed by importing the module
directly, so the suite was green and the documentation said ✅ while the feature did not
exist for any caller. Two full review rounds had read that code.

So nothing in this file imports an endpoint function. Every assertion is made against:

* the **mounted application's** route table and its generated OpenAPI document - so a
  router that exists but is not included is red, which is the failure mode above;
* an actual **request through the transport**, so a route that is registered but unroutable
  is red too;
* the **live exception-handler registry**, because the error envelope is a capability like
  any other and *"the handler function exists"* is not evidence that it is installed.

:data:`EXPECTED_ROUTES` is compared in **both directions**. A missing route is the defect
above. An *extra* route is also a failure, and deliberately so: a surface that appears
without anyone declaring it here is a surface nobody reviewed, and the manifest would then
be a partial list that reads as exhaustive.

Task 10.2's control for this file: **delete an ``include_router`` line, run the suite,
require a red, restore it.** Recorded in the report when run.
"""

from __future__ import annotations

import re

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import health as health_module
from api.app import create_app, status_for
from api.route_table import mounted_routes as route_table
from core.config import reset_settings
from core.errors import Refusal, refuse

#: Every route this application serves, as ``(method, path)``. **The single declaration of
#: the surface.** A feature adding an endpoint adds a line here; if it does not, this file
#: goes red, which is the only reason the comparison is two-way.
EXPECTED_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/health"),
        ("GET", "/api/v1/runs/{run_id}"),
        # F1 (R-02.46, R-02.14, R-02.13). Submit refuses before allocating; cancel sets a
        # column and never `revoke(terminate=True)`; the step log is keyset-paginated
        # newest-first; `models/sources` lists every acquisition mode, each unavailable one
        # BY NAME - an option missing from a response is indistinguishable from a broken one.
        ("POST", "/api/v1/runs"),
        ("POST", "/api/v1/runs/{run_id}/cancel"),
        ("GET", "/api/v1/runs/{run_id}/steps"),
        # F6 (R-02.24 - R-02.27, R-02.53). `validate` and `define` are ONE validator with
        # two callers: a second, laxer copy behind the submit button is the drift this
        # project pays for, and a `validate` route the create path never calls is an
        # unreachable capability whose unit tests all pass. `define` is R-02.53's create
        # path and F1's `POST /runs` is the acquisition half that predates it - a run with
        # no recorded definition is refused in the worker, so only one of the two can
        # produce a run that trains. There is deliberately **no** F6 stop route: F1's
        # `cancel` is the stop and F6 adds the condition, not a second mechanism.
        ("POST", "/api/v1/runs/validate"),
        ("POST", "/api/v1/runs/define"),
        ("GET", "/api/v1/runs/{run_id}/definition"),
        ("GET", "/api/v1/runs/{run_id}/stop-conditions"),
        ("GET", "/api/v1/runs/{run_id}/boundaries"),
        # F6's two REGISTRY listings, and they exist because F9 was transcribing F6's prose
        # instead of reading its registries - the drift that produced `DB_RUN_TYPE` and the
        # stale `RUN_TYPES`. Not under `/runs/`: `GET /runs/{run_id}` would match
        # `stop-conditions` as a run id, so a collection there would depend on the order of
        # include_router lines.
        ("GET", "/api/v1/stop-conditions"),
        ("GET", "/api/v1/refresh-modes"),
        ("GET", "/api/v1/models/sources"),
        # F9 task 14.0: the three read-only listings the pickers read. `GET /runs` is the
        # collection the path lacked - a GET there answered 405 until 2026-10-18.
        ("GET", "/api/v1/runs"),
        ("GET", "/api/v1/models/local"),
        ("GET", "/api/v1/gpus"),
        # The tokenizer a create run ships (2026-10-20): the tokenizer directories under the
        # model roots and the Hub revision check the New run form's tokenizer picker reads.
        ("GET", "/api/v1/tokenizers/local"),
        ("GET", "/api/v1/hub/revision"),
        # The shell: host readings for the resource strip, and the attachment chip's live read
        # (2026-10-19). Both are on every screen.
        ("GET", "/api/v1/host/resources"),
        ("GET", "/api/v1/attachment"),
        # F12 (BRD-MIFORGE-MCP-001): the approval queue - three reads an agent polls, two
        # decisions only the operator may make - and the settings surface (FR-012.7, .8).
        ("GET", "/api/v1/approvals"),
        ("GET", "/api/v1/approvals/actions"),
        ("GET", "/api/v1/approvals/{approval_id}"),
        ("POST", "/api/v1/approvals/{approval_id}/approve"),
        ("POST", "/api/v1/approvals/{approval_id}/reject"),
        ("GET", "/api/v1/settings"),
        ("GET", "/api/v1/settings/history"),
        ("PUT", "/api/v1/settings/{key}"),
        ("DELETE", "/api/v1/settings/{key}"),
        # F3's two read paths (FR-003.12). No write path: F3 authors no configuration, and
        # an endpoint accepting one would be a second authoring surface and a second hash,
        # which R-02.10 forbids.
        ("GET", "/api/v1/selectors"),
        ("POST", "/api/v1/selectors/evaluate"),
        # F10 (R-02.36, R-02.37, R-02.50). The part index and its validator - whose product
        # IS the refusal, so it is a first-class route rather than a debug helper - plus the
        # miDataworks integration's own status, its read-only listing, and one run's review
        # counts. **No adjudication route**, and that absence is itself asserted by
        # `test_review_queue_no_local_queue.py`: the queue, the decision history and the
        # review screen are miDataworks' (R-02.36), and a decision-recording endpoint here
        # would be the second answer to the one question miDataworks already answers.
        ("POST", "/api/v1/parts/import"),
        ("POST", "/api/v1/parts/validate"),
        ("GET", "/api/v1/parts"),
        ("GET", "/api/v1/parts/{part_id}"),
        # Archive (operator decision 2026-10-20): visibility only, deletes nothing.
        ("POST", "/api/v1/runs/{run_id}/archive"),
        ("POST", "/api/v1/runs/{run_id}/unarchive"),
        ("POST", "/api/v1/parts/{part_id}/archive"),
        ("POST", "/api/v1/parts/{part_id}/unarchive"),
        # Purge (operator decision 2026-10-20): permanent delete of ARCHIVED items, operator only.
        ("POST", "/api/v1/runs/purge/plan"),
        ("POST", "/api/v1/runs/purge"),
        ("POST", "/api/v1/runs/{run_id}/purge"),
        ("POST", "/api/v1/parts/purge/plan"),
        ("POST", "/api/v1/parts/purge"),
        ("POST", "/api/v1/parts/{part_id}/purge"),
        ("GET", "/api/v1/purges"),
        ("GET", "/api/v1/midataworks/status"),
        ("GET", "/api/v1/midataworks/available-parts"),
        ("GET", "/api/v1/runs/{run_id}/review"),
        # F4 (R-02.16 - R-02.19, R-02.49). The live provider registry and its probe - read
        # by F7's dry-run and F9's New-run form before a run exists - plus the score series
        # and the out-of-cadence evaluation, read by F9's curves after one does. The probe
        # exists because **the refusal is the product** when a provider cannot be used.
        ("GET", "/api/v1/providers"),
        ("POST", "/api/v1/providers/{name}/check"),
        ("GET", "/api/v1/runs/{run_id}/scores"),
        ("GET", "/api/v1/runs/{run_id}/scores/{step}"),
        ("POST", "/api/v1/runs/{run_id}/evaluate"),
        # F2 (R-02.4 - R-02.8). The live source registry with each unsatisfied requirement
        # NAMED - an option missing from a response is indistinguishable from a broken one -
        # its dry pre-flight, which F7's pack validation calls before a run exists, and the
        # two read paths over a run's batch records. **No write path for a batch**: a batch
        # is produced by a run, and an endpoint that accepted one would be a second
        # provenance path, which is two answers to what a checkpoint was trained on.
        ("GET", "/api/v1/signal-sources"),
        ("POST", "/api/v1/signal-sources/preflight"),
        ("GET", "/api/v1/runs/{run_id}/batches"),
        ("GET", "/api/v1/runs/{run_id}/batches/{batch_sha256}"),
        # F5 (R-02.20 - R-02.23, R-02.51). Eight routes. **Addressed by `(run_id, step)`,
        # never by a surrogate id**: ids are reassigned by a rebuild and foundation 4.4
        # demands identical responses across one, so an exposed id would make ADR-004's
        # claim untestable. No export route and no delete route, deliberately: PPRD section
        # 7 names an "Export checkpoint" button and no requirement defines the capability
        # (F5's AMB-3), and nothing defines pruning or deletion either - a checkpoint is a
        # plain directory, so `scp` and `tar` are the current answer.
        ("POST", "/api/v1/runs/{run_id}/checkpoints"),
        ("GET", "/api/v1/checkpoints"),
        ("GET", "/api/v1/runs/{run_id}/checkpoints"),
        ("GET", "/api/v1/runs/{run_id}/checkpoints/{step}"),
        ("POST", "/api/v1/runs/{run_id}/checkpoints/{step}/verify"),
        ("GET", "/api/v1/runs/{run_id}/checkpoints/{step}/lineage"),
        ("POST", "/api/v1/runs/{run_id}/continue"),
        ("POST", "/api/v1/runs/{run_id}/checkpoints/{step}/branch"),
        # F8 (R-02.41 - R-02.45). The run-type list and the budget preview are what a New
        # run form reads BEFORE a run exists; the adapter routes are what F9's tree and a
        # merge request read after one has produced an artifact. No `POST /adapters`: an
        # adapter is produced by a run's save, never uploaded, and a creation endpoint
        # would be a second way for an artifact to appear with no manifest behind it.
        #
        # `{name:path}` and not `{name}`, and the converter is load-bearing: an adapter's
        # name is `<base_model>:<run_id>:adapter-<step>` and `base_model` for a Hub base is
        # a repo id CONTAINING A SLASH (`LiquidAI/LFM2.5-1.2B-Instruct`), so the default
        # converter would 404 every adapter trained on a Hub model - the ordinary case.
        ("GET", "/api/v1/run-types"),
        ("POST", "/api/v1/budget/estimate"),
        ("GET", "/api/v1/adapters"),
        ("GET", "/api/v1/merges"),
        ("GET", "/api/v1/adapters/{name:path}"),
        ("POST", "/api/v1/adapters/{name:path}/merge"),
        # F7 (R-02.35, R-02.38 - R-02.40). Three surfaces: the registry and a pack version's
        # validity, lineage, export and import; the seven-step authoring flow with its
        # try-panels and its dry-run; and the one read that turns an indexed part into a
        # pinnable slot reference. **No queue route, and that absence is a requirement** -
        # the in-run review queue is miDataworks' (R-02.36) and the authoring dry-run is
        # synchronous and in-request, which `test_pack_no_review_queue.py` asserts against
        # this same route table rather than by reading the endpoint modules.
        #
        # The two `…/validity` routes are **GETs that describe invalidity with a 200**:
        # asking whether a pack is valid is a successful question, and the refusals fire
        # where an action is attempted (commit, or a run start). The draft's is NOT called
        # `/review`: in this repository that word names miDataworks' human-review queue, and
        # F10 ships a guard that the only route mentioning it is its own.
        ("GET", "/api/v1/packs"),
        ("POST", "/api/v1/packs"),
        ("POST", "/api/v1/packs/import"),
        ("GET", "/api/v1/packs/{pack_id}/versions/{version}"),
        ("DELETE", "/api/v1/packs/{pack_id}/versions/{version}"),
        ("GET", "/api/v1/packs/{pack_id}/versions/{version}/validity"),
        ("GET", "/api/v1/packs/{pack_id}/versions/{version}/lineage"),
        ("GET", "/api/v1/packs/{pack_id}/versions/{version}/export"),
        ("POST", "/api/v1/packs/{pack_id}/versions/{version}/clone"),
        ("POST", "/api/v1/pack-drafts"),
        ("GET", "/api/v1/pack-drafts"),
        ("GET", "/api/v1/pack-drafts/{draft_id}"),
        ("DELETE", "/api/v1/pack-drafts/{draft_id}"),
        ("PUT", "/api/v1/pack-drafts/{draft_id}/steps/{number}"),
        ("POST", "/api/v1/pack-drafts/{draft_id}/samples/{slot}"),
        ("POST", "/api/v1/pack-drafts/{draft_id}/samples/{slot}/score"),
        ("POST", "/api/v1/pack-drafts/{draft_id}/dry-run"),
        ("POST", "/api/v1/pack-drafts/{draft_id}/commit"),
        ("GET", "/api/v1/pack-drafts/{draft_id}/validity"),
        ("GET", "/api/v1/pack-part-refs/{part_id}"),
        # F9 (R-02.29). ONE route: the artifact forest across runs, two node kinds joined on
        # two different edges, with a merged checkpoint carrying both. It is the only backend
        # code F9 writes. Deleting its `include_router` line must make this file red - the
        # composer, the three edge resolutions and the whole of `schemas/tree.py` would still
        # be implemented, unit-tested and documented, and the checkpoint tree screen - one of
        # the two screens R-02.28/.29 promise - would have no data source. That is the
        # sixteen-unregistered-MCP-tools failure, and `test_tree_api.py` runs the control
        # explicitly rather than trusting that it would.
        ("GET", "/api/v1/tree"),
        # F6, chaining (R-02.27). Four routes around the sweep, which is itself not a route:
        # FR-006.45 makes the `maintenance` sweep the one advance path, so nothing here starts
        # a stage but stage 0 at creation.
        ("POST", "/api/v1/chains"),
        ("GET", "/api/v1/chains"),
        ("GET", "/api/v1/chains/{chain_id}"),
        ("POST", "/api/v1/chains/{chain_id}/stop"),
        # The documentation surface. Listed rather than filtered out: ADR-013 pins these
        # paths outside the version prefix for the same reason health is pinned there, and
        # a silent move would break whatever is pointed at them.
        ("GET", "/api/openapi.json"),
        ("GET", "/api/docs"),
        ("GET", "/docs/oauth2-redirect"),
    }
)


@pytest.fixture
def app():
    reset_settings()
    return create_app()


@pytest.fixture
def client(app, monkeypatch):
    monkeypatch.setitem(
        health_module.DEPENDENCY_PROBES, "postgres", lambda: {"status": "ok", "detail": None}
    )
    monkeypatch.setitem(
        health_module.DEPENDENCY_PROBES, "redis", lambda: {"status": "ok", "detail": None}
    )
    return TestClient(app, raise_server_exceptions=False)


def _without_converters(path: str) -> str:
    """``/a/{name:path}`` -> ``/a/{name}``, the form OpenAPI renders.

    Only for the OpenAPI comparison. The mounted-app comparison uses the real path,
    converter included, because the converter is behaviour: dropping it changes which
    requests the route matches.
    """
    return re.sub(r"\{([^{}:]+):[^{}]+\}", r"{\1}", path)


def mounted_routes(app) -> set[tuple[str, str]]:
    """``(method, path)`` for every route on the **mounted** app.

    ``HEAD`` is dropped because Starlette adds it alongside ``GET`` for free; keeping it
    would double the manifest with entries nobody declares.
    """
    found: set[tuple[str, str]] = set()
    for route in route_table(app):
        for method in getattr(route, "methods", set()) or set():
            if method != "HEAD":
                found.add((method, route.path))
    return found


class TestTheSurfaceIsExactlyWhatIsDeclared:
    def test_no_declared_route_is_missing(self, app):
        """**The sixteen-unregistered-tools test.** A router written, tested and documented
        but never included fails here and nowhere else."""
        missing = EXPECTED_ROUTES - mounted_routes(app)
        assert missing == set(), (
            f"declared but not mounted: {sorted(missing)}. A route that is not on the "
            f"mounted app does not exist for any caller, however well its module is tested."
        )

    def test_no_undeclared_route_is_mounted(self, app):
        """A surface nobody declared is a surface nobody reviewed, and it would make this
        file's manifest a partial list that reads as exhaustive."""
        extra = mounted_routes(app) - EXPECTED_ROUTES
        assert extra == set(), (
            f"mounted but not declared: {sorted(extra)}. Add it to EXPECTED_ROUTES so the "
            f"surface stays reviewed."
        )


class TestOpenApiIsGenerated:
    def test_the_document_generates(self, app):
        schema = app.openapi()
        assert schema["info"]["title"] == "miForge"
        assert schema["openapi"].startswith("3.")

    def test_every_declared_route_appears_in_the_document(self, app):
        """OpenAPI is derived from the mounted app, so a route absent here is absent for
        every generated client as well as for the docs page."""
        schema = app.openapi()
        documented = {
            (method.upper(), path)
            for path, operations in schema["paths"].items()
            for method in operations
        }
        api_routes = {
            # Starlette's path converter is part of the mounted route's path and is NOT
            # part of the OpenAPI path, so `{name:path}` is declared with its converter in
            # EXPECTED_ROUTES and compared without it here. The converter stays in the
            # manifest deliberately: an adapter's name contains a slash whenever its base
            # is a Hub repo id, so a route that silently lost `:path` would 404 the
            # ordinary case, and the two-way comparison against the mounted app is the
            # only place that would notice.
            (m, _without_converters(p))
            for m, p in EXPECTED_ROUTES
            if p not in {"/api/openapi.json", "/api/docs", "/docs/oauth2-redirect"}
        }
        assert api_routes <= documented, f"not in OpenAPI: {sorted(api_routes - documented)}"

    def test_the_document_is_served_over_the_transport(self, client):
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"] == "miForge"


class TestTheRoutesActuallyAnswer:
    """Registered is not the same as routable."""

    def test_health_answers(self, client):
        assert client.get("/api/health").status_code == 200

    def test_the_run_route_answers_rather_than_404ing_on_the_path(self, client, monkeypatch):
        """A route whose *path* does not match answers 404 with ``http_404``; a route that
        matched and found no run answers 404 with ``run_not_found``. **The distinction is
        the test**: without it, a deleted ``include_router`` would look exactly like a
        missing run."""
        from api.deps import session_dependency
        from tests.support.jobs import memory_engine

        engine = memory_engine()
        client.app.dependency_overrides[session_dependency] = lambda: (
            yield from _one_session(engine)
        )
        try:
            response = client.get("/api/v1/runs/does-not-exist")
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "run_not_found"
        finally:
            client.app.dependency_overrides.clear()

    def test_an_unrouted_path_is_a_different_404(self, client):
        response = client.get("/api/v1/nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "http_404"

    def test_the_run_route_serves_the_heartbeat_block(self, client):
        """**Foundation 6.4a, asserted through the transport.** F6 carries a mutation
        control for dropping this block (M-18); without it the run screen must trust the
        socket - which shows a stalled run as live - or invent a threshold from the 60 s
        write interval, which shows a healthy run as stalled."""
        from api.deps import session_dependency
        from tests.support.jobs import insert_run, memory_engine, session_for

        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(session, "r1", status="running")

        client.app.dependency_overrides[session_dependency] = lambda: (
            yield from _one_session(engine)
        )
        try:
            body = client.get("/api/v1/runs/r1").json()
        finally:
            client.app.dependency_overrides.clear()

        assert body["run_id"] == "r1"
        assert set(body["heartbeat"]) == {
            "heartbeat_at",
            "age_seconds",
            "longest_gap_seconds",
            "lapse_threshold_seconds",
            "lapsed",
        }


class TestTheErrorEnvelopeIsInstalled:
    """The envelope is a capability, and *"the handler function exists"* is not evidence
    that it is wired."""

    @pytest.mark.parametrize(
        "exception_type",
        [Refusal, RequestValidationError, StarletteHTTPException, Exception],
        ids=["refusal", "validation", "http", "unhandled"],
    )
    def test_the_handler_is_in_the_live_registry(self, app, exception_type):
        assert exception_type in app.exception_handlers, (
            f"no handler registered for {exception_type.__name__}: a failure of that kind "
            f"would leave as a bare body, and the API would have two error shapes."
        )

    def test_a_refusal_leaves_as_the_envelope_over_the_transport(self, client):
        """Asserted through a real response, not by calling the handler."""
        from api.deps import session_dependency

        def refusing():
            raise refuse("teapot_not_configured", "Set a teapot.", hint="any teapot")
            yield  # pragma: no cover

        client.app.dependency_overrides[session_dependency] = refusing
        try:
            response = client.get("/api/v1/runs/anything")
        finally:
            client.app.dependency_overrides.clear()

        assert response.status_code == 503, "a *_not_configured code maps to 503 by suffix"
        body = response.json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"code", "message", "details"}
        assert body["error"]["code"] == "teapot_not_configured"
        assert body["error"]["details"]["hint"] == "any teapot"

    def test_success_bodies_are_not_wrapped(self, client):
        """ADR-013: the envelope is for failures only. A uniform wrapper makes every client
        unwrap twice and tells it nothing."""
        body = client.get("/api/health").json()
        assert "error" not in body

    def test_an_unmapped_refusal_code_is_a_client_error_not_a_server_error(self):
        """The documented default is **400**. Reporting a deliberate, named refusal as 500
        sends the reader looking for a bug in miForge instead of reading the message, which
        says what to do next."""
        assert status_for("something_nobody_mapped") == 400

    def test_every_not_configured_code_is_503_by_suffix_without_a_table_entry(self):
        """ADR-010 fixes that spelling for every optional sibling, so a new one must not
        need an edit in ``REFUSAL_STATUS`` to be reported as 503 - a missing entry would
        turn *"this sibling is not configured"* into *"your request was bad"*."""
        assert status_for("someservice_not_configured") == 503

    def test_the_mapped_codes_are_what_they_claim(self):
        assert status_for("run_not_found") == 404
        assert status_for("gpu_lease_held") == 409
        assert status_for("database_unavailable") == 503


def _one_session(engine):
    """A ``session_dependency`` replacement over a test engine."""
    from tests.support.jobs import session_for

    with session_for(engine) as session:
        yield session
