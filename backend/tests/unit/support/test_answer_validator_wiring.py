# Origin: miDataworks (Onegaishimas/miDataworks)
#   backend/tests/unit/support/test_answer_validator_wiring.py @ 648434ee
# Mode: adapt (docs/REUSE.md). Kept: the registration check, the wrap check, the inner run red and
# green, the strict-form, message and status checks, report mode's strict expected failure.
# Changed: miForge's route (GET /api/v1/packs, a closed model), its route table, and its conftest's
# registration (pytest_configure, not pytest_plugins: backend/tests is not the rootdir); no census
# of app-building files (the class-level hook is what covers them).
"""The answer validator is wired and it fires (021 FR-021.12; FTASKS 2.1 to 2.4).

- It is registered by the root conftest, in the LIVE session, under its module's name.
- Every FastAPI app is wrapped: a fresh ``create_app()`` and a bare ``FastAPI()``.
- It saw answers in this session (so it is not merely installed).
- It fires: an inner pytest run in which one real route answers off its contract fails, naming the
  route, the key and the model; the same run with a conforming answer passes, and so does the
  off-contract run without the plugin (so the red is the validator's).
- Its strict form refuses an undeclared key that the published form accepts.
- A backend route the contract lacks fails; the WSP worker's internal app is skipped by name.
- Report mode is a branch-only switch (FTASKS 8.1 turns it off); a strict expected failure holds it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.support import answer_validator

TESTS = Path(__file__).resolve().parents[2]
BACKEND = TESTS.parent
PLUGIN = "tests.support.answer_validator"
PACKS = "GET /api/v1/packs"


def test_the_plugin_is_registered_by_the_root_conftest(request: pytest.FixtureRequest) -> None:
    manager = request.config.pluginmanager
    assert manager.get_plugin(PLUGIN) is answer_validator, "the validator is not a live plugin"
    conftest = next(
        plugin
        for plugin in manager.get_plugins()
        if Path(getattr(plugin, "__file__", "") or "").resolve() == TESTS / "conftest.py"
    )
    assert getattr(conftest, "ANSWER_VALIDATOR", None) == PLUGIN
    assert answer_validator.installed()


def test_every_fastapi_app_is_wrapped() -> None:
    from api.app import create_app

    for app in (create_app(), FastAPI()):
        assert isinstance(app, FastAPI), "the app object must stay a FastAPI"
        assert isinstance(app.build_middleware_stack(), answer_validator.ValidatingASGI)


def test_the_validator_saw_an_answer_in_this_session() -> None:
    from api.app import create_app

    before = sum(answer_validator.SESSION["answers"].values())
    with TestClient(create_app()) as client:
        assert client.get("/api/v1/selectors").status_code == 200
    assert sum(answer_validator.SESSION["answers"].values()) == before + 1


_INNER = """
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from tests.support import answer_validator


def test_one_real_route_answers():
    answer_validator.REPORT_ONLY = False  # P2 only: FTASKS 8.1 turns report mode off
    from api.app import create_app
    from api.route_table import mounted_api_routes

    app = create_app()
    ctx = next(
        c for c in mounted_api_routes(app)
        if c.path_format == "/api/v1/packs" and "GET" in (c.methods or ())
    )
    ctx._effective_route.app = JSONResponse({body})
    with TestClient(app) as client:
        assert client.get("/api/v1/packs").status_code == 200
"""


def _inner_run(
    tmp_path: Path, body: dict[str, Any], with_plugin: bool
) -> subprocess.CompletedProcess[str]:
    test = tmp_path / "test_inner.py"
    test.write_text(textwrap.dedent(_INNER.format(body=repr(body))), encoding="utf-8")
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist"]
    if with_plugin:
        args += ["-p", PLUGIN]
    env = {**os.environ, "PYTHONPATH": f"{BACKEND / 'src'}{os.pathsep}{BACKEND}"}
    env.pop("PYTEST_XDIST_WORKER", None)
    return subprocess.run(
        [*args, str(test)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_an_answer_off_its_contract_fails_the_test_that_produced_it(tmp_path: Path) -> None:
    red = _inner_run(tmp_path, {"packs": [], "unexpected": True}, with_plugin=True)
    assert red.returncode != 0, red.stdout[-3000:]
    assert f"{PACKS} answered 200 off its contract" in red.stdout, red.stdout[-3000:]
    assert "undeclared key 'unexpected' (model PackListingDoc)" in red.stdout, red.stdout[-3000:]


def test_a_conforming_answer_passes_and_without_the_plugin_nothing_checks(tmp_path: Path) -> None:
    green = _inner_run(tmp_path, {"packs": []}, with_plugin=True)
    assert green.returncode == 0, green.stdout[-3000:]
    unchecked = _inner_run(tmp_path, {"packs": [], "unexpected": True}, with_plugin=False)
    assert unchecked.returncode == 0, unchecked.stdout[-3000:]


def test_the_strict_form_refuses_an_undeclared_key_the_published_form_accepts() -> None:
    check = answer_validator.answer_check()
    assert check.check(PACKS, 200, {"packs": []}) == []
    failures = check.check(PACKS, 200, {"packs": [], "unexpected": True})
    assert [f.message for f in failures] == ["undeclared key 'unexpected' (model PackListingDoc)"]
    published = check.contract
    schema = {"$ref": "#/responses/" + PACKS.replace("/", "~1") + "/200", **published}
    jsonschema.Draft202012Validator(schema).validate({"packs": [], "unexpected": True})


def test_a_missing_field_a_wrong_type_and_an_undeclared_status_are_named() -> None:
    check = answer_validator.answer_check()
    assert ["'packs' is a required property"] == [f.message for f in check.check(PACKS, 200, {})]
    wrong = check.check(PACKS, 200, {"packs": "all"})
    assert [f.json_path for f in wrong] == ["packs"]
    assert "status 202 is not declared" in check.check(PACKS, 202, {"packs": []})[0].message
    assert "a JSON answer from a route absent" in check.check("GET /nowhere", 200, {})[0].message
    empty_204 = "DELETE /api/v1/pack-drafts/{draft_id}"
    assert check.check(empty_204, 204, None) == [], "a 204's empty answer has no body to check"
    assert "status 204 answers no body" in check.check(empty_204, 204, {"x": 1})[0].message
    tensors = "GET /api/v1/wsp/candidates/{candidate_id}/tensors"
    assert "a route excluded" in check.check(tensors, 200, {"x": 1})[0].message
    assert check.check_media(tensors, 200, "application/octet-stream") == []
    assert "is not declared" in check.check_media(tensors, 200, "text/plain")[0].message
    patch = "GET /api/v1/wsp/candidates/{candidate_id}/patch/{name}"
    assert check.check_media(patch, 200, "application/octet-stream") == []
    assert check.check_media(empty_204, 204, "") == []


def test_a_backend_route_the_contract_lacks_fails_and_the_internal_app_is_skipped() -> None:
    class Route:
        def __init__(self, endpoint: Any) -> None:
            self.endpoint = endpoint
            self.path = "/somewhere"

    def unknown() -> None:  # pragma: no cover - never called
        return None

    unknown.__module__ = "api.v1.endpoints.not_a_router"
    key = answer_validator.route_key("GET", Route(unknown))
    assert key is not None and not answer_validator.answer_check().knows(key)
    unknown.__module__ = "api.wsp_internal_app"
    assert answer_validator.route_key("GET", Route(unknown)) is None
    unknown.__module__ = "tests.unit.something"
    assert answer_validator.route_key("GET", Route(unknown)) is None


def test_every_route_of_the_app_is_known_to_the_map() -> None:
    from fastapi.routing import APIRoute

    from api.app import create_app
    from api.route_table import mounted_api_routes

    check = answer_validator.answer_check()
    for ctx in mounted_api_routes(create_app()):
        if not isinstance(ctx.original_route, APIRoute) or not ctx.original_route.include_in_schema:
            continue
        for method in ctx.methods or ():
            key = answer_validator.route_key(method, ctx.original_route)
            assert key is not None and check.knows(key), (method, ctx.path_format, key)


def test_the_check_reads_the_committed_contract() -> None:
    from schemas.answers_contract import CONTRACT_PATH

    committed = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert answer_validator.answer_check().contract == committed


def test_report_mode_is_off() -> None:
    assert answer_validator.REPORT_ONLY is False
