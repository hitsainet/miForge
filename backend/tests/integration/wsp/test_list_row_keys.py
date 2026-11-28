"""The keys of the WSP list rows miAssist vendors (feature 020, FR-020.26; the 020-P2 round).

miAssist pins miForge's OpenAPI document as its drift guard
(``tests/cross_repo/test_openapi_live.py``).
``GET /api/v1/wsp/endpoints`` and ``GET /api/v1/wsp/runs`` answer an untyped ``dict``, so that
document names no row field, and the route index carries only methods, paths, content types and
gate markers: 020-P1 added ``history_mode`` and ``runs_refused`` to every endpoint row and
``metered`` to every run row, and miAssist's live test stayed green against it. miAssist keeps each
row's keys by hand instead, as closed objects in ``tests/fixtures/openapi/response_sources/
miforge.json`` (its FU-10 overlays, which its route profiles read for a list's items and id).

So the producer pins them (``tests/support/wsp_list_keys.py``, which miAssist's cross-repo test
reads at the commit it pins) and asserts them here against the routes' real answers: a row that
gains or loses a key goes red in this repo, at the commit that changes it, and the message says
what to re-pin. The cure the FU-10 record names (typing these answers with a ``response_model``)
would retire both.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from tests.integration.wsp.test_gateway import Rig, _running
from tests.integration.wsp.test_gateway import rig as rig
from tests.integration.wsp.test_internal_app_routes import inner as inner
from tests.integration.wsp.test_run_cause import _client
from tests.support.wsp_list_keys import ENDPOINT_ROW_KEYS, RUN_ROW_KEYS

REPIN = (
    "re-pin miAssist: its tests/fixtures/openapi/response_sources/miforge.json overlay for {route} "
    "lists these keys (closed), then scripts.snapshot_shapes and eval/scripts/"
    "write_route_profiles.py, as its newest miForge re-pin record says"
)


@pytest.fixture()
def api(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(rig.world, monkeypatch)


def _keys(rows: list[dict[str, Any]]) -> set[str]:
    assert rows, "the route answered no row: nothing would be compared"
    keys = {frozenset(row) for row in rows}
    assert len(keys) == 1, f"rows of one list carry different keys: {keys}"
    return set(next(iter(keys)))


def test_every_endpoint_row_carries_exactly_the_vendored_keys(api: TestClient) -> None:
    answer = api.get("/api/v1/wsp/endpoints")
    assert answer.status_code == 200, answer.text
    keys = _keys(answer.json()["endpoints"])
    assert keys == ENDPOINT_ROW_KEYS, (
        f"added {sorted(keys - ENDPOINT_ROW_KEYS)}, removed {sorted(ENDPOINT_ROW_KEYS - keys)}: "
        + REPIN.format(route="GET /api/v1/wsp/endpoints")
    )


def test_every_run_row_carries_exactly_the_vendored_keys(rig: Rig, api: TestClient) -> None:
    _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    _running(rig.world, "ep-local", tokens=10_000_000)
    answer = api.get("/api/v1/wsp/runs")
    assert answer.status_code == 200, answer.text
    keys = _keys(answer.json()["runs"])
    assert keys == RUN_ROW_KEYS, (
        f"added {sorted(keys - RUN_ROW_KEYS)}, removed {sorted(RUN_ROW_KEYS - keys)}: "
        + REPIN.format(route="GET /api/v1/wsp/runs")
    )
