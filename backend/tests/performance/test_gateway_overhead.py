"""The gateway's overhead, measured on its real path (feature 016; FTASKS 16.5; FPRD §8: under
50 ms p95 per forwarded request, excluding the upstream).

The deployed path, over real loopback HTTP: a client posts to the gateway (``create_gateway_app``
served by uvicorn); the gateway looks the token up and reserves through the REAL internal
application (``create_internal_app``, served by uvicorn, on a migrated database: SQLite here,
PostgreSQL with ``MIFORGE_TEST_DATABASE_URL``), checks the body, forwards to ``FakeUpstream`` and
settles. The upstream's own time is measured inside its handler and subtracted, so the figure is
the gateway's alone (its two internal calls and the database included). Starlette's
``TestClient`` is NOT used on this path: its per-request thread portal costs about 28 ms by
itself (measured 2026-11-19) and would be counted as gateway time.

A wall-clock bound is a measurement, not a CI gate (suite lesson): this runs only with
``MIFORGE_MEASURE_PERF=1`` and skips by name otherwise. It prints the figures it measured.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import numpy as np
import pytest

from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from tests.integration.wsp.test_gateway import Rig, _running, oai, resolver
from tests.integration.wsp.test_gateway import rig as rig
from tests.integration.wsp.test_internal_app_routes import AUTH, SERVICE
from tests.integration.wsp.test_internal_app_routes import inner as inner
from tests.support.served import served

pytestmark = pytest.mark.skipif(
    os.environ.get("MIFORGE_MEASURE_PERF") != "1",
    reason="test_gateway_overhead: a wall-clock measurement; set MIFORGE_MEASURE_PERF=1 to run it",
)

REQUESTS = 300
WARMUP = 30
P95_BUDGET_MS = 50.0


def test_the_gateways_overhead_is_under_its_budget(rig: Rig) -> None:
    from api.wsp_internal_app import create_internal_app
    from wsp_gateway.app import Gateway, create_gateway_app
    from wsp_gateway.settings import GatewaySettings

    upstream_s: list[float] = []

    def timed(request: httpx.Request) -> Any:
        started = time.perf_counter()
        try:
            return rig.upstream(request)
        finally:
            upstream_s.append(time.perf_counter() - started)

    _run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    with served(create_internal_app()) as internal_url:
        internal_ms: list[tuple[str, float]] = []

        def _start(request: httpx.Request) -> None:
            request.extensions["t0"] = time.perf_counter()

        def _end(response: httpx.Response) -> None:
            t0 = response.request.extensions["t0"]
            internal_ms.append((response.request.url.path, (time.perf_counter() - t0) * 1000))

        internal = httpx.Client(
            base_url=internal_url,
            headers=AUTH,
            timeout=10,
            event_hooks={"request": [_start], "response": [_end]},
        )
        gateway = Gateway(
            GatewaySettings(
                wsp_internal_url=internal_url,
                wsp_gateway_service_token=SERVICE,
                wsp_gateway_private_key=rig.private,
            ),
            internal=internal,
            upstream_client=httpx.Client(transport=httpx.MockTransport(timed)),
            resolver=resolver,
        )
        gateway.registry.rebuild()
        with (
            served(create_gateway_app(gateway)) as gateway_url,
            httpx.Client(base_url=gateway_url, timeout=10) as client,
        ):
            overhead_ms: list[float] = []
            body = httpx.Request("POST", "/", json=oai(max_tokens=1024)).content
            for i in range(REQUESTS + WARMUP):
                before = len(upstream_s)
                started = time.perf_counter()
                answer = client.post(
                    "/v1/chat/completions", content=body, headers={"x-api-key": token}
                )
                total = time.perf_counter() - started
                assert answer.status_code == 200, (i, answer.text)
                assert len(upstream_s) == before + 1  # forwarded exactly once
                if i >= WARMUP:
                    overhead_ms.append((total - upstream_s[-1]) * 1000)
        internal.close()
    figures = np.percentile(overhead_ms, [50, 95, 99])
    by_path: dict[str, list[float]] = {}
    for path, ms in internal_ms[WARMUP * 2 :]:
        by_path.setdefault(path.rsplit("/", 1)[0] if "by-token" in path else path, []).append(ms)
    for path, values in sorted(by_path.items()):
        print(f"\n  internal {path}: n {len(values)}, p50 {np.percentile(values, 50):.2f} ms")
    print(
        f"\ngateway overhead on {rig.world.engine.dialect.name}, {REQUESTS} requests over "
        f"loopback HTTP: p50 {figures[0]:.2f} ms, p95 {figures[1]:.2f} ms, "
        f"p99 {figures[2]:.2f} ms, max {max(overhead_ms):.2f} ms"
    )
    assert figures[1] < P95_BUDGET_MS
