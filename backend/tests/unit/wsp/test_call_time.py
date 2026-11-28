"""One editor call's time limits are ordered (fix 2026-11-22, D128 round 7).

Production, 2026-11-21: ``miwsp_engine``'s client and the gateway's upstream both waited 900 s
and raced (run ``wsp_e571f6971fd2`` lost its session to ``ReadTimeout``). Every worker client
must outlast the gateway, so the gateway answers first. What is asserted is the CONSTRUCTED
clients' timeouts (the engine's ``GatewayClient`` and the gateway's own production ``build``),
never the source text. (The retired ``weightspace`` adapter's SDK clients were checked here
until 018-P4.)
"""

from __future__ import annotations

from wsp_gateway.__main__ import build
from wsp_gateway.settings import GatewaySettings
from wsp_worker import call_time
from wsp_worker.miwsp_engine.client import GatewayClient

GATEWAY = {"ip": "10.152.183.20", "port": 8080, "token": "tok"}


def _phases(timeout: object) -> list[float | None]:
    return [getattr(timeout, p) for p in ("read", "connect", "write", "pool")]


def test_the_engine_client_outlasts_any_upstream_timeout_the_gateway_admits() -> None:
    ceiling = GatewaySettings.model_fields["wsp_gateway_upstream_timeout_s"].default
    assert ceiling == call_time.GATEWAY_UPSTREAM_TIMEOUT_S
    for kind in ("openai_compatible", "anthropic"):
        client = GatewayClient(kind, "m", GATEWAY)
        try:
            phases = _phases(client._http.timeout)
        finally:
            client.close()
        assert phases == [call_time.CLIENT_TIMEOUT_S] * 4, kind
        assert all(p is not None and p > ceiling for p in phases), kind


def test_the_gateways_production_upstream_uses_its_setting_inside_the_clients() -> None:
    for value in (call_time.GATEWAY_UPSTREAM_TIMEOUT_S, 300.0):
        gateway = build(GatewaySettings(wsp_gateway_upstream_timeout_s=value))
        try:
            assert _phases(gateway.upstream.timeout) == [value] * 4
        finally:
            gateway.upstream.close()
        assert value < call_time.CLIENT_TIMEOUT_S


def test_the_deadline_sits_inside_the_gateways_upstream_timeout() -> None:
    deadline = call_time.call_deadline(call_time.GATEWAY_UPSTREAM_TIMEOUT_S, None)
    assert deadline is not None
    assert call_time.MIN_CALL_TIME_S <= deadline < call_time.GATEWAY_UPSTREAM_TIMEOUT_S
    assert deadline < call_time.CLIENT_TIMEOUT_S
