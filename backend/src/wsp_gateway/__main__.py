"""``python -m wsp_gateway``: rebuild the registry from the backend, then serve on 8080."""

from __future__ import annotations

import logging
import sys

import httpx

from wsp_gateway.app import LOGGER_NAME, Gateway, create_gateway_app
from wsp_gateway.settings import GatewaySettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(LOGGER_NAME)


def build(settings: GatewaySettings) -> Gateway:
    service = settings.wsp_gateway_service_token
    # Without the service token every lookup is refused by the backend, so every session is
    # refused here too: the pod serves /health until the operator's FTASKS 11.2 step.
    headers = {} if service is None else {"authorization": f"Bearer {service.get_secret_value()}"}
    internal = httpx.Client(
        base_url=settings.wsp_internal_url or "http://wsp-internal.not-configured.invalid",
        headers=headers,
        timeout=30.0,
        follow_redirects=False,
        trust_env=False,  # review OR3-016-4: no proxy from the environment, as for the upstream
    )
    upstream = httpx.Client(
        timeout=settings.wsp_gateway_upstream_timeout_s, follow_redirects=False, trust_env=False
    )
    return Gateway(settings, internal=internal, upstream_client=upstream)


def main() -> int:
    import uvicorn

    settings = GatewaySettings()
    gateway = build(settings)
    # Spend is the backend's (R1B-M7): the registry is a cache of it. A failed rebuild leaves it
    # empty, and every token then resolves through by-token (with spend) or is refused.
    from wsp_gateway.registry import RegistryUnavailable

    try:
        logger.info("registry rebuilt with %d active sessions", gateway.registry.rebuild())
    except RegistryUnavailable:
        logger.warning("registry not rebuilt: the internal application did not answer")
    # The value filter guards every handler, not only this package's logger (R1B-M22).
    for handler in logging.getLogger().handlers:
        handler.addFilter(gateway.secrets)
    uvicorn.run(
        create_gateway_app(gateway),
        host=settings.wsp_gateway_host,
        port=settings.wsp_gateway_port,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
