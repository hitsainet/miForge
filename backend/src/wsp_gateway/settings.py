"""Gateway settings: environment only, secrets hidden from reprs and validation errors.

Unlike ``wsp.settings.WSPSettings`` (which has no private-key field, D15, R1B-M9), the gateway's
settings carry ``WSP_GATEWAY_PRIVATE_KEY``: the gateway Deployment is the one object that mounts
the Secret ``miforge-wsp-gateway`` (``test_wsp_gateway_manifest.py``).
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", case_sensitive=False, hide_input_in_errors=True
    )

    wsp_gateway_host: str = (
        "0.0.0.0"  # a pod listens on its own IP; the policy admits only the worker
    )
    wsp_gateway_port: int = 8080
    #: The backend's internal application (``miforge-wsp-internal:8001``).
    wsp_internal_url: str = ""
    wsp_gateway_service_token: SecretStr | None = Field(default=None, repr=False)
    wsp_gateway_private_key: SecretStr | None = Field(default=None, repr=False)
    wsp_external_host_allowlist: str = "api.anthropic.com"
    wsp_local_host_allowlist: str = "miserver-backend.miserver.svc"
    #: Decrypted keys are cached in memory only, for at most this long (FTDD §5.5).
    wsp_gateway_key_cache_s: float = 300.0
    #: Upstream timeout; one call, never retried (PF-49).
    wsp_gateway_upstream_timeout_s: float = 900.0
