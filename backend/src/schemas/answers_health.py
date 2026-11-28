"""The answer of ``GET /api/health`` (feature 021, P3): ``api/health.py``.

Documentation only (MF9): the route keeps ``response_model=None`` and serves what it builds. The
body is the same at 200 and 503.
"""

from pydantic import BaseModel


class HealthDependency(BaseModel):
    """``api/health.py:_probe_postgres`` / ``_probe_redis`` / ``_probe_within_deadline``."""

    status: str
    detail: str | None


class HealthDependencies(BaseModel):
    """``api/health.py:DEPENDENCY_PROBES``: the required dependencies, each probed separately."""

    postgres: HealthDependency
    redis: HealthDependency


class HealthIntegration(BaseModel):
    """``api/health.py:health_payload``: one optional integration, configured or not."""

    configured: bool
    code: str | None
    capabilities_unavailable: list[str]


class HealthIntegrations(BaseModel):
    """``core/config.py:Settings.configured_integrations``: every optional integration by name."""

    millm: HealthIntegration
    midataworks: HealthIntegration
    activation_source: HealthIntegration
    wsp: HealthIntegration
    miworkbench: HealthIntegration


class HealthLegacyNameInUse(BaseModel):
    """``services/legacy_names/store.py:block``: an ``in_use`` entry."""

    kind: str
    old: str
    new: str | None
    last_seen: str


class HealthLegacyName(BaseModel):
    """``services/legacy_names/store.py:block``: an ``also_set``/``conflicts``/``retired_but_set``
    entry."""

    old: str
    new: str | None


class HealthLegacyNames(BaseModel):
    """``services/legacy_names/store.py:block``: exactly four lists, each sorted by ``old``."""

    in_use: list[HealthLegacyNameInUse]
    also_set: list[HealthLegacyName]
    conflicts: list[HealthLegacyName]
    retired_but_set: list[HealthLegacyName]


class HealthLegacyNamesMeta(BaseModel):
    """``services/legacy_names/store.py:meta``."""

    error: str | None
    in_use_window_hours: int


class HealthGetHealthAnswer(BaseModel):
    """``api/health.py:get_health`` (``health_payload``)."""

    status: str
    environment: str
    dependencies: HealthDependencies
    integrations: HealthIntegrations
    unconfigured_integrations: list[str]
    degraded_dependencies: list[str]
    legacy_names: HealthLegacyNames
    legacy_names_meta: HealthLegacyNamesMeta
