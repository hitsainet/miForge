"""The provider registry (FR-016.8, WR-2).

Explicit registration, availability by setting, no fallback.

A provider is an **id string** (``weightspace``) with a version and the operations and styles it
supports.
The backend never imports a provider library (the fence in ``test_wsp_fences.py``): the library runs
only
in the worker image. A provider is *available* when its id is in ``WSP_PROVIDERS_AVAILABLE``; asking
for an
unregistered id is ``provider_unknown`` and for a registered but unavailable one
``provider_unavailable``.
**Nothing falls back to another provider**: a run made by a provider nobody asked for is a
fabricated
provenance (ADR-018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import refuse
from wsp.settings import WSPSettings


@dataclass(frozen=True)
class ProviderDescriptor:
    id: str
    version: str
    operations: frozenset[str]
    styles: frozenset[str]

    def public(self, available: bool) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "operations": sorted(self.operations),
            "styles": sorted(self.styles),
            "available": available,
        }


PROVIDERS: dict[str, ProviderDescriptor] = {}


def register(descriptor: ProviderDescriptor) -> None:
    PROVIDERS[descriptor.id] = descriptor


#: The first provider: the pinned ``weightspace`` 0.3.0, in the worker image only (D4, D6, PF-25).
register(
    ProviderDescriptor(
        id="weightspace",
        version="0.3.0",
        operations=frozenset({"inspect", "create", "repair"}),
        styles=frozenset({"session", "program"}),
    )
)


#: The second provider: ``miwsp_engine``, miStudio's own engine in the worker image (feature 018;
#: FR-018.1; D23, D28). Available only when named in ``WSP_PROVIDERS_AVAILABLE``; the default
#: stays ``WSP_DEFAULT_PROVIDER`` until the switch record exists (FR-018.17). The version is the
#: engine's own ``wsp_worker.miwsp_engine.VERSION`` (the backend never imports the worker; a test
#: pins the two equal).
register(
    ProviderDescriptor(
        id="miwsp_engine",
        version="0.1.0",
        operations=frozenset({"inspect", "create", "repair"}),
        styles=frozenset({"session", "program"}),
    )
)


def listing(settings: WSPSettings) -> list[dict[str, Any]]:
    on = settings.providers_available()
    return [PROVIDERS[k].public(k in on) for k in sorted(PROVIDERS)]


def get(provider_id: str, settings: WSPSettings) -> ProviderDescriptor:
    descriptor = PROVIDERS.get(provider_id)
    if descriptor is None:
        raise refuse(
            "provider_unknown",
            f"No WSP provider is registered as {provider_id!r}; "
            "GET /api/v1/wsp/providers lists them.",
            provider=provider_id,
        )
    if provider_id not in settings.providers_available():
        raise refuse(
            "provider_unavailable",
            f"WSP provider {provider_id!r} is not enabled here; the operator enables it in "
            "WSP_PROVIDERS_AVAILABLE. No other provider is used in its place.",
            provider=provider_id,
        )
    return descriptor


def require_operation(descriptor: ProviderDescriptor, operation: str, style: str) -> None:
    if operation not in descriptor.operations or style not in descriptor.styles:
        raise refuse(
            "provider_operation_unsupported",
            f"Provider {descriptor.id!r} does not support {operation} in {style} style.",
        )
