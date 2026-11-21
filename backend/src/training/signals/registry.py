"""The one registry of signal sources  (FR-002.1; BRD-02 §5.2's preamble).

**One resolution path, and deleting a registration must make a test red.** The step, the
loop, both API routes and pre-flight all resolve through :func:`get`. There is no
``importlib`` path, no module-path resolution and no fallback - because a scan makes
*"is it registered?"* untestable and an unregistered source silent, which is the failure
this project's whole review discipline was written after: a sibling shipped **sixteen
fully-implemented, unit-tested, documented MCP tools that were never registered with the
server**, so the suite was green while the feature did not exist for any caller.

## The four names are the manifest's, and that is not a style choice

``checkpoint-manifest-v1.json``'s ``signal_source`` is a **required** field whose enum is
exactly ``corpus``, ``self-generation``, ``preference-pairs``, ``reward`` - verified by
parsing the file, not by reading about it. A source whose name cannot be written into a
manifest **cannot be saved**, so :func:`names` is asserted equal to that enum by test.

That is also why the first release registers in-tree only (FPRD OQ-8, answered upstream):
the enum is closed, so a fifth source would be a schema change in two repositories plus a
plugin - *"the plugin interface is open, the recorded vocabulary is not."*
"""

from __future__ import annotations

from collections.abc import Mapping

from core.errors import refuse
from training.signals.base import SourceCapability, SourceFactory

_FACTORIES: dict[str, SourceFactory] = {}
_CAPABILITIES: dict[str, SourceCapability] = {}


def register(name: str, factory: SourceFactory, capability: SourceCapability) -> None:
    """Register one source. **Raises on a duplicate name, loudly, at import.**

    Not last-one-wins: two plugins claiming one name is a real conflict whose resolution
    would otherwise depend on import order, and the symptom would be a run trained by a
    source nobody chose. :func:`training.signals.ensure_registered` is idempotent for the
    repeated-import case, which is the one that is normal.
    """
    if name in _FACTORIES:
        raise refuse(
            "signal_source_already_registered",
            f"A signal source named {name!r} is already registered. Refusing rather than "
            f"replacing it: last-one-wins would make which source trains a run depend on "
            f"import order.",
            name=name,
            registered=sorted(_FACTORIES),
        )
    if capability.name != name:
        raise refuse(
            "signal_source_capability_mismatch",
            f"Source {name!r} declares the capability of {capability.name!r}. The two must "
            f"agree: the capability is what GET /api/v1/signal-sources serves, and a "
            f"listing that names a different source than the registry resolves is a "
            f"listing nobody can act on.",
            name=name,
            declared=capability.name,
        )
    _FACTORIES[name] = factory
    _CAPABILITIES[name] = capability


def get(name: str) -> SourceFactory:
    """The factory for ``name``. **Refuses an unknown name, listing the four** (E-1).

    Raises:
        Refusal: ``signal_source_unknown``.
    """
    factory = _FACTORIES.get(name)
    if factory is None:
        raise refuse(
            "signal_source_unknown",
            f"No signal source named {name!r}. miForge ships with {sorted(_FACTORIES)} - "
            f"the four values of signal_source in checkpoint-manifest-v1.json, which is a "
            f"required field with a closed enum, so a run using a fifth source could not "
            f"be saved at all. Choose one of the four.",
            name=name,
            registered=sorted(_FACTORIES),
        )
    return factory


def capability(name: str) -> SourceCapability:
    """What ``name`` needs. Refuses an unknown name through :func:`get`'s one refusal."""
    get(name)
    return _CAPABILITIES[name]


def names() -> tuple[str, ...]:
    """Every registered source name, sorted. Asserted equal to the manifest's enum."""
    return tuple(sorted(_FACTORIES))


def capabilities() -> Mapping[str, SourceCapability]:
    """Every capability, for the listing route. A copy: the registry is not a mutable view."""
    return dict(_CAPABILITIES)


def _reset_for_tests() -> None:
    """Empty the registry. **Only a test calls this**, and the reachability control does.

    Named with the leading underscore and the reason, because a production caller of this
    would make every source unresolvable at an arbitrary later moment.
    """
    _FACTORIES.clear()
    _CAPABILITIES.clear()
