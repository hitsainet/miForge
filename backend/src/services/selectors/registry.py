"""The one selector/grader registry  (FR-003.1, FR-003.12; R-02.9).

**One registry, one resolution path.** Every consumer - F2's signal sources, F4's scoring,
F7's dry-run, the two endpoints - resolves a type through :func:`get`. There is no
``importlib`` path, no name-to-module mapping and no fallback.

## Why registration is explicit and not a directory scan

A scan makes *"is it registered?"* untestable and an unregistered plugin silent. That is
the exact failure this project's review discipline was written after: a sibling shipped
**sixteen fully-implemented, unit-tested, documented MCP tools that were never registered
with the server.** Every one of their tests passed by importing the module directly, so the
suite was green and the docs said ✅ while the feature did not exist for any caller.

So: one explicit registration block, in :mod:`services.selectors`, and the reachability
test asserts the five names are present **in this live object** as the service resolves it.
Deleting one ``register(...)`` line must make that test red - observed, not assumed.

## This is not the provider registry

The provider registry is **F4's** (its FR-004.1) and lives in ``providers/``. This one maps
selector *kinds* to factories. Two different objects; F3 owns only this one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from core.errors import refuse
from schemas.selector import SelectorKind
from services.selectors.types import Capability, CapabilityView, Requirement

if TYPE_CHECKING:  # pragma: no cover - typing only
    from services.selectors.resolve import ResolvedSelector
    from services.selectors.settings import SelectorSettings

#: A factory turns a resolved configuration into something satisfying
#: :class:`schemas.selector.Selector` (and, where the type offers one,
#: :class:`schemas.selector.Grader`). Typed loosely here because the resolved object is
#: defined in :mod:`services.selectors.resolve`, which imports this module.
PluginFactory = Callable[["ResolvedSelector"], object]

_factories: dict[str, PluginFactory] = {}
_capabilities: dict[str, Capability] = {}


def register(kind: SelectorKind, factory: PluginFactory, capability: Capability) -> None:
    """Register one selector type. **Re-registering a name raises.**

    A silent overwrite is how two plugins answer to one name and the second one wins
    depending on import order - a failure that reproduces only sometimes, which is the
    worst kind.

    Raises:
        ValueError: if ``kind`` is already registered, or if the capability disagrees with
            the name it is registered under. The second check exists because a copy-pasted
            registration line with the wrong ``Capability`` would register a type that
            describes a different one, and the listing endpoint would then be confidently
            wrong.
    """
    if kind in _factories:
        raise ValueError(
            f"selector kind {kind!r} is already registered; a silent overwrite is how two "
            f"plugins answer to one name and import order decides which"
        )
    if capability.kind != kind:
        raise ValueError(
            f"capability describes {capability.kind!r} but is registered under {kind!r}; "
            f"the listing endpoint would report the wrong requirement for this type"
        )
    _factories[kind] = factory
    _capabilities[kind] = capability


def registered_kinds() -> tuple[str, ...]:
    """The registered names, sorted. What a refusal lists so the operator can see them."""
    return tuple(sorted(_factories))


def get(kind: str) -> PluginFactory:
    """The factory for ``kind``. **Refuses an unknown name, listing what is registered.**

    Raises:
        Refusal: ``selector_type_unknown``. A generic validation failure would make the
            operator guess; ADR-018 forbids the code guessing, and it is no better to make
            the reader do it.
    """
    factory = _factories.get(kind)
    if factory is None:
        raise refuse(
            "selector_type_unknown",
            f"No selector type named {kind!r} is registered. Use one of "
            f"{list(registered_kinds())}, or correct the configuration's 'kind'.",
            requested=kind,
            registered=list(registered_kinds()),
        )
    return factory


def capability(kind: str) -> Capability:
    """The declaration for ``kind``. Refuses an unknown name the same way :func:`get` does."""
    declared = _capabilities.get(kind)
    if declared is None:
        get(kind)  # raises the named refusal, with the same message and details
        raise AssertionError("unreachable")  # pragma: no cover - defensive
    return declared


def list_capabilities(settings: SelectorSettings) -> list[CapabilityView]:
    """Every registered type with its **live** availability (FR-003.12).

    ``settings`` is an **argument**, not a module global. Reading a global would freeze
    availability at import, so configuring miDataworks would not change the answer until
    something restarted - and an operator reading "unavailable" after configuring the
    service would reasonably conclude the configuration was wrong.
    """
    from services.selectors.availability import availability_of

    views: list[CapabilityView] = []
    for kind in registered_kinds():
        declared = _capabilities[kind]
        available, reason = availability_of(declared.requires, settings)
        views.append(
            CapabilityView(
                name=declared.kind,
                forms=tuple(sorted(form.value for form in declared.forms)),
                requires=declared.requires,
                summary=declared.summary,
                available=available,
                unavailable_reason=reason,
            )
        )
    return views


def _reset_for_tests() -> None:
    """Empty the registry. **Tests only**, and the name says so.

    Needed by the reachability test's own negative control: it must be able to prove that
    an *empty* registry fails, so that the green it reports against a full one means
    something. Without this the control would have to edit the source file.
    """
    _factories.clear()
    _capabilities.clear()


# Re-exported so callers do not need a second import to describe a requirement.
__all__ = [
    "Capability",
    "CapabilityView",
    "PluginFactory",
    "Requirement",
    "capability",
    "get",
    "list_capabilities",
    "register",
    "registered_kinds",
]
