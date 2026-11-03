"""How F3 reaches a generation provider  (FR-003.5, FR-003.17; ADR-009, R-02.49, R-02.11).

**F3 resolves a provider; it never constructs one.** The ratified split (F4's FR-004.1,
foundation task 3.8) is:

* the **interface** - :mod:`providers.base`, the foundation's;
* the **OpenAI-compatible HTTP implementation** - F3's, at
  :mod:`providers.openai_compatible`, because R-02.11 makes the in-process provider
  unusable as a judge and so F3 cannot wait on F4 for it;
* the **registry, the provider identity and the in-process path** - F4's.

``selector-config-v1.json`` says ``judge_model.provider`` is *"a provider key from F4's
registry, never a sibling's name"*. F4 does not exist yet, so this module declares the
narrow port its registry will satisfy - the same pattern as
:mod:`services.selectors.review_port` for F10 - and nothing under ``services/selectors/``
constructs a provider. A static test asserts that, and it matters for a reason beyond
tidiness: a second construction path in this package would be exactly the
import-by-module-path resolution the selector registry's own rule forbids, one directory
over.

**With no resolver bound the judge type is unavailable by name**, not silently in-process.
That is R-02.11 enforced by absence: there is nothing to fall back to, which is the point.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from providers.base import InferenceProvider


@runtime_checkable
class ProviderResolver(Protocol):
    """Resolve a provider key to a provider. **F4's registry satisfies this.**"""

    def available(self, key: str) -> bool:
        """Whether ``key`` names a provider that can be used right now."""
        ...

    def resolve(self, key: str) -> InferenceProvider:
        """Return the provider for ``key``, or raise a named ``Refusal``.

        Refuses rather than returning ``None``: a caller handed ``None`` writes
        ``provider or default_provider``, and on the judge path the only default available
        is the model being trained.
        """
        ...


# --------------------------------------------------------------------------------------
# The binding. One module-level slot, set by the application wiring (or by a test).
# --------------------------------------------------------------------------------------

_resolver: ProviderResolver | None = None


def bind_provider_resolver(resolver: ProviderResolver | None) -> None:
    """Install the resolver F4's registry provides. ``None`` unbinds it.

    A function rather than a mutable public global so that the unbind path exists and is
    used by tests: a test that configured a judge and could not configure it *away* would
    make the standalone case (AC20) unassertable.
    """
    global _resolver
    _resolver = resolver


def current_provider_resolver() -> ProviderResolver | None:
    """The bound resolver, or ``None`` when F4's registry has not been wired.

    ``None`` is a legitimate, expected state - it is the standalone path - so callers
    report ``judge_provider_not_configured`` by name rather than treating it as a bug.
    """
    return _resolver
