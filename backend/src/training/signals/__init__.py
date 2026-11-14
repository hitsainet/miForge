"""Signal sources - **F2**  (R-02.4 - R-02.8).

The four things that can push the weights, and the record of which one did. A signal source
answers one question - *what is the next batch?* - and four implementations answer it behind
one Protocol, so F1's step never learns which one it is talking to.

## THE REGISTRATION BLOCK BELOW IS THE WIRING, AND IT IS THE POINT

Four explicit :func:`training.signals.registry.register` calls, in one place, run as an
import side effect of this package. **Deleting one must make the reachability test red.** A
directory scan was rejected, and the reason is the one the selector registry gives one
directory over: a scan makes *"is it registered?"* untestable and a missing source
**silent** - which is the failure this project's review discipline was written after, when a
sibling shipped sixteen fully-implemented, unit-tested, documented MCP tools that were never
registered with its server. Every one of their tests passed by importing the module
directly, so the suite was green and the documentation said done while the feature did not
exist for any caller.

## The four names are ``checkpoint-manifest-v1.json``'s, exactly

``signal_source`` is a **required** field there with a closed four-value enum, so **a source
whose name cannot be written into a manifest cannot be saved**, and a test asserts the
registry's key set equals that enum rather than retyping it. That is also why the first
release registers in-tree only: adding a fifth source is a schema change in two repositories
plus a plugin - *the plugin interface is open, the recorded vocabulary is not.*

:func:`ensure_registered` exists because an import side effect that nobody triggers is not a
side effect. Anything resolving a source calls it first, and it is idempotent.
"""

from __future__ import annotations

from training.signals import corpus, preference_pairs, registry, reward, self_generation

#: The four sources. The tuple is the single declaration of the surface: a source absent
#: here does not exist for any caller, however well its module is tested.
_REGISTRATIONS = (
    (corpus.NAME, corpus.factory, corpus.CAPABILITY),
    (self_generation.NAME, self_generation.factory, self_generation.CAPABILITY),
    (preference_pairs.NAME, preference_pairs.factory, preference_pairs.CAPABILITY),
    (reward.NAME, reward.factory, reward.CAPABILITY),
)


def ensure_registered() -> None:
    """Register the four built-in sources, idempotently.

    Idempotent rather than raising on a second call, while
    :func:`training.signals.registry.register` itself **does** raise on a duplicate name.
    A repeated *import* is normal and must not fail; a second plugin claiming one name is a
    real conflict whose resolution would otherwise depend on import order.
    """
    existing = set(registry.names())
    for name, factory, capability in _REGISTRATIONS:
        if name not in existing:
            registry.register(name, factory, capability)


ensure_registered()

__all__ = ["ensure_registered", "registry"]
