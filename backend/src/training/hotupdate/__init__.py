"""BRD-02 §5.10 - the optional in-place hot-update mode. **The whole of F11 lives here.**

*Requirements:* R-02.1, R-02.2, R-02.3, R-02.30, R-02.31, R-02.32, R-02.33, R-02.34 and
**R-02.52**. *Normative for the wire contract:* BRD-01 (miLLM - Attachable Weights), cited
by requirement ID; **no field list of that document is restated anywhere in miForge.**

## What this mode is, and why it is fenced

Everywhere else in miForge, improving a model means writing a new checkpoint directory -
and atomicity is free: temp directory, ``fsync``, ``rename`` (ADR-015). Here the parameters
live in another process's GPU memory and a multi-tensor write to them is not atomic, so
atomicity has to be **built out of a protocol**: quiesce, a bracket that suspends the
availability timeout, a separate watchdog that fails closed, and a recovery path for when it
does not. **Every hard correctness problem this project ever had is in this one package, by
design** - that is where the 2026-10-13 standalone reframe put it.

ADR-022 fences it, and the fence is a structural property rather than a convention:

* **One importer.** Exactly one module in ``backend/src/`` reaches this package, and it
  reaches it **dynamically**: ``training/acquisition/__init__.py`` imports it by name inside
  a guard. A *static* import would be a dependency edge from the core path into §5.10, which
  F1's own fence test forbids by AST - and rightly, because the standalone test runs with
  this directory **deleted**.
* **Two seams.** That registration, and one settings block in ``core/config.py``. A third is
  a build failure, not a review comment: ``tests/unit/test_hotupdate_fence.py`` asserts the
  count and carries a self-test proving its scanner sees each form a reference can take.
* **Nothing outside this package names a weights endpoint.** Not a path, not an f-string,
  not a joined URL. The same guard asserts that too.

## Reachability

:func:`install` is what makes the mode exist for a caller. **Removing the call to it must
turn ``tests/unit/test_reachability_f11.py`` red** - the whole review discipline of this
project exists because a sibling shipped sixteen fully-implemented, unit-tested, documented
MCP tools that were never registered with its server, so every test passed by importing the
module directly while the feature existed for nobody.

## What it is built against

miLLM has **no** weight attach/detach/quiesce/resume/commit/save/load code: BRD-01 is
specced and unimplemented, and its only ``attach`` is SAE attach, which is unrelated. So
every response shape this package reads is **BRD-01's requirement text**, the fake surface
in ``tests/support/fake_weights_surface.py`` cites the requirement behind each route, and
the live integration is marked ``upstream`` and does not run in CI. *The fake is this
feature's reading of a contract with no implementation; the citation is the only thing a
reviewer can check it against.*
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from training.hotupdate.settings import (
    NOT_AVAILABLE_REASON,
    UNAVAILABLE_CAPABILITIES,
    HotUpdateSettings,
)
from training.hotupdate.settings import (
    load as load_settings,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from training.hotupdate.attach import AttachAcquirer

__all__ = [
    "NOT_AVAILABLE_REASON",
    "UNAVAILABLE_CAPABILITIES",
    "AttachAcquirer",
    "HotUpdateSettings",
    "install",
    "load_settings",
]


def __getattr__(name: str) -> Any:
    """Export :class:`AttachAcquirer` **lazily** (PEP 562).

    ``from training.hotupdate import AttachAcquirer`` still works, and nothing is imported
    until something asks. The eager form was a real cycle rather than a hypothetical one:
    this package's ``attach`` module is reached *from* ``training/acquisition/__init__.py``,
    so importing it at the top of this file made the direction depend on which of the two
    packages a caller happened to import first.
    """
    if name == "AttachAcquirer":
        from training.hotupdate.attach import AttachAcquirer

        return AttachAcquirer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def install(register: object) -> AttachAcquirer:
    """Register the ``attach`` acquirer into F1's registry. **The one seam, called once.**

    Takes F1's ``register`` callable as an argument rather than importing
    ``training.acquisition`` itself: this function runs *while* that package is being
    imported (it is called from the bottom of its ``__init__``), so importing it back would
    re-enter a partially initialised module. Passing the callable also makes the seam
    explicit in both directions - one line there, one function here - and lets a test call
    it with a recording double to assert **the payload and the call count**, not merely
    that something was registered.
    """
    from training.hotupdate.attach import AttachAcquirer

    acquirer = AttachAcquirer()
    register(acquirer)  # type: ignore[operator]
    return acquirer
