"""The **one** provider resolution path  (FR-004.1, FR-004.6, FR-004.10; R-02.49, R-02.11, ADR-009).

The ratified split (foundation task 3.8, which **ratifies rather than redesigns** what the
features settled on):

* the **interface** - :mod:`providers.base`, the foundation's;
* the **OpenAI-compatible HTTP implementation** - F3's, at
  :mod:`providers.openai_compatible`. **F4 reuses it and does not fork it**: two HTTP
  providers would be two sets of timeout semantics and two answers to ``capabilities()``.
  This module imports and registers it. There is no subclass and no wrapper;
* the **registry, the provider identity and the in-process path** - F4's, here and in
  :mod:`providers.in_process`.

Every consumer - F3's judge, F4's scoring, F7's dry-run - resolves from this one registry.
**There is no second construction path and no import-by-module-path resolution.**

## THE REGISTRATION BLOCK IS THE WIRING, AND IT IS THE POINT

:func:`ensure_registered` makes explicit :func:`register` calls, in one place. A directory
scan was rejected for the reason F3 records one directory over: *a scan makes "is it
registered?" untestable and an unregistered implementation silent* - which is the failure the
whole review discipline was written after, when a sibling shipped sixteen fully-implemented,
unit-tested, documented MCP tools that were never registered with the server.

**Deleting one ``register(...)`` line below must make a reachability test red.** That red is
observed and recorded, not assumed - see ``test_provider_registry_reachability.py``.

## `get()` never returns a different provider

A fallback here is mutation control **M-9** shipped on purpose. An unreachable endpoint
refuses by name; it does **not** quietly become the in-process provider, because that would
rewrite the meaning of a run's curve halfway through it and nothing would say so. The same
rule is why :func:`assert_pin_unchanged` exists: a run's test curve is a comparison of one
number against the number before it, and changing the instrument halfway shows a change in
the instrument.

## R-02.11 is enforced by what the judge resolver cannot reach

:func:`judge_resolver` is the view F3's ``bind_provider_resolver`` is given, and **it has no
in-process row at all.** ADR-009 defines the in-process provider as generating from the
parameters miForge holds, which *is* the model under training, and a model grading itself is
not a measurement. The refusal is spelled ``judge_provider_not_configured`` - **identically**
to the reason F3's listing publishes for an unavailable judge - so the listing and the
refusal cannot drift and an operator is not told two different things about one state.

That is the first of two lines of defence and the weaker one; F3's own
``identity.check_judge_identity`` refuses an in-process judge unconditionally at ``prepare``.
Both exist because *a constraint made unreachable beats one merely stated.*
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.errors import refuse
from providers.base import InferenceProvider, ProviderIdentity
from providers.in_process import PROVIDER_KIND as IN_PROCESS_KIND
from providers.model_handle import AcquiredModelHandle, ModelHandle
from providers.openai_compatible import PROVIDER_KIND as OPENAI_COMPATIBLE_KIND
from providers.settings import HttpProviderRow, http_provider_rows, in_process_tuning

#: The capability vocabulary. **``activations`` is deliberately not here**, and that absence
#: is the implementation of R-02.19's own amended text: activation statistics are *"not a
#: capability of R-02.49's inference provider, whose scope is generation and which exposes
#: no activations"*. A declared-but-unused slot is how the next reader concludes the provider
#: is the route - which is exactly what DEF-10 reported.
#:
#: ``embeddings`` stays in the **vocabulary** while no implementation declares it: whether an
#: embeddings capability belongs on this interface at all is F3's open question, not F4's to
#: close, and R-02.16 asks only for *"exact or fuzzy match"* (DEF-3).
KNOWN_CAPABILITIES: frozenset[str] = frozenset({"generation", "embeddings"})

#: The reason an unconfigured judge is reported with. **The same string F3's
#: ``services.selectors.availability`` publishes**, imported nowhere so this package keeps no
#: dependency on ``services/`` - the agreement is pinned by a test instead, which is the only
#: form of agreement that survives an edit on either side.
JUDGE_PROVIDER_NOT_CONFIGURED = "judge_provider_not_configured"


@dataclass(frozen=True, slots=True)
class ProviderRow:
    """One row of the registry as a listing sees it. **Never carries a credential.**"""

    name: str
    kind: str
    model: str | None
    base_url: str | None
    capabilities: frozenset[str]
    available: bool
    unavailable_reason: str | None
    #: Whether this row may serve as a judge (R-02.11). False for every in-process row.
    judge_eligible: bool

    def payload(self) -> dict[str, Any]:
        """The API body. ``ProviderIdentity`` has no credential field and neither has this."""
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "capabilities": sorted(self.capabilities),
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "judge_eligible": self.judge_eligible,
        }


Factory = Callable[[], InferenceProvider]

#: ``kind`` -> how to build one from a row. Registered, not scanned. See the module
#: docstring: deleting a line here must turn a reachability test red.
_FACTORIES: dict[str, Callable[[HttpProviderRow], InferenceProvider]] = {}

#: The in-process handle this process holds, if any. Bound by the ``train`` worker at
#: acquisition and **never** in the API process: a web process holds no parameters, so the
#: in-process row is listed as unavailable *by name* rather than being absent from the list -
#: an option missing from a response is indistinguishable from one that is broken (ADR-013).
_handle: ModelHandle | None = None
_handle_lock = threading.Lock()

#: The in-process row's name. One spelling, because a selector configuration can name it and
#: a refusal has to quote it back.
IN_PROCESS_NAME = IN_PROCESS_KIND


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------


def register(kind: str, factory: Callable[[HttpProviderRow], InferenceProvider]) -> None:
    """Register one provider kind. **Raises on a duplicate.**

    Raises on a second registration while :func:`ensure_registered` is idempotent, and the
    two are not in tension: a repeated *import* is normal and must not fail, while a second
    implementation claiming one kind is a real conflict whose resolution would otherwise
    depend on import order.
    """
    if kind in _FACTORIES:
        raise refuse(
            "inference_provider_duplicate_kind",
            f"A provider kind {kind!r} is already registered. Two implementations of one "
            f"kind is two answers to 'which provider produced this score', resolved by "
            f"import order.",
            kind=kind,
        )
    _FACTORIES[kind] = factory


def registered_kinds() -> tuple[str, ...]:
    """Every registered kind, sorted. What a reachability test reads."""
    return tuple(sorted(_FACTORIES))


def _build_openai_compatible(row: HttpProviderRow) -> InferenceProvider:
    """Build **F3's** HTTP provider from a configured row. No subclass, no wrapper.

    The only F4 addition to that module is this row and the recording of its identity.
    """
    from providers.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        base_url=row.base_url,
        model=row.model,
        api_key=row.api_key,
        connect_timeout_s=row.connect_timeout_s,
        read_timeout_s=row.read_timeout_s,
        # WIRED, not merely passed through. `redacted()` publishes this figure on
        # GET /api/v1/providers, and until 2026-10-16 nothing enforced it: the provider had
        # no such parameter, so every endpoint ran at an effective bound of 1 while the API
        # reported 4. A published figure that nothing enforces is worse than an absent one.
        concurrency=row.concurrency,
    )


def ensure_registered() -> None:
    """Register the built-in provider kinds, idempotently.

    An import side effect that nobody triggers is not a side effect, so everything resolving
    a provider calls this first - F3's ``ensure_registered`` exists for the same reason.
    """
    if OPENAI_COMPATIBLE_KIND not in _FACTORIES:
        # F3's implementation, REGISTERED not forked. Delete this line and the HTTP provider
        # stops existing for every caller while remaining fully unit-tested.
        register(OPENAI_COMPATIBLE_KIND, _build_openai_compatible)


ensure_registered()


# --------------------------------------------------------------------------------------
# The in-process handle
# --------------------------------------------------------------------------------------


def bind_model_handle(handle: ModelHandle | None) -> None:
    """Install the parameter set this process holds. ``None`` unbinds it.

    Called by the ``train`` worker once the model is acquired (R-02.46). **Not called in the
    API process**, which is why ``GET /api/v1/providers`` reports the in-process row
    unavailable there - a web process has no parameters to generate from, and saying so by
    name is better than offering a row that would refuse on first use.

    A function rather than a mutable public global so the unbind path exists and is used:
    a test that configured a provider and could not configure it *away* would make the
    standalone case (AC20) unassertable.
    """
    global _handle
    with _handle_lock:
        _handle = handle


def bind_acquired_model(acquired: Any, *, step: int) -> ModelHandle:
    """Bind F1's acquisition as this process's in-process handle, **for ``step``**.

    The one production entry point for :func:`bind_model_handle`, and ``step`` is
    **required** on purpose.

    ## Why this exists at all

    ``bind_model_handle`` and :class:`providers.model_handle.AcquiredModelHandle` were
    called in fourteen test files and **nowhere in ``backend/src/``**, so
    :func:`get` refused ``inference_provider_unavailable`` in every process - including the
    training worker, whose refusal message *"available in the training worker once the
    model is acquired (R-02.46)"* was therefore false. ADR-009's **default** inference
    provider did not exist for any caller, which took test and retention scoring
    (R-02.16, R-02.17) with it on the standalone path (AC20). ADR-016 #1 exactly: *a
    capability is not shipped until a test FAILS when its wiring is removed.*

    ## Why ``step`` has no default

    ``AcquiredModelHandle``'s ``step=None`` renders the digest ``<hash>@as-acquired``,
    which is right for a handle nobody steps and **wrong for every score after the first
    update**: ``parameter_set_digest``'s own docstring says that stamping one digest on a
    whole run is the *"ours is not an identity"* failure the field exists to prevent. A
    binding made once at acquisition would have traded a loud refusal for a silently wrong
    provenance, which is strictly worse. So the production seam cannot express it: the
    caller must say which parameter set this is, and :func:`training.tasks.drive_run`
    rebinds after **every** update.

    Args:
        acquired: F1's ``AcquiredModel``. Typed ``Any`` for the reason
            :class:`providers.model_handle.AcquiredModelHandle` gives - naming the type
            would pull torch into this module's import.
        step: The number of completed steps these parameters represent. ``0`` for a fresh
            run's as-acquired weights, the resumed step on a continue.

    Returns:
        The handle that was bound, so a caller can read the digest it will record.
    """
    handle = AcquiredModelHandle(acquired, step=step)
    bind_model_handle(handle)
    return handle


def current_model_handle() -> ModelHandle | None:
    """The bound handle, or ``None``. ``None`` is a legitimate, expected state."""
    return _handle


# --------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------


def _in_process_row() -> ProviderRow:
    """The in-process row - **always listed**, available only where parameters are held."""
    handle = current_model_handle()
    available = handle is not None
    model: str | None = None
    if handle is not None:
        try:
            model = handle.model_identity()
        except Exception:
            available = False
    return ProviderRow(
        name=IN_PROCESS_NAME,
        kind=IN_PROCESS_KIND,
        model=model,
        base_url=None,
        capabilities=frozenset({"generation"}),
        available=available,
        unavailable_reason=None if available else "inference_provider_unavailable",
        # R-02.11, structurally: there is no configuration that makes this True.
        judge_eligible=False,
    )


def _http_rows() -> list[tuple[ProviderRow, HttpProviderRow]]:
    """Configured HTTP rows, from **live** settings - never cached at import."""
    out: list[tuple[ProviderRow, HttpProviderRow]] = []
    for row in http_provider_rows():
        out.append(
            (
                ProviderRow(
                    name=row.name,
                    kind=OPENAI_COMPATIBLE_KIND,
                    model=row.model,
                    base_url=row.base_url,
                    capabilities=frozenset({"generation"}),
                    available=OPENAI_COMPATIBLE_KIND in _FACTORIES,
                    unavailable_reason=(
                        None
                        if OPENAI_COMPATIBLE_KIND in _FACTORIES
                        else "inference_provider_unavailable"
                    ),
                    judge_eligible=True,
                ),
                row,
            )
        )
    return out


def list_rows() -> list[ProviderRow]:
    """Every provider row, computed from **live** settings on each call (FR-004.6).

    The in-process row is always present. With nothing configured it is the only one, which
    is the standalone case acceptance criterion 20 pins.
    """
    ensure_registered()
    return [_in_process_row(), *(row for row, _ in _http_rows())]


def row_for(name: str) -> ProviderRow | None:
    """One row by name, or ``None``."""
    for row in list_rows():
        if row.name == name:
            return row
    return None


def get(name: str) -> InferenceProvider:
    """**The one resolution path.** Returns the named provider or refuses by name.

    Raises:
        Refusal: ``inference_provider_unavailable`` when the name is unknown, when the
            in-process row is asked for in a process that holds no parameters, or when a
            kind has no registered implementation.

    It **never returns a different provider.** A fallback here is mutation control M-9
    shipped on purpose: a silent substitution rewrites a run's curve mid-run and nothing
    says so.
    """
    ensure_registered()
    if name == IN_PROCESS_NAME:
        handle = current_model_handle()
        if handle is None:
            raise refuse(
                "inference_provider_unavailable",
                "The in-process provider was selected but this process holds no "
                "parameters to generate from. It is available in the training worker "
                "once the model is acquired. miForge will not substitute a "
                "different provider: a score's number means nothing without the "
                "instrument that produced it.",
                provider=name,
                kind=IN_PROCESS_KIND,
            )
        from providers.in_process import InProcessProvider

        tuning = in_process_tuning()
        return InProcessProvider(
            handle,
            kv_bytes_per_token=tuning.kv_bytes_per_token,
            batch_bound=tuning.batch_bound,
        )

    for row, http in _http_rows():
        if row.name != name:
            continue
        factory = _FACTORIES.get(row.kind)
        if factory is None:
            raise refuse(
                "inference_provider_unavailable",
                f"Provider {name!r} is configured as kind {row.kind!r}, which has no "
                f"registered implementation. Refusing rather than resolving a different "
                f"kind.",
                provider=name,
                kind=row.kind,
                registered=list(registered_kinds()),
            )
        return factory(http)

    raise refuse(
        "inference_provider_unavailable",
        f"No provider named {name!r} is configured. Set PROVIDER_{name.upper()}_BASE_URL "
        f"and PROVIDER_{name.upper()}_MODEL, or use the in-process provider "
        f"({IN_PROCESS_NAME!r}). miForge will not fall back to a different provider.",
        provider=name,
        configured=[row.name for row in list_rows()],
    )


# --------------------------------------------------------------------------------------
# Pinning  (FR-004.10, E-4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pin:
    """A run's pinned provider: the name it was resolved by and the identity it resolved to.

    **Both**, because they answer different questions. The name is what configuration said;
    the identity is what answered. An endpoint re-pointed at a different model keeps its
    name and changes its identity, and that is precisely the change a curve must break at.
    """

    name: str
    identity: ProviderIdentity

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, **self.identity.model_dump(mode="json")}


def pin(name: str) -> Pin:
    """Resolve ``name`` once, at run start, and record what it resolved to."""
    return Pin(name=name, identity=get(name).identity)


def assert_pin_unchanged(pinned: Pin, current: ProviderIdentity, *, repin: bool = False) -> bool:
    """Refuse a mid-run provider change, or record a discontinuity when re-pinned.

    Args:
        pinned: What the run pinned at start.
        current: What the name resolves to now.
        repin: The operator's explicit instruction to adopt the new identity.

    Returns:
        ``True`` when the identity changed and was accepted as a **recorded discontinuity**,
        ``False`` when nothing changed. F9 breaks the curve wherever this returned ``True``
        rather than joining two instruments' numbers.

    Raises:
        Refusal: ``provider_changed_mid_run`` carrying both identities.

    The ``engine_version`` field is part of the comparison, which matters only for the
    in-process provider: it carries the parameter-set digest, which changes at **every**
    step by design. So the in-process kind compares on ``(kind, model, base_url)`` alone -
    stated here rather than discovered, because the alternative is a discontinuity reported
    on every step of every run, which would train every reader to ignore the marker.
    """
    if _comparable(pinned.identity) == _comparable(current):
        return False
    if repin:
        return True
    raise refuse(
        "provider_changed_mid_run",
        f"The provider pinned as {pinned.name!r} at run start now resolves to a different "
        f"identity. A run's test curve compares each number with the one before it, so a "
        f"changed instrument would show as a changed model. Re-pin explicitly to continue "
        f"with a recorded discontinuity, or restore the configuration.",
        provider=pinned.name,
        pinned=pinned.identity.model_dump(mode="json"),
        current=current.model_dump(mode="json"),
    )


def _comparable(identity: ProviderIdentity) -> tuple[str, str, str | None, str | None]:
    """The identity fields a pin comparison reads. See :func:`assert_pin_unchanged`."""
    engine = None if identity.kind == IN_PROCESS_KIND else identity.engine_version
    return (identity.kind, identity.model, identity.base_url, engine)


# --------------------------------------------------------------------------------------
# The judge view  (R-02.11)
# --------------------------------------------------------------------------------------


class JudgeProviderResolver:
    """The registry as F3's ``ProviderResolver`` sees it. **No in-process row exists here.**

    Satisfies :class:`services.selectors.provider_port.ProviderResolver` structurally - a
    ``Protocol``, so this package declares no dependency on ``services/`` and the layering
    stays one-way. The agreement is pinned by a test that checks this class against that
    Protocol at runtime, which is the only form of agreement that survives an edit on either
    side.
    """

    def available(self, key: str) -> bool:
        """Whether ``key`` names a judge-eligible provider that can be used right now."""
        row = row_for(key)
        return bool(row and row.judge_eligible and row.available)

    def resolve(self, key: str) -> InferenceProvider:
        """The provider for ``key``, or ``judge_provider_not_configured`` by name.

        Refuses rather than returning ``None``: a caller handed ``None`` writes
        ``provider or default_provider``, and on the judge path the only default available
        is the model being trained.
        """
        row = row_for(key)
        if row is not None and not row.judge_eligible:
            raise refuse(
                JUDGE_PROVIDER_NOT_CONFIGURED,
                f"Provider {key!r} is the in-process provider, which generates from the "
                f"parameters miForge is training - a model grading itself is not a "
                f"measurement. Point the judge at an OpenAI-compatible endpoint "
                f"serving a different model. miForge will not use it as a judge under any "
                f"configuration.",
                provider_key=key,
                service="judge_provider",
                capabilities_unavailable=["judge_model selector", "judge_model grader"],
            )
        if row is None or not row.available:
            raise refuse(
                JUDGE_PROVIDER_NOT_CONFIGURED,
                f"No generation provider is available for key {key!r}, so the judge-model "
                f"selector cannot run. Set PROVIDER_{key.upper()}_BASE_URL and "
                f"PROVIDER_{key.upper()}_MODEL. miForge will NOT fall back to the model it "
                f"is training: a model grading itself is not a measurement.",
                provider_key=key,
                service="judge_provider",
                capabilities_unavailable=["judge_model selector", "judge_model grader"],
            )
        return get(key)


def judge_resolver() -> JudgeProviderResolver:
    """The resolver F3's ``bind_provider_resolver`` is given. See :mod:`services.scoring.wiring`."""
    return JudgeProviderResolver()
