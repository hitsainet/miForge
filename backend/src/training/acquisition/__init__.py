"""The acquisition mode registry  (R-02.46, R-02.48; FR-001.1, FR-001.2, FR-001.15).

**There is no ``if mode == "attach"`` anywhere in this feature, and that is the design.**
BRD-02 section 5.10's hot-update mode adds a *fourth registration* from
``training/hotupdate/`` (F11's FR-011.1 - *"a registration, not a branch"*), so ADR-022's
fence holds by construction: the core path has nothing to import and nothing to special-case,
and the standalone test runs with that package **absent from the tree**.

## Unavailable is a first-class answer

``attach``'s default entry is a :class:`~training.acquisition.base.Availability` that names
its own reason, so FR-001.15's refusal is a **lookup** rather than a special case. That
matters for two reasons R-02.48 states directly:

* ``GET /api/v1/models/sources`` must list an unavailable mode **with its reason**, not omit
  it. An option missing from a response is indistinguishable from one that is broken.
* Requesting ``attach`` in a build without the mode is **refused by name**, and must never
  be *"quietly satisfied by loading the same model from disk"* - that would train a
  different set of tensors than the operator asked for and report success.

## Two acquirers share the ``load`` mode

``load`` covers a local directory and a Hub repo (R-02.46 case 1 names both), so the
registry dispatches on the mode **and** the source the spec names. The alternative - two
modes, ``load_local`` and ``load_hub`` - would have invented mode names R-02.46 does not
have, and the mode is recorded on the run and travels into the record.
"""

from __future__ import annotations

from core.errors import refuse
from training.acquisition.base import (
    DEFAULT_MODE,
    AcquiredModel,
    Acquirer,
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    ArchitectureConfig,
    Availability,
    SourceKind,
    lineage_for_creation,
    lineage_for_first_load,
    lineage_for_loaded_checkpoint,
)
from training.acquisition.hub import HubAcquirer
from training.acquisition.local_dir import LocalDirAcquirer
from training.acquisition.scratch import ScratchAcquirer

__all__ = [
    "DEFAULT_MODE",
    "AcquiredModel",
    "Acquirer",
    "AcquisitionMode",
    "AcquisitionRecord",
    "AcquisitionSpec",
    "ArchitectureConfig",
    "Availability",
    "SourceKind",
    "availability",
    "lineage_for_creation",
    "lineage_for_first_load",
    "lineage_for_loaded_checkpoint",
    "modes",
    "register",
    "registered_modes",
    "resolve",
    "unavailable",
]


class _UnavailableMode:
    """A mode this build does not have. Registered rather than absent (FR-001.15).

    **Registered, not omitted.** An absent entry makes an unavailable mode
    indistinguishable from a mode that does not exist, and ``GET /models/sources`` has to
    report the difference: one is a build that lacks a capability, the other is a typo.
    """

    def __init__(self, mode: AcquisitionMode, availability: Availability) -> None:
        self._mode = mode
        self._availability = availability

    @property
    def mode(self) -> AcquisitionMode:
        return self._mode

    def available(self) -> Availability:
        return self._availability

    def preflight(self, spec: AcquisitionSpec) -> AcquisitionSpec:
        """Refuse **by name**, before anything is leased or allocated."""
        raise self._refusal()

    def acquire(self, spec: AcquisitionSpec) -> AcquiredModel:
        """Refuse. **Never a fallback to a load** - R-02.48 forbids substitution."""
        raise self._refusal()

    def _refusal(self) -> Exception:
        return refuse(
            "acquisition_mode_unavailable",
            self._availability.message
            or (
                f"The {self._mode.value!r} acquisition mode is not available in this "
                f"build. Refusing rather than substituting another mode: a different "
                f"acquisition trains a different set of tensors than was asked for."
            ),
            mode=self._mode.value,
            reason=self._availability.reason,
            capabilities_unavailable=list(self._availability.unavailable_capabilities),
        )


#: ``attach``'s default state. F11 **replaces** this entry by calling :func:`register` from
#: ``training/hotupdate/``; until then the mode is present and reports why it is not usable.
ATTACH_UNAVAILABLE = Availability(
    available=False,
    reason="hot_update_mode_not_available",
    message=(
        "The 'attach' acquisition mode belongs to BRD-02 §5.10, the optional hot-update "
        "mode, which is not present in this build. miForge trains the parameters it holds "
        "itself: use the 'load' mode to train the same weights from disk. Refusing rather "
        "than doing that silently - attaching and loading train different tensors."
    ),
    unavailable_capabilities=(
        "attach to a serving process's resident tensors",
        "update weights in place while they are being served",
    ),
)

_REGISTRY: dict[AcquisitionMode, Acquirer] = {}


def register(acquirer: Acquirer) -> None:
    """Register (or replace) the acquirer for a mode.

    **Replacement is allowed on purpose**: it is the mechanism by which F11 turns ``attach``
    from an unavailable stub into a working mode, with no edit to any file in this feature.
    """
    _REGISTRY[acquirer.mode] = acquirer


def unavailable(mode: AcquisitionMode, availability: Availability) -> None:
    """Register ``mode`` as present-but-unavailable, with its reason."""
    if availability.available:
        raise ValueError("unavailable() registers an unavailable mode; use register()")
    _REGISTRY[mode] = _UnavailableMode(mode, availability)


def _install_defaults() -> None:
    """Three registrations, run at import. ``attach``'s is the unavailable stub, unless
    §5.10's package is present - see :func:`_install_optional_attach_mode`.

    Deliberately **not** importing ``training.hotupdate`` *statically*: ADR-022 requires the
    standalone test to run with that package absent from the tree, so a static import of it
    from the core path would be a dependency edge on an optional package and would fail the
    suite with the package deleted - which is the point of the fence.
    """
    register(ScratchAcquirer())
    _REGISTRY[AcquisitionMode.LOAD] = LocalDirAcquirer()
    unavailable(AcquisitionMode.ATTACH, ATTACH_UNAVAILABLE)
    _install_optional_attach_mode()


def _install_optional_attach_mode() -> None:
    """**SEAM 1.** Let BRD-02 §5.10 replace ``attach``'s stub, if this build has it.

    This is the one place in ``backend/src/`` that reaches ``training/hotupdate/``, and it
    reaches it **by name at runtime** rather than with an ``import`` statement. Both halves
    of that are deliberate:

    * **Dynamic**, because a static import is a dependency edge from the core path into the
      optional package, and ``test_acquisition_registry.py`` forbids one by AST for the
      reason ADR-022 gives - the standalone test deletes that directory and the suite must
      still pass. A guarded static import would not *break* with the package absent, but it
      would make the core path's import graph depend on §5.10's existence, which is the
      property the fence is about.
    * **A function call into the package**, not a flag. A settings flag can be set in a
      build where the package is absent, and then the mode claims to exist. Module present
      or absent is the only answer that cannot be wrong.

    With the package absent, ``attach`` keeps :data:`ATTACH_UNAVAILABLE` and every other
    mode is untouched. With it present but miLLM unconfigured, the acquirer itself reports
    unavailable with the same ``reason`` and names the missing configuration in its message
    (see that package's ``settings.py`` on why the reason code does not change).

    **Deleting this call must turn ``tests/unit/test_reachability_f11.py`` red.** A mode
    whose own tests pass by importing its module directly is a mode no caller can reach.
    """
    import importlib

    try:
        package = importlib.import_module("training.hotupdate")
    except ImportError:
        # ABSENT AND NORMAL (ADR-010), not an error: §5.10 is optional, and ten of eleven
        # features complete with nothing else running. The stub registered above already
        # reports the mode unavailable by name.
        return
    package.install(register)


def resolve(spec: AcquisitionSpec) -> Acquirer:
    """The acquirer for ``spec``'s mode **and** source.

    Raises:
        Refusal: ``acquisition_mode_unknown`` for a mode outside R-02.46's three. Not a
            ``KeyError``: an unknown mode is an operator-facing refusal with a list of what
            is available, and a ``KeyError`` would surface as a 500 with a traceback.
    """
    if spec.mode is AcquisitionMode.LOAD:
        # One mode, two sources (R-02.46 case 1 names both). Dispatching on the source the
        # spec actually carries beats inventing `load_local`/`load_hub` mode names the
        # requirement does not have and the record would then have to carry.
        return HubAcquirer() if spec.repo is not None else LocalDirAcquirer()

    acquirer = _REGISTRY.get(spec.mode)
    if acquirer is None:
        raise refuse(
            "acquisition_mode_unknown",
            f"{spec.mode!r} is not an acquisition mode. miForge acquires parameters by "
            f"'load', 'create' or 'attach' (R-02.46).",
            mode=str(spec.mode),
            known=[m.value for m in registered_modes()],
        )
    return acquirer


def registered_modes() -> tuple[AcquisitionMode, ...]:
    """Every registered mode, in R-02.46's own order."""
    order = (AcquisitionMode.LOAD, AcquisitionMode.CREATE, AcquisitionMode.ATTACH)
    return tuple(mode for mode in order if mode in _REGISTRY)


def availability(mode: AcquisitionMode) -> Availability:
    """Whether ``mode`` can be used in this build. Touches neither network nor GPU."""
    acquirer = _REGISTRY.get(mode)
    if acquirer is None:
        return Availability(
            available=False,
            reason="acquisition_mode_unknown",
            # `str(mode)` rather than `mode.value`: this function is the one that answers
            # for a mode it does not recognise, and a value that is not an AcquisitionMode
            # at all has no `.value` - raising an AttributeError here would turn "unknown
            # mode" into a 500 with a traceback, which is the opposite of reporting it.
            message=f"{str(mode)!r} is not an acquisition mode miForge knows.",
        )
    return acquirer.available()


def modes() -> list[dict[str, object]]:
    """The payload behind ``GET /api/v1/models/sources`` (FR-001.15).

    Every mode, available or not, each unavailable one carrying **its reason and the
    capabilities it would have provided**. The default is marked so the UI does not have to
    know which one R-02.46 names.
    """
    payload: list[dict[str, object]] = []
    for mode in registered_modes():
        state = availability(mode)
        payload.append(
            {
                "mode": mode.value,
                "available": state.available,
                "default": mode is DEFAULT_MODE,
                # Present on an available mode too, as nulls: a key that appears only on
                # failure makes the success shape and the failure shape two different
                # types for one endpoint.
                "reason": state.reason,
                "message": state.message,
                "capabilities_unavailable": list(state.unavailable_capabilities),
                "sources": _sources_for(mode),
            }
        )
    return payload


def _sources_for(mode: AcquisitionMode) -> list[str]:
    """Which source kinds a mode accepts. ``load`` has two; the others have one each."""
    if mode is AcquisitionMode.LOAD:
        return [SourceKind.LOCAL_DIR.value, SourceKind.HUB.value]
    if mode is AcquisitionMode.CREATE:
        return [SourceKind.CONFIG.value]
    return [SourceKind.ATTACHED.value]


_install_defaults()
