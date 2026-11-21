"""The run-type registry  (FR-008.1, FR-008.19; R-02.41, R-02.43).

Three registrations, populated at import, resolved by name. **The reachability test
resolves through :func:`resolve`; it does not import the three modules** - importing proves
a module parses, resolving proves a caller can reach it. That distinction is the lesson of
the sibling that shipped sixteen fully-implemented, unit-tested, documented MCP tools which
were never registered with its server: every one of those tests passed by importing the
module directly, so the suite was green while the feature did not exist for any caller.

**Do not add a fourth registration for ``attach``.** Section 5.10 changes the *acquisition
mode* - which is F1's registry - and not the run type. An adapter run under the optional
hot-update mode is still the ``adapter`` run type (ADR-022).
"""

from __future__ import annotations

from core.errors import refuse
from training.runtypes.base import (
    RUN_TYPE_NAMES,
    STEP,
    RunType,
    SaveContext,
    TrainablePlan,
    all_parameters,
)

__all__ = [
    "RUN_TYPE_NAMES",
    "STEP",
    "RunType",
    "SaveContext",
    "TrainablePlan",
    "all_parameters",
    "known",
    "register",
    "resolve",
]

_REGISTRY: dict[str, RunType] = {}


def register(run_type: RunType) -> None:
    """Add one run type. A duplicate name is a programming error, not a replacement.

    Raising on a duplicate rather than overwriting is deliberate: a second registration of
    ``adapter`` would silently win and the earlier one's behaviour would vanish with nothing
    going red.
    """
    if run_type.name in _REGISTRY:
        raise RuntimeError(
            f"duplicate run type {run_type.name!r}: a second registration would silently "
            f"replace the first"
        )
    _REGISTRY[run_type.name] = run_type


def resolve(name: str) -> RunType:
    """The run type called ``name``, or a refusal that **lists what is known**.

    A bare ``KeyError`` reaching the API is a 500 where a 400 naming the three valid values
    belongs (ADR-013: refuse with a reason, and say what to do next).
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise refuse(
            "run_type_unknown",
            f"There is no run type named {name!r}. Choose one of the three: 'change' "
            f"updates every parameter of a model that exists, 'create' makes one from an "
            f"architecture config, 'adapter' trains a low-rank adapter beside the base.",
            requested=name,
            known=sorted(_REGISTRY),
        ) from None


def known() -> dict[str, RunType]:
    """The live registry, copied. What ``GET /api/v1/run-types`` answers from."""
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------- the three
# At import, and at the bottom so the three modules can import this package's `base`
# without a cycle. Removing any one of these lines must turn `test_runtype_registry.py`
# and `test_reachability.py` red (M12) - the module would still parse, still be
# unit-tested and still be documented, and would exist for no caller.
from training.runtypes.adapter import AdapterRunType  # noqa: E402
from training.runtypes.change import ChangeRunType  # noqa: E402
from training.runtypes.create import CreateRunType  # noqa: E402

register(ChangeRunType())
register(CreateRunType())
register(AdapterRunType())
