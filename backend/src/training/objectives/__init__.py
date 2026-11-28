"""The objective registry - **THE registration block**  (FR-002.3).

Three explicit :func:`training.objectives.base.register` calls, in one place, run as an
import side effect of this package. **Deleting one must make a test red.** A directory scan
was rejected for the reason the selector registry one directory over states: a scan makes
*"is it registered?"* untestable and a missing objective silent - which is the failure this
project's review discipline was written after, when a sibling shipped sixteen
fully-implemented, unit-tested, documented MCP tools that were never registered.

**This package is the ``ObjectiveRegistry`` F1's step resolves through.** ``training/step.py``
imports it lazily and casts the module to its Protocol, so the module itself must export
``get`` - which it does, from :mod:`training.objectives.base`. ``get`` **raises** on an
unknown name and never returns ``None``: an ``Optional`` invites ``or default``, and the
only default available is the loss nobody chose.

Three objectives, four sources: ``sft`` serves **both** ``corpus`` and ``self-generation``,
because R-02.5's accepted completions are *"supervised examples"* - R-02.4's objective on
different material. One implementation, not two.
"""

from __future__ import annotations

from training.objectives import dpo, grpo, sft
from training.objectives.base import (
    Objective,
    get,
    names,
    register,
    registered,
    torch_module,
)

#: Which objective each signal source names on its batches. Stated here, beside the
#: registrations, so the four-to-three mapping is one fact in one place rather than four
#: string literals in four source modules.
SOURCE_OBJECTIVE: dict[str, str] = {
    "corpus": sft.NAME,
    "self-generation": sft.NAME,
    "preference-pairs": dpo.NAME,
    "reward": grpo.NAME,
}


def ensure_registered() -> None:
    """Register the three built-in objectives, idempotently.

    Idempotent rather than raising on a second call, while
    :func:`training.objectives.base.register` itself **does** raise on a duplicate name.
    The two are not in tension: a repeated *import* is normal and must not fail, while a
    second implementation claiming one name is a real conflict whose resolution would
    otherwise depend on import order.
    """
    existing = set(names())
    for objective in (sft.OBJECTIVE, dpo.OBJECTIVE, grpo.OBJECTIVE):
        if objective.name not in existing:
            register(objective)


ensure_registered()

__all__ = [
    "SOURCE_OBJECTIVE",
    "Objective",
    "ensure_registered",
    "get",
    "names",
    "register",
    "registered",
    "torch_module",
]
