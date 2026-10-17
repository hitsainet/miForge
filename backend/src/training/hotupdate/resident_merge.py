"""A merge into a serving process's resident tensors  (FR-011.61; BRD criterion 16's clause).

**R-02.45 is F8's requirement and is not implemented here.** F8 computes the merge - the
arithmetic, the ``merged_adapter`` manifest field, and criterion 17's refusal when no
checkpoint of the unmerged base exists. **F11 wraps a write it does not compute.** If this
module ever needs to know an adapter's format, the seam has been drawn in the wrong place.

## The same bracket, with no exemption

A merge into resident tensors is **one more multi-tensor write to tensors miForge does not
own**, which is the entire definition of what the bracket protects. So it is
``barrier.commit_bracket`` - the same stages in the same order, the same watchdog, the same
``torn`` outcome when the write is cut off - and **not a second code path**. *"A merge is not
a step"* is the sentence that will tempt someone to route around it, which is why **MUT-20**
writes the merged tensors before ``commit-begin`` and must turn the test red.

**Interrupting it leaves the set ``torn`` and serving refused, and the merge is NOT
retried** - FR-011.43 applies unchanged. A half-applied merge re-applied compounds the
damage exactly as a half-applied AdamW update does, and a merge *cannot be undone by a
second merge*: once the base has moved, the adapter's contribution cannot be subtracted back
out. Recovery is ``recovery.py``'s, through R-02.30, rather than a merge-specific path.

## The default-path merge is F8's in full

A merge that writes a *directory* is an atomic directory write under ADR-015 and has no
bracket; that is why criterion 16 is split at all. Nothing here touches it.
"""

from __future__ import annotations

import logging
from typing import Any

from core.errors import refuse
from training.hotupdate import barrier, state

log = logging.getLogger(__name__)


def apply(
    client: Any,
    merged_tensors: dict[str, Any],
    *,
    timings: Any,
    expected_identity: state.ResidentIdentity | None = None,
    writer: Any = None,
) -> dict[str, Any]:
    """Write **finished** tensors into the resident parameter set, inside the bracket.

    Args:
        client: the one call site.
        merged_tensors: F8's output. **Computed elsewhere**; this module reads them and
            writes them, and refuses an empty set rather than opening a bracket around
            nothing - an empty merge "succeeds" and changes nothing, which is the vacuous
            pass this project keeps finding.
        timings: the hold and commit duration go to the same place a step's do. A merge is
            held serving time like any other commit, and the watchdog reads the same
            distribution.
        expected_identity: the resident identity the caller believes it is merging into. The
            state precondition runs **before** the bracket, as it does for a step.
        writer: how a tensor is written into the resident set. Injected so the ordering and
            the call counts are assertable with no card: the real writer copies into the
            attached tensors in place.

    Raises:
        Refusal: ``resident_merge_empty`` for an empty tensor set; ``weights_not_clean`` from
            the precondition; ``weights_commit_interrupted`` from the bracket when
            ``commit-end`` does not land - and **the merge is not retried**.
    """
    if not merged_tensors:
        raise refuse(
            "resident_merge_empty",
            "A resident merge was asked to write no tensors. Refusing rather than opening "
            "a commit bracket around nothing: an empty merge reports success, changes "
            "nothing, and leaves an operator believing an adapter was merged.",
        )
    state.require_clean(client, expected_identity)
    written: list[str] = []
    with barrier.commit_bracket(client, timings, what="resident adapter merge"):
        for name, tensor in merged_tensors.items():
            if writer is None:
                _write_in_place(name, tensor)
            else:
                writer(name, tensor)
            written.append(name)
    log.info(
        "an adapter was merged into the resident parameter set",
        extra={
            "code": "resident_merge_applied",
            "tensors": len(written),
            "bracket": "barrier.commit_bracket - the same one a step uses",
            "retry_policy": "an interrupted merge is NEVER retried (R-02.31)",
        },
    )
    return {"tensors_written": len(written), "names": written}


def _write_in_place(name: str, tensor: Any) -> None:
    """Copy one merged tensor into the resident parameter it replaces.

    ``copy_`` rather than an assignment: the resident tensor's **address must not move**
    (BRD-01 R-01.5 keeps the parameters at fixed addresses for the lifetime of the load, and
    every IPC handle depends on it). Rebinding the name would leave miLLM serving the old
    storage and miForge training a new one - two models, both reporting success.
    """
    target = tensor.get("target") if isinstance(tensor, dict) else None
    source = tensor.get("source") if isinstance(tensor, dict) else None
    if target is None or source is None:
        raise refuse(
            "resident_merge_tensor_unusable",
            f"The merged entry for {name!r} does not carry both the resident target and "
            f"the computed source, so it cannot be written in place. Refusing rather than "
            f"rebinding the name: that leaves miLLM serving the old storage while miForge "
            f"trains a new one.",
            tensor=name,
        )
    target.copy_(source)
