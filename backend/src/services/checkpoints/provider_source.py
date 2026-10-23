"""Which provider produced a score, and **which source said so**  (FR-005.16, MUT-13).

R-02.16 requires a score be recorded *"together with which provider produced it"* and
R-02.49 repeats the obligation. ``ScoreRecord`` now carries a nullable ``provider`` in both
vendored manifests, so the manifest answers for itself and F5 discharges the obligation by
passing F4's record through **verbatim** - no ``provider`` argument of its own, nothing
derived from the run, nothing defaulted.

What this module adds is the second source and its **label**. The run log's *evaluation*
block - F4's, under ``step-record-v1.json`` - carries the provider too, along with what the
manifest has no room for: the aggregation method, the decode parameters and the subsample
seed, which BRD-02 section 9 keeps together as one later decision rather than amending one
field twice. So a manifest written before the field existed, or one whose ``provider`` is
honestly null, can still be answered from the log.

**The label is the load-bearing part.** A step-record provider presented as the manifest's
is a claim the artifact does not make: the artifact is the thing that travels, and criterion
2's exact case - *deterministic decoding and the same provider* - is only checkable from the
artifact when the artifact says so. So every response says which of the two answered, and
``null`` with no source named is a different answer from ``null`` because nothing recorded
one. MUT-13 removes the label; the test for it must go red.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.errors import Refusal
from training import run_log

log = logging.getLogger("miforge.checkpoints.provider")

__all__ = ["MANIFEST", "NONE", "STEP_RECORD", "provider_sources"]

#: The three answers to *"where did this provider come from?"*. Strings rather than a
#: boolean because there are three facts, and because a response that carried
#: ``from_manifest: false`` could not distinguish the log from nothing.
MANIFEST = "manifest"
STEP_RECORD = "step_record"
NONE = "none"


def _evaluation_provider(log_path: Path, step: int) -> str | None:
    """The provider the run log's evaluation block records for ``step``, if any.

    The **last** matching block wins: a step can be evaluated more than once (a re-score
    after a provider change), and the later line is the later fact.
    """
    if not log_path.is_file():
        return None
    try:
        blocks = [
            block
            for block in run_log.read_blocks(log_path, block="evaluation")
            if block.get("step") == step
        ]
    except Refusal as refusal:
        # A corrupt run log must not take the manifest read with it: the manifest is the
        # authority for the score and this join is additive. Logged rather than swallowed.
        log.warning(
            "run log unreadable while joining a provider",
            extra={"code": refusal.code, "path": str(log_path), "step": step},
        )
        return None
    if not blocks:
        return None
    provider = blocks[-1].get("provider")
    return str(provider) if isinstance(provider, str) and provider else None


def provider_sources(
    doc: Mapping[str, Any],
    *,
    log_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """For ``test`` and ``retention``: the provider and **the source that said so**.

    Args:
        doc: the manifest, as read from disk.
        log_path: the run's log, for the second source. ``None`` means no join is attempted,
            which is reported as such rather than as "no provider recorded".

    Returns:
        ``{"test": {"provider": ..., "source": "manifest"|"step_record"|"none"}, ...}``.
        A record that is itself ``null`` - no evaluation ran at that step - is reported
        with ``"record": null`` so a reader cannot mistake *"no test ran"* for *"a test ran
        and its provider is unknown"*.
    """
    step = doc.get("step")
    joined: dict[str, dict[str, Any]] = {}

    for slot in ("test", "retention"):
        record = doc.get(slot)
        if not isinstance(record, Mapping):
            joined[slot] = {"record": None, "provider": None, "source": NONE}
            continue

        provider = record.get("provider")
        if isinstance(provider, str) and provider:
            # The artifact answers for itself. This is the case criterion 2's exact
            # comparison needs, and it is the only one where the provider travels with the
            # weights.
            joined[slot] = {"record": dict(record), "provider": provider, "source": MANIFEST}
            continue

        from_log = (
            _evaluation_provider(log_path, int(step))
            if log_path is not None and isinstance(step, int)
            else None
        )
        joined[slot] = {
            "record": dict(record),
            "provider": from_log,
            # NEVER `MANIFEST` here. A step-record provider presented as the manifest's is a
            # claim the artifact does not make, and an artifact moved away from its run log
            # would then appear to carry provenance it does not have.
            "source": STEP_RECORD if from_log is not None else NONE,
        }

    return joined
