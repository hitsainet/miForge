"""One save: the resave rule, the order, and what is announced  (FR-005.1, 6, 42).

## Durable before announced

Index row, then the structured log line, then the socket event. **Never the other way
round.** ADR-007 states the reason in one sentence - *progress reported only over the
socket is invisible to the janitor* - and this project carries the receipt: miStudio reaped
a live 5.8-hour job as "worker lost" because its progress existed only on a socket. The
order is recorded in :attr:`SaveOutcome.effects` so a test can assert the **sequence**
rather than the end state, which is the only way to catch an inversion whose consequence
appears solely after a crash.

## The resave rule is not an edge case

Run end landing on a cadence boundary makes a second save of the same step **ordinary**.
``os.rename`` onto an existing non-empty directory fails with ``ENOTEMPTY``, so the naive
atomic write does not merely race here - it breaks outright. And the obvious repair, remove
the old directory then rename, reopens exactly the window the atomic write closes: between
the removal and the rename there is no checkpoint at that step at all, and a crash there
loses a published artifact that was perfectly good.

So: **no-op when the hash matches, named refusal when it does not, and never a partial
replace.** The matching case is idempotence - a retried task is harmless. The differing
case is two parameter sets claiming one name, and the step is part of the identity, so one
of them has to be refused.

Knowing which case it is costs a full serialisation (see
:func:`services.checkpoints.writer.candidate_weights_sha256`): the hash is of serialised
bytes and is not knowable from the tensors in memory. Paying it is the only alternative to
guessing.

## What is not written here

**No run-log line.** ``step-record-v1.json`` is a discriminated union of five blocks -
step, evaluation, boundary, definition, batch - every one closed to unknown keys, and none
of them is a save. F5 was settled as a *reader* of that contract rather than an author of a
sixth block, so the durable record of a save is the manifest on disk and the index row, and
the line below is ADR-021's structured application log with ``run_id`` bound. Reported
rather than resolved by inventing a block: a sixth block is foundation task 3.3's to add.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.errors import refuse
from services.checkpoints import index as index_module
from services.checkpoints import manifest as manifest_module
from services.checkpoints import writer as writer_module
from services.checkpoints.manifest import ManifestFields
from services.checkpoints.paths import checkpoint_dir, resolve_latest, run_dir

log = logging.getLogger("miforge.checkpoints.service")

__all__ = ["CHECKPOINT_SAVED_EVENT", "SaveOutcome", "save"]

#: The Socket.IO event a completed save publishes into ``miforge/run/<id>``. **Declared
#: once, here, and never spelled at a call site.** F9's store subscribes to this string: a
#: screen subscribed to an event the emitter does not emit is silently dead while looking
#: connected, and that typo is invisible in review. F6's room vocabulary cites this name
#: rather than redefining it.
CHECKPOINT_SAVED_EVENT = "checkpoint_saved"


@dataclass
class SaveOutcome:
    """What one save did.

    ``status`` is ``written`` or ``noop`` - the two outcomes FR-005.6 allows; the third
    possibility is a refusal and refusals are raised, not returned, because a caller that
    can ignore a return value can ignore a refusal.
    """

    status: str
    directory: Path
    manifest: dict[str, Any]
    #: The order the side effects actually happened in. Not telemetry: it is what lets the
    #: ordering test assert the sequence instead of the result.
    effects: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def step(self) -> int:
        return int(self.manifest["step"])


def save(
    model: Any,
    tokenizer: Any,
    *,
    session: Any,
    fields: ManifestFields,
    root: Path | None = None,
    emit: Any = None,
    heartbeat: Any = None,
) -> SaveOutcome:
    """Write one checkpoint for ``fields.run_id`` at ``fields.step``, or refuse by name.

    Args:
        model: anything with ``save_pretrained``. Typed loosely on purpose - ADR-023's
            layout keeps torch out of this module so every refusal path is testable without
            a card, and this function makes no CUDA call.
        tokenizer: likewise.
        session: an open session. **The caller commits**, so a save whose transaction is
            rolled back leaves no row for a checkpoint it did not keep.
        fields: the manifest's borrowed values. The one place another feature's value
            enters a manifest.
        emit: the socket emitter (``api.sockets.emit_run_event``), injected so a worker with
            no Redis reachable still completes a save - the socket is a convenience.
        heartbeat: the foundation's throttled database heartbeat, or ``None``.

    Raises:
        Refusal: ``checkpoint_step_exists`` when a different parameter set already claims
            this step; ``save_refused_missing_field`` when a required manifest value has no
            owner value; ``checkpoint_root_not_atomic`` when the destination's filesystem
            cannot give an atomic rename.
    """
    dest = checkpoint_dir(fields.run_id, fields.step, root=root)
    outcome: SaveOutcome

    if dest.exists():
        outcome = _resave(model, tokenizer, dest=dest, fields=fields, root=root)
        if outcome.status == "noop":
            return outcome
    else:
        result = writer_module.write(model, tokenizer, dest=dest, fields=fields)
        outcome = SaveOutcome(status="written", directory=dest, manifest=result.manifest)

    # ---- durable first ----
    index_module.upsert_from_manifest(session, dest, root=root)
    outcome.effects.append("index")

    log.info(
        "checkpoint saved",
        extra={
            "code": "checkpoint_saved",
            "run_id": fields.run_id,
            "step": fields.step,
            # `checkpoint`, not `name`: `name` is a reserved LogRecord attribute.
            "checkpoint": outcome.name,
            "path": str(dest),
            "weights_sha256": outcome.manifest["weights_sha256"],
            "optimizer_state": outcome.manifest["optimizer"].get("state"),
        },
    )
    outcome.effects.append("log")

    # ---- announced second, and only after the record exists ----
    if emit is not None:
        emit(fields.run_id, CHECKPOINT_SAVED_EVENT, _event_payload(outcome, root=root))
        outcome.effects.append("socket")

    if heartbeat is not None:
        heartbeat.touch(session)
        outcome.effects.append("heartbeat")

    return outcome


def _event_payload(outcome: SaveOutcome, *, root: Path | None) -> dict[str, Any]:
    """What ``checkpoint_saved`` carries: the step, the name, and what ``latest`` resolves to.

    ``latest`` is resolved rather than assumed to be this step: a save of an earlier step
    (a branch's step-0, a repair) does not move the pointer, and a curve marking the wrong
    step is a quietly wrong answer.
    """
    try:
        latest = resolve_latest(outcome.manifest["run_id"], root=root)
    except Exception:  # the event is a convenience; a save is not undone by it
        latest = None
    return {
        "run_id": outcome.manifest["run_id"],
        "step": outcome.step,
        "name": outcome.name,
        "status": outcome.status,
        "latest_step": latest,
    }


def _resave(
    model: Any,
    tokenizer: Any,
    *,
    dest: Path,
    fields: ManifestFields,
    root: Path | None,
) -> SaveOutcome:
    """FR-005.6: no-op on a matching hash, ``checkpoint_step_exists`` otherwise.

    **Never remove-then-rename.** See the module docstring: the repair that looks obvious
    opens the window the whole design closes.
    """
    existing = manifest_module.read(dest)
    candidate = writer_module.candidate_weights_sha256(
        model, tokenizer, parent=run_dir(fields.run_id, root=root), fields=fields
    )

    if candidate == existing["weights_sha256"]:
        log.info(
            "checkpoint already saved; save is a no-op",
            extra={
                "code": "checkpoint_resave_noop",
                "run_id": fields.run_id,
                "step": fields.step,
                "weights_sha256": candidate,
            },
        )
        return SaveOutcome(status="noop", directory=dest, manifest=existing, effects=["noop"])

    raise refuse(
        "checkpoint_step_exists",
        f"A checkpoint already exists for step {fields.step} of run {fields.run_id} with "
        f"different weights. The step is part of a checkpoint's identity, so two parameter "
        f"sets cannot claim one name and the existing directory is never partially "
        f"replaced. Save to a new step, or remove that checkpoint deliberately.",
        run_id=fields.run_id,
        step=fields.step,
        existing_weights_sha256=existing["weights_sha256"],
        candidate_weights_sha256=candidate,
        directory=str(dest),
    )
