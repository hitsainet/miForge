"""Resuming an attached run: the served model must BE the checkpoint  (D52; F11 FTASKS 18.0).

A crashed attached run is continued from a checkpoint miLLM saved for it, with **its own
optimizer state restored** (R-02.22, acceptance criterion 4). An AdamW moment buffer belongs
to the parameters it was computed against, so restoring it onto any other weights does not
crash - it *runs*, with a loss curve that looks continuous and is not. That is why this check
exists, why it refuses rather than warns, and why it runs **before** anything is restored.

## What "the served model IS the checkpoint" means, field by field

Read from **one** response - the status body at pre-flight, the attach response at
acquisition (the response that carried the handles this run will train) - never two calls
that could straddle a load.

* **``name``** equals the checkpoint manifest's ``name``. miLLM names a resident set loaded
  from (or saved as) a checkpoint by that checkpoint's qualified name.
* **``manifest_hash``** equals SHA-256 of the checkpoint's ``manifest.json`` bytes - miLLM's
  own definition (``weights_checkpoint_service._verify`` hashes the file it read). The
  manifest in turn carries ``weights_sha256``, which F5's ``resume_plan`` verified against
  the bytes on disk, and miLLM's load verified before touching a tensor (R-01.14). So the
  chain is: served set -> manifest hash -> manifest -> weights hash -> bytes.
* **``steps_since_load == 0``.** The one that is easy to miss and the one the live case
  needs. miLLM's identity **survives commits**: a save sets ``name`` and ``manifest_hash`` to
  the checkpoint it wrote, and every later commit leaves them alone while moving the
  counter. A run killed 2 s after its step-100 save (F11 14.5, f11w2-r4) leaves a resident
  set NAMED step-100 that is step-100 plus whatever was committed after it. Only a fresh
  load resets the counter, so ``0`` is the only reading that proves the resident bytes are
  the checkpoint's. A non-zero counter refuses even when the name and hash match - and the
  operator action is the same: load the checkpoint.
* **the architecture config** (attach response only; the status body carries none): every
  key both the served config and the checkpoint's ``config.json`` state must agree, apart
  from serialisation metadata. ``config.json`` is written with ``save_pretrained``'s
  difference-from-defaults form, so a key absent from it is not a disagreement.

Any mismatch refuses ``resume_served_model_mismatch`` **by name**, listing each field with
both values and the operator action: ``POST /api/weights/load {"name": <checkpoint>}``.
**miForge does not perform that load itself** (D52's design decision, FTDD §6.9): a load
replaces whatever is resident, including commits nothing saved, and that is an operator's
decision to take, not a trainer's to infer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.errors import refuse

log = logging.getLogger(__name__)

#: Config keys that are serialisation metadata rather than architecture. Kept short on
#: purpose: every key added here is one more way two different models compare equal.
CONFIG_METADATA_KEYS: frozenset[str] = frozenset(
    {"transformers_version", "_name_or_path", "_commit_hash", "torch_dtype", "dtype"}
)

#: The refusal code. One name for every mismatch, with the fields in the details, so an
#: operator acts on one message rather than on whichever field happened to be checked first.
MISMATCH = "resume_served_model_mismatch"


@dataclass(frozen=True)
class Expected:
    """What the served model must be, from :class:`services.checkpoints.resume.ResumeTarget`."""

    run_id: str
    step: int
    name: str
    manifest_file_sha256: str
    config: Mapping[str, Any] | None = None
    #: The checkpoint's directory as F5's resolver found it (primary root or a read root).
    #: Once :func:`check_served` has passed, this is the directory of the model being SERVED,
    #: so its root-level tokenizer files are the serving tokenizer's (see
    #: ``attach.verify_tokenizer_source``).
    directory: Path | None = None

    @classmethod
    def of(cls, target: Any) -> Expected:
        directory = getattr(target, "directory", None)
        return cls(
            run_id=str(target.run_id),
            step=int(target.step),
            name=str(target.name),
            manifest_file_sha256=str(target.manifest_file_sha256),
            config=target.config,
            directory=None if directory is None else Path(directory),
        )


@dataclass(frozen=True)
class Verified:
    """What the check proved, carried on the acquisition so the loop can require it."""

    run_id: str
    step: int
    name: str
    manifest_file_sha256: str
    stage: str
    config_keys_compared: int

    def record(self) -> dict[str, Any]:
        # `checkpoint_name`, not `name`: this dict is passed as a log record's `extra`, and
        # `name` is a LogRecord attribute - logging raises KeyError on the collision, which
        # turned every verified resume into a 500 (found by test_hotupdate_resume.py).
        return {
            "run_id": self.run_id,
            "step": self.step,
            "checkpoint_name": self.name,
            "manifest_file_sha256": self.manifest_file_sha256,
            "stage": self.stage,
            "config_keys_compared": self.config_keys_compared,
        }


def config_differences(
    served: Mapping[str, Any] | None, saved: Mapping[str, Any] | None
) -> tuple[list[dict[str, Any]], int]:
    """``(differences, keys_compared)`` between the served and the saved config.

    ``model_type`` must be present and equal on both. Every other key both state, apart from
    :data:`CONFIG_METADATA_KEYS`, must be equal.
    """
    if served is None or saved is None:
        return [], 0
    diffs: list[dict[str, Any]] = []
    if served.get("model_type") != saved.get("model_type") or not saved.get("model_type"):
        diffs.append(
            {
                "key": "model_type",
                "served": served.get("model_type"),
                "saved": saved.get("model_type"),
            }
        )
    shared = sorted((set(served) & set(saved)) - CONFIG_METADATA_KEYS - {"model_type"})
    for key in shared:
        if served[key] != saved[key]:
            diffs.append({"key": key, "served": served[key], "saved": saved[key]})
    return diffs, len(shared) + 1


def check_served(
    expected: Expected,
    *,
    model_block: Any,
    steps_since_load: Any,
    config: Mapping[str, Any] | None = None,
    stage: str,
) -> Verified:
    """Refuse unless the served model IS ``expected``'s checkpoint. See the module docstring.

    Args:
        model_block: miLLM's ``{model_id, name, manifest_hash}`` - ``model`` on attach,
            ``resident_model`` on status.
        steps_since_load: the same response's counter.
        config: the attach response's architecture config, or ``None`` at pre-flight.
        stage: ``preflight`` | ``attach`` - which response was read, for the record.

    Raises:
        Refusal: ``resume_served_model_mismatch``.
    """
    block = model_block if isinstance(model_block, Mapping) else {}
    observed_name = block.get("name")
    observed_hash = block.get("manifest_hash")
    mismatches: list[dict[str, Any]] = []
    if observed_name != expected.name:
        mismatches.append({"field": "name", "served": observed_name, "expected": expected.name})
    if observed_hash != expected.manifest_file_sha256:
        mismatches.append(
            {
                "field": "manifest_hash",
                "served": observed_hash,
                "expected": expected.manifest_file_sha256,
            }
        )
    # `isinstance(..., bool)` first: True is an int and would compare equal to 1.
    if isinstance(steps_since_load, bool) or steps_since_load != 0:
        mismatches.append(
            {
                "field": "steps_since_load",
                "served": steps_since_load,
                "expected": 0,
                "why": (
                    "commits since the set became resident; a save does not reset it, so a "
                    "set named after the checkpoint can carry later steps"
                ),
            }
        )
    diffs, compared = config_differences(config, expected.config)
    if diffs:
        mismatches.append({"field": "config", "differences": diffs[:20]})
    if mismatches:
        raise refuse(
            MISMATCH,
            f"The model miLLM is serving is not checkpoint {expected.name!r}, so this run will "
            f"not restore step {expected.step}'s optimizer state onto it: an AdamW moment "
            f"buffer belongs to the weights it was computed against, and restoring it onto "
            f"others trains on silently, with a loss curve that only looks continuous. "
            f"Differing: {', '.join(m['field'] for m in mismatches)}. Operator action: load "
            f"the checkpoint into miLLM (POST /api/weights/load with "
            f'{{"name": "{expected.name}"}}, operator identity), then submit the continue '
            f"again. miForge does not load it itself. Refused at {stage}, before any "
            f"optimizer state was restored.",
            stage=stage,
            mismatches=mismatches,
            checkpoint={"run_id": expected.run_id, "step": expected.step, "name": expected.name},
            operator_action={
                "call": "POST /api/weights/load",
                "body": {"name": expected.name},
                "identity": "operator",
            },
            miforge_can_perform_it=False,
        )
    verified = Verified(
        run_id=expected.run_id,
        step=expected.step,
        name=expected.name,
        manifest_file_sha256=expected.manifest_file_sha256,
        stage=stage,
        config_keys_compared=compared,
    )
    log.info(
        "the served model is the checkpoint being resumed",
        extra={"code": "resume_served_model_verified", **verified.record()},
    )
    return verified


def expected_for(resume_from: Any) -> Expected:
    """The target of a spec's ``resume_from``, through F5's resolver (one reader)."""
    from services.checkpoints import resume as resume_module

    return Expected.of(resume_module.resolve_resume_from(resume_from))


__all__ = [
    "CONFIG_METADATA_KEYS",
    "MISMATCH",
    "Expected",
    "Verified",
    "check_served",
    "config_differences",
    "expected_for",
]
