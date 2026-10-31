"""Submitting a continue  (R-02.22; D52, F11 FTASKS 18.0).

``POST /runs/{run_id}/continue`` **planned** a continue and stopped there (F5): the route's own
docstring said submission "stays F1's", and no submission existed for a run whose checkpoints
are another process's (F11 14.5, OQ-8). This module is the submission, for the one mode D52
decided: an **attached** run resuming from a checkpoint miLLM saved for it.

## The order, and every step refuses having changed nothing durable

1. the row is terminal (a pending or running run is not continued);
2. the recorded definition is an attach run (``continue_submit_mode_unsupported`` otherwise -
   a load-mode continue submission is tracked debt, FTDD §6.9);
3. **F5's plan**: resolve the checkpoint through the read roots, **verify its weights hash**,
   decide trainer state;
4. the step is the run's latest artifact (``resume_not_from_latest``);
5. the row moves back to ``pending`` - **inside the request transaction**, so a refusal below
   rolls it back with everything else;
6. :func:`services.run_launch.launch` with the new definition: the same validator (whose
   attach pre-flight checks the SERVED model is the checkpoint), the run log's new definition
   block carrying ``resolved.resume`` - the discontinuity - then the lease, commit, enqueue.

**The same run id, a new attempt.** Step numbering continues from the checkpoint's step; the
worker reads the last definition block, so it sees ``resume_from``.
"""

from __future__ import annotations

import logging
from typing import Any

from core.errors import refuse
from schemas.run_definition import RunDefinition
from services import run_launch
from services.checkpoints import resume as resume_module
from training import run_state
from workers.run_record import TERMINAL_STATES, require_run

log = logging.getLogger(__name__)

#: The acquisition modes whose continue this module submits. One today (D52).
SUBMITTABLE_MODES: frozenset[str] = frozenset({"attach"})


def submit_continue(
    session: Any,
    run_id: str,
    *,
    step: int | None = None,
    configured_config_hash: str | None = None,
    enqueue: Any | None = None,
    lease: Any | None = None,
    log_root: Any = None,
    **spies: Any,
) -> dict[str, Any]:
    """Plan, verify and submit a continue of ``run_id``. Returns the plan and the launch.

    Raises:
        Refusal: ``run_not_continuable``, ``continue_submit_mode_unsupported``, F5's
            checkpoint refusals, ``resume_not_from_latest``, and whatever the validator
            refuses - in particular ``resume_served_model_mismatch`` from the attach
            pre-flight, naming the operator action.
    """
    status = require_run(session, run_id)
    if status not in TERMINAL_STATES:
        raise refuse(
            "run_not_continuable",
            f"Run {run_id!r} is {status}, not finished, so there is nothing to continue: a "
            f"pending or running run is already going.",
            run_id=run_id,
            status=status,
        )

    record = run_launch.read_definition(session, run_id, log_root=log_root)
    submitted = dict(record["submitted"])
    source = dict(submitted.get("model_source") or {})
    mode = str(source.get("mode") or "load")
    if mode not in SUBMITTABLE_MODES:
        raise refuse(
            "continue_submit_mode_unsupported",
            f"Run {run_id!r} acquired its model by {mode!r}, and submitting a continue is "
            f"built for attached runs only (D52). Its plan is available without "
            f"`submit`; to train on from a checkpoint, define a run whose model_source "
            f"loads that checkpoint directory.",
            run_id=run_id,
            mode=mode,
            submittable_modes=sorted(SUBMITTABLE_MODES),
        )

    # F5'S PLAN: resolve through the read roots (D47), VERIFY THE WEIGHTS HASH, decide
    # trainer state. Refuses having changed nothing.
    plan = resume_module.continue_plan(
        run_id, step=step, configured_config_hash=configured_config_hash
    )
    resume_module.refuse_unless_latest(run_id, plan.step)

    new_source = {k: v for k, v in source.items() if k != "resume_from"}
    new_source["resume_from"] = {"run_id": run_id, "step": plan.step}
    definition = RunDefinition(**{**submitted, "run_id": run_id, "model_source": new_source})

    last_logged = run_launch.last_logged_step(run_id, log_root=log_root)
    previous = run_state.reopen_for_continue(session, run_id)
    target = resume_module.resolve_resume_from(
        definition_resume_from(new_source), root=plan.directory.parent.parent
    )
    resume_record = {
        "resumed_from": {
            "run_id": run_id,
            "step": plan.step,
            "name": plan.parent,
            "weights_sha256": plan.verified_weights_sha256,
            "manifest_file_sha256": target.manifest_file_sha256,
            "directory": str(plan.directory),
        },
        "optimizer": {"state": plan.optimizer_state, "reason": plan.reason},
        # THE DISCONTINUITY: the steps the previous attempt recorded after the checkpoint,
        # which this attempt writes again. Null when there are none.
        "superseded_steps": (
            {"first": plan.step + 1, "last": last_logged}
            if last_logged is not None and last_logged > plan.step
            else None
        ),
        "previous_attempt": previous,
    }
    launched = run_launch.launch(
        definition,
        session=session,
        enqueue=enqueue,
        lease=lease,
        log_root=log_root,
        extra_resolved={"resume": resume_record},
        **spies,
    )
    # Logged only now: the reopen above is part of this submission's transaction, and a
    # refused submission rolls it back (F11 window 5 logged it for refused continues too).
    log.info(
        "run reopened for a continue",
        extra={"code": "run_reopened_for_continue", "run_id": run_id, "previous": previous},
    )
    log.info(
        "continue submitted",
        extra={
            "code": "continue_submitted",
            "run_id": run_id,
            "step": plan.step,
            "optimizer_state": plan.optimizer_state,
            "superseded_steps": resume_record["superseded_steps"],
        },
    )
    return {"plan": plan.as_dict(), "submitted": True, "resume": resume_record, "run": launched}


def definition_resume_from(source: dict[str, Any]) -> Any:
    """The model source's ``resume_from``, as the spec's own type parses it."""
    from training.acquisition.base import ResumeFrom

    return ResumeFrom(**source["resume_from"])


__all__ = ["SUBMITTABLE_MODES", "definition_resume_from", "submit_continue"]
