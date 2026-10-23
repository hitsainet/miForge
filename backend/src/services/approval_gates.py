"""What each gated route checks before an agent's request is stored, and what the operator sees.

F12 (FR-012.11, .15, .18; FTID section 3.3). One block per gated route:

* a **predicate** where the class depends on the request (the attach escalation, continue's
  ``submit``), reading the request exactly as the route itself does;
* a **validate** hook that runs **the route's own refusal code** - the one validator
  ``POST /runs/validate`` serves, the merge preflight, the chain stage validation, the continue
  plan - so the operator is never asked to approve something that would refuse, and an
  invalid request is answered with the route's own envelope and stores nothing;
* a **summary** in the operator's terms: for GPU work the run, run type, model, GPU and step
  budget; for an attach the miLLM target; for a delete the target; for a setting the old and
  new values.

Nothing here allocates, leases or enqueues. The hooks read: the lease is taken by the route
itself when the approved request executes, so a card leased in between ends the approval
``failed`` with the route's own lease refusal.

**ADR-022.** The attach summary reads the served model through
``training.acquisition.attachment_status()`` - the registry, which is the one importer of the
fenced package - never ``training.hotupdate`` directly.
"""

from __future__ import annotations

from typing import Any

from core.errors import refuse

#: The acquisition mode that escalates a GPU start to ``hotupdate_attach`` (FR-012.15).
ATTACH = "attach"


# ------------------------------------------------------------------------- shared helpers


def mode_of(model_source: Any) -> str:
    """The acquisition mode as ``runs.py:_acquisition_mode`` reads it: missing is ``load``."""
    if not isinstance(model_source, dict):
        return "load"
    return str(model_source.get("mode") or "load")


def step_budget(stop_conditions: Any) -> int | None:
    """The ``step_count`` total, or ``None`` when the run stops on another condition only."""
    for condition in stop_conditions or ():
        name = getattr(condition, "condition", None)
        config = getattr(condition, "config", None)
        if isinstance(condition, dict):
            name, config = condition.get("condition"), condition.get("config")
        if name == "step_count" and isinstance(config, dict) and "total" in config:
            try:
                return int(config["total"])
            except (TypeError, ValueError):
                return None
    return None


def model_label(model_source: Any) -> str | None:
    """A model source in one string: the path, the repo, ``attach``, or ``create``."""
    if not isinstance(model_source, dict):
        return None
    mode = mode_of(model_source)
    if mode == ATTACH:
        return "the model miLLM is serving"
    for key in ("path", "repo", "config_path"):
        if model_source.get(key):
            return str(model_source[key])
    return mode


def attach_target() -> dict[str, Any]:
    """Where an attach would land: miLLM's base URL and the model it serves now.

    Read through the acquisition registry (ADR-022). A failure to read is reported in the
    summary, never raised: the operator should still see the request.
    """
    from core.runtime_settings import effective_settings

    target: dict[str, Any] = {"millm_base_url": effective_settings().millm_base_url}
    try:
        from training import acquisition

        status = dict(acquisition.attachment_status())
    except Exception as exc:
        target["served_model"] = None
        target["served_model_unreadable"] = type(exc).__name__
        return target
    target["served_model"] = status.get("resident_model")
    if status.get("reachable") is False:
        target["served_model_unreadable"] = status.get("reason")
    return target


# ---------------------------------------------------------------- POST /runs (F1 submit)


def submit_is_attach(values: dict[str, Any]) -> bool:
    payload = values["payload"]
    return str(getattr(payload.mode, "value", payload.mode)) == ATTACH


def validate_submit(values: dict[str, Any], session: Any) -> dict[str, Any]:
    """F1's own pre-flight: the spec and config refusals, the device, the budget guard."""
    del session
    from services import acquisition as acquisition_service

    payload = values["payload"]
    spec = payload.to_spec()
    cfg = payload.to_config()
    device = acquisition_service.resolve_submit_device(spec.gpu_request)
    enriched, _availability = acquisition_service.preflight(spec, cfg, device=device)
    return {
        "mode": enriched.mode.value,
        "param_count": enriched.param_count,
        "gpu_uuid": None if device is None else device.uuid,
    }


def summarise_submit(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    payload = values["payload"]
    source = {
        "mode": str(getattr(payload.mode, "value", payload.mode)),
        "path": payload.path,
        "repo": payload.repo,
    }
    block: dict[str, Any] = {
        "route": "POST /api/v1/runs",
        "run_id": payload.run_id,
        "run_type": "change",
        "model": model_label(source),
        "gpu_request": payload.gpu_request,
        "gpu_uuid": (validated or {}).get("gpu_uuid"),
        # F1's submit records no stop condition, so it names no step budget.
        "step_budget": None,
    }
    if source["mode"] == ATTACH:
        block["attach"] = attach_target()
    return block


# ------------------------------------------------------------- POST /runs/define (R-02.53)


def define_is_attach(values: dict[str, Any]) -> bool:
    return mode_of(values["definition"].model_source) == ATTACH


def validate_define(values: dict[str, Any], session: Any) -> dict[str, Any]:
    """The one validator - the code behind ``POST /runs/validate`` (FR-006.62)."""
    from services import run_definition as validator_module

    outcome = validator_module.validate(values["definition"], session=session)
    outcome.raise_if_invalid()
    return dict(outcome.resolved)


def summarise_definition(definition: Any, resolved: dict[str, Any] | None) -> dict[str, Any]:
    budget = (resolved or {}).get("budget") or {}
    block: dict[str, Any] = {
        "run_id": definition.run_id,
        "run_type": definition.run_type,
        "model": model_label(definition.model_source),
        "gpu_request": definition.gpu_request,
        "gpu_uuid": budget.get("gpu_uuid"),
        "step_budget": step_budget(definition.stop_conditions),
        "pack": (
            None
            if definition.pack is None
            else f"{definition.pack.pack_id}@{definition.pack.version}"
        ),
    }
    if mode_of(definition.model_source) == ATTACH:
        block["attach"] = attach_target()
    return block


def summarise_define(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    return {
        "route": "POST /api/v1/runs/define",
        **summarise_definition(values["definition"], validated),
    }


# --------------------------------------------------------------------- POST /chains
#
# The chain route's predicate, validator and summary live in ``api/v1/endpoints/chains.py``,
# not here: FR-006.46 requires that deleting the chain modules removes the feature, so no
# module outside them may reach the chain code (``test_chain_deletability.py``).


# ------------------------------------------------------- POST /runs/{run_id}/continue


def continue_submits(values: dict[str, Any]) -> bool:
    """Gated only when it submits (D52 continue is attached-only, hence ``hotupdate_attach``)."""
    body = values.get("body")
    return body is not None and body.submit is True


def validate_continue(values: dict[str, Any], session: Any) -> dict[str, Any]:
    """The continue plan - the route's own ``submit: false`` path - before anything is stored."""
    from services.checkpoints import resume as resume_module
    from workers.run_record import require_run

    run_id = values["run_id"]
    body = values["body"]
    require_run(session, run_id)
    plan = resume_module.continue_plan(
        run_id, step=body.step, configured_config_hash=body.optimizer_config_hash
    )
    return plan.as_dict()


def summarise_continue(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    plan = validated or {}
    return {
        "route": "POST /api/v1/runs/{run_id}/continue",
        "run_id": values["run_id"],
        "from_checkpoint": plan.get("name"),
        "step": plan.get("step"),
        "optimizer_state": (plan.get("optimizer") or {}).get("state"),
        "attach": attach_target(),
    }


# ------------------------------------------------------- POST /runs/{run_id}/evaluate


def validate_evaluate(values: dict[str, Any], session: Any) -> dict[str, Any]:
    from models.run import Run

    run = session.get(Run, values["run_id"])
    if run is None:
        raise refuse(
            "run_not_found", f"No run with id {values['run_id']!r}.", run_id=values["run_id"]
        )
    return {"step": run.last_step, "run_type": run.run_type}


def summarise_evaluate(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    return {
        "route": "POST /api/v1/runs/{run_id}/evaluate",
        "run_id": values["run_id"],
        "run_type": (validated or {}).get("run_type"),
        "at_step": (validated or {}).get("step"),
        "kinds": list(values.get("kinds") or ("test", "retention")),
    }


# ---------------------------------------------------------- POST /adapters/{name}/merge


def validate_merge(values: dict[str, Any], session: Any) -> dict[str, Any]:
    """AC17's preflight - the half of a merge that refuses with no card."""
    from services import runtypes_service

    return dict(runtypes_service.preflight_merge(session, values["name"]))


def summarise_merge(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    return {
        "route": "POST /api/v1/adapters/{name}/merge",
        "adapter": values["name"],
        "into_run_id": values["run_id"],
        "at_step": values["step"],
        "admitted": validated,
    }


# ---------------------------------------------- DELETE /packs/{pack_id}/versions/{version}


def validate_pack_delete(values: dict[str, Any], session: Any) -> dict[str, Any]:
    """The pack exists and no run cites it - the delete's own refusal, read-only."""
    from packs import registry
    from packs.service import pack_service

    pack_id, version = values["pack_id"], values["version"]
    pack_service.get(session, pack_id, version)  # refuses pack_not_found
    runs = registry.usage_map(session).get(registry.row_id(pack_id, version), ())
    if runs:
        raise refuse(
            "pack_in_use",
            f"Pack version {pack_id}@{version} was used by run(s) {list(runs)}, so it cannot "
            f"be deleted; asking the operator to approve a delete that would refuse is noise.",
            pack_id=pack_id,
            version=version,
            runs=list(runs),
        )
    return {"pack_id": pack_id, "version": version}


def summarise_pack_delete(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    return {
        "route": "DELETE /api/v1/packs/{pack_id}/versions/{version}",
        "target": f"pack {values['pack_id']}@{values['version']}",
    }


# ------------------------------------------------------- DELETE /pack-drafts/{draft_id}


def validate_draft_delete(values: dict[str, Any], session: Any) -> dict[str, Any]:
    del session
    from packs.service import pack_service

    draft = pack_service.draft(values["draft_id"])
    payload = draft.payload()
    return {"name": payload.get("name"), "current_step": payload.get("current_step")}


def summarise_draft_delete(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    return {
        "route": "DELETE /api/v1/pack-drafts/{draft_id}",
        "target": f"pack draft {values['draft_id']}",
        "name": (validated or {}).get("name"),
        "current_step": (validated or {}).get("current_step"),
    }


# ----------------------------------------------------------------- /settings/{key}


def _require_runtime_key(key: str) -> None:
    from core.config import NEVER_WRITABLE, RUNTIME_KEYS

    if key not in RUNTIME_KEYS:
        raise refuse(
            "setting_not_runtime",
            f"{key!r} cannot be changed at runtime. Writable: {', '.join(sorted(RUNTIME_KEYS))}."
            + (
                " Database and broker URLs, artifact roots, the trainer token and the "
                "environment are never writable through the API."
                if key in NEVER_WRITABLE
                else ""
            ),
            key=key,
            runtime_keys=sorted(RUNTIME_KEYS),
        )


def validate_setting_write(values: dict[str, Any], session: Any) -> dict[str, Any]:
    """Allow-listed key, and a value the field's own type and validators accept."""
    del session
    from core import runtime_settings

    key = values["key"]
    _require_runtime_key(key)
    raw = values["body"].value
    try:
        value = runtime_settings.validated_value(key, raw)
    except ValueError as exc:
        # `hide_input_in_errors` keeps the rejected value out of the message (ADR-020).
        raise refuse(
            "setting_value_invalid",
            f"{key!r} does not accept that value: {type(exc).__name__}. Read the field's type "
            f"in GET /api/v1/settings.",
            key=key,
        ) from exc
    current = runtime_settings.effective_settings().model_dump(mode="json")[key]
    return {"value": value, "previous": current}


def summarise_setting_write(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    from api.v1.endpoints.settings import served_value

    key = values["key"]
    return {
        "route": "PUT /api/v1/settings/{key}",
        "key": key,
        "old": served_value(key, (validated or {}).get("previous")),
        "new": served_value(key, (validated or {}).get("value")),
        "reason": values["body"].reason,
    }


def validate_setting_reset(values: dict[str, Any], session: Any) -> dict[str, Any]:
    from core.config import get_settings
    from models.setting import SettingOverride

    key = values["key"]
    _require_runtime_key(key)
    row = session.get(SettingOverride, key)
    if row is None:
        raise refuse(
            "setting_not_overridden",
            f"{key!r} has no override; it already reads the environment value.",
            key=key,
        )
    return {"override": row.value, "env": get_settings().model_dump(mode="json")[key]}


def summarise_setting_reset(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    from api.v1.endpoints.settings import served_value

    key = values["key"]
    return {
        "route": "DELETE /api/v1/settings/{key}",
        "key": key,
        "old": served_value(key, (validated or {}).get("override")),
        "new": served_value(key, (validated or {}).get("env")),
        "reason": values.get("reason"),
    }
