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
        return "the model miServer is serving"
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
    from services import hub_access

    device = acquisition_service.resolve_submit_device(spec.gpu_request)
    # Runs only for an agent's filing, inside the gate's requesting-actor scope: the token is
    # False and the fetch metadata-only (017 FR-017.33, .34).
    token = hub_access.hub_token()
    enriched, _availability = acquisition_service.preflight(
        spec, cfg, device=device, token=token, fetch=hub_access.fetch_for(token)
    )
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


def _require_within_pool_maximum(key: str, raw: Any) -> None:
    """019 P3 (F19-D31): the pool size is bounded by the workers provisioned. Checked here, at
    the write and its gate's ``validate``, so an over-maximum write files no approval.
    Non-integers fall through to ``validated_value``, which refuses them as for any field."""
    from wsp.settings import get_wsp_settings

    maximum = get_wsp_settings().wsp_worker_pool_max
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > maximum:
        raise refuse(
            "wsp_pool_size_above_maximum",
            f"The WSP pool size is at most WSP_WORKER_POOL_MAX, {maximum} here; raising it is a "
            "GitOps change to the worker StatefulSet and the miforge-config ConfigMap.",
            key=key,
            maximum=maximum,
        )


def validate_setting_write(values: dict[str, Any], session: Any) -> dict[str, Any]:
    """Allow-listed key, and a value the field's own type and validators accept."""
    del session
    from core import runtime_settings

    key = values["key"]
    _require_runtime_key(key)
    raw = values["body"].value
    if key == "wsp_pool_size":
        _require_within_pool_maximum(key, raw)
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


def _setting_label(key: str) -> str:
    """The words the Settings screen shows for ``key`` (D79's confirm lists it by label)."""
    from core.settings_catalog import CATALOG

    info = CATALOG.get(key)
    return key if info is None else info.label


def summarise_setting_write(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    from api.v1.endpoints.settings import served_value

    key = values["key"]
    return {
        "route": "PUT /api/v1/settings/{key}",
        "label": _setting_label(key),
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
        "label": _setting_label(key),
        "key": key,
        "old": served_value(key, (validated or {}).get("override")),
        "new": served_value(key, (validated or {}).get("env")),
        "reason": values.get("reason"),
    }


# ------------------------------------------------------------- WSP sessions (feature 016)
# FR-016.37, FTDD §5.3. ``decide(values)`` gets no session, so each predicate opens its OWN read
# session and derives the endpoint's locality from the STORED row (P52, R1B-M12); a body field
# is never trusted. Both fail closed: no readable endpoint gates the call, and escalates it.


def _wsp_endpoint_locality(values: dict[str, Any]) -> str | None:
    """The stored endpoint's locality, or ``None`` when the body names no live endpoint."""
    from db.session import session_scope
    from wsp import runs

    body = values.get("body")
    endpoint_id = body.get("endpoint_id") if isinstance(body, dict) else None
    with session_scope() as session:
        row = runs.live_endpoint(session, endpoint_id)
        return None if row is None else runs.endpoint_locality(row)


def wsp_run_needs_approval(values: dict[str, Any]) -> bool:
    """False only for an inspect through a stored LOCAL endpoint (no GPU in phases 1-4, P2)."""
    body = values.get("body")
    kind = body.get("kind") if isinstance(body, dict) else None
    return not (kind == "inspect" and _wsp_endpoint_locality(values) == "local")


def wsp_run_is_external(values: dict[str, Any]) -> bool:
    """Escalate to ``wsp_session_external`` unless the stored endpoint is local."""
    return _wsp_endpoint_locality(values) != "local"


def _wsp_caller(values: dict[str, Any]) -> str:
    agent = values.get("x_miforge_agent")
    return agent if isinstance(agent, str) and agent else "operator"


def validate_wsp_run(values: dict[str, Any], session: Any) -> Any:
    """The route's own refusal code: ``wsp.runs.prepare_start``, exactly as the route runs it."""
    from wsp import runs

    return runs.prepare_start(session, values.get("body"), _wsp_caller(values))  # type: ignore[arg-type]


# Feature 018 (FR-018.8): the protocol test is an inspect session on the endpoint in its path, so
# its predicates ARE the start's, applied to that body: none for a local endpoint,
# wsp_session_external for an external (or unreadable) one.


def _protocol_test_values(values: dict[str, Any]) -> dict[str, Any]:
    return {"body": {"kind": "inspect", "endpoint_id": values.get("endpoint_id")}}


def wsp_protocol_test_needs_approval(values: dict[str, Any]) -> bool:
    return wsp_run_needs_approval(_protocol_test_values(values))


def wsp_protocol_test_is_external(values: dict[str, Any]) -> bool:
    return wsp_run_is_external(_protocol_test_values(values))


def validate_wsp_protocol_test(values: dict[str, Any], session: Any) -> Any:
    """The route's own refusal code: ``wsp.protocol_test.prepare``, exactly as the route runs it."""
    from wsp import protocol_test

    return protocol_test.prepare(
        session, str(values.get("endpoint_id")), values.get("body"), _wsp_caller(values)
    )


def summarise_wsp_protocol_test(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    return {**summarise_wsp_run(values, validated), "protocol_test": True}


def summarise_wsp_run(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    """What the operator sees (FR-016.37), and the endpoint fingerprint execution re-checks."""
    request = validated.frozen.request
    endpoint = validated.endpoint
    return {
        "kind": request["kind"],
        "mode": request["mode"],
        "task": request.get("task"),
        "bundle": {
            "owner": validated.bundle.owner,
            "bundle_id": validated.bundle.bundle_id,
            "bundle_sha256": validated.bundle.bundle_sha256,
        },
        "sensitivity": validated.frozen.sensitivity_value,
        "endpoint": {
            "id": endpoint.id,
            "model": endpoint.model,
            "base_url": endpoint.base_url,
            "locality": validated.frozen.locality,
        },
        "budgets": request["budgets"],
        "provider": request["provider"],
        # 2026-11-21: the thinking in effect, frozen in the request the operator approves.
        "thinking": request.get("thinking"),
        "thinking_tokens": request.get("thinking_tokens"),
        "output_cap": request.get("output_cap", "on"),
        # 2026-11-21: the paging in effect (null: the provider does not page).
        "paging": request.get("paging"),
        # FU-8 (F18-D15): the python tool in effect (off: the editor gets none).
        "python": request.get("python", "on"),
        "request_sha256": validated.frozen.request_sha256,
        "endpoint_fingerprint": validated.fingerprint,
    }


# ------------------------------------------------- an agent's editor endpoint proposal (2026-11-20)
# ``POST /wsp/endpoints`` from an agent (``forge_wsp_register_endpoint``): a LOCAL endpoint with no
# key and no prices, filed as ``wsp_endpoint_register``. The validator is the agent rule, then the
# registration's own checks, storing nothing; the summary is exactly what approving will register.


def validate_wsp_endpoint_proposal(values: dict[str, Any], session: Any) -> dict[str, Any]:
    """``wsp.endpoints.check_agent_proposal`` (key, prices, external, id), then
    ``wsp.endpoints.validate_registration``, exactly as the route runs them."""
    from wsp import endpoints
    from wsp.settings import get_wsp_settings

    body = values.get("body")
    if not isinstance(body, dict):
        raise refuse("validation_failed", "The body is a JSON object.")
    endpoints.check_agent_proposal(body)
    fields, _key = endpoints.validate_registration(session, body, settings=get_wsp_settings())
    return fields


def summarise_wsp_endpoint_proposal(values: dict[str, Any], validated: Any) -> dict[str, Any]:
    """What the operator approves: the endpoint exactly as it will be registered."""
    fields = validated or {}
    return {
        "route": "POST /api/v1/wsp/endpoints",
        "endpoint_id": fields.get("id"),
        "model": fields.get("model"),
        "base_url": fields.get("base_url"),
        "action_protocol": fields.get("action_protocol"),
        "kind": fields.get("kind"),
        "serving": fields.get("serving"),
        "locality": fields.get("locality"),
        "context_window_tokens": fields.get("context_window_tokens"),
        "thinking": fields.get("thinking"),
        "thinking_tokens": fields.get("thinking_tokens"),
        "output_cap": fields.get("output_cap"),
        "paging": fields.get("paging"),
        "output_cap_field": fields.get("output_cap_field"),
        "key_and_prices": "none: an agent cannot send an API key or set prices",
    }
