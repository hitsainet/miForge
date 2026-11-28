"""Freezing a start request (FR-016.22, P24, WR-13; FTDD §3.3).

``freeze`` turns miWorkbench's start body into the ``FrozenRequest`` of ``miforge-wsp-run-v1.json``
and
its ``request_sha256`` (RFC 8785), with what the run will need: the approval action (or none), the
endpoint's derived locality, the effective sensitivity and the budgets after defaults. ``POST
/wsp/runs/validate`` returns it; ``POST /wsp/runs`` stores exactly what it validates, so changing
any
field makes a new digest and a new run.

Refusals, in order: a bundle not yet imported (``wsp_bundle_not_imported``); a bundle larger than
the worker's ``/work`` can hold (``wsp_bundle_too_large_for_worker``, ``wsp.capacity``); the
caller's queue (``wsp_queue_limit_reached``, R1B-M23); a declared sensitivity that is not the
bundle's (``wsp_sensitivity_mismatch``, PF-36); an unknown endpoint; **a sensitive or unknown
bundle with an external endpoint** (``wsp_sensitive_data_external_endpoint``, WR-13, P11); a cost
cap above
``WSP_COST_CAP_MAX_USD`` (``wsp_cost_cap_above_maximum``, D16); a ``time_s`` under half of what the
``forward_passes`` need (``wsp_time_budget_too_small``, ``wsp.capacity.time_refusal``, fix
2026-11-21); for ``miwsp_engine``, a ``thinking`` the endpoint cannot carry
(``wsp_endpoint_thinking_unsupported``); and any field outside the schema (``validation_failed``).
``caller_ref`` is stored beside the request, never inside it (R4A-8).

**Thinking (2026-11-21).** For ``miwsp_engine`` the request freezes the thinking IN EFFECT, so
``request_sha256`` and a replay carry it: the start body's ``thinking`` (``off``, ``on`` or
``default``) overrides the endpoint's (:func:`thinking_in_effect`); ``default`` freezes as
``null`` (the engine sends nothing). With ``on``, ``thinking_tokens`` (the per-turn reasoning
allowance: the body's, else the endpoint's, else :data:`wsp.endpoints.DEFAULT_THINKING_TOKENS`)
is frozen too, and is present only then. (Until its retirement, FR-018.18, a ``weightspace`` run
froze the body's ``thinking`` as its own effort level; the registry now refuses that provider
before anything is frozen, so every request here is the engine's.)

**The output cap (operator, 2026-11-21).** The start body's ``output_cap``
(``on`` | ``off``) overrides the endpoint's; the value in effect is frozen as ``output_cap:
"off"`` when off and absent when on (today's behaviour, so every request frozen before it reads
the same), so ``request_sha256`` and a replay carry it (:func:`output_cap_in_effect`).

**Paging (operator, 2026-11-21), for ``miwsp_engine`` only.** The start body's ``paging`` (``on`` |
``off``) overrides the endpoint's (``on`` by default); the value in effect is frozen as
``paging: "off"`` when off and absent when on (:func:`paging_in_effect`), as ``output_cap`` is, so
``request_sha256`` and the approval carry it and every default run stays valid under a vendored
copy of the contract from before it. A replay pages by the original transcript's header, so a run
made before paging existed replays unpaged.

**Python (FU-8, F18-D15, 2026-11-25), for ``miwsp_engine`` only.** The start body's ``python``
(``on`` | ``off``; omitted, ``on``) switches the editor's python tool off (FR-018.5: "a request can
switch Python off"). There is no endpoint setting: it is the run's. Frozen as ``python: "off"``
when off and absent when on (:func:`python_in_effect`), the ``paging`` pattern, so
``request_sha256``, the approval and a replay carry it and every default run stays valid under a
vendored copy of the contract from before it. The engine then offers no python tool and starts no
kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import refuse
from wsp import capacity
from wsp.endpoints import (
    DEFAULT_THINKING_TOKENS,
    check_output_cap,
    check_paging,
    check_thinking,
    check_thinking_tokens,
)
from wsp.models import EFFORTS
from wsp.schemas import validate_frozen_request
from wsp.settings import WSPSettings
from wsp.weight_ops.hashing import jcs_sha256

#: Defaults for budgets the start body leaves out (OVN-wsp-forge-12), by the endpoint's locality
#: (operator 2026-11-21, :func:`default_budgets`): ``tokens``, ``max_steps`` and ``time_s`` from
#: ``WSP_DEFAULT_{TOKENS,MAX_STEPS,TIME_BUDGET_S}_{LOCAL,EXTERNAL}``; ``cost_usd`` from
#: ``WSP_COST_CAP_DEFAULT_USD`` (external only); ``forward_passes`` 0, for every endpoint.
DEFAULT_FORWARD_PASSES = 0


def default_budgets(locality: str, settings: WSPSettings) -> dict[str, Any]:
    """The budgets a start body leaves out. A local editor's are safety stops (5,000,000 tokens,
    100 steps, 2 hours by default); an external one keeps 100,000 tokens, 50 steps and 1 hour.
    Anything but ``external`` is local, as the cost cap reads it."""
    if locality == "external":
        return {
            "forward_passes": DEFAULT_FORWARD_PASSES,
            "tokens": settings.wsp_default_tokens_external,
            "max_steps": settings.wsp_default_max_steps_external,
            "time_s": settings.wsp_default_time_budget_s_external,
        }
    return {
        "forward_passes": DEFAULT_FORWARD_PASSES,
        "tokens": settings.wsp_default_tokens_local,
        "max_steps": settings.wsp_default_max_steps_local,
        "time_s": settings.wsp_default_time_budget_s_local,
    }


START_MODES = ("session", "program")
#: The worker's observation-tool catalogue, (name, version) -> cost (FR-016.18). A literal here
#: because the backend does not import the worker; ``test_observation_tools.py`` pins it equal to
#: ``wsp_worker.observation_tools.CATALOGUE``. Every version stays (fix 2026-11-21): a body may
#: still name version 1, and a replay of a run recorded before version 2 must run version 1.
#: Version 3 (``stratified_false_alarms`` 2; fix 2026-11-22, D128 round 7) measures each head at
#: its own bar re-cut to the target false-alarm rate, as the held-out verdict does; a run frozen
#: before it names version 2 and replays version 2. Version 4 (plan A1, 2026-11-22) gives a
#: difference its interval and measures on validation by default; version 3 replays as version 3.
#: ``supplement_metrics`` 1 (2026-11-25, FU-7a C2) reports a bundle's validation-only supplements.
OBSERVATION_TOOLS: dict[tuple[str, str], int] = {
    ("practice_metrics", "1"): 1,
    ("practice_metrics", "2"): 1,
    ("practice_metrics", "3"): 1,
    ("practice_metrics", "4"): 1,
    ("stratified_false_alarms", "1"): 1,
    ("stratified_false_alarms", "2"): 1,
    ("direction_cosines", "1"): 1,
    ("firing_positions", "1"): 1,
    ("compare_to_parent", "1"): 2,
    ("compare_to_parent", "2"): 2,
    ("compare_to_parent", "3"): 2,
    ("compare_to_parent", "4"): 2,
    ("supplement_metrics", "1"): 2,
}
#: The version a start body gets for a tool it names without one: the newest (pinned equal to
#: ``wsp_worker.observation_tools.CURRENT``).
CURRENT_OBSERVATION_TOOLS: dict[str, str] = {
    "practice_metrics": "4",
    "stratified_false_alarms": "2",
    "direction_cosines": "1",
    "firing_positions": "1",
    "compare_to_parent": "4",
    "supplement_metrics": "1",
}


#: The provider whose request carries the endpoint's thinking (``wsp.protocol_test.PROVIDER``).
ENGINE_PROVIDER = "miwsp_engine"
#: A start body's ``thinking`` for ``miwsp_engine``: ``null`` or absent takes the endpoint's.
#: Feature 020 (FR-020.4): the three efforts, frozen as themselves (the run schema's
#: ``thinking`` already admits them), and checked with the endpoint's kind and locality.
ENGINE_THINKING = ("off", "on", "default", *EFFORTS)


def thinking_in_effect(
    body: dict[str, Any], endpoint: dict[str, Any]
) -> tuple[str | None, int | None]:
    """``(thinking, thinking_tokens)`` to freeze. See the module docstring."""
    asked = body.get("thinking")
    tokens = body.get("thinking_tokens")
    if asked is not None and asked not in ENGINE_THINKING:
        raise refuse(
            "validation_failed",
            f"For {ENGINE_PROVIDER}, thinking is one of {', '.join(ENGINE_THINKING)}, or null "
            "for the endpoint's setting.",
        )
    value = check_thinking(
        endpoint.get("thinking", "off") if asked is None else asked,
        kind=str(endpoint.get("kind", "")),
        serving=str(endpoint.get("serving", "")),
        locality=str(endpoint.get("locality", "local")),
    )
    if value != "on":
        if tokens is not None:
            raise refuse(
                "validation_failed",
                f"thinking_tokens applies when thinking is on; the thinking in effect is {value}.",
            )
        return (None if value == "default" else value), None
    allowance = check_thinking_tokens(tokens, window=None)
    if allowance is None:
        allowance = endpoint.get("thinking_tokens") or DEFAULT_THINKING_TOKENS
    return "on", int(allowance)


def output_cap_in_effect(body: dict[str, Any], endpoint: dict[str, Any]) -> str:
    """The start body's ``output_cap``, else the endpoint's (``on`` when it states none)."""
    asked = body.get("output_cap")
    return check_output_cap(endpoint.get("output_cap", "on") if asked is None else asked)


def paging_in_effect(body: dict[str, Any], endpoint: dict[str, Any]) -> str:
    """The ``paging`` to freeze: the start body's, else the endpoint's (``on`` when it states
    none). Feature 020 (FR-020.6, F20-D4): on an ``append_only`` endpoint it is always ``off``,
    the run contract's only paging value (the engine reads the mode from ``spec.editor``), and a
    start body's ``on`` there is refused ``wsp_history_mode_conflict`` before anything queues."""
    asked = body.get("paging")
    if endpoint.get("history_mode", "paging") == "append_only":
        if asked is not None and check_paging(asked) == "on":
            raise refuse(
                "wsp_history_mode_conflict",
                "paging on applies to an endpoint whose history_mode is paging; this endpoint is "
                "append_only, which never rewrites what it sent. Leave paging out.",
                history_mode="append_only",
            )
        return "off"
    return check_paging(endpoint.get("paging", "on") if asked is None else asked)


#: FU-8: a start body's ``python``.
PYTHON_VALUES = ("on", "off")


def python_in_effect(body: dict[str, Any]) -> str:
    """The start body's ``python`` (``on`` when omitted or null). Anything else is refused."""
    asked = body.get("python")
    if asked is None:
        return "on"
    if asked not in PYTHON_VALUES:
        raise refuse(
            "validation_failed",
            f"python is one of {', '.join(PYTHON_VALUES)} (omitted: on); off gives the editor no "
            "python tool.",
        )
    return str(asked)


@dataclass(frozen=True)
class Frozen:
    request: dict[str, Any]
    request_sha256: str
    approval_action: str | None
    locality: str
    sensitivity_value: str
    sensitivity_effective: str
    caller_ref: str | None


def effective_sensitivity(value: str | None) -> str:
    """P11: anything but ``not_sensitive`` (absent, ``unknown``) is treated as sensitive."""
    return "not_sensitive" if value == "not_sensitive" else "sensitive"


def _observation_tool(tool: Any) -> dict[str, Any]:
    """A start body's tool entry, frozen whole: a name the worker knows, its version (the newest
    when the body names none) and that version's cost (the catalogue's when the body names none).
    Anything else is refused, naming the catalogue."""
    entry = dict(tool) if isinstance(tool, dict) else {}
    name = str(entry.get("name"))
    version = entry.get("version", CURRENT_OBSERVATION_TOOLS.get(name))
    cost = OBSERVATION_TOOLS.get((name, str(version)))
    if (
        set(entry) - {"name", "version", "cost"}
        or cost is None
        or not isinstance(version, str)
        or entry.get("cost", cost) != cost
    ):
        raise refuse(
            "validation_failed",
            f"Observation tool {tool!r} is not in the worker's catalogue: "
            + ", ".join(
                f"{n} version {v!r} cost {c}" for (n, v), c in sorted(OBSERVATION_TOOLS.items())
            )
            + ". A tool named without a version gets the newest; without a cost, its own.",
        )
    return {"name": name, "version": version, "cost": cost}


def approval_action(operation: str, locality: str) -> str | None:
    """FR-016.37: external sessions always ask; a local inspect does not; other local ones do."""
    if locality == "external":
        return "wsp_session_external"
    if operation == "inspect":
        return None
    return "wsp_session"


def freeze(
    body: dict[str, Any],
    *,
    bundle: dict[str, Any],
    bundle_imported: bool,
    endpoint: dict[str, Any] | None,
    provider_version: str,
    queued_for_caller: int,
    settings: WSPSettings,
) -> Frozen:
    """Validate and freeze ``body``; ``endpoint`` is the stored row ``{id, locality, model}``."""
    if not bundle_imported:
        raise refuse(
            "wsp_bundle_not_imported",
            "The bundle's import has not completed; start the run once GET /wsp/bundles/{id} "
            "reports it imported.",
        )
    # Production defect 2026-11-20: a bundle the worker's /work cannot hold is refused here, so
    # the operator is never asked to approve a run that cannot succeed.
    capacity.require_fits(capacity.manifest_bytes(bundle), settings)
    if queued_for_caller >= settings.wsp_max_queued_per_caller:
        raise refuse(
            "wsp_queue_limit_reached",
            f"This caller already has {queued_for_caller} queued WSP runs (the limit is "
            f"{settings.wsp_max_queued_per_caller}). Wait for one of them to start, then file "
            "this request again: the limit is checked when an approved request executes too, and "
            "an approval refused here is not retried.",
        )
    sensitivity = bundle["sensitivity"]["value"]
    if body.get("sensitivity") != sensitivity:
        raise refuse(
            "wsp_sensitivity_mismatch",
            "The request's sensitivity is not the bundle's; a caller's claim never loosens the "
            "operator's mark.",
            bundle=sensitivity,
        )
    mode = body.get("mode")
    if mode not in START_MODES:
        raise refuse("validation_failed", "mode must be session or program.")
    if endpoint is None:
        raise refuse("wsp_endpoint_unknown", "No such editor endpoint; register it first.")
    locality = endpoint["locality"]
    effective = effective_sensitivity(sensitivity)
    if locality == "external" and effective == "sensitive":
        raise refuse(
            "wsp_sensitive_data_external_endpoint",
            "A sensitive or unknown-sensitivity bundle cannot be sent to an external editor; use "
            "a local endpoint.",
        )
    budgets: dict[str, Any] = default_budgets(locality, settings)
    budgets.update(body.get("budgets") or {})
    if "cost_usd" not in budgets:
        budgets["cost_usd"] = settings.wsp_cost_cap_default_usd if locality == "external" else None
    if budgets["cost_usd"] is not None and budgets["cost_usd"] > settings.wsp_cost_cap_max_usd:
        raise refuse(
            "wsp_cost_cap_above_maximum",
            f"A session's cost cap is at most ${settings.wsp_cost_cap_max_usd:g}.",
        )
    # Fix 2026-11-21: a time budget clearly too short for the forward passes asked for.
    capacity.require_time(budgets, bundle, settings)
    kind = body.get("kind")
    if kind not in ("inspect", "create", "repair"):
        raise refuse("validation_failed", "kind must be inspect, create or repair.")
    sets = {s["name"] for s in bundle["heldout_ref"]["sets"]}
    primary = list(body.get("primary_sets") or [])
    if not primary or not set(primary) <= sets:
        raise refuse("validation_failed", "primary_sets must name held-out sets of the bundle.")
    limits: dict[str, Any] = {
        "max_changed_entries": None,
        "allowed_tensors": None,
        "frozen_tensors": [],
    }
    limits.update(body.get("change_limits") or {})
    tools = [_observation_tool(tool) for tool in list(body.get("observation_tools") or [])]
    names = [t["name"] for t in tools]
    if len(set(names)) != len(names):
        raise refuse(
            "validation_failed",
            "An observation tool is named twice; name each tool once, at one version.",
        )
    parent = body.get("parent") or {"kind": "bundle", "ref": bundle["bundle_sha256"]}
    if (parent or {}).get("kind") != "bundle" or body.get("chain"):
        # Review 016-P2, R16P2-3: FR-016.5's chaining is not wired (the push carries only the
        # bundle's files, the adapter diffs against the bundle, and the intake refuses a patch
        # whose parent is not the request's), so such a run would fail after its approval and its
        # whole budget. Refused here, before either, until chaining is wired.
        raise refuse(
            "wsp_phase_not_enabled",
            "A run on a candidate (parent kind candidate, or chain: true) is not available yet: "
            "start from the bundle (omit parent and chain).",
        )
    if parent.get("ref") != bundle["bundle_sha256"]:
        raise refuse(
            "validation_failed",
            "parent.ref must be this bundle's bundle_sha256 (or omit parent).",
        )
    provider_id = (body.get("provider") or {}).get("id", settings.wsp_default_provider)
    thinking, thinking_tokens = thinking_in_effect(body, endpoint)
    paging = paging_in_effect(body, endpoint)
    python = python_in_effect(body)
    request: dict[str, Any] = {
        "kind": kind,
        "goal": body.get("goal"),
        "mode": mode,
        "task": body.get("task"),
        "bundle": body.get("bundle"),
        "sensitivity_declared": sensitivity,
        "parent": parent,
        "chain": bool(body.get("chain", False)),
        "keep_weights": bool(body.get("keep_weights", False)),
        "provider": {"id": provider_id, "version": provider_version},
        "endpoint": {"id": endpoint["id"], "locality": locality, "model": endpoint["model"]},
        "thinking": thinking,
        "budgets": budgets,
        "change_limits": limits,
        "observation_tools": sorted(tools, key=lambda x: str(x["name"])),
        "primary_sets": primary,
        "program_sha256": None,
        "support_set": None,
        "of_run_id": None,
    }
    if output_cap_in_effect(body, endpoint) == "off":
        # Present only when off (the run schema's second optional field): absent means on.
        request["output_cap"] = "off"
    if paging == "off":
        # The engine's only, present only when off (the run schema's third optional field):
        # absent means on.
        request["paging"] = "off"
    if python == "off":
        # FU-8 (F18-D15): the run schema's fourth optional field, present only when off.
        request["python"] = "off"
    if thinking_tokens is not None:
        # Present only with thinking on (the run schema's one optional field): every other
        # request stays valid under a copy of the contract from before it existed.
        request["thinking_tokens"] = thinking_tokens
    if request["keep_weights"] and kind != "create":
        raise refuse("validation_failed", "keep_weights applies to create only.")
    if (request["bundle"] or {}).get("bundle_sha256") != bundle["bundle_sha256"]:
        raise refuse("validation_failed", "bundle.bundle_sha256 is not the imported bundle's.")
    validate_frozen_request(request)
    caller_ref = body.get("caller_ref")
    if caller_ref is not None and (not isinstance(caller_ref, str) or len(caller_ref) > 128):
        raise refuse("validation_failed", "caller_ref is a string of at most 128 characters.")
    return Frozen(
        request=request,
        request_sha256=jcs_sha256(request),
        approval_action=approval_action(kind, locality),
        locality=locality,
        sensitivity_value=sensitivity,
        sensitivity_effective=effective,
        caller_ref=caller_ref,
    )
