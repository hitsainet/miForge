"""F4's read surface, plus the out-of-cadence evaluation  (FPRD §7; ADR-012, ADR-013, ADR-018).

Five routes in one module - two routers, because they are two surfaces and FastAPI prefixes
are per-router. Named ``scoring.py`` rather than ``providers.py`` + ``scores.py`` so this
feature adds **one** file to a directory several features write into concurrently.

## Every number names its scale and its sample  (ADR-012)

Every score element carries ``n`` and its provider, so **F9 cannot render a bare ``0.84``**
even by accident. That is enforced by the payload rather than by F9's diligence, and it is
why the obligation lives in this feature: *a bare* ``0.84`` *is a defect in F4's output, not
a gap in F9's rendering.*

The series also carries ``provider_discontinuity`` wherever the run's provider was re-pinned,
so F9 **breaks** the curve there rather than joining two instruments' numbers.

## The refusal is the product

``POST /providers/{name}/check`` exists to tell an operator *why* a provider cannot be used.
When it fails, the named refusal with its ``details`` **is** the useful answer - not a
degraded success.

## Never a credential, and never a response default

``ProviderIdentity`` has no credential field and :meth:`providers.settings.HttpProviderRow.redacted`
has none either, so the only way to leak one is to add it by hand. Do not. And no response
model carries a default that could stand in for a missing value (ADR-018): a
``score: float = 0.0`` default is a fabricated measurement.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from api.agent_origin import requires_approval_when_agent
from api.deps import session_dependency
from core.errors import Refusal, refuse
from providers import registry
from providers.base import GenerationParams
from services import approval_gates
from services.scoring import entrypoint
from services.scoring.settings import exposure_window_steps, get_scoring_settings
from services.scoring.tables import SCORE_ITEMS, SCORES

log = logging.getLogger("miforge.scoring")

providers_router = APIRouter(prefix="/providers", tags=["providers"])
scores_router = APIRouter(prefix="/runs", tags=["scores"])


# --------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------


@providers_router.get("")
def list_providers() -> dict[str, Any]:
    """The **live** registry: every row, available or not, **named either way**.

    Computed from live settings on each call rather than cached at import, as F3's capability
    listing is, so an operator who configures an endpoint sees it without restarting this
    layer.

    An unavailable row is **listed with its reason**, never omitted: *an option missing from
    a response is indistinguishable from one that is broken* (ADR-013).
    """
    rows = registry.list_rows()
    return {
        "providers": [row.payload() for row in rows],
        # Stated on the listing because the judge path is the one place a reader could
        # reasonably expect the in-process provider to serve and it never may (R-02.11).
        "judge_eligible": [row.name for row in rows if row.judge_eligible and row.available],
        "capability_vocabulary": sorted(registry.KNOWN_CAPABILITIES),
    }


@providers_router.post("/{name}/check")
def check_provider(name: str, prompt: Annotated[str, Body(embed=True)] = "ping") -> dict[str, Any]:
    """A one-prompt liveness and capability probe. **The refusal is the product.**

    Returns the resolved identity, the declared capabilities and the completion's length -
    never the completion text, which is untrusted data read back from a provider and has no
    reason to be echoed into an API response.
    """
    provider = registry.get(name)
    params = GenerationParams(max_tokens=16)
    completions = provider.generate([prompt], params)
    # The capabilities come from the REGISTRY ROW, not from the provider object - and that is
    # a finding rather than a preference. :class:`providers.base.InferenceProvider` declares
    # ``identity``, ``available`` and ``generate`` and **no ``capabilities``**, although
    # FR-004.6 makes capability declaration a requirement and both implementations provide
    # one. Reading it off the provider would depend on a method the gate contract does not
    # promise; the row is the authority this feature actually owns. Reported for foundation
    # task 3.8.
    row = registry.row_for(name)
    return {
        "name": name,
        "identity": provider.identity.model_dump(mode="json"),
        "capabilities": sorted(row.capabilities) if row else [],
        "available": provider.available(),
        "completion_chars": len(completions[0].text) if completions else 0,
    }


# --------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------


@scores_router.get("/{run_id}/scores")
def run_scores(
    run_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """One run's score series, with everything a curve needs to be honest.

    ``exposure_window_steps`` is reported as **a number** (FR-004.26, AC-004.15): with a
    retention cadence of N, up to N-1 steps of training can occur between a breach happening
    and being seen, and an operator who was not told that reads the floor as a guarantee it
    is not.
    """
    rows = session.execute(
        sa.select(SCORES).where(SCORES.c.run_id == run_id).order_by(SCORES.c.step, SCORES.c.kind)
    ).mappings()
    settings = get_scoring_settings()
    scores = [_score_payload(row) for row in rows]
    return {
        "run_id": run_id,
        "scores": scores,
        "test_every": settings.test_every,
        "retention_every": settings.retention_every,
        "exposure_window_steps": exposure_window_steps(settings.retention_every),
        # The steps at which the instrument changed. F9 breaks the curve here rather than
        # joining two providers' numbers into one line (FR-004.10).
        "provider_discontinuities": sorted(
            {int(score["step"]) for score in scores if score["provider_discontinuity"]}
        ),
    }


@scores_router.get("/{run_id}/scores/{step}")
def run_scores_at_step(
    run_id: str, step: int, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """One step's evaluations, with per-prompt detail **where it was retained**.

    Per-prompt detail is prunable and the set score is not (FPRD OQ-6), so an empty ``items``
    list means *pruned or never retained*, not *scored nothing*. Said in the payload with
    ``items_retained`` rather than left for a reader to infer from an empty list.
    """
    rows = list(
        session.execute(
            sa.select(SCORES)
            .where(SCORES.c.run_id == run_id, SCORES.c.step == step)
            .order_by(SCORES.c.kind, SCORES.c.created_at)
        ).mappings()
    )
    if not rows:
        raise refuse(
            "score_not_found",
            f"Run {run_id!r} has no recorded evaluation at step {step}. A step with no "
            f"evaluation is normal on a cadence coarser than every step; this is not an "
            f"error about the run.",
            run_id=run_id,
            step=step,
        )
    payloads: list[dict[str, Any]] = []
    for row in rows:
        items = list(
            session.execute(
                sa.select(SCORE_ITEMS)
                .where(SCORE_ITEMS.c.score_id == row["id"])
                .order_by(SCORE_ITEMS.c.prompt_index)
            ).mappings()
        )
        payload = _score_payload(row)
        payload["items_retained"] = len(items)
        payload["items"] = [
            {
                "prompt_index": item["prompt_index"],
                "verdict": item["verdict"],
                "scalar": item["scalar"],
                "reason_code": item["reason_code"],
                "completion_excerpt": item["completion_excerpt"],
            }
            for item in items
        ]
        payloads.append(payload)
    return {"run_id": run_id, "step": step, "scores": payloads}


@scores_router.post("/{run_id}/evaluate")
@requires_approval_when_agent(
    "gpu_start",
    validate=approval_gates.validate_evaluate,
    summary=approval_gates.summarise_evaluate,
)
def evaluate_now(
    run_id: str,
    session: Annotated[Session, Depends(session_dependency)],
    kinds: Annotated[list[str] | None, Body(embed=True)] = None,
) -> dict[str, Any]:
    """Evaluate now, out of cadence, against the run's **pinned** provider.

    Recorded with ``out_of_cadence: true`` so it **never silently joins the step series** - an
    operator's ad-hoc check becoming a data point in a curve is how a curve acquires a point
    nobody can explain.
    """
    from models.run import Run

    run = session.get(Run, run_id)
    if run is None:
        raise refuse("run_not_found", f"No run with id {run_id!r}.", run_id=run_id)
    step = run.last_step if run.last_step is not None else 0
    block = entrypoint.evaluate_run(
        run_id=run_id,
        step=int(step),
        out_of_cadence=True,
        kinds=tuple(kinds) if kinds else ("test", "retention"),
        session=session,
    )
    return {"run_id": run_id, "step": int(step), "out_of_cadence": True, "evaluation": block}


def _score_payload(row: Any) -> dict[str, Any]:
    """One ``mf_scores`` row as a response element. **``n`` and the provider are always here.**"""
    return {
        "step": row["step"],
        "kind": row["kind"],
        "set": row["set_rendered"],
        "score": row["score"],
        "n": row["n"],
        "floor": row["floor"],
        "floor_layer": row["floor_layer"],
        "breached": bool(row["breached"]),
        "provider": row["provider"],
        "provider_kind": row["provider_kind"],
        "provider_model": row["provider_model"],
        "aggregation": row["aggregation"],
        "decode": row["decode"],
        "subsample_seed": row["subsample_seed"],
        "split_markers": row["split_markers"],
        "reproducible": bool(row["reproducible"]),
        "not_reproducible_reason": row["not_reproducible_reason"],
        "out_of_cadence": bool(row["out_of_cadence"]),
        "provider_discontinuity": bool(row["provider_discontinuity"]),
        "message": _message_for(row),
    }


def _message_for(row: Any) -> str:
    """The one sentence F9 renders beside the number, built **here** so it exists once.

    ADR-012: every number names its scale and sample. ADR-021: user-facing text says what to
    do next. A breach therefore names the set, the score, the floor, the sample and the step
    and states the consequence - rather than describing how the operator should feel.
    """
    base = (
        f"{row['kind'].capitalize()} {row['score']:.3f} on {row['n']} prompts "
        f"via {row['provider']}"
    )
    if row["floor"] is not None:
        base = f"{base} (floor {row['floor']:.3f})"
    if row["breached"]:
        return (
            f"{base}. Below the floor: the run's stop condition reads this signal. Continue "
            f"from the last checkpoint above the floor, or lower the floor deliberately."
        )
    return f"{base}."


__all__ = ["Refusal", "providers_router", "scores_router"]
