"""What the submitted weights measure, beside what the editor claims (plan A2, 2026-11-22).

Run ``wsp_4bd06d60acf0`` claimed figures it never measured. Prose is not parsed: at the end of a
session that can produce a candidate, the worker measures the submitted weights against the parent
on practice validation, as ``compare_to_parent`` 4 does, and appends one transcript line
``{event: submitted_measured}``. It spends none of the editor's budget and is never fed back.

It runs AFTER the patch and result are written, only with time to spare before the supervisor's
kill, and a failure is recorded, never raised: a run cannot be lost to its own check. A run frozen
before the version-4 tools gets no line, so it replays as it ran.

Standard library and numpy only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from wsp_worker import judging, observation_tools

EVENT = "submitted_measured"
#: FU-2 (operator, 2026-11-25): the version of ``submit``'s claimed ``figures``, in the transcript
#: header of every session that offers them (a replay offers them by it).
FIGURES_VERSION = 1
#: The figures an editor may claim, in the order a check lists them.
CLAIMABLE = ("auroc", "recall", "false_alarms")
#: Decimal places read from a claimed number's shortest form, at most.
MAX_PLACES = 12
#: Kept free before the supervisor's kill, beside the estimate.
MARGIN_S = 5.0
#: The estimate when no forward pass was timed: the practice bytes at this rate (conservative).
BYTES_PER_S = 1.0e9


def applies(request: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    function = manifest.get("function") or {}
    return (
        request.get("kind") in ("repair", "create")
        and function.get("kind") == "linear_probe/v1"
        and judging.shows_uncertainty(request)
    )


def _figures(side: Mapping[str, Any]) -> dict[str, Any]:
    own = side["at_own_bar"]
    return {
        "auroc": side["auroc"],
        "recall": own["recall"],
        "false_alarms": own["false_alarms"],
        "false_alarm_rate": own["false_alarm_rate"],
    }


def places(value: float) -> int:
    """Decimal places in ``value``'s shortest round-trip form (``0.9904`` -> 4, ``1`` -> 0,
    ``1e-05`` -> 5), at most :data:`MAX_PLACES`: the precision the editor wrote the figure at."""
    text = repr(float(value)).lower()
    mantissa, _, exponent = text.partition("e")
    digits = len(mantissa.partition(".")[2].rstrip("0")) if "." in mantissa else 0
    shift = -int(exponent) if exponent else 0
    return max(0, min(MAX_PLACES, digits + shift))


def matches(figure: str, claimed: float, measured: float) -> bool:
    """A claimed figure matches when the measured one rounds to it at the claim's own precision
    (half a unit in its last place); ``false_alarms`` is a count, so it matches exactly. Run
    ``wsp_4bd06d60acf0`` claimed AUROC 0.9904 where 0.990304 was measured: it differs."""
    if figure == "false_alarms":
        return int(claimed) == int(measured)
    half = 0.5 * 10.0 ** -places(claimed)
    return abs(float(claimed) - float(measured)) <= half * (1 + 1e-9)


def compare_claims(claimed: Mapping[str, Any] | None, line: Mapping[str, Any]) -> dict[str, Any]:
    """FU-2: the editor's claimed figures beside the measured ones. ``compared`` only when both
    exist on the same split; ``differs`` is then whether any stated figure does not match, and
    ``None`` otherwise (nothing to compare is not "no difference")."""
    stated = claimed is not None and any(name in claimed for name in CLAIMABLE)
    split = None if claimed is None else claimed.get("split")
    compared = bool(stated and line.get("measured") is True and split == line.get("split"))
    now = line.get("now") or {}
    checks = []
    for name in CLAIMABLE:
        if claimed is None or name not in claimed:
            continue
        measured = now.get(name) if compared else None
        checks.append(
            {
                "figure": name,
                "claimed": claimed[name],
                "measured": measured,
                "matches": None if measured is None else matches(name, claimed[name], measured),
            }
        )
    return {
        "stated": stated,
        "stated_split": split if stated else None,
        "compared": compared,
        "differs": any(c["matches"] is False for c in checks) if compared else None,
        "checks": checks,
    }


def event(
    make_context: Callable[[], observation_tools.Context],
    submitted: Mapping[str, np.ndarray],
    *,
    left_s: float,
    estimate_s: float,
    claims: bool = False,
    claimed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The transcript line. ``submitted``: the tensors as the patch carries them. With
    ``claims`` (FU-2: the session offered ``submit``'s figures), the line carries ``claims``,
    the comparison with ``claimed`` (the accepted submit's figures, or ``None``); without it the
    line is exactly as before FU-2, so a run recorded before it replays its own line."""
    line = _measure(make_context, submitted, left_s=left_s, estimate_s=estimate_s)
    if claims:
        line["claims"] = compare_claims(claimed, line)
    return line


def _measure(
    make_context: Callable[[], observation_tools.Context],
    submitted: Mapping[str, np.ndarray],
    *,
    left_s: float,
    estimate_s: float,
) -> dict[str, Any]:
    if left_s < estimate_s + MARGIN_S:
        return {"event": EVENT, "measured": False, "reason": "no_time"}
    try:
        ctx = make_context()
        weights = {**ctx.parent, **{n: v.astype(np.float64) for n, v in submitted.items()}}
        changed = any(
            not np.array_equal(v, ctx.parent[n], equal_nan=True) for n, v in weights.items()
        )
        measured = observation_tools.compare_to_parent_v4(
            replace(ctx, current=lambda: dict(weights))
        )
    except Exception as exc:  # recorded, never raised: see the module docstring
        return {"event": EVENT, "measured": False, "reason": "failed", "error": type(exc).__name__}
    return {
        "event": EVENT,
        "measured": True,
        "changed": changed,
        "split": measured["split"],
        "now": _figures(measured["now"]),
        "parent": _figures(measured["parent"]),
        "auroc_against_parent": measured["auroc_against_parent"],
    }
