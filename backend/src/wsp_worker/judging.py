"""How the editor's result is judged, told to the editor (fix 2026-11-22, D128 round 7).

In round 7 the editor optimised the false-alarm rate at the parent's FIXED shipped bar, a figure
the held-out verdict never uses: the verdict re-cuts every head's bar to the target false-alarm
rate, so the three bias-only candidates equalled the parent bit for bit. Nothing had told the
editor how it would be judged. This module writes that down from the run's own evaluation
configuration (the frozen request's ``primary_sets``; the bundle's ``heldout_ref``,
``baselines_available`` and ``operating_point``; the function's rule), never as fixed prose about
one bundle, and holds the two other texts the fix adds: the tools' invariance note and the submit
warning.

**Both providers got the same text** while both ran (D27's bake-off compared the engine, not a
prompt): it is appended to the ``## Context`` block ``miwsp_engine`` builds
(:func:`context_with_judging`), as the ``weightspace`` adapter appended it until 018-P4.

**Replay** (FR-016.21, FR-018.13): a run recorded before this section existed must replay as it
ran. The engine records :data:`VERSION` in its transcript header (``judging``) and a replay
follows the header. (A ``weightspace`` run's replay followed whether its transcript held
:data:`HEADING`; since 018-P4 such a replay is refused before it reaches the worker.)

The verdict's constants are restated here because the worker imports nothing from ``wsp`` (FTDD
§6.1's fence); ``tests/unit/wsp/test_judging_pins.py`` pins them equal to ``wsp.verdict``'s.

Standard library only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: The transcript header's ``judging`` value while the section and the submit warning are on.
VERSION = 1
HEADING = "## How your result is judged"
#: ``wsp.verdict``'s, restated (pinned by test).
REFITS = ("logistic_regression", "lda", "diff_of_means")
OPERATING_POINT = ("threshold_recut", "affine_recalibration")
FALSE_ALARM_ROLES = ("everyday_text", "nonsense_text")
CI = 0.95
PRIMARY_METRIC = "auroc"
#: A set name shown to the editor must be an identifier, never free text from a manifest.
SET_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

#: The observation tools' note on their judged figures (``observation_tools``), by rule. Fixed
#: texts: ``numbers_and_names_only`` admits these strings and no other free text.
INVARIANCE = (
    "A uniform shift of the bias (or a positive rescale of weight and bias together) changes "
    "none of these figures: each head's bar is re-cut to the target false-alarm rate, as the "
    "held-out verdict re-cuts it, and AUROC depends only on how rows rank."
)
INVARIANCE_SOFTMAX = (
    "A uniform shift of the bias changes none of these figures: each head's bar is re-cut to the "
    "target false-alarm rate, as the held-out verdict re-cuts it, and AUROC depends only on how "
    "rows rank."
)
SAME_VERDICT = (
    "These weights differ from the parent only by a uniform bias shift or a positive rescale, so "
    "their held-out verdict cannot differ from the parent's. Only a change to the weight vector's "
    "direction changes how rows rank."
)
#: A compared AUROC difference whose interval includes zero (plan A1, 2026-11-22).
WITHIN_NOISE = (
    "The AUROC difference from the parent is within noise: its interval includes zero, so this "
    "is not evidence of a gain or of a loss."
)
FIXED_TEXTS = frozenset({INVARIANCE, INVARIANCE_SOFTMAX, SAME_VERDICT, WITHIN_NOISE})
#: The section's line for a run frozen with the version-4 tools (plan A1).
CHOOSE_ON_VALIDATION = (
    "- Choose edits on the validation split; train is what the parent was fitted on. Count a gain "
    "only when 'auroc_against_parent' is not within_noise, and remember that each further look at "
    "validation makes a lucky gain likelier."
)
#: The tool version from which comparisons carry their uncertainty.
UNCERTAINTY_FROM = 4

#: The submit warning (once per session; the editor may then submit the same edit again).
AFFINE_SUBMIT = (
    "Not submitted yet. This edit changes the parent only by {what}, so every figure the held-out "
    "verdict uses (AUROC, and recall and false alarms at each head's own re-cut bar) is the "
    "parent's exactly: the verdict cannot differ from the parent's. Change the direction of the "
    "weight vector and test it, or call `{tool}` again to submit this edit as it is (for "
    "example, if you conclude that no better edit exists). You are told this once."
)


def invariance_note(rule: str) -> str:
    return INVARIANCE_SOFTMAX if rule == "softmax" else INVARIANCE


def _names(names: Sequence[str]) -> str:
    shown = [n if SET_NAME.match(n) else f"<set {i + 1}>" for i, n in enumerate(names)]
    return ", ".join(shown) if shown else "(none)"


def shows_uncertainty(request: Mapping[str, Any]) -> bool:
    """Whether the run froze the tools that give a difference its interval (version 4 on). The
    frozen request decides, so a replay of an older run is told and shown what it was."""
    for tool in request.get("observation_tools") or []:
        version = str(tool.get("version", ""))
        if (
            tool.get("name") in ("practice_metrics", "compare_to_parent")
            and version.isdecimal()
            and int(version) >= UNCERTAINTY_FROM
        ):
            return True
    return False


def section(request: Mapping[str, Any], manifest: Mapping[str, Any]) -> str | None:
    """The judging section for this run, or ``None`` where no candidate is judged this way (an
    ``inspect``, or a bundle whose function is not a linear probe). Kept short: an editor with an
    8,192-token window must still fit its task."""
    function = manifest.get("function") or {}
    if request.get("kind") == "inspect" or function.get("kind") != "linear_probe/v1":
        return None
    rule = str((function.get("linear_probe") or {}).get("rule") or "")
    point = manifest.get("operating_point") or {}
    target = point.get("target_fpr")
    sets = list((manifest.get("heldout_ref") or {}).get("sets") or [])
    primary = [str(n) for n in request.get("primary_sets") or []]
    alarms = [str(s.get("name", "")) for s in sets if s.get("role") in FALSE_ALARM_ROLES]
    available = [str(b) for b in manifest.get("baselines_available") or []]
    refits = [b for b in REFITS if b in available]
    beat = ["the parent's", "the control's (your changes permuted at random)"]
    if refits:
        # Each refit (operator, 2026-11-26): ``wsp.verdict.decide`` compares every one.
        beat.append(f"each practice-refit baseline's ({', '.join(refits)})")
    lines = [
        HEADING,
        "Held-out rows you never see score your weights; the primary metric is AUROC. A "
        f"difference counts only when its {CI:.0%} paired-bootstrap interval excludes zero.",
        f"- 'improved' needs AUROC above {', '.join(beat[:-1])} and {beat[-1]}, on every primary "
        f"set: {_names(primary)}.",
    ]
    if isinstance(target, int | float) and not isinstance(target, bool):
        where = (
            "the held-out calibration set"
            if point.get("threshold_source") == "calibration"
            else "the practice validation negatives"
        )
        checked = "recall on the primary sets" + (
            f", and false alarms on {_names(alarms)}," if alarms else ""
        )
        recut = (
            " nor than baseline:threshold_recut's (else not 'improved')"
            if "threshold_recut" in available
            else ""
        )
        lines.append(
            f"- Every head's bar is re-cut on {where} to the target false-alarm rate "
            f"{float(target):.4g}; none is held at the parent's bar. At its own bar, {checked} "
            f"must be no worse than the parent's (else 'regressed'){recut}."
        )
    if "affine_recalibration" in available:
        lines.append(
            "- 'improved' is also blocked if baseline:affine_recalibration beats you on recall "
            "and on false alarms at once."
        )
    if rule == "softmax":
        lines.append(
            "- So a uniform bias shift gets the parent's verdict exactly. Only how the head ranks "
            "rows (the weight vector's direction and size) can change it."
        )
    else:
        also = " or the attention query" if rule == "attention" else ""
        lines.append(
            "- So a uniform bias shift, or a positive rescale of weight and bias together, gets "
            "the parent's verdict exactly. Only how the head ranks rows (the weight vector's "
            f"direction{also}) can change it."
        )
    lines.append(
        "- The tools' 'auroc' and 'at_own_bar' (each head at its own bar, re-cut on the practice "
        "validation negatives) measure the same way on the practice rows."
    )
    if shows_uncertainty(request):
        lines.append(CHOOSE_ON_VALIDATION)
    return "\n".join(lines)


def context_with_judging(
    context: str, request: Mapping[str, Any], manifest: Mapping[str, Any], *, on: bool
) -> str:
    """``context`` (the function and its tensors) with the judging section after it, when on."""
    text = section(request, manifest) if on else None
    return context if text is None else f"{context}\n\n{text}"
