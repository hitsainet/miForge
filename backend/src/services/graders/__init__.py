"""Graders  (FR-003.9; R-02.9, consumed by R-02.7 in F2 and R-02.16/17 in F4).

**A grader is a form, not a sixth type.** R-02.9 puts graders behind the same fixed
interface as selectors, and :mod:`schemas.selector` gives them their own Protocol -
:class:`schemas.selector.Grader` - because the two answer different questions. So this
package holds no plugins of its own: it adapts a resolved selector to the grader Protocol,
and the type that answers is the same object under the same resolution path, out of the
same registry.

Why that matters rather than being tidy: **a grader and a selector over one configuration
can never disagree about what a completion scored.** The judge's threshold comparison and
the judge's scalar read the same reply; reference-match's hit/miss and its similarity are
the same number. Two code paths would be two answers, and the symptom would be a run whose
reward and whose admission disagreed with no way to tell which was right.

## The range is 0..1, and it is the contract's

FPRD OQ-5 asked whether a grader's scalar has a declared range and who declares it. The
answer is settled by the contract at both ends: ``Verdict.score`` is ``ge=0.0, le=1.0`` and
:meth:`schemas.selector.Grader.grade` is documented as *"a score in 0..1"*. A value outside
it is **refused with the value and the range, never clamped** - a clamp turns a grader that
misread its scale into a confident number.
"""

from __future__ import annotations

from typing import Any

from core.errors import refuse
from services.selectors.resolve import ResolvedSelector
from services.selectors.types import SelectorForm

#: The range every grader scalar lies in. Fixed by the contract, not by configuration.
SCORE_RANGE = (0.0, 1.0)


class GraderAdapter:
    """A resolved selector seen through :class:`schemas.selector.Grader`.

    Thin on purpose. It exists so that *"the same registry under the same resolution
    path"* (FR-003.9) is a fact about the code rather than a sentence in a document, and so
    that the range check lives in exactly one place.
    """

    def __init__(self, resolved: ResolvedSelector) -> None:
        self._resolved = resolved

    @property
    def name(self) -> str:
        return self._resolved.name

    def grade(self, *, prompt: str, completion: str, context: dict[str, Any]) -> float:
        """The scalar, range-checked. **Refuses out of range; never clamps.**"""
        score = float(
            self._resolved.plugin.grade(prompt=prompt, completion=completion, context=context)
        )
        low, high = SCORE_RANGE
        if not (low <= score <= high):
            raise refuse(
                "grader_score_out_of_range",
                f"Grader {self.name!r} returned {score:g}, outside the required "
                f"{low:g}..{high:g} range. Refusing rather than clamping: a clamp turns a "
                f"grader that misread its scale into a confident number, and every reward "
                f"computed from it would be silently wrong.",
                value=score,
                range=[low, high],
                grader=self.name,
            )
        return score


def grader_for(resolved: ResolvedSelector) -> GraderAdapter:
    """The grader form of a resolved configuration. **Refuses a type that has none.**

    Raises:
        Refusal: ``grader_form_unavailable``, naming the type and the forms it does offer.
            Refused rather than synthesised from the verdict, because a 1.0/0.0 derived
            from an admission decision is not a grade - and a GRPO group of those has zero
            variance by construction, which is the degenerate case this feature raises a
            named signal for.
    """
    if SelectorForm.GRADER not in resolved.capability.forms:
        raise refuse(
            "grader_form_unavailable",
            f"Selector type {resolved.kind!r} offers no grader form, so it cannot produce "
            f"a scalar. It offers: "
            f"{sorted(form.value for form in resolved.capability.forms)}. Use a type with "
            f"a grader form, or a selector where an admission decision is what you want.",
            kind=resolved.kind,
            forms=sorted(form.value for form in resolved.capability.forms),
        )
    if not hasattr(resolved.plugin, "grade"):  # pragma: no cover - registration guarantees it
        raise refuse(
            "grader_form_unavailable",
            f"Selector type {resolved.kind!r} declares a grader form but its plugin has no "
            f"grade() method. This is a registration defect, not a configuration error.",
            kind=resolved.kind,
        )
    return GraderAdapter(resolved)


__all__ = ["SCORE_RANGE", "GraderAdapter", "grader_for"]
