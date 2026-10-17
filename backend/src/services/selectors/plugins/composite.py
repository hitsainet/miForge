"""The composite selector  (FR-003.8; R-02.9).

Combines child selectors with ``all``, ``any`` or ``majority``.

## ``undecided`` PROPAGATES, and this is the one place the logic is non-obvious

The contract states the rule rather than leaving it to an implementer, and quotes the
reason: *"``all`` with one undecided child cannot be accept (the child may yet reject) and
must not be reject (nobody rejected)."* So:

* **all** - the first ``reject`` decides it (short-circuit). Otherwise, any ``undecided``
  child makes the composite ``undecided``, naming that child. Only an all-accept set
  accepts.
* **any** - the first ``accept`` decides it (short-circuit). Otherwise, any ``undecided``
  child makes the composite ``undecided``. Only an all-reject set rejects.
* **majority** - a strict majority of children. Once ``accept`` or ``reject`` reaches
  ``n // 2 + 1`` the outcome cannot change, so it short-circuits there. Below that, a
  pending child could still tip it, so the composite is ``undecided``. With every child
  decided and no majority either way the composite **rejects**: that is a decided outcome
  (nothing more will arrive, and nothing admitted it), and the reason says so with the
  counts.

## Short-circuit is a correctness feature and a kindness

Children are ordered **cheapest first** where the configuration does not pin an order, so
``rule`` runs before ``judge_model`` and both before ``human_review``. An ``all`` composite
of a rule and a human review can reject on a forbidden term **without ever posting the
candidate to a person** - fewer judge calls, fewer candidates in a queue, and nobody asked
to read something a rule already settled. The test asserts the queue call count is zero.

## The composite's hash is its own

``checkpoint-manifest-v1.json`` holds **one scalar per kind** for ``selector_hash`` -
verified against the vendored file - so a composite publishes its own hash and its children
are enumerated in the batch record. A list is not expressible there.

## There is no weighted sum, and no ``composite_threshold_missing``

The FPRD described *"AND, OR or a weighted sum"* with an explicit threshold. The settled
contract's ``combine`` enum is ``["all", "any", "majority"]`` with **no weights field and
no threshold field**, so a weighted sum is not expressible and ``majority`` takes its place.
Reported as a document/contract divergence rather than reintroduced here: adding a weight
vector would be F3 adding a field to a schema it does not own (FR-003.2).
"""

from __future__ import annotations

from typing import Any

from core.errors import refuse
from schemas.selector import SelectorKind, Verdict, VerdictValue, is_admitted
from services.selectors.resolve import ResolvedSelector

#: Evaluation cost, cheapest first. Used to order children where the configuration does not
#: pin an order. ``human_review`` is last by a wide margin: it costs a person's attention.
_COST: dict[str, int] = {
    "rule": 0,
    "reference_match": 1,
    "composite": 2,
    "judge_model": 3,
    "human_review": 4,
}


def order_children(children: tuple[ResolvedSelector, ...]) -> tuple[ResolvedSelector, ...]:
    """Cheapest first, **stably**: equal-cost children keep their configured order.

    Stability matters because the reason names the deciding child, and a reason that
    depends on sort instability is a reason that changes between runs over the same input.
    """
    return tuple(sorted(children, key=lambda child: _COST.get(child.kind, 2)))


class CompositeSelector:
    """Satisfies :class:`schemas.selector.Selector`. Delegates to its resolved children."""

    kind: SelectorKind = "composite"

    def __init__(self, resolved: ResolvedSelector) -> None:
        self._resolved = resolved

    @property
    def name(self) -> str:
        return self._resolved.name

    @property
    def combine(self) -> str:
        return str(self._resolved.body["combine"])

    def prepare(self, resolved: ResolvedSelector, *, run_target: Any = None) -> None:
        """Order the children and refuse a composite with nothing to combine.

        The children themselves were already resolved - and their own ``prepare`` already
        ran - by :func:`services.selectors.resolve.resolve`, which is also where a cycle is
        refused. This method only does what needs the whole child set.
        """
        del run_target
        if not resolved.children:
            raise refuse(
                "composite_children_missing",
                f"Composite {resolved.name!r} resolved to no children, so it has nothing "
                f"to combine and would answer the same way for every completion.",
                selector=resolved.name,
            )
        resolved.children = order_children(resolved.children)
        # A child's caveat is the composite's caveat: a judge sharing the model's lineage
        # three levels down still bears on every verdict this composite produces.
        inherited: list[str] = list(resolved.caveats)
        for child in resolved.children:
            inherited.extend(child.caveats)
        resolved.caveats = tuple(dict.fromkeys(inherited))
        resolved.prepared = {"combine": self.combine, "n": len(resolved.children)}

    def evaluate(self, *, prompt: str, completion: str, context: dict[str, Any]) -> Verdict:
        """Combine the children's verdicts, short-circuiting where the outcome is fixed."""
        children = self._resolved.children
        combine = self.combine
        total = len(children)
        accepted: list[str] = []
        rejected: list[str] = []
        pending: list[str] = []

        if combine == "majority":
            needed = total // 2 + 1
        else:
            needed = 0

        for child in children:
            verdict = child.plugin.evaluate(prompt=prompt, completion=completion, context=context)
            # `is_admitted` rather than a comparison: it is the one place admission is
            # decided (foundation 3.1a), so "no path from undecided to admitted" is
            # enforced once rather than trusted at each of the five plugins.
            if is_admitted(verdict):
                accepted.append(child.name)
                if combine == "any":
                    return self._decide(True, f"{child.name} accepted: {verdict.reason}")
                if combine == "majority" and len(accepted) >= needed:
                    return self._decide(
                        True, f"majority accepted ({len(accepted)} of {total}, needs {needed})"
                    )
            elif verdict.value is VerdictValue.REJECT:
                rejected.append(child.name)
                if combine == "all":
                    return self._decide(False, f"{child.name} rejected: {verdict.reason}")
                if combine == "majority" and len(rejected) >= needed:
                    return self._decide(
                        False, f"majority rejected ({len(rejected)} of {total}, needs {needed})"
                    )
            else:
                pending.append(child.name)

        # No short-circuit fired, so every child was visited. An undecided child could
        # still change the answer, so the composite is undecided - never collapsed to one
        # of the two decided values, which is precisely what the contract forbids.
        if pending:
            return self._pending(pending, accepted, rejected, total)

        if combine == "all":
            return self._decide(True, f"all {total} children accepted")
        if combine == "any":
            return self._decide(False, f"all {total} children rejected")
        # majority, every child decided, neither side reached the majority: a decided
        # outcome, not a pending one. Nothing further will arrive and nothing admitted it.
        return self._decide(
            False,
            f"no majority ({len(accepted)} of {total} accepted, needs {needed})",
        )

    def _decide(self, admitted: bool, reason: str) -> Verdict:
        return Verdict(
            value=VerdictValue.ACCEPT if admitted else VerdictValue.REJECT,
            selector=self.name,
            label=f"composite_{self.combine}_{'accept' if admitted else 'reject'}",
            reason=reason,
            score=1.0 if admitted else 0.0,
        )

    def _pending(
        self, pending: list[str], accepted: list[str], rejected: list[str], total: int
    ) -> Verdict:
        """``undecided``, **naming the pending child** - and carrying no score.

        A score here would be refused by the contract's own validator, and rightly: a score
        on a pending candidate is how a consumer reconstructs the route from undecided to
        admitted.
        """
        return Verdict(
            value=VerdictValue.UNDECIDED,
            selector=self.name,
            label=f"composite_{self.combine}_pending",
            reason=(
                f"waiting on {', '.join(pending)} "
                f"({len(accepted)} of {total} accepted, {len(rejected)} rejected)"
            ),
        )


def factory(resolved: ResolvedSelector) -> CompositeSelector:
    return CompositeSelector(resolved)
