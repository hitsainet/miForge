"""A scalar orders two completions; a verdict does not  (FR-002.19, FR-002.22).

R-02.6 admits pairs *"from the selector ranking two generations."* **A selector as R-02.9
defines it returns a verdict - accept, reject or undecided - which does not order two
completions**: two accepts give no preference, and three unordered values order a pair no
better than two do. The ranking operation R-02.6 names does not exist in the interface
R-02.9 defines, which is the same shape as BRD-01's missing load call: a capability named
in prose with no mechanism (FPRD §15 finding 2, reported upstream and not fixed here).

So the ordering comes from a **scalar** - a grader's score, or the optional ``score`` a
verdict carries where its configuration produces one - and **a tie is never broken.**
*A pair ordered by a coin flip trains the model toward noise, and its loss curve looks
exactly like one that is working.*

## ``Pair | NoPair``, not ``Optional[Pair]``

An optional invites a caller to read ``None`` as *nothing happened*. :class:`NoPair`
carries a reason code that lands on the record as a **count**, so a batch that produced
three pairs out of eight says so rather than silently being a batch of three.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.errors import refuse
from training.signals.base import PreferenceExample

#: Why a pair was not produced. Codes rather than prose, because F6's stop conditions and
#: F9's step log key on them and a UI keyed on wording breaks when the wording changes.
TIE = "tie"
NO_SCALAR = "no_scalar"
IDENTICAL = "identical"
EMPTY = "empty"

NO_PAIR_REASONS: tuple[str, ...] = (TIE, NO_SCALAR, IDENTICAL, EMPTY)


@dataclass(frozen=True)
class NoPair:
    """No pair, with the reason. **A normal, recorded outcome - not an error.**"""

    prompt_id: str
    reason_code: str

    def __post_init__(self) -> None:
        if self.reason_code not in NO_PAIR_REASONS:
            raise ValueError(f"unknown NoPair reason {self.reason_code!r}")


def from_generations(
    prompt_id: str,
    prompt: str,
    first: str,
    second: str,
    *,
    first_scalar: float | None,
    second_scalar: float | None,
    ordered_by: str,
) -> PreferenceExample | NoPair:
    """Order two generated completions by a scalar, or return :class:`NoPair`.

    Args:
        first_scalar: The scalar for ``first`` - a grader's score, or a verdict's optional
            ``score``. ``None`` when the configuration produces none.
        ordered_by: ``grader_score`` or ``verdict_score``. Recorded, because a pair ordered
            by a grader and one ordered by a verdict's confidence are not the same evidence.

    Returns:
        A :class:`training.signals.base.PreferenceExample` with the margin that ordered it,
        or a :class:`NoPair` whose reason is counted on the batch record.
    """
    malformed = _malformed(first, second)
    if malformed is not None:
        return NoPair(prompt_id=prompt_id, reason_code=malformed)
    if first_scalar is None or second_scalar is None:
        # No scalar is not a tie and is counted separately: a tie means the selector
        # answered and could not separate them; this means nothing answered at all, which
        # is a configuration problem the operator can fix.
        return NoPair(prompt_id=prompt_id, reason_code=NO_SCALAR)
    if first_scalar == second_scalar:
        return NoPair(prompt_id=prompt_id, reason_code=TIE)
    if first_scalar > second_scalar:
        chosen, rejected, margin = first, second, first_scalar - second_scalar
    else:
        chosen, rejected, margin = second, first, second_scalar - first_scalar
    return PreferenceExample(
        prompt_id=prompt_id,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        ordered_by=ordered_by,
        margin=margin,
    )


def from_row(
    prompt_id: str, row: Mapping[str, object], *, chosen_column: str, rejected_column: str
) -> PreferenceExample | NoPair:
    """One pair from a ``preference_pairs`` part's row, by its **declared** columns.

    The columns were bound from the contract's declared ``chosen`` and ``rejected``
    semantics (:mod:`training.signals.semantics`), never from their names - the decisive
    case being a part whose columns are *called* ``chosen`` and ``rejected`` while their
    declared semantics are ``other``.
    """
    chosen = _text(row.get(chosen_column))
    rejected = _text(row.get(rejected_column))
    malformed = _malformed(chosen, rejected)
    if malformed is not None:
        return NoPair(prompt_id=prompt_id, reason_code=malformed)
    return PreferenceExample(
        prompt_id=prompt_id,
        prompt="",
        chosen=chosen,
        rejected=rejected,
        ordered_by="file",
        margin=None,
    )


def _malformed(chosen: str, rejected: str) -> str | None:
    """``identical``, ``empty`` or ``None``. Both are **refused and counted** (E-10).

    A byte-identical pair expresses no preference and a direct preference objective over it
    has no gradient to give; an empty side trains the model toward producing nothing. A
    silently dropped pair is a batch that is smaller than its record says.
    """
    if not chosen.strip() or not rejected.strip():
        return EMPTY
    if chosen == rejected:
        return IDENTICAL
    return None


def _text(value: object) -> str:
    """A cell as text, refusing a non-string rather than coercing one.

    ``str(value)`` on a list or a null would produce ``"None"`` or ``"['a']"`` as a training
    example - a fabricated value that trains perfectly well (ADR-018).
    """
    if value is None or not isinstance(value, str):
        raise refuse(
            "pair_column_not_text",
            f"A preference column holds {type(value).__name__}, not text. Refusing rather "
            f"than coercing it: str() of a null or a list is a training example nobody "
            f"wrote, and it would train as happily as a real one.",
            value_type=type(value).__name__,
        )
    return value


def split_counts(results: Sequence[PreferenceExample | NoPair]) -> tuple[int, int]:
    """``(dropped, ties)`` over a batch's results - the two counts the record carries.

    Separated because they mean different things: ``dropped`` is material that was
    malformed, ``ties`` is material the selector could not separate. Collapsing them would
    make a selector that never discriminates look like a dataset full of empty rows.
    """
    dropped = sum(
        1 for r in results if isinstance(r, NoPair) and r.reason_code in (IDENTICAL, EMPTY)
    )
    ties = sum(1 for r in results if isinstance(r, NoPair) and r.reason_code in (TIE, NO_SCALAR))
    return dropped, ties
