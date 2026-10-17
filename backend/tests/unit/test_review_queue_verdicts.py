"""Silence is not consent  (AC-010.9; FR-010.24, R-02.36).

**The no-decision case is written first, deliberately.** It is the one case that is an
*absence* in the data and a *value* in the model, so it is the one a reader of this file
should meet before any of the others - and the one an implementer would have reached for a
default on.

Exhaustive over miDataworks' real decision vocabulary, which is read from its
``services/review/decision_rules.py`` rather than from this chain's summary of it. That is
how the ``skip``/``flag`` discrepancy below was found: by checking the gate contract's map
against the **source**, not against a document about the source.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from schemas.selector import DECISION_TO_VERDICT, VerdictValue, is_admitted
from services.review.verdicts import (
    MIDATAWORKS_DECISIONS,
    SELECTOR_NAME,
    admits,
    verdict_for,
    verdict_for_absent,
)


class TestNoDecisionYet:
    """The case that is an absence in the data and a value in the model."""

    def test_no_decision_is_undecided(self):
        verdict = verdict_for_absent()
        assert verdict.value is VerdictValue.UNDECIDED

    def test_it_is_not_admitted(self):
        assert admits(verdict_for_absent()) is False

    def test_it_is_not_a_rejection_either(self):
        """**Never counted as rejected.** A pending candidate counted as rejected would make
        an unattended queue look like a selector that rejects everything, and the acceptance
        rate on the batch record would be a lie about the model rather than about the human."""
        assert verdict_for_absent().value is not VerdictValue.REJECT

    def test_it_carries_no_score(self):
        """The gate contract refuses an ``undecided`` verdict with a score, because a score
        on a pending candidate is how a consumer reconstructs the forbidden path from
        undecided to admitted. Asserted here so this feature cannot be the consumer that
        tries."""
        assert verdict_for_absent().score is None

    def test_the_selector_is_named_on_the_verdict(self):
        """For the batch record's provenance. One string, so a batch cannot disagree with
        the registry about which selector produced a verdict."""
        assert verdict_for_absent().selector == SELECTOR_NAME == "human_review"


class TestTheMappedDecisions:
    def test_accept_admits(self):
        verdict, anomaly = verdict_for("accept")
        assert verdict.value is VerdictValue.ACCEPT
        assert is_admitted(verdict) is True
        assert anomaly is None

    def test_reject_rejects(self):
        verdict, _ = verdict_for("reject")
        assert verdict.value is VerdictValue.REJECT
        assert is_admitted(verdict) is False

    def test_override_is_undecided_and_carries_its_label(self):
        """**Settled in the gate contract, against this feature's earlier draft.**

        Only one layer of one feature had ever named a value for ``override`` and it said
        ``accept``; this feature's own FPRD, its FTDD and both of F3's statements named a
        label and no value at all. ``undecided`` is the answer because miDataworks'
        ``label_set`` is 2-64 **free strings** - for a set that is not accept/reject-shaped,
        reading "override" as an admission is precisely the silent default FR-010.24
        forbids.
        """
        verdict, anomaly = verdict_for(
            "override", label="needs-rework", label_set=["needs-rework", "fine"]
        )
        assert verdict.value is VerdictValue.UNDECIDED
        assert verdict.label == "needs-rework"
        assert is_admitted(verdict) is False
        assert anomaly is None

    def test_an_override_label_outside_the_label_set_is_an_anomaly_not_an_admission(self):
        """E-16/E-17's shape: recorded, never admitted, never silently dropped."""
        verdict, anomaly = verdict_for(
            "override", label="something-else", label_set=["a", "b"], external_id="x1"
        )
        assert verdict.value is VerdictValue.UNDECIDED
        assert anomaly is not None
        assert anomaly.code == "override_label_outside_label_set"
        assert anomaly.external_id == "x1"

    def test_there_is_no_route_from_undecided_to_admitted(self):
        """The invariant the third value exists to protect, asserted over every case."""
        for decision in ("override", "flag"):
            verdict, _ = verdict_for(decision, label="x", label_set=["x", "y"])
            assert is_admitted(verdict) is False
        assert is_admitted(verdict_for_absent()) is False

    def test_an_unrecognised_decision_refuses_rather_than_defaulting(self):
        """**No ``else: ACCEPT``, no ``.get(..., ACCEPT)``.** Guessing here changes what the
        model trains on, which is why the gate contract's function raises."""
        with pytest.raises(Refusal) as caught:
            verdict_for("approve-ish")
        assert caught.value.code == "review_decision_unrecognised"
        assert "approve-ish" in caught.value.details["decision"]


class TestF10DeclaresNoVocabularyOfItsOwn:
    def test_the_map_is_the_gate_contracts(self):
        """An earlier draft of this feature's FTID specified a **fourth** vocabulary - a
        local ``accept|reject|label|pending`` enum - which its own FPRD and FTDD forbade.
        This is the assertion that stops it coming back."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "src" / "services" / "review" / "verdicts.py"
        ).read_text("utf-8")
        tree = ast.parse(source)
        defined_enums = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id in {"StrEnum", "Enum"} for base in node.bases
            )
        ]
        assert defined_enums == [], f"{defined_enums} is a fourth verdict vocabulary"
        assert "DECISION_TO_VERDICT" in source
        assert "map_decision" in source

    def test_no_module_in_this_feature_redeclares_a_verdict_value(self):
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        for package in ("services/review", "services/parts", "clients"):
            for path in (src / package).rglob("*.py"):
                text = path.read_text("utf-8")
                assert "VerdictValue.PENDING" not in text
                assert "class Verdict" not in text, f"{path} declares its own Verdict"


class TestTheGateContractAgainstMiDataworksRealVocabulary:
    """**A verified finding, pinned so it cannot be forgotten.**

    ``DECISION_TO_VERDICT`` holds ``accept``, ``reject``, ``override`` and **``skip``**.
    miDataworks' shipped vocabulary is ``accept``, ``override``, **``flag``** and ``reject``:
    ``DECISIONS = ("accept", "override", "flag", "reject")`` in
    ``services/review/decision_rules.py``, ``AGENT_DECISIONS = {"accept", "flag"}``,
    ``NEEDS_REASON = {"override", "flag", "reject"}``, and its request body type is
    ``Literal["accept", "override", "flag", "reject"]``. **The string ``skip`` occurs nowhere
    in that repository's review code.**

    So the gate contract maps a kind that does not exist and does not map one that does -
    and ``flag``, the decision that most obviously means *undecided* (*"any flag leaves the
    row flagged until an operator decides again"*), currently arrives as a **refusal**.
    """

    def test_midataworks_vocabulary_is_what_this_module_says_it_is(self):
        assert MIDATAWORKS_DECISIONS == ("accept", "override", "flag", "reject")

    def test_the_invented_decision_is_gone(self):
        """``skip`` was never a miDataworks decision and is no longer mapped.

        This asserted the *opposite* until 2026-10-14 - that ``skip`` **was** mapped and
        **was not** theirs - as a faithful record of the discrepancy. That was the right
        shape while the fix sat in a file F10 did not own, but **a test that pins a defect
        goes stale the instant the defect is fixed**, and a stale pin is worse than none: it
        keeps the suite green only while the bug survives.
        """
        assert "skip" not in DECISION_TO_VERDICT
        assert "skip" not in MIDATAWORKS_DECISIONS

    def test_every_mapped_decision_is_one_miDataworks_actually_sends(self):
        """The general form, which would have caught ``skip`` on the day it was written.

        Compares against that repository's own ``DECISIONS`` tuple rather than against a
        comment about it - the check the gate contract skipped.
        """
        unknown = sorted(set(DECISION_TO_VERDICT) - set(MIDATAWORKS_DECISIONS))
        assert not unknown, (
            f"mapped but never sent: {unknown}. An invented key is a silent fiction - it "
            f"maps a decision that cannot arrive while a real one is refused."
        )

    def test_every_decision_miDataworks_sends_is_mapped(self):
        """The direction that actually bit.

        ``flag`` was missing, and ``AGENT_DECISIONS = {"accept", "flag"}`` makes it **the**
        non-accept outcome for an agent reviewer - so the commonest decision on that
        surface arrived as a refusal.
        """
        unmapped = sorted(set(MIDATAWORKS_DECISIONS) - set(DECISION_TO_VERDICT))
        assert not unmapped, (
            f"sent but unmapped: {unmapped}. Each arrives as review_decision_unrecognised "
            f"- loud and safe, and wrong."
        )

    # The strict xfail that guarded this is removed. It fired the moment the one-line fix
    # landed in the gate contract - which is exactly what a strict xfail is for, and a
    # better handover than a note in a report nobody re-reads.
    def test_flag_maps_to_undecided(self):
        verdict, _ = verdict_for("flag")
        assert verdict.value is VerdictValue.UNDECIDED

    def test_today_a_flag_refuses_loudly_rather_than_defaulting(self):
        """The behaviour that ships meanwhile, and why it is the safe direction.

        A named refusal an operator can see beats a verdict invented on miDataworks' behalf.
        This test asserts the *property* - it refuses rather than admitting - so it keeps
        passing after the gate is fixed only if the fix is wrong; a correct fix makes the
        xfail above XPASS and fails the suite, which is the attention this needs.
        """
        try:
            verdict, _ = verdict_for("flag")
        except Refusal as refusal:
            assert refusal.code == "review_decision_unrecognised"
        else:  # pragma: no cover - reached once the gate contract is fixed
            assert verdict.value is not VerdictValue.ACCEPT, (
                "a flag must never admit a candidate: miDataworks' own rule is that a flag "
                "leaves the row flagged until an operator decides again"
            )
