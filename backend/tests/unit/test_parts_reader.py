"""A held-out split cannot become training material  (AC-010.7; FR-010.18).

miDataworks guarantees that a held-out split is written as its own file and *"no code path
merges them into another split"*. **miForge must not be that code path**, and the rule lives
in one function because F2 and F4 both call it: a second copy in F2's loader would drift, and
the drift would be silent.

**The refusal rather than an empty list is the point.** An empty list becomes a zero-row
batch, and a zero-row batch is a step that trains on nothing while every log line looks
healthy. That is this feature's design criterion applied: *where there was a choice between a
design that could be used correctly and one that cannot be used incorrectly, the second was
taken.*
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from services.parts.contract_validator import Split
from services.parts.reader import evaluation_splits, splits_for, training_splits


def split(name: str, *, held_out: bool = False, evaluation_only: bool = False) -> Split:
    return Split(
        name=name,
        path=f"data/{name}.parquet",
        rows=10,
        bytes=100,
        sha256="a" * 64,
        logical_digest="b" * 64,
        held_out=held_out,
        evaluation_only=evaluation_only,
        label_counts={},
    )


TRAIN = split("train")
TEST = split("test", held_out=True)
EVAL = split("eval", evaluation_only=True)
BOTH = split("both", held_out=True, evaluation_only=True)


class TestTraining:
    def test_a_held_out_split_is_excluded(self):
        assert training_splits([TRAIN, TEST]) == [TRAIN]

    def test_an_evaluation_only_split_is_excluded(self):
        assert training_splits([TRAIN, EVAL]) == [TRAIN]

    def test_a_split_that_is_both_is_excluded(self):
        assert training_splits([TRAIN, BOTH]) == [TRAIN]

    @pytest.mark.parametrize("excluded", [TEST, EVAL, BOTH])
    def test_asking_for_an_excluded_split_by_name_refuses(self, excluded: Split):
        """**Refused, not filtered.** A caller that named a split and got an empty list
        would turn it into a zero-row batch, which is silent."""
        with pytest.raises(Refusal) as caught:
            splits_for([TRAIN, excluded], "train", named=[excluded.name])
        assert caught.value.code == "split_not_available_for_training"
        assert caught.value.details["split"] == excluded.name
        assert caught.value.details["held_out"] == excluded.held_out
        assert caught.value.details["evaluation_only"] == excluded.evaluation_only

    def test_the_refusal_says_what_to_use_it_for_instead(self):
        with pytest.raises(Refusal) as caught:
            splits_for([TRAIN, TEST], "train", named=["test"])
        assert "test or retention set" in caught.value.message

    def test_a_part_with_no_trainable_split_refuses(self):
        """A test set offered as a corpus. Refused rather than returning nothing."""
        with pytest.raises(Refusal) as caught:
            training_splits([TEST, EVAL])
        assert caught.value.code == "no_training_splits"
        assert sorted(caught.value.details["splits"]) == ["eval", "test"]

    def test_an_unknown_split_name_refuses(self):
        """A typo'd name that quietly yielded nothing is the same silent zero-row batch by
        another route."""
        with pytest.raises(Refusal) as caught:
            splits_for([TRAIN], "train", named=["trian"])
        assert caught.value.code == "split_unknown"
        assert caught.value.details["available"] == ["train"]

    def test_no_splits_at_all_is_an_empty_answer_not_a_refusal(self):
        """A config part has no splits by construction, and asking it for training material
        is a different question from asking a rows part that has only held-out ones."""
        assert training_splits([]) == []


class TestEvaluation:
    def test_held_out_is_the_correct_answer_for_evaluation(self):
        """Which is why this is one function with a purpose rather than two functions, one
        of which someone would call from the wrong place."""
        assert evaluation_splits([TRAIN, TEST, EVAL]) == [TRAIN, TEST, EVAL]

    def test_naming_a_held_out_split_for_evaluation_is_fine(self):
        assert splits_for([TRAIN, TEST], "evaluate", named=["test"]) == [TEST]

    def test_an_unknown_name_still_refuses_for_evaluation(self):
        with pytest.raises(Refusal) as caught:
            splits_for([TRAIN], "evaluate", named=["nope"])
        assert caught.value.code == "split_unknown"


class TestTheRuleLivesInOnePlace:
    def test_no_other_module_decides_training_eligibility(self):
        """FR-010.18's structural half: the rule in two places is the rule in one place
        plus a bug waiting for the two to disagree.

        Asserted by searching for the *combination* of both flags outside this module - a
        module that reads one of them (to display it, say) is fine; a module that reads both
        together is making the decision.
        """
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        offenders: list[str] = []
        for path in src.rglob("*.py"):
            relative = str(path.relative_to(src))
            if relative in {"services/parts/reader.py", "services/parts/contract_validator.py"}:
                continue
            tree = ast.parse(path.read_text("utf-8"))
            names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            if {"held_out", "evaluation_only"} <= names:
                offenders.append(relative)
        # Two names expected, each for a stated reason, so a THIRD is a finding:
        #   * `tables.py` declares the two columns, which is where the flags are stored;
        #   * `index.py` derives `training_eligible` onto the API payload from the same rule,
        #     and is therefore the one place that could legitimately drift.
        # Listed rather than excluded silently, because a reader should know they exist.
        assert sorted(offenders) == [
            "services/parts/index.py",
            "services/parts/tables.py",
        ], offenders
