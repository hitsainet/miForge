"""F4's block of the run log, against the contract itself  (FR-004.28; foundation 3.3a).

``step-record-v1.json``'s ``EvaluationBlock`` exists for a reason its own description states:
*"THIS BLOCK EXISTS BECAUSE FOUR OF ITS FIELDS CHANGE WHAT A SCORE MEANS AND THE MANIFESTS DO
NOT CARRY THEM."* So every assertion here is made **against the vendored schema**, through
F1's real writer-side validator, rather than against a shape this feature believes in.

The subtle one is :func:`services.scoring.evaluation_block.kind_of_entry`: ``ScoreEntry``
carries **no kind**, so ``reindex`` has to derive test-vs-retention from ``floor``. That rule
is written once and asserted here, because two derivations would be two answers and the
symptom would be a rebuilt index that quietly disagrees with the live one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from core.errors import Refusal
from providers.base import GenerationParams
from services.scoring import evaluation_block as blocks
from training import run_log

SCHEMA = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "step-record-v1.json"


def a_block(**overrides):
    payload = {
        "run_id": "run-1",
        "step": 7,
        "scores": [
            blocks.score_entry(
                set_ref="test_set/general@1",
                score=0.84,
                n=200,
                floor=None,
                provider="in_process:lfm2-1.2b",
            ),
            blocks.score_entry(
                set_ref="retention_set/general@1",
                score=0.70,
                n=180,
                floor=0.72,
                provider="in_process:lfm2-1.2b",
            ),
        ],
        "provider": "in_process:lfm2-1.2b",
        "aggregation": "mean",
        "decode_params": blocks.decode_params_payload(GenerationParams(max_tokens=32)),
        "subsample_seed": None,
        "split": ["evaluation_only", "held_out"],
        "retention_floor_breached": True,
        "ts": datetime(2026, 10, 14, tzinfo=UTC),
    }
    payload.update(overrides)
    return blocks.evaluation_block(**payload)


class TestTheBlockConformsToTheVendoredContract:
    def test_it_validates_through_f1s_real_validator(self) -> None:
        """The same validator the writer uses, so a conforming test cannot mean a
        non-conforming write."""
        run_log.validate_block(a_block())

    def test_f4_is_the_declared_writer_of_this_block(self) -> None:
        assert run_log.BLOCK_WRITERS["evaluation"] == "F4"
        assert run_log.BLOCK_DEFINITIONS["evaluation"] == "EvaluationBlock"

    def test_every_required_field_is_present(self) -> None:
        required = set(
            json.loads(SCHEMA.read_text("utf-8"))["$defs"]["EvaluationBlock"]["required"]
        )
        assert required <= set(a_block())

    def test_the_block_carries_no_key_the_schema_does_not_declare(self) -> None:
        """``additionalProperties: false``, so this is also what the validator enforces -
        asserted directly so a failure says *which* key rather than *is not valid*."""
        declared = set(
            json.loads(SCHEMA.read_text("utf-8"))["$defs"]["EvaluationBlock"]["properties"]
        )
        assert set(a_block()) <= declared

    def test_an_empty_scores_array_refuses_with_a_readable_message(self) -> None:
        """``minItems: 1`` would catch it at the writer; refusing here names the real fault.

        *An empty block reads as "this pass ran and found nothing", which is a different and
        false claim* from "this pass failed".
        """
        with pytest.raises(Refusal) as caught:
            a_block(scores=[])
        assert caught.value.code == "evaluation_block_invalid"

    def test_a_non_finite_score_is_refused_at_assembly(self, tmp_path: Path) -> None:
        """A NaN validates against JSON Schema's ``number`` and then breaks the round trip.

        **And the shared writer does not catch this one** -
        ``training.run_log._assert_finite_numbers`` walks the record's *top-level* values, so
        a NaN nested inside ``scores[i].score`` passes it and surfaces later as a bare
        ``ValueError`` from the canonical serialiser. Reported as a finding on that writer;
        guarded here meanwhile, which is why the assertion is on the assembler rather than on
        ``append``.
        """
        with pytest.raises(Refusal) as caught:
            a_block(
                scores=[
                    blocks.score_entry(
                        set_ref="s", score=float("nan"), n=1, floor=None, provider="p:m"
                    )
                ]
            )
        assert caught.value.code == "evaluation_block_invalid"
        assert caught.value.details["field"] == "score"

    def test_the_shared_writers_finite_check_is_top_level_only(self, tmp_path: Path) -> None:
        """Pins the finding so it is not rediscovered, and so it goes red when it is fixed.

        If ``run_log.append`` ever starts refusing a nested NaN, this test fails and the
        guard above becomes redundant - which is the right moment to learn that.
        """
        hand_built = {
            "block": "evaluation",
            "run_id": "run-1",
            "step": 1,
            "ts": "2026-10-14T00:00:00+00:00",
            "scores": [
                {"set": "s", "score": float("nan"), "n": 1, "floor": None, "provider": "p:m"}
            ],
            "provider": "p:m",
            "aggregation": "mean",
            "decode_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8},
            "subsample_seed": None,
            "split": [],
            "retention_floor_breached": False,
        }
        run_log.validate_block(hand_built)  # passes: the nested NaN is invisible to it


class TestOnePassIsOneBlock:
    def test_test_and_retention_share_one_block(self) -> None:
        """FR-004.24's pairing, and the contract agrees: ``scores`` is an array with no
        ``kind`` on its entries, so the two cannot be taken with different parameters."""
        block = a_block()
        assert len(block["scores"]) == 2
        assert block["provider"] == "in_process:lfm2-1.2b"
        assert block["aggregation"] == "mean"

    def test_the_decode_parameters_come_from_the_gate_interface_type(self) -> None:
        """Built from ``GenerationParams`` rather than from loose keywords, so the values
        recorded are **the values that were sent**."""
        params = GenerationParams(max_tokens=64, temperature=0.0, top_p=0.9, stop=("END",))
        payload = blocks.decode_params_payload(params)
        assert payload == {
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 64,
            "stop": ["END"],
        }

    def test_the_subsample_seed_is_not_a_decode_parameter(self) -> None:
        """The schema puts it at the block level, and it is *which n prompts* - not how the
        tokens were chosen."""
        payload = blocks.decode_params_payload(GenerationParams(seed=7))
        assert "seed" not in payload

    def test_a_null_seed_is_written_present(self) -> None:
        """*Null is an assertion that no seed was set, which makes the number
        irreproducible - worth recording as a fact rather than hiding.*"""
        block = a_block(subsample_seed=None)
        assert "subsample_seed" in block and block["subsample_seed"] is None


class TestTheKindIsDerivedFromTheFloorExactlyOnce:
    def test_an_entry_with_a_floor_is_retention(self) -> None:
        assert blocks.kind_of_entry({"floor": 0.72}) == "retention"

    def test_an_entry_without_one_is_test(self) -> None:
        assert blocks.kind_of_entry({"floor": None}) == "test"

    def test_the_rule_holds_because_a_retention_set_always_has_a_floor(self) -> None:
        """The premise, asserted so the derivation cannot quietly stop being sound.

        FR-004.27 refuses a retention set with no floor and the table carries a check
        constraint saying the same; if either were relaxed, ``reindex`` would start
        classifying retention rows as test rows with nothing going red.
        """
        from services.scoring.tables import SCORES

        names = {
            constraint.name
            for constraint in SCORES.constraints
            if getattr(constraint, "name", None)
        }
        assert "ck_mf_scores_retention_has_a_floor" in names


class TestTheProjectionToIndexRows:
    def test_one_block_becomes_one_row_per_entry(self) -> None:
        rows = blocks.score_rows_from_block(a_block())
        assert [row["kind"] for row in rows] == ["test", "retention"]

    def test_the_breach_attaches_to_the_retention_row_only(self) -> None:
        """A test row has no floor to breach, and the check constraint refuses the pair."""
        rows = blocks.score_rows_from_block(a_block())
        assert rows[0]["breached"] is False
        assert rows[1]["breached"] is True

    def test_the_split_markers_are_carried_through(self) -> None:
        rows = blocks.score_rows_from_block(a_block())
        assert rows[0]["split_markers"] == ["evaluation_only", "held_out"]

    def test_the_projection_is_the_one_both_writers_use(self) -> None:
        """The live writer and the rebuild call the same function.

        Two projections would agree until one was changed, and the reindex-equivalence test
        compares API responses - so the disagreement would surface as a rebuild that quietly
        returns different numbers.
        """
        from services.scoring import reindex_scores

        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "services"
            / "scoring"
            / "reindex_scores.py"
        ).read_text("utf-8")
        assert "blocks.score_rows_from_block(block)" in source
        assert reindex_scores.blocks is blocks
