"""Three stop reasons can never advance a chain  (FR-006.43, DEF-7; **M-8**).

**R-02.27 names no filter at all.** Verified against the requirement: *"the stop condition
of one run triggers the start of another"* reads as though **any** stop advances it. So this
is a derived safety rule, reported as DEF-7, and kept in three layers - the ``Literal``
type, an insert-time check and the runtime filter - because each catches a different way
the rule could be lost.

Advancing past a floor breach would start the next stage **from the checkpoint whose
general capability had just collapsed**, and push it further in the same direction. The
PPRD names that as a project risk and names the floor as its only mitigation; a chain that
advanced through it would undo the mitigation one stage at a time, **and the suite would be
green for every stage.**

*The table-backed orchestration - ``mf_chains``, ``mf_chain_stages``, the sweeper, the four
routes - is owed on an Alembic revision F6 does not own, and the DDL is in the feature
report. What is tested here is the half that is safety-critical and needs no table.*
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.errors import Refusal
from schemas.chain import (
    NEVER_ADVANCES,
    ChainSpec,
    StageSpec,
    advance_filter,
    assert_advanceable,
    halt_reason,
)
from training.stop.base import STOP_REASONS

DEFINITION = {"run_type": "change"}


class TestLayerOneTheTypeExcludesThem:
    @pytest.mark.parametrize("reason", ["retention_floor", "operator_stop", "failed"])
    def test_a_stage_cannot_DECLARE_a_never_advancing_reason(self, reason: str) -> None:
        """The first of FR-006.43's three layers, and the one a later edit cannot forget:
        the configuration is refused by **the type** before any code runs."""
        with pytest.raises(ValidationError):
            StageSpec(definition=DEFINITION, advance_on=(reason,))

    @pytest.mark.parametrize("reason", ["step_count", "test_threshold", "loss_plateau"])
    def test_the_three_that_MAY_advance_are_accepted(self, reason: str) -> None:
        """The positive half. Without it the test above would pass against a type that
        admitted nothing, which is a chain that can never advance."""
        assert StageSpec(definition=DEFINITION, advance_on=(reason,)).advance_on == (reason,)

    def test_a_stage_that_advances_on_NOTHING_is_refused(self) -> None:
        """A chain that silently stops at a stage reads as a chain that completed."""
        with pytest.raises(ValidationError):
            StageSpec(definition=DEFINITION, advance_on=())

    def test_a_chain_of_one_stage_is_refused(self) -> None:
        """A "chain" of one is a run, and accepting one would make the chain machinery the
        path a single run took - the opposite of FR-006.46's deletability property."""
        with pytest.raises(ValidationError):
            ChainSpec(
                name="c", stages=(StageSpec(definition=DEFINITION, advance_on=("step_count",)),)
            )

    def test_the_type_and_the_never_set_agree_about_R0225s_five(self) -> None:
        """Asserted as a test as well as at import: a sixth stop reason added to R-02.25
        and to neither would be silently excluded from every chain configuration, which is
        the safe direction **silently**."""
        assert NEVER_ADVANCES & set(STOP_REASONS) == {"retention_floor", "operator_stop"}
        assert "failed" in NEVER_ADVANCES
        assert "failed" not in STOP_REASONS, (
            "a failed stage has an error code, not one of R-02.25's five - it is a terminal "
            "cause, not a stop condition"
        )


class TestLayerTwoTheInsertTimeCheck:
    @pytest.mark.parametrize("reason", ["retention_floor", "operator_stop", "failed"])
    def test_plain_strings_are_refused_too(self, reason: str) -> None:
        """Catches a configuration that arrived as plain strings - from JSON, from a
        migration, or from a later edit that widened the type."""
        with pytest.raises(Refusal) as caught:
            assert_advanceable(["step_count", reason])
        assert caught.value.code == "chain_stage_invalid"
        assert reason in caught.value.details["offenders"]

    def test_the_refusal_says_WHY_and_what_to_do_instead(self) -> None:
        """ADR-021: *"a floor breach would start the next stage from the checkpoint whose
        general capability had just collapsed"* is the half an operator can act on."""
        with pytest.raises(Refusal) as caught:
            assert_advanceable(["retention_floor"])
        assert "general capability" in caught.value.message
        assert caught.value.details["advanceable"] == [
            "step_count",
            "test_threshold",
            "loss_plateau",
        ]

    def test_an_advanceable_set_passes_silently(self) -> None:
        assert_advanceable(["step_count", "test_threshold", "loss_plateau"])


class TestLayerThreeTheRuntimeFilter:
    """**M-8's target.** The check that holds in *every* configuration."""

    @pytest.mark.parametrize("reason", ["retention_floor", "operator_stop", "failed"])
    def test_a_never_advancing_reason_does_not_advance_EVEN_IF_DECLARED(self, reason: str) -> None:
        """**The point of the third layer.** The type and the insert both refuse such a
        configuration, so this case can only arise from a row that predates either or from
        a widened type - and that is exactly the case the rule has to hold in.

        M-8 removes this check, and then a chain advances on a floor breach the moment any
        configuration declares one.
        """
        assert advance_filter(reason, ["step_count", reason]) is False, (
            f"{reason!r} advanced the chain because the configuration declared it. A floor "
            f"breach must never advance, in any configuration - the next stage would start "
            f"from the checkpoint whose general capability had just collapsed."
        )

    @pytest.mark.parametrize("reason", ["step_count", "test_threshold", "loss_plateau"])
    def test_a_declared_advanceable_reason_DOES_advance(self, reason: str) -> None:
        assert advance_filter(reason, [reason]) is True

    def test_an_UNDECLARED_reason_does_not_advance(self) -> None:
        """E-15: the chain halts, naming the reason it stopped on."""
        assert advance_filter("loss_plateau", ["step_count"]) is False

    def test_a_stage_that_ended_with_NO_reason_does_not_advance(self) -> None:
        """A stage with no reason cannot say it finished what it was for."""
        assert advance_filter(None, ["step_count", "test_threshold"]) is False

    @pytest.mark.parametrize(
        ("reason", "code"),
        [
            ("retention_floor", "chain_halted_on_floor_breach"),
            ("operator_stop", "chain_halted_on_operator_stop"),
            ("failed", "chain_halted_on_failed_stage"),
            (None, "chain_halted_on_failed_stage"),
            ("loss_plateau", "chain_not_advanceable"),
        ],
    )
    def test_the_halt_is_NAMED_rather_than_generic(self, reason, code) -> None:
        """*"The chain stopped"* and *"the chain stopped because general capability fell
        below the floor"* are different sentences, and only one of them is actionable."""
        assert halt_reason(reason) == code


class TestChainingIsDeletable:
    def test_nothing_outside_the_chain_modules_depends_on_a_chain_existing(self) -> None:
        """FR-006.46, AMB-5 / OQ-3. **BRD §9 asks whether chained runs should be a
        first-release feature or a script on top of single runs, and it is unanswered.**

        So the implementation is shaped so either answer is cheap: deleting the chain
        modules removes a feature without unpicking the loop. Asserted as the absence of
        any chain import from the loop, the boundary, the stop conditions or the validator.
        """
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "backend" / "src"
        core_of_f6 = [
            src / "training" / "loop.py",
            src / "training" / "boundary.py",
            src / "training" / "run_state.py",
            src / "training" / "run_driver.py",
            src / "services" / "run_definition.py",
            src / "services" / "run_launch.py",
            *sorted((src / "training" / "stop").rglob("*.py")),
            *sorted((src / "training" / "refresh").rglob("*.py")),
        ]
        offenders: list[str] = []
        for path in core_of_f6:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.endswith("chain") or ".chain" in name:
                        offenders.append(f"{path.name}:{node.lineno} -> {name}")
        assert offenders == [], (
            f"the loop must not depend on a chain existing, or deleting chaining means "
            f"unpicking the loop: {offenders}"
        )
