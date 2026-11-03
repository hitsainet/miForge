"""The stop registry: exactly five, explicitly registered, every one evaluated.

**M-1's home.** Removing ``retention_floor`` from ``_REGISTRATIONS`` - or making its
``evaluate`` always answer not-fired - must make this file and
``integration/test_retention_floor_stops_the_run.py`` red. If neither goes red, F4's entire
feature is a log line: it detects and names the breach and nothing stops the run.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.loop_fakes import AlwaysFires, NeverFires
from training import stop
from training.stop.base import STOP_REASONS, StopContext, StopReason


class TestTheRegistryIsExactlyRwithinTheRequirement:
    def test_it_holds_exactly_R0225s_five_names(self) -> None:
        """R-02.25's list is **closed**. Asserted against the enum, which is derived from
        the requirement, rather than against a literal retyped here - two spellings of one
        vocabulary is this project's signature defect."""
        assert set(stop.names()) == set(STOP_REASONS)
        assert len(stop.names()) == 5

    def test_the_five_are_the_names_the_requirement_uses(self) -> None:
        """A literal here on purpose, once: it is the one place the code is checked against
        the words of R-02.25 rather than against another part of the code."""
        assert set(stop.names()) == {
            "step_count",
            "test_threshold",
            "retention_floor",
            "loss_plateau",
            "operator_stop",
        }

    def test_every_registered_name_resolves_to_a_condition_of_that_name(self) -> None:
        """A condition registered under a name it does not declare would be resolvable and
        would record the wrong condition on every boundary record it appears in."""
        for name in stop.names():
            assert stop.resolve(name).name == name

    def test_an_unknown_name_is_refused_LISTING_the_five(self) -> None:
        """E-2. Listed rather than merely named, because the set is closed and the useful
        next step is to pick one of the five, not to go looking for a sixth."""
        with pytest.raises(Refusal) as caught:
            stop.resolve("gpu_too_hot")
        assert caught.value.code == "stop_condition_unknown"
        assert caught.value.details["known"] == list(stop.names())
        for name in STOP_REASONS:
            assert name in caught.value.message

    def test_there_is_no_scan_based_lookup_path(self) -> None:
        """No ``importlib``, no ``__subclasses__``, no directory walk.

        A scan makes *"is it registered?"* untestable and a missing condition **silent** -
        and here the cost is worse than a missing feature: a stop condition that is
        configured and never evaluated is indistinguishable from a run that is going well.
        """
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "stop"
        banned = {"importlib", "pkgutil", "__subclasses__", "iter_modules", "rglob", "glob"}
        for path in sorted(src.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    assert node.attr not in banned, f"{path.name} resolves by scan"
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names]
                    if isinstance(node, ast.ImportFrom) and node.module:
                        names.append(node.module)
                    for name in names:
                        assert name.split(".")[0] not in banned, f"{path.name} scans"


class TestEvaluateAllDoesNotShortCircuit:
    """FR-006.16: **every** enabled condition at **every** boundary, all verdicts kept."""

    def test_a_non_firing_condition_still_appears_in_the_verdicts(self) -> None:
        """M-1's unit half. Without this a reader of a boundary record cannot tell "did not
        fire" from "was not evaluated", and the second is the defect this feature exists to
        prevent."""
        ctx = StopContext(run_id="r", step=5, round=0)
        verdicts = stop.evaluate_all({"step_count": {"total": 100}, "retention_floor": {}}, ctx)
        assert [v.condition for v in verdicts] == ["step_count", "retention_floor"]
        assert all(v.triggered is False for v in verdicts)

    def test_a_fired_condition_does_not_stop_the_others_being_evaluated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No short-circuit. It is also what makes precedence meaningful: you cannot pick
        the most significant of two reasons if you stopped looking at the first."""
        monkeypatch.setitem(stop._CONDITIONS, "step_count", AlwaysFires("step_count"))
        monkeypatch.setitem(stop._CONDITIONS, "loss_plateau", AlwaysFires("loss_plateau"))
        ctx = StopContext(run_id="r", step=5, round=0)
        verdicts = stop.evaluate_all({"step_count": {}, "loss_plateau": {}}, ctx)
        assert len(verdicts) == 2
        assert all(v.triggered for v in verdicts)

    def test_the_verdict_order_is_the_registry_order_not_the_dict_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So the boundary record's array is stable across runs and machines. A dict's
        insertion order is the request's, which would make two identical runs produce two
        different records."""
        monkeypatch.setitem(stop._CONDITIONS, "operator_stop", NeverFires("operator_stop"))
        ctx = StopContext(run_id="r", step=1, round=0)
        verdicts = stop.evaluate_all({"operator_stop": {}, "step_count": {"total": 9}}, ctx)
        assert [v.condition for v in verdicts] == ["step_count", "operator_stop"]

    def test_a_condition_that_is_not_enabled_is_not_evaluated(self) -> None:
        """The other half: a condition the operator did not enable must not fire."""
        ctx = StopContext(run_id="r", step=1_000_000, round=0)
        assert stop.evaluate_all({}, ctx) == []


class TestTheVerdictIsTheContractsShape:
    def test_as_evaluation_has_exactly_the_contracts_four_keys(self) -> None:
        """``StopConditionEvaluation`` is closed to unknown keys, so a fifth would make the
        whole boundary block fail validation at the writer. Verified against the vendored
        schema, not against a literal."""
        import json

        from training import run_log

        schema = json.loads(run_log.schema_path().read_text(encoding="utf-8"))
        item = schema["$defs"]["BoundaryBlock"]["properties"]["stop_conditions"]["items"]
        assert item["additionalProperties"] is False

        ctx = StopContext(run_id="r", step=3, round=0)
        verdict = stop.evaluate_all({"step_count": {"total": 10}}, ctx)[0]
        assert set(verdict.as_evaluation()) <= set(item["properties"])
        assert set(item["required"]) <= set(verdict.as_evaluation())

    def test_the_structured_values_are_NOT_in_the_contracts_object(self) -> None:
        """``values`` is for the API's stop-conditions view. Putting it in the log would
        make every boundary block invalid - stated in the dataclass and asserted here so
        nobody tries."""
        ctx = StopContext(run_id="r", step=3, round=0)
        verdict = stop.evaluate_all({"step_count": {"total": 10}}, ctx)[0]
        assert verdict.values
        assert "values" not in verdict.as_evaluation()


class TestUnconfiguredConditionsTakeNoConfiguration:
    def test_the_floor_and_the_operator_stop_take_none(self) -> None:
        """A ``floor`` parameter on ``retention_floor`` would be a **third** floor layer,
        and the two that exist already needed a dedicated field to tell them apart (F4
        FR-004.23)."""
        assert stop.UNCONFIGURED == {"retention_floor", "operator_stop"}

    def test_the_floor_condition_ignores_any_configuration_handed_to_it(self) -> None:
        """Belt and braces: even if a definition carried a floor here, this condition must
        not read it. F4 compares; F6 stops."""
        ctx = StopContext(run_id="r", step=1, round=0)
        verdict = stop.resolve("retention_floor").evaluate(ctx, {"floor": 0.99})
        assert verdict.triggered is False
        assert "floor" not in verdict.values or verdict.values.get("breached") is False


class TestTheEnumIsTheVocabulary:
    def test_stop_reason_members_are_their_own_wire_values(self) -> None:
        """A ``StrEnum`` so the recorded string and the member are one thing. An enum whose
        value differed from its name would need a translation table, which is a second
        source of truth for the vocabulary a run log is written in."""
        for reason in StopReason:
            assert reason.value == str(reason)
