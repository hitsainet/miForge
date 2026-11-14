"""The evaluation service  (FR-003.11, FR-003.13; E-16).

Three properties the rest of the system joins against:

* **input order**, as a list - two identical completions are two candidates, and a mapping
  keyed on text would silently merge them and misattribute every verdict after the
  collision;
* **cooperative cancellation at a completion boundary** - evaluated and recorded, or not
  started; never half;
* **no Celery task and no queue inside the service** - the placement is the caller's (F6's
  AMB-1), and a task added here would hand a ``--pool=solo -c 1`` worker a cross-queue
  ``.get()``.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from services.selectors.service import evaluate
from services.selectors.settings import reset_selector_settings
from services.selectors.types import EvaluationContext
from tests.support.selector_stubs import resolved, rule_config


@pytest.fixture(autouse=True)
def _settings():
    reset_selector_settings()
    yield
    reset_selector_settings()


def rule(**checks):
    return resolved(rule_config(**(checks or {"forbidden_terms": ["bad"]})))


class TestInputOrder:
    def test_verdicts_come_back_in_input_order(self) -> None:
        result = evaluate(["good", "bad", "good", "bad"], rule())
        assert [record.verdict.value.value for record in result.records] == [
            "accept",
            "reject",
            "accept",
            "reject",
        ]

    def test_two_identical_completions_are_two_records(self) -> None:
        """A dict keyed on completion text would merge them."""
        result = evaluate(["same", "same"], rule())
        assert len(result.records) == 2

    def test_the_prompt_is_joined_positionally(self) -> None:
        result = evaluate(["a", "b"], rule(), EvaluationContext(), prompts=["Q1", "Q2"])
        assert len(result.records) == 2

    def test_a_prompt_count_mismatch_is_refused_rather_than_padded(self) -> None:
        """A positional join over mismatched lists misattributes every verdict after the
        gap."""
        with pytest.raises(Refusal) as exc:
            evaluate(["a", "b"], rule(), EvaluationContext(), prompts=["only one"])
        assert exc.value.code == "evaluation_prompt_count_mismatch"
        assert exc.value.details == {"prompts": 1, "completions": 2}

    def test_per_item_context_is_joined_positionally_too(self) -> None:
        selector = resolved(
            {
                "kind": "reference_match",
                "name": "ref",
                "config_sha256": None,
                "mode": "exact",
            }
        )
        result = evaluate(
            ["alpha", "beta"],
            selector,
            EvaluationContext(items=({"reference": "alpha"}, {"reference": "zzz"})),
        )
        assert [record.verdict.value.value for record in result.records] == ["accept", "reject"]


class TestCancellation:
    def test_a_cancelled_evaluation_refuses_by_name(self) -> None:
        """E-16, ADR-006/007. ``revoke(terminate=True)`` is never used."""
        with pytest.raises(Refusal) as exc:
            evaluate(["a", "b"], rule(), EvaluationContext(is_cancelled=lambda: True))
        assert exc.value.code == "evaluation_cancelled"
        assert exc.value.details == {"completed": 0, "total": 2}

    def test_cancellation_happens_at_a_completion_boundary(self) -> None:
        """Each completion is evaluated and recorded, or not started."""
        seen = {"n": 0}

        def cancel_after_two() -> bool:
            seen["n"] += 1
            return seen["n"] > 2

        with pytest.raises(Refusal) as exc:
            evaluate(["a", "b", "c", "d"], rule(), EvaluationContext(is_cancelled=cancel_after_two))
        assert exc.value.details["completed"] == 2

    def test_no_cancellation_check_means_no_cancellation(self) -> None:
        result = evaluate(["a"], rule(), EvaluationContext(is_cancelled=None))
        assert len(result.records) == 1


class TestTheBound:
    def test_an_oversized_batch_is_refused_with_the_cap_named(self, monkeypatch) -> None:
        monkeypatch.setenv("EVALUATE_MAX_COMPLETIONS", "3")
        reset_selector_settings()
        with pytest.raises(Refusal) as exc:
            evaluate(["a"] * 4, rule())
        assert exc.value.code == "evaluation_too_large"
        assert exc.value.details == {"count": 4, "limit": 3}


class TestTheRecordCarriesProvenance:
    def test_every_record_names_the_configuration_that_produced_it(self) -> None:
        selector = rule()
        result = evaluate(["good"], selector)
        record = result.records[0]
        assert record.config_sha256 == selector.config_sha256
        assert record.config_kind == "rule"

    def test_the_reason_code_mirrors_the_verdicts_label(self) -> None:
        """``label`` doubles as the machine reason code - the contract's own comment says
        so. Mirrored so a reader of this record need not know that."""
        result = evaluate(["bad"], rule())
        assert result.records[0].reason_code == result.records[0].verdict.label

    def test_a_record_with_no_judge_says_so_by_absence(self) -> None:
        """Absence is the distinction, not a missing value."""
        assert evaluate(["good"], rule()).records[0].judge is None

    def test_reference_match_records_its_mode_and_metric(self) -> None:
        selector = resolved(
            {
                "kind": "reference_match",
                "name": "ref",
                "config_sha256": None,
                "mode": "fuzzy",
                "threshold": 0.5,
            }
        )
        result = evaluate(["alpha"], selector, EvaluationContext(items=({"reference": "alpha"},)))
        assert result.records[0].match_mode == "fuzzy"
        assert result.records[0].match_metric == "difflib.SequenceMatcher.ratio"

    def test_exact_mode_records_no_metric_because_there_is_none(self) -> None:
        selector = resolved(
            {"kind": "reference_match", "name": "ref", "config_sha256": None, "mode": "exact"}
        )
        result = evaluate(["alpha"], selector, EvaluationContext(items=({"reference": "alpha"},)))
        assert result.records[0].match_mode == "exact"
        assert result.records[0].match_metric is None


class TestHealthAndHashesTravelWithTheResult:
    def test_the_health_record_is_part_of_the_result(self) -> None:
        result = evaluate(["good", "bad"], rule())
        assert result.health.n == 2
        assert result.health.accepted == 1

    def test_the_published_hashes_are_part_of_the_result(self) -> None:
        selector = rule()
        result = evaluate(["good"], selector)
        assert result.hashes.selector_hash == selector.config_sha256
        assert result.hashes.selector_hash_provenance == "local"

    def test_the_signal_sink_is_injected_and_its_payload_assertable(self) -> None:
        calls: list[tuple[str, dict]] = []
        evaluate(
            ["good"] * 6,
            rule(),
            EvaluationContext(emit_signal=lambda e, p: calls.append((e, p))),
        )
        assert [event for event, _ in calls] == ["selection_accepts_everything"]
        assert calls[0][1]["n"] == 6

    def test_the_verdicts_property_is_the_contract_type(self) -> None:
        from schemas.selector import Verdict

        result = evaluate(["good"], rule())
        assert all(isinstance(verdict, Verdict) for verdict in result.verdicts)


class TestItDeclaresNoQueue:
    def test_evaluate_is_a_plain_synchronous_callable(self) -> None:
        """F6's AMB-1: the loop calls this in-process and synchronously, and the placement
        is the caller's. A coroutine here would make the placement F3's decision."""
        import inspect

        assert not inspect.iscoroutinefunction(evaluate)

    def test_no_module_in_this_package_imports_celery_or_declares_a_task(self) -> None:
        """An AST scan, not a word search: a docstring explaining why there is no task
        must not fail the test that there is no task."""
        import ast
        from pathlib import Path

        package = Path(__file__).resolve().parents[2] / "src" / "services" / "selectors"
        offenders: list[str] = []
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(alias.name.split(".")[0] == "celery" for alias in node.names):
                        offenders.append(f"{path.name}:{node.lineno} import celery")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] in {"celery", "workers"}:
                        offenders.append(f"{path.name}:{node.lineno} from {node.module}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for decorator in node.decorator_list:
                        rendered = ast.unparse(decorator)
                        if "task" in rendered:
                            offenders.append(f"{path.name}:{node.lineno} @{rendered}")
        assert offenders == [], (
            f"F3 declares no Celery task and no queue, but found {offenders}. A task "
            f"inside this service would make the placement F3's decision rather than its "
            f"caller's, and would hand a --pool=solo -c 1 worker a cross-queue .get()."
        )
