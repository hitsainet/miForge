"""BRD criterion 16's ``(§5.10 only)`` clause: a merge inside the **same** bracket.

**AC-011.19, FR-011.61; MUT-20.** R-02.45 is **F8's requirement and is not implemented
here**: F8 computes the merge - the arithmetic, the ``merged_adapter`` manifest field, and
criterion 17's refusal when no checkpoint of the unmerged base exists. **F11 wraps a write it
does not compute.**

**Why MUT-20 exists at all.** *"A merge is not a step"* is the sentence that will tempt
someone to route around the bracket, and a merge into resident tensors is exactly the thing
the bracket protects: one more multi-tensor write to tensors miForge does not own. An
interrupted merge leaves a parameter set **no checkpoint describes** while serving still
answers from it - MUT-1's defect on a path a reader assumes is exempt. And a merge **cannot
be undone by a second merge**: once the base has moved, the adapter's contribution cannot be
subtracted back out, so the never-retry rule matters more here than anywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import barrier, recovery, resident_merge
from training.hotupdate.recovery import Interruption, Recovery


class Tensor:
    """A resident tensor with ``copy_``, which is what a write in place looks like.

    ``copy_`` rather than rebinding: **the resident tensor's address must not move** (BRD-01
    R-01.5 keeps the parameters at fixed addresses and every IPC handle depends on it), and
    rebinding the name would leave miLLM serving the old storage while miForge trained a new
    one - two models, both reporting success.
    """

    def __init__(self, value: int) -> None:
        self.value = value
        self.copies = 0

    def copy_(self, other: Tensor) -> None:
        self.value = other.value
        self.copies += 1


def merged(n: int = 2) -> dict[str, Any]:
    """What F8 hands over: **finished** tensors, with their resident targets."""
    return {
        f"model.layers.{index}.self_attn.q_proj.weight": {
            "target": Tensor(index),
            "source": Tensor(100 + index),
        }
        for index in range(n)
    }


class TestTheMergeRunsInsideTheBracket:
    def test_the_stage_order_is_the_steps_and_the_write_is_between(self, harness: Harness) -> None:
        """**MUT-20.** The same four calls in the same order - *no second code path and no
        exemption.*"""
        harness.attach()
        harness.fake.calls.clear()
        tensors = merged()
        states: list[str] = []

        def writer(name: str, entry: dict[str, Any]) -> None:
            states.append(harness.fake.weights_state)
            entry["target"].copy_(entry["source"])

        outcome = resident_merge.apply(
            harness.client, tensors, timings=harness.timings, writer=writer
        )
        assert harness.fake.ops() == ["status", *barrier.STAGE_ORDER]
        assert states == ["writing", "writing"], "every tensor was written inside the bracket"
        assert outcome["tensors_written"] == 2
        assert harness.fake.weights_state == "clean"

    def test_the_write_happens_in_place_through_copy_(self, harness: Harness) -> None:
        harness.attach()
        tensors = merged(1)
        target = tensors["model.layers.0.self_attn.q_proj.weight"]["target"]
        resident_merge.apply(harness.client, tensors, timings=harness.timings)
        assert target.copies == 1
        assert target.value == 100

    def test_an_entry_without_both_halves_is_refused_rather_than_rebound(
        self, harness: Harness
    ) -> None:
        harness.attach()
        with pytest.raises(Refusal) as caught:
            resident_merge.apply(
                harness.client,
                {"model.x.weight": {"source": Tensor(1)}},
                timings=harness.timings,
            )
        assert caught.value.code == "resident_merge_tensor_unusable"
        assert "rebinding the name" in caught.value.message

    def test_an_empty_merge_is_refused_rather_than_bracketed(self, harness: Harness) -> None:
        """*An empty merge reports success, changes nothing, and leaves an operator
        believing an adapter was merged.*"""
        harness.attach()
        harness.fake.calls.clear()
        with pytest.raises(Refusal) as caught:
            resident_merge.apply(harness.client, {}, timings=harness.timings)
        assert caught.value.code == "resident_merge_empty"
        assert harness.fake.call_count("quiesce") == 0

    def test_the_state_precondition_runs_before_the_bracket(self, harness: Harness) -> None:
        harness.attach()
        harness.fake.weights_state = "torn"
        harness.fake.calls.clear()
        with pytest.raises(Refusal) as caught:
            resident_merge.apply(harness.client, merged(), timings=harness.timings)
        assert caught.value.code == "weights_not_clean"
        assert harness.fake.call_count("quiesce") == 0

    def test_the_commit_duration_is_recorded_like_a_steps(self, harness: Harness) -> None:
        """A merge is held serving time like any other commit, and the watchdog reads the
        same distribution - so it belongs in the same timings, not a second series."""
        harness.attach()
        resident_merge.apply(harness.client, merged(), timings=harness.timings)
        assert harness.timings.commits_ms and harness.timings.holds_ms


class TestAnInterruptedMergeIsNotRetried:
    def test_it_leaves_the_set_torn_and_serving_refused(self, harness: Harness) -> None:
        """Identical to an interrupted step, **which is the point**."""
        harness.attach()
        harness.fake.refuse_always["commit-end"] = (503, "WEIGHTS_TORN")
        with pytest.raises(Refusal) as caught:
            resident_merge.apply(harness.client, merged(), timings=harness.timings)
        assert caught.value.code == "weights_commit_interrupted"
        assert caught.value.details["retried"] is False
        assert caught.value.details["what"] == "resident adapter merge"

    def test_the_write_is_attempted_exactly_once(self, harness: Harness) -> None:
        harness.attach()
        harness.fake.refuse_always["commit-end"] = (503, "WEIGHTS_TORN")
        tensors = merged(1)
        target = tensors["model.layers.0.self_attn.q_proj.weight"]["target"]
        with pytest.raises(Refusal):
            resident_merge.apply(harness.client, tensors, timings=harness.timings)
        assert target.copies == 1, "a retried merge compounds a change that cannot be undone"
        assert harness.fake.call_count("commit-begin") == 1

    def test_recovery_goes_through_the_same_module_as_a_steps(self, harness: Harness) -> None:
        """*Recovery is R-02.30's, rather than a merge-specific path.* A second recovery
        path would be a second set of rules for the same parameter set."""
        plan = recovery.plan(
            Interruption(
                same_process=True,
                holds_update=False,
                attachment_id="a",
                commit_begun=True,
                last_checkpoint_step=100,
            )
        )
        assert plan.outcome is Recovery.ROLLBACK
        assert (
            recovery.is_retryable(
                Interruption(
                    same_process=True,
                    holds_update=False,
                    attachment_id="a",
                    commit_begun=True,
                    last_checkpoint_step=100,
                )
            )
            is False
        )


class TestF11ComputesNothing:
    """17.2a: *nothing in ``training/hotupdate/`` computes an adapter or a merge.*

    The live risk after the two clauses were assigned here is no longer that they go
    unbuilt - it is that **the requirements come with them**, which is undecided scope
    growth and exactly what ADR-022 exists to prevent.

    **Checked on the code, not on the text**, and the first draft of this file got that
    wrong: a substring scan reported these modules' own *disclaimers* as violations, because
    every one of them explains at length that the merge arithmetic is F8's. That is this
    project's recorded lesson - a fresh checker reported 18 scope-creep failures and all 18
    were documents correctly disclaiming a foreign requirement - and it happened again here,
    in the test written to honour it.
    """

    @staticmethod
    def _code_only(source: str) -> str:
        """The module's source with docstrings and comments removed.

        ``ast.unparse`` drops comments, and the docstrings are stripped node by node, so
        what is left is what executes.
        """
        import ast

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    node.body = body[1:] or [ast.Pass()]
        return ast.unparse(tree)

    def test_the_module_contains_no_merge_arithmetic(self) -> None:
        import ast
        import inspect

        tree = ast.parse(self._code_only(inspect.getsource(resident_merge)))
        arithmetic = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Div))
        ]
        assert arithmetic == [], f"merge arithmetic appeared in F11: lines {arithmetic}"

    def test_no_module_in_the_package_knows_an_adapters_format(self) -> None:
        """``merged_adapter``, ``save_pretrained``, a rank, an alpha or a scaling rule in the
        **code** would mean the seam was drawn in the wrong place."""
        from pathlib import Path

        package = Path(__file__).resolve().parents[2] / "src" / "training" / "hotupdate"
        offenders: dict[str, list[str]] = {}
        for path in sorted(package.glob("*.py")):
            code = self._code_only(path.read_text("utf-8"))
            hits = [
                token
                for token in ("merged_adapter", "save_pretrained", "lora_", "scaling")
                if token in code
            ]
            if hits:
                offenders[path.name] = hits
        assert offenders == {}, (
            f"{offenders}: F8 owns the adapter arithmetic, the manifest field and criterion "
            f"17's refusal. F11 wraps a write it does not compute."
        )

    def test_the_scan_sees_a_real_violation(self) -> None:
        """Otherwise the two tests above pass vacuously - and a disclaimer-blind scan would
        have been "fixed" by deleting the explanations, which is the worse outcome."""
        planted = "def apply():\n    weight = base * alpha + delta\n    merged_adapter = 1\n"
        code = self._code_only(planted)
        assert "merged_adapter" in code
        import ast

        assert any(isinstance(node, ast.BinOp) for node in ast.walk(ast.parse(code)))

    def test_a_disclaimer_in_prose_is_not_a_violation(self) -> None:
        planted = '"""The merged_adapter field and the scaling rule are F8\'s."""\nx = 1\n'
        assert "merged_adapter" not in self._code_only(planted)
