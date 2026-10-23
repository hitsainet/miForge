"""Prompt refresh  (R-02.26; FR-006.31 - .38; AC-006.24 - .28).

Two mutation controls live here and both are about the same thing from two sides:

* **M-5** includes ``undecided`` in the seed set. R-02.9's verdict is **three-valued** and
  carries the prohibition *"no consumer may construct a path from ``undecided`` to
  admitted"* - seeding a prompt from a pending completion is that path wearing a different
  hat: the unattended review queue becomes an unselected run, one round later, with nothing
  in the record to say so.
* **M-6** calls the inference provider from the refresh path. R-02.26 closes *"The resident
  model does not write its own prompt set"* and BRD §8 makes it a non-goal, so the template
  is a **mechanical** transformation and asking a model to invent prompts from it is the
  forbidden thing with an extra step.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from core.errors import Refusal
from schemas.selector import VerdictValue
from tests.support.loop_fakes import SeedSource
from training import refresh
from training.refresh.base import RefreshContext, Seed


def _ctx(**overrides) -> RefreshContext:
    defaults = {"run_id": "run-r", "step": 200, "round": 1}
    defaults.update(overrides)
    return RefreshContext(**defaults)  # type: ignore[arg-type]


class TestTheRegistryHasExactlyTwoModes:
    def test_exactly_R0226s_two(self) -> None:
        """R-02.26 names two: *"a new prompt set is loaded, or the selector's accepted
        completions become seeds for new prompts through a templating rule."* **There is no
        third.**"""
        assert refresh.names() == ("new_prompt_set", "template_seeds")

    def test_an_unknown_mode_is_refused_LISTING_the_two(self) -> None:
        with pytest.raises(Refusal) as caught:
            refresh.resolve("ask_the_model")
        assert caught.value.code == "refresh_mode_unknown"
        assert caught.value.details["known"] == ["new_prompt_set", "template_seeds"]

    def test_each_registered_mode_declares_the_name_it_is_registered_under(self) -> None:
        for name in refresh.names():
            assert refresh.resolve(name).name == name


class TestRefreshAppliesToSelfGenerationOnly:
    def test_self_generation_is_applicable(self) -> None:
        refresh.assert_applicable("self-generation")

    @pytest.mark.parametrize("source", ["corpus", "preference-pairs", "reward"])
    def test_every_other_source_is_REFUSED_not_ignored(self, source: str) -> None:
        """E-12, FR-006.31. R-02.26 opens *"For self-generation runs."* **Not silently
        ignored**: an ignored refresh leaves the operator believing the prompt set rotated
        when it did not, which is a wrong answer delivered quietly."""
        with pytest.raises(Refusal) as caught:
            refresh.assert_applicable(source)
        assert caught.value.code == "refresh_not_applicable"
        assert caught.value.details["signal_source"] == source

    def test_the_applicable_source_name_is_F2s_registry_key(self) -> None:
        """The four source names are ``checkpoint-manifest-v1.json``'s closed enum. A
        refresh gated on a fifth spelling would be a refresh that never applies."""
        from training import signals

        signals.ensure_registered()
        assert refresh.SELF_GENERATION in signals.registry.names()


class TestOnlyAcceptSeeds:
    """**M-5.** The filter is one positive comparison and there is no branch around it."""

    RULE: ClassVar[dict[str, str]] = {"template": "Write another question like: {prompt}"}

    def test_accepted_completions_seed(self) -> None:
        source = SeedSource(
            seeds=[
                Seed(prompt_id="p1", prompt="a", completion="A"),
                Seed(prompt_id="p2", prompt="b", completion="B"),
            ]
        )
        outcome = refresh.run(_ctx(source=source), "template_seeds", self.RULE)
        assert outcome.size == 2
        assert outcome.seed_counts == {"accept": 2, "reject": 0, "undecided": 0}

    def test_an_UNDECIDED_seed_is_never_used_in_any_configuration(self) -> None:
        """E-14. Not past any deadline, not under any parameter - **there is none that
        could**. F2 writes its own half the same way: *"there is no configuration value,
        deadline or timeout from which an ``undecided`` becomes an accept."*"""
        source = SeedSource(
            seeds=[
                Seed(
                    prompt_id="p1",
                    prompt="a",
                    completion="A",
                    verdict=VerdictValue.UNDECIDED,
                )
            ]
        )
        with pytest.raises(Refusal) as caught:
            refresh.run(_ctx(source=source), "template_seeds", self.RULE)
        assert caught.value.code == "refresh_produced_no_prompts"
        assert caught.value.details["seed_counts"]["undecided"] == 1

    def test_a_REJECTED_seed_is_never_used_either(self) -> None:
        source = SeedSource(
            seeds=[Seed(prompt_id="p1", prompt="a", completion="A", verdict=VerdictValue.REJECT)]
        )
        with pytest.raises(Refusal) as caught:
            refresh.run(_ctx(source=source), "template_seeds", self.RULE)
        assert caught.value.code == "refresh_produced_no_prompts"

    def test_a_mixed_set_seeds_only_from_the_accepts_and_COUNTS_the_rest(self) -> None:
        """The counts matter: the ``undecided`` number is what explains a small refresh to
        an operator, and an absent count is indistinguishable from a count of nothing."""
        source = SeedSource(
            seeds=[
                Seed(prompt_id="p1", prompt="a", completion="A"),
                Seed(prompt_id="p2", prompt="b", completion="B", verdict=VerdictValue.UNDECIDED),
                Seed(prompt_id="p3", prompt="c", completion="C", verdict=VerdictValue.REJECT),
            ]
        )
        outcome = refresh.run(_ctx(source=source), "template_seeds", self.RULE)
        assert outcome.size == 1
        assert outcome.seed_counts == {"accept": 1, "reject": 1, "undecided": 1}

    def test_the_comparison_is_written_POSITIVELY_in_the_source(self) -> None:
        """``== ACCEPT``, never ``!= REJECT``. The verdict is three-valued, so the negative
        spelling admits ``undecided`` - and the two forms are behaviourally identical on a
        two-valued verdict, which is why reading the code is not enough and this is a test.
        """
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3]
            / "backend"
            / "src"
            / "training"
            / "refresh"
            / "template_seeds.py"
        ).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare):
                continue
            mentions_verdict = any(
                isinstance(operand, ast.Attribute) and operand.attr == "verdict"
                for operand in [node.left, *node.comparators]
            )
            if mentions_verdict:
                assert all(isinstance(op, ast.Eq) for op in node.ops), (
                    "the seed filter must be a positive == ACCEPT; != REJECT admits "
                    "undecided on a three-valued verdict"
                )


class TestTheRefreshPathIsNonGenerative:
    """**M-6.** ``training/refresh/`` imports nothing from ``providers/``."""

    FORMS = (
        "import providers",
        "from providers import base",
        "from providers.base import GenerationParams",
        "import providers.registry",
        "from . import providers",
        "from .. import providers",
    )

    def _provider_imports(self, path) -> list[str]:
        import ast

        found: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if name.split(".")[0] == "providers":
                    found.append(f"{path.name}:{node.lineno} -> {name}")
        return found

    def test_no_refresh_module_imports_the_provider_package(self) -> None:
        """R-02.26's closing sentence and BRD §8's non-goal together make this a
        **prohibition, not a preference**. *The F2/F4 boundary is kept the same way: by
        what a module cannot reach, not by discipline.*"""
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "refresh"
        offenders = [
            site for path in sorted(src.rglob("*.py")) for site in self._provider_imports(path)
        ]
        assert not offenders, f"the refresh path reaches the inference provider: {offenders}"

    @pytest.mark.parametrize("form", FORMS)
    def test_the_scanner_sees_EACH_form_of_the_import(self, form: str, tmp_path) -> None:
        """**The self-test.** *A scanner that cannot see one form is a guard that is not
        there* - and the relative forms are the two a reader would forget."""
        path = tmp_path / "planted.py"
        path.write_text(form, encoding="utf-8")
        assert self._provider_imports(path), f"the scanner would not catch {form!r}"

    def test_the_scanner_does_not_flag_an_unrelated_import(self, tmp_path) -> None:
        path = tmp_path / "planted.py"
        path.write_text("from services import prompt_sets", encoding="utf-8")
        assert not self._provider_imports(path)

    def test_the_template_is_deterministic(self) -> None:
        """Nothing is sampled, so the same rule over the same seeds produces the same
        prompts - which is what makes the recorded ``rule_sha256`` mean anything."""
        source = SeedSource(seeds=[Seed(prompt_id="p1", prompt="a", completion="A")])
        first = refresh.run(_ctx(source=source), "template_seeds", {"template": "{prompt}!"})
        second = refresh.run(_ctx(source=source), "template_seeds", {"template": "{prompt}!"})
        assert first.rule_sha256 == second.rule_sha256
        assert first.size == second.size


class TestTheSeedSurfaceIsOwedByF2AndTheRefusalSaysSo:
    def test_a_source_that_publishes_no_seeds_is_refused_BY_NAME(self) -> None:
        """**Reported, not papered over.** F2's self-generation source builds
        ``SelectionOutcome.admitted`` inside ``next()`` and retains it nowhere, and the
        batch record carries identifiers and counts but **no text** (ADR-020) - so there is
        no path to read accepted completions from.

        The refusal names what is missing and who owes it, rather than seeding from a batch
        hash or from a pending candidate. *Refuse with a reason rather than guess.*
        """
        with pytest.raises(Refusal) as caught:
            refresh.run(_ctx(source=object()), "template_seeds", {"template": "{prompt}"})
        assert caught.value.code == "refresh_seeds_unavailable"
        assert "signal sources" in caught.value.details["owed_by"]

    def test_the_refusal_is_NOT_the_empty_seed_refusal(self) -> None:
        """An empty list would fall through to E-14 and blame the review queue for a
        mechanism that does not exist - *a fabricated diagnosis is worse than a missing one,
        because the operator tries to fix it.*"""
        with pytest.raises(Refusal) as caught:
            refresh.run(_ctx(source=None), "template_seeds", {"template": "{prompt}"})
        assert caught.value.code != "refresh_produced_no_prompts"


class TestTheTemplateRule:
    def test_a_missing_template_is_refused(self) -> None:
        """A default template would invent the prompts the next round trains on."""
        with pytest.raises(Refusal) as caught:
            refresh.run(_ctx(source=SeedSource()), "template_seeds", {})
        assert caught.value.code == "refresh_rule_incomplete"

    def test_a_template_naming_an_unknown_field_is_refused_not_blanked(self) -> None:
        """*A prompt with a silently missing clause is a prompt nobody wrote.*"""
        with pytest.raises(Refusal) as caught:
            refresh.run(
                _ctx(source=SeedSource(seeds=[Seed("p", "a", "A")])),
                "template_seeds",
                {"template": "{prompt} {grade}"},
            )
        assert caught.value.details["unknown_fields"] == ["grade"]

    def test_the_rule_hash_includes_the_MODE(self) -> None:
        """The same body under the two modes is two different rules, and a run log
        recording one hash for both could not say which happened."""
        body = {"template": "{prompt}"}
        assert refresh.rule_hash("template_seeds", body) != refresh.rule_hash(
            "new_prompt_set", body
        )


class TestAFailedRefreshIsAFailureNotASixthCondition:
    def test_an_empty_refreshed_set_refuses_by_name(self) -> None:
        """FR-006.38, E-13: *"it does not continue on the old set"* - the run log would then
        record a refresh that did not happen and every subsequent batch's part reference
        would be wrong."""
        with pytest.raises(Refusal) as caught:
            refresh.run(_ctx(source=SeedSource(seeds=[])), "template_seeds", {"template": "x"})
        assert caught.value.code == "refresh_produced_no_prompts"

    def test_the_refusal_code_is_not_a_stop_reason(self) -> None:
        """FR-006.15's set stays closed at five."""
        from training.stop.base import STOP_REASONS

        assert "refresh_produced_no_prompts" not in STOP_REASONS


class TestTheRecordedRefresh:
    def test_it_records_the_mode_the_rule_hash_the_size_and_the_round(self) -> None:
        """FR-006.36, R-02.26's own words: *"refresh rules are configuration and are
        recorded."*"""
        outcome = refresh.run(
            _ctx(round=3, source=SeedSource(seeds=[Seed("p", "a", "A")])),
            "template_seeds",
            {"template": "{prompt}?"},
        )
        payload = outcome.payload()
        assert payload["mode"] == "template_seeds"
        assert len(payload["rule_sha256"]) == 64
        assert payload["round"] == 3
        assert payload["size"] == 1
        assert payload["seed_counts"] == {"accept": 1, "reject": 0, "undecided": 0}

    def test_a_templated_set_carries_MIFORGES_OWN_provenance_not_a_contract_reference(
        self,
    ) -> None:
        """R-02.40, FR-006.33: *miForge does not silently promote an external part to a
        contract reference, and does not claim miDataworks provenance for anything
        miDataworks did not publish.* These prompts were generated here."""
        outcome = refresh.run(
            _ctx(source=SeedSource(seeds=[Seed("p", "a", "A")])),
            "template_seeds",
            {"template": "{prompt}?"},
        )
        assert outcome.set_reference is not None
        assert "external_provenance" in outcome.set_reference
        assert "reference" not in outcome.set_reference

    def test_no_prompt_or_completion_TEXT_is_in_the_record(self) -> None:
        """ADR-020, FR-006.50's security half, F2 FR-002.17's rule: identifiers, counts and
        hashes - never content."""
        outcome = refresh.run(
            _ctx(source=SeedSource(seeds=[Seed("p1", "SECRET PROMPT", "SECRET COMPLETION")])),
            "template_seeds",
            {"template": "{prompt}?"},
        )
        rendered = repr(outcome.payload())
        assert "SECRET PROMPT" not in rendered
        assert "SECRET COMPLETION" not in rendered


class TestThePlateauWindowNeverSpansARefresh:
    """**FR-006.22, and a test finding from F6's own sweep.**

    Mutation **M-10** - letting the window span a refresh - **SURVIVED** the first sweep:
    the clearing was implemented in :func:`training.loop._on_refresh` and nothing depended
    on it. Written here as the regression, and M-10 re-run against it as the negative
    control.

    Why it matters: a refresh changes the batch distribution, so the losses either side of
    it are not comparable, and a detector spanning the boundary measures **the refresh
    rather than the model** - in *either* direction. A harder refreshed set raises the loss
    and resets a plateau that had genuinely arrived; an easier one lowers it and hides one.
    Both are wrong and neither shows on a curve.
    """

    #: ``min_steps >= window`` is enforced by the condition itself, so the window is 3 and
    #: the minimum is 3. The *shape* is what the test needs: a window that fills by step 4
    #: on an unbroken series and does not fill by step 4 if it was cleared at step 2.
    PLATEAU: ClassVar[dict[str, object]] = {
        "metric": "raw",
        "window": 3,
        "min_relative_improvement": 0.01,
        "min_steps": 3,
    }

    def _run_four_flat_steps(self, *, refresh_every: int | None):
        """Four steps - one improving, then flat - with a refresh at step 2 when asked for.

        The first loss is higher so that the step-3 window still shows improvement: the
        window has to be *fillable and flat* at step 4 and not before, or the test would
        pass for the wrong reason.
        """
        from tests.support.loop_fakes import FakeStep, SeedSource, boundary_context
        from training import loop
        from training.refresh.base import Seed

        return loop.run(
            boundary_context(
                # A flat loss from the first step, so a FULL window plateaus.
                take_step=FakeStep(losses=[0.9, 0.5, 0.5, 0.5, 0.5, 0.5]),
                save_every=None,
                stop_conditions={"loss_plateau": self.PLATEAU},
                refresh_every=refresh_every,
                refresh_mode="template_seeds" if refresh_every else None,
                refresh_rule={"template": "{prompt}?"},
                source=SeedSource(seeds=[Seed("p1", "a", "A")]),
            ),
            max_steps=4,
            evaluate_step_zero=False,
            run_end_save=False,
        )

    def test_without_a_refresh_a_flat_window_DOES_fire(self) -> None:
        """The positive control. Without it the test below would pass against a plateau
        condition that never fires at all, which protects nothing."""
        result = self._run_four_flat_steps(refresh_every=None)
        assert result.stopped_because == "loss_plateau"
        assert result.last_step == 4

    def test_a_refresh_at_step_2_RESETS_the_window_so_step_4_does_not_fire(self) -> None:
        """**M-10's target.** With the window cleared at the refresh, step 4 has only two
        of its four observations and the condition cannot fire - which is correct, because
        the two losses before the refresh were taken on a different prompt set."""
        result = self._run_four_flat_steps(refresh_every=2)
        assert result.stopped_because != "loss_plateau", (
            "the plateau fired across a refresh. The detector measured the refresh, not "
            "the model - and the losses either side of a refresh are not comparable."
        )
        assert result.last_step == 4

    def test_the_round_increments_at_the_refresh_and_is_on_the_boundary_record(
        self, tmp_path
    ) -> None:
        """FR-006.37, AMB-4: **a "round" is ``refresh_every`` completed steps**, and F6
        defines it because *"round" is defined nowhere in BRD-02* - which is what makes
        acceptance criterion 3's "three rounds" a checkable statement."""
        from tests.support.loop_fakes import FakeStep, SeedSource, boundary_context
        from training import loop, run_log
        from training.refresh.base import Seed

        log = tmp_path / "run.jsonl"
        loop.run(
            boundary_context(
                log_path=log,
                take_step=FakeStep(losses=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
                save_every=None,
                stop_conditions={"step_count": {"total": 6}},
                refresh_every=2,
                refresh_mode="template_seeds",
                refresh_rule={"template": "{prompt}?"},
                source=SeedSource(seeds=[Seed("p1", "a", "A")]),
            ),
            max_steps=6,
            evaluate_step_zero=False,
            run_end_save=False,
        )
        rounds = [b["round"] for b in run_log.read_blocks(log, block="boundary")]
        # Steps 1-2 are round 0, 3-4 round 1, 5-6 round 2 - three rounds over six steps at
        # refresh_every=2, which is criterion 3's shape made checkable.
        assert rounds == [0, 0, 1, 1, 2, 2], rounds

    def test_the_reset_is_LOGGED_with_the_round(self, caplog) -> None:
        """Invisible otherwise: a window that reset and a window that never filled look the
        same from outside, and only one of them is a refresh."""
        import logging

        with caplog.at_level(logging.INFO):
            self._run_four_flat_steps(refresh_every=2)
        resets = [r for r in caplog.records if getattr(r, "code", "") == "plateau_window_reset"]
        assert resets, "the window reset was silent"
        assert getattr(resets[0], "round", None) is not None

    def test_a_refresh_is_SKIPPED_when_the_run_is_stopping(self) -> None:
        """FR-006.38, 6.7: the refresh is **last** in the boundary and does not happen on a
        boundary that ends the run - a refresh recorded on a run that then ended would say
        the prompt set rotated for a round that never happened."""
        from tests.support.loop_fakes import FakeEmit, FakeStep, SeedSource, boundary_context
        from training import loop
        from training.refresh.base import Seed
        from training.run_events import EVENT_REFRESH

        emit = FakeEmit()
        result = loop.run(
            boundary_context(
                emit=emit,
                take_step=FakeStep(),
                save_every=None,
                stop_conditions={"step_count": {"total": 2}},
                refresh_every=2,
                refresh_mode="template_seeds",
                refresh_rule={"template": "{prompt}?"},
                source=SeedSource(seeds=[Seed("p1", "a", "A")]),
            ),
            max_steps=10,
            evaluate_step_zero=False,
            run_end_save=False,
        )
        assert result.stopped_because == "step_count"
        assert emit.named(EVENT_REFRESH) == [], "a stopping run must record no refresh"
