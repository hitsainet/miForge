"""Two guards that existed and were called by nothing  (ADR-016 #1; R-02.48, R-02.11).

``services.run_definition`` is R-02.53's validator and the **one** path a run reaches a card
through. Before this file:

* **``services.scoring.entrypoint.check_submission``** had no caller anywhere in
  ``backend/src``. The validator imported ``DEFINITION_KEY`` *from its module* and never the
  guard, so a run could bind a part version as the **test set** and as a **training signal
  source** at once and nothing refused. The refusal's own message is why that is the worst
  available failure: *"training on the test set raises the test score for the wrong reason,
  and the rise is indistinguishable from success on every surface miForge produces."* It is
  not a crash and not a bad number - it is a **good** number that means nothing, on every
  screen, for the whole life of the run.
* **``services.selectors.identity.guard_at_preflight``** - call site 2 of three - had none
  either. The one wired site, ``guard_at_resolution``, runs inside F2's source pre-flight,
  and ``training/run_driver._resolve_source`` builds its ``RunContext`` with
  ``provider=None``. So ``run_target`` was ``None`` there, the comparison took
  ``check_judge_identity``'s early ``return ()``, and **R-02.11's comparison had never once
  been made in production**: only the unconditional in-process refusal fired. A judge
  configured during pack authoring - when no run target existed - and then used in a run
  whose target *is* that judge was never caught (AC-003.5). R-02.11 exists because *a model
  that grades its own outputs and trains on the winners converges on whatever it already
  likes*: the loss falls, the test score does not move, and nothing errors.

## The false-refusal direction is tested FIRST and deliberately

Assembling ``check_submission``'s four part-version lists wrongly fails in the worst
available direction - **refusing every run at submit** - so
:class:`TestAValidRunStillSubmits` drives one run of each shape (loose parts, a pack, a
retention set with a floor, no retention material at all) through the live step and requires
``outcome.ok``. Three vocabularies reach these lists and *they are not aliases*
(``packs/slots.py``), which is exactly how a list silently comes out empty or - worse -
matching the wrong thing.

## What is NOT enforced here, and why that is recorded rather than decided

E-12's *unconditional* reading - every run must declare a retention set with a floor - is
**deferred to the operator**, and :class:`TestTheE12DeferralIsVisibleRatherThanSilent` pins
the deferral so it cannot become invisible. R-02.17 and F4's AC-004.13 support it; two
measurements argue against taking it here: F6's own ``run_driver._evaluator`` states that *"a
run may legitimately have no test or retention set"*, and enforcing it refuses **74
currently-passing tests across five features** - among them every chain stage, since
``services/chains/service.py`` builds and launches a ``RunDefinition`` per hop. What *is*
enforced is E-12 wherever retention is part of the run's declared shape (a pack, a declared
retention slot, or an enabled ``retention_floor`` condition), plus E-16 unconditionally.

## The controls that were run

* **MUT-C** - the ``_check_submission_guards`` call replaced by ``resolved[...] = {}`` in
  :func:`services.run_definition.validate`, which is the wiring rather than the rule.
  **10 failed** - every E-16 and E-12 assertion here plus the two in
  ``test_run_definition.py`` - and **not one** of
  :class:`TestAValidRunStillSubmits`'s four, which is the shape a reachability control
  should have: removing a guard must not change what a valid run does.
* **MUT-D** - ``guard_at_preflight(judge, run_target)`` swapped for
  ``guard_at_resolution(judge, None)``: the exact production state this file closes, where
  the guard still *runs* and runs with no target. **3 failed** - the two lineage-caveat
  assertions and the payload/count one. ``test_an_in_process_judge_is_refused_at_submit``
  **survives**, and that is the whole finding: the refusal that did work was never the gap,
  so a test suite built around it would have reported R-02.11 as covered.
* **MUT-E** - ``_run_target_identity`` returning ``None`` always. **3 failed**, the same
  three. Kept separate from MUT-D because the two defects are reached by different edits and
  only one of them looks like a simplification.

All three restored, with the file's bytes compared against the backup after each restore and
``git diff`` read for that path - *a ``replace()`` that matches nothing is indistinguishable
from a survivor.*
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from services import run_definition as validator
from services.selectors.identity import CAVEAT_SHARES_LINEAGE
from tests.support.jobs import memory_engine, session_for

RUN = "run-guards"

#: A part named by contract reference, in the form **both** sides of the overlap render.
HELD_OUT_REFERENCE = {"type": "prompt-set", "name": "support-tickets", "version": "3"}

GOOD: dict[str, Any] = {
    "run_id": RUN,
    "model_source": {"mode": "load", "repo": "LiquidAI/LFM2-1.2B"},
    "run_type": "change",
    "run_configuration": {
        "learning_rate": 1e-5,
        "warmup_steps": 2,
        "batch_size": 2,
        "gradient_accumulation": 1,
    },
    "signal_source": {"source": "corpus"},
    "parts": [{"part_id": "part-corpus", "kind": "corpus"}],
    "cadences": {"test_every": 2, "retention_every": 1},
    "stop_conditions": [{"condition": "step_count", "config": {"total": 100}}],
    "domain": "legal",
}


def definition(**overrides: Any):
    from schemas.run_definition import RunDefinition

    return RunDefinition(**{**GOOD, **overrides})


def _passing() -> dict[str, Any]:
    """Every *other* collaborator stubbed, so a test is about the two guards.

    The gate, the part resolver, the budget and the service check are stubbed; the two
    guards under test are **not**, which is what makes this a reachability test of them
    rather than of a spy.
    """
    return {
        "gate": lambda *a, **k: _PackDouble(),
        "resolve_parts": lambda *a, **k: {"source": "parts", "parts": []},
        "budget": lambda *a, **k: {"param_count": 10, "gpu_uuid": None},
        "services": lambda *a, **k: {"needed": []},
    }


class _PackDouble:
    """F7's ``ResolvedPack``, as far as this step reads it: the published ``payload()``.

    Carries ``retention_floor`` because a pack that passed the gate **cannot** be floorless -
    a double that could not express that would make the valid-pack assertion vacuous.
    """

    class _Definition:
        domain = "legal"

    definition = _Definition()

    def __init__(self, parts: list[dict[str, Any]] | None = None) -> None:
        self._parts = parts or []

    def payload(self) -> dict[str, Any]:
        return {
            "pack_id": "legal",
            "version": "1",
            "retention_floor": 0.72,
            "parts": self._parts,
        }

    def part_records(self) -> list[Any]:
        return []


def _validate(session: Any, defn: Any, **overrides: Any):
    return validator.validate(defn, session=session, **{**_passing(), **overrides})


@pytest.fixture
def session():
    engine = memory_engine()
    with session_for(engine) as session:
        yield session


# ---------------------------------------------------------------- the direction that hurts


class TestAValidRunStillSubmits:
    """**Run first.** A wrong derivation refuses every run, which is worse than none."""

    def test_a_loose_parts_run_with_no_retention_material(self, session) -> None:
        outcome = _validate(session, definition())
        assert outcome.ok, [failure.payload() for failure in outcome.failures]
        assert "submission_guards" in outcome.steps_run

    def test_a_pack_run(self, session) -> None:
        outcome = _validate(
            session, definition(pack={"pack_id": "legal", "version": "1"}, parts=None)
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]

    def test_a_run_declaring_a_retention_set_and_a_floor(self, session) -> None:
        outcome = _validate(
            session,
            definition(
                scoring={"retention_slot": {"part_id": "part-retention"}},
                retention_floor_override=0.6,
            ),
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]

    def test_a_run_whose_evaluation_and_signal_parts_merely_SHARE_A_KIND(self, session) -> None:
        """Different parts of the same kind are not an overlap. The guard intersects part
        **identities**, and conflating kind with identity would refuse every honest run that
        draws its test set and its corpus from the same kind of source."""
        outcome = _validate(
            session,
            definition(
                parts=[{"part_id": "part-corpus", "kind": "corpus"}],
                signal_source={
                    "source": "corpus",
                    "parts": [{"part_id": "part-corpus", "kind": "corpus"}],
                },
                scoring={"test_slot": {"part_id": "part-test"}},
            ),
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]


# ---------------------------------------------------------------------------- E-16


class TestAHeldOutSetBoundForTrainingIsRefused:
    def test_by_part_id_naming_the_part(self, session) -> None:
        outcome = _validate(
            session,
            definition(
                signal_source={
                    "source": "corpus",
                    "parts": [{"part_id": "part-test", "kind": "corpus"}],
                },
                scoring={"test_slot": {"part_id": "part-test"}},
            ),
        )
        assert [failure.code for failure in outcome.failures] == ["held_out_set_used_for_training"]
        failure = outcome.failures[0]
        assert failure.step == "submission_guards"
        # The PAYLOAD: a refusal that does not name the part leaves nothing to fix.
        assert failure.details["parts"] == ["part:part-test"]

    def test_by_contract_reference_across_the_two_renderings(self, session) -> None:
        """The same part named as a **contract reference** on the signal side and on the
        scoring side. One identity function on both sides is the whole mechanism: two
        derivations would intersect to nothing and the guard would read as having run."""
        outcome = _validate(
            session,
            definition(
                signal_source={
                    "source": "corpus",
                    "parts": [{"reference": HELD_OUT_REFERENCE, "kind": "corpus"}],
                },
                scoring={"retention_slot": {"reference": HELD_OUT_REFERENCE}},
                retention_floor_override=0.6,
            ),
        )
        assert [failure.code for failure in outcome.failures] == ["held_out_set_used_for_training"]
        assert failure_details(outcome)["parts"] == ["prompt-set/support-tickets@3"]

    def test_a_PACK_test_slot_trained_on_as_a_loose_signal_part(self, session) -> None:
        """The cross-layer case: the pack says *test set*, the run's signal block names the
        same part. Neither side alone is wrong, which is why nothing else catches it."""
        pack = _PackDouble(
            parts=[{"kind": "test_set", "contract_reference": HELD_OUT_REFERENCE, "external": None}]
        )
        outcome = _validate(
            session,
            definition(
                pack={"pack_id": "legal", "version": "1"},
                parts=None,
                signal_source={
                    "source": "corpus",
                    "parts": [{"reference": HELD_OUT_REFERENCE, "kind": "corpus"}],
                },
            ),
            gate=lambda *a, **k: pack,
        )
        assert [failure.code for failure in outcome.failures] == ["held_out_set_used_for_training"]

    def test_part_versions_on_a_scoring_slot_are_read_too(self, session) -> None:
        """``part_versions`` is the field F4's own ``_resolve_slot`` reads and calls *"for
        E-16's guard"*, so it is read here as well rather than only the slot's ``part_id``."""
        outcome = _validate(
            session,
            definition(
                signal_source={
                    "source": "corpus",
                    "parts": [{"part_id": "v-77", "kind": "corpus"}],
                },
                scoring={"test_slot": {"part_versions": ["part:v-77"]}},
            ),
        )
        assert [failure.code for failure in outcome.failures] == ["held_out_set_used_for_training"]


def failure_details(outcome: Any) -> dict[str, Any]:
    return dict(outcome.failures[0].details)


# ---------------------------------------------------------- the call, its payload, its count


class TestTheGuardIsCalledOnceWithTheRunsOwnLists:
    def test_the_payload_and_the_count(self, session, monkeypatch: pytest.MonkeyPatch) -> None:
        """*"Was called" passes against a call sending the wrong arguments* - and the wrong
        argument available here is another run's part list, which would report no overlap."""
        calls: list[dict[str, Any]] = []
        real = validator.__dict__.get("check_submission")
        del real

        import services.scoring.entrypoint as entrypoint

        original = entrypoint.check_submission

        def spy(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            return original(**kwargs)

        monkeypatch.setattr(entrypoint, "check_submission", spy)
        outcome = _validate(
            session,
            definition(
                signal_source={
                    "source": "corpus",
                    "parts": [{"part_id": "part-corpus", "kind": "corpus"}],
                },
                scoring={"retention_slot": {"part_id": "part-retention"}},
                retention_floor_override=0.6,
            ),
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]
        assert len(calls) == 1, f"one call per submission, not {len(calls)}"
        assert calls[0]["signal_part_versions"] == {"part:part-corpus"}
        assert calls[0]["retention_part_versions"] == {"part:part-retention"}
        assert calls[0]["retention_set_present"] is True
        assert calls[0]["retention_floor"] == 0.6


class TestTheE12DeferralIsVisibleRatherThanSilent:
    def test_a_run_with_no_retention_material_records_the_deferral(self, session) -> None:
        """So an operator reading ``POST /runs/validate`` sees that E-12 was not asked,
        rather than reading its silence as a pass. **A deferral nobody can see is the
        stale-finding loop with a validator attached.**"""
        outcome = _validate(session, definition())
        assert outcome.ok
        block = outcome.resolved["submission_guards"]
        assert block["retention_guard_deferred"] == "retention_set_missing"
        assert block["checked"] == ["held_out_set_used_for_training"]

    def test_E16_STILL_REFUSES_on_the_deferred_path(self, session) -> None:
        """The deferral must not take the other guard with it. This is the assertion that
        would catch an implementation that returned early instead."""
        outcome = _validate(
            session,
            definition(
                signal_source={
                    "source": "corpus",
                    "parts": [{"part_id": "part-test", "kind": "corpus"}],
                },
                scoring={"test_slot": {"part_id": "part-test"}},
            ),
        )
        assert [failure.code for failure in outcome.failures] == ["held_out_set_used_for_training"]

    def test_a_declared_retention_set_is_NOT_deferred(self, session) -> None:
        outcome = _validate(
            session,
            definition(
                scoring={"retention_slot": {"part_id": "part-retention"}},
                retention_floor_override=0.6,
            ),
        )
        assert "retention_guard_deferred" not in outcome.resolved["submission_guards"]


# ----------------------------------------------------------------- R-02.11, at pre-flight


JUDGE_SELECTOR = {
    "kind": "judge_model",
    "name": "quality",
    "provider": "judge",
    "threshold": 0.6,
}


@pytest.fixture
def judge_endpoint(monkeypatch: pytest.MonkeyPatch):
    """An OpenAI-compatible judge row, serving a **different** model from the run's."""
    monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "qwen-7b-instruct")
    yield


class TestTheJudgeIsCheckedAgainstTheRunsOwnTarget:
    def test_an_in_process_judge_is_refused_at_submit(self, session) -> None:
        """R-02.11, unconditionally: ADR-009's in-process provider generates from the
        parameters miForge is training, so it **is** the resident model."""
        outcome = _validate(
            session,
            definition(
                signal_source={
                    "source": "corpus",
                    "selector": {**JUDGE_SELECTOR, "provider": "in_process"},
                }
            ),
        )
        assert [failure.code for failure in outcome.failures] == ["judge_is_resident_model"]
        assert outcome.failures[0].step == "judge_identity"
        assert outcome.failures[0].details["judge"]["kind"] == "in_process"

    def test_a_judge_serving_THE_MODEL_UNDER_TRAINING_raises_the_lineage_caveat(
        self, session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The case that was never caught.** Same model name, different endpoint: almost
        certainly a different parameter set of the same lineage, which FPRD OQ-3 permits
        *while recording that it happened*. It was never recorded, because production passed
        no run target at all - so the caveat could not be produced by any configuration."""
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "LiquidAI/LFM2-1.2B")
        outcome = _validate(
            session,
            definition(signal_source={"source": "corpus", "selector": JUDGE_SELECTOR}),
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]
        judge = outcome.resolved["judge"]
        assert judge["caveats"] == [CAVEAT_SHARES_LINEAGE]
        assert judge["compared"] == ["judge"]
        assert judge["run_target"]["model"] == "LiquidAI/LFM2-1.2B"
        assert judge["run_target"]["kind"] == "in_process"

    def test_a_judge_serving_a_different_model_raises_nothing(
        self, session, judge_endpoint
    ) -> None:
        outcome = _validate(
            session,
            definition(signal_source={"source": "corpus", "selector": JUDGE_SELECTOR}),
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]
        assert outcome.resolved["judge"]["caveats"] == []

    def test_a_grader_is_checked_as_well_as_a_selector(self, session, monkeypatch) -> None:
        """Both forms of a judge-model configuration. R-02.11 is about the call, not about
        which of the two slots made it."""
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "LiquidAI/LFM2-1.2B")
        outcome = _validate(
            session,
            definition(signal_source={"source": "corpus", "grader": JUDGE_SELECTOR}),
        )
        assert outcome.resolved["judge"]["caveats"] == [CAVEAT_SHARES_LINEAGE]

    def test_the_guard_is_called_once_per_judge_with_both_identities(
        self, session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The payload **and** the count. The wrong argument available here is ``None`` for
        the run target - which is precisely the production defect, and which leaves every
        assertion about a *refusal* passing while the comparison does not happen."""
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "qwen-7b-instruct")
        calls: list[tuple[Any, Any]] = []

        import services.selectors.identity as identity_module

        original = identity_module.guard_at_preflight

        def spy(judge: Any, run_target: Any) -> Any:
            calls.append((judge, run_target))
            return original(judge, run_target)

        monkeypatch.setattr(identity_module, "guard_at_preflight", spy)
        outcome = _validate(
            session,
            definition(signal_source={"source": "corpus", "selector": JUDGE_SELECTOR}),
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]
        assert len(calls) == 1, f"once per configured judge, not {len(calls)}"
        judge, run_target = calls[0]
        assert (judge.kind, judge.model, judge.base_url) == (
            "openai_compatible",
            "qwen-7b-instruct",
            "http://judge.test:8000",
        )
        assert run_target is not None, (
            "the run's own target must be supplied - with None the comparison silently "
            "does not happen, which is the production defect this test closes"
        )
        assert (run_target.kind, run_target.model) == ("in_process", "LiquidAI/LFM2-1.2B")


class TestWhatCouldNotBeCheckedIsNamed:
    def test_a_run_with_no_judge_records_nothing(self, session) -> None:
        outcome = _validate(session, definition())
        assert "judge" not in outcome.resolved

    def test_an_unconfigured_judge_provider_is_NAMED_not_refused_here(self, session) -> None:
        """Step 6 reports an absent optional service by name. Refusing twice for one fact
        would tell the operator two different things about one state."""
        outcome = _validate(
            session,
            definition(signal_source={"source": "corpus", "selector": JUDGE_SELECTOR}),
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]
        assert outcome.resolved["judge"]["unchecked"] == ["judge:provider_not_configured"]
        assert outcome.resolved["judge"]["compared"] == []

    def test_a_composite_selector_is_named_as_unchecked(self, session) -> None:
        """Its children resolve against a catalog this step does not hold. *A step that
        quietly checked nothing reads exactly like a step that found nothing wrong.*"""
        outcome = _validate(
            session,
            definition(
                signal_source={
                    "source": "corpus",
                    "selector": {"kind": "composite", "name": "both", "children": ["a", "b"]},
                }
            ),
        )
        assert outcome.resolved["judge"]["unchecked"] == [
            "selector:composite_children_not_resolvable_at_submit"
        ]


class TestNeitherStepAllocatesAnything:
    def test_no_lease_is_taken_and_no_device_is_resolved_by_these_steps(
        self, session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The validator's defining property. The judge's identity comes off the registry's
        **row** - kind, model, base_url - rather than from a constructed provider, so
        nothing is built and no endpoint is contacted."""
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "qwen-7b-instruct")
        built: list[str] = []

        import providers.registry as provider_registry

        monkeypatch.setattr(
            provider_registry,
            "get",
            lambda name: built.append(name) or pytest.fail("a provider was constructed"),
        )
        outcome = _validate(
            session,
            definition(signal_source={"source": "corpus", "selector": JUDGE_SELECTOR}),
        )
        assert outcome.ok, [failure.payload() for failure in outcome.failures]
        assert built == []


class TestTheStepsAreInTheValidatorsOwnSequence:
    def test_both_steps_are_declared_and_run_before_the_budget(self, session) -> None:
        """Cheapest refusals first (ADR-023): these read two dicts and a settings row, while
        the budget step resolves a device."""
        assert validator.STEPS.index("submission_guards") < validator.STEPS.index("budget")
        assert validator.STEPS.index("judge_identity") < validator.STEPS.index("budget")
        outcome = _validate(session, definition())
        assert outcome.steps_run == list(validator.STEPS)

    def test_the_submission_guards_WAIT_on_the_pack_gate(self, session) -> None:
        """Rather than reporting *"this run defines no retention set"* about a run whose
        pack did not resolve - a fabricated second failure the operator would go and fix."""
        from core.errors import refuse

        outcome = _validate(
            session,
            definition(pack={"pack_id": "legal", "version": "1"}, parts=None),
            gate=lambda *a, **k: (_ for _ in ()).throw(refuse("pack_incomplete", "missing")),
        )
        skipped = [entry.step for entry in outcome.skipped]
        assert "submission_guards" in skipped
        assert [failure.code for failure in outcome.failures] == ["pack_incomplete"]


class TestTheRefusalsReachTheApiEnvelope:
    def test_raise_if_invalid_carries_the_overlap_refusal_with_its_own_code(self, session) -> None:
        """Each refusal keeps **its own code** - never re-wrapped, because re-wrapping hides
        which layer refused."""
        outcome = _validate(
            session,
            definition(
                signal_source={
                    "source": "corpus",
                    "parts": [{"part_id": "part-test", "kind": "corpus"}],
                },
                scoring={"test_slot": {"part_id": "part-test"}},
            ),
        )
        with pytest.raises(Refusal) as caught:
            outcome.raise_if_invalid()
        assert caught.value.code == "run_definition_invalid"
        codes = [failure["code"] for failure in caught.value.details["failures"]]
        assert codes == ["held_out_set_used_for_training"]
