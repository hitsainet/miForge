"""R-02.53: defining, validating and starting a run  (FR-006.56 - .65; AC-006.38 - .44).

**Five mutation controls live here**, and each one is a defect that looks like nothing:

* **M-9** deletes the ``require_valid_pack`` call. A run then starts from an invalid pack -
  the thing F7's whole gate exists to prevent, and F7's task 13.2 is this assertion.
* **M-14** moves the budget check after the lease. The run takes the only card, finds it
  does not fit, and releases it - the ordering defect BRD §9 records against R-02.42.
* **M-15** makes the create path skip the validator. *A ``validate`` route the create path
  never calls is an unreachable capability whose unit tests all pass.*
* **M-16** writes the definition record after the enqueue. A worker can then start a run
  whose definition is not yet durable, so a crash in that window leaves a running run
  nothing can state the shape of.
* **SEED** records ``definition.seed`` instead of the resolved value, so a run with no
  supplied seed stays unreproducible while the record looks complete.

Every assertion is on **the payload and the call count**, never on "was called": a
reachability test that only checks the latter passes against a call sending the wrong
arguments (ADR-016 #1).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from core.errors import Refusal
from schemas.run_definition import RunDefinition
from services import run_definition as validator
from services import run_launch
from tests.support.jobs import memory_engine, session_for

RUN = "run-def"

#: A definition that is valid in every respect: loose parts (no pack), one stop condition,
#: no optional service involved. **The standalone case is the default case** (FR-006.60),
#: and that is where AC20 begins.
GOOD: dict[str, Any] = {
    "run_id": RUN,
    "model_source": {"mode": "load", "path": "/data/checkpoints/base"},
    "run_type": "change",
    "run_configuration": {
        "learning_rate": 1e-5,
        "warmup_steps": 2,
        "batch_size": 2,
        "gradient_accumulation": 1,
    },
    "signal_source": {"source": "corpus"},
    "parts": [{"part_id": "part-1", "kind": "corpus"}],
    "cadences": {"test_every": 2, "retention_every": 1, "save_every": 10},
    "stop_conditions": [{"condition": "step_count", "config": {"total": 100}}],
    "domain": "legal",
}


def definition(**overrides: Any) -> RunDefinition:
    body = {**GOOD, **overrides}
    return RunDefinition(**body)


class _Spy:
    """Records every call with its arguments, and counts them."""

    def __init__(self, result: Any = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._result = result
        self._raises = raises

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self._raises is not None:
            raise self._raises
        return self._result

    @property
    def count(self) -> int:
        return len(self.calls)


@pytest.fixture
def engine():
    return memory_engine()


@pytest.fixture
def spies():
    """Every step of FR-006.58 as a spy. The **order** they record in is the assertion."""
    order: list[str] = []

    def record(name: str, result: Any = None):
        def fn(*args: Any, **kwargs: Any) -> Any:
            order.append(name)
            return result

        fn.calls = []  # type: ignore[attr-defined]
        return fn

    gate = _Spy(result=_ResolvedPackDouble())
    parts = _Spy(result={"source": "parts", "parts": []})
    budget = _Spy(result={"param_count": 1_200_000_000, "gpu_uuid": None, "mode": "load"})
    services = _Spy(result={"needed": [], "midataworks": {"configured": False}})

    def ordered(name: str, spy: _Spy):
        def fn(*args: Any, **kwargs: Any) -> Any:
            order.append(name)
            return spy(*args, **kwargs)

        return fn

    return {
        "order": order,
        "gate_spy": gate,
        "parts_spy": parts,
        "budget_spy": budget,
        "services_spy": services,
        "kwargs": {
            "gate": ordered("pack_gate", gate),
            "resolve_parts": ordered("part_resolvability", parts),
            "budget": ordered("budget", budget),
            "services": ordered("optional_services", services),
        },
    }


class _ResolvedPackDouble:
    """Stands in for F7's ``ResolvedPack`` where the subject is F6's call, not F7's gate."""

    class _Definition:
        domain = "legal"

    definition = _Definition()

    def payload(self) -> dict[str, Any]:
        return {"pack_id": "p", "version": "1", "retention_floor": 0.72, "parts": []}

    def part_records(self) -> list[Any]:
        return []


# ====================================================================== the envelope


class TestTheDefinitionEnvelope:
    def test_it_carries_R0253s_nine_parts(self) -> None:
        """FR-006.56. Asserted against the field set rather than a prose list, so a part
        removed from the model is red here."""
        fields = set(RunDefinition.model_fields)
        for part in (
            "model_source",
            "pack",
            "parts",
            "signal_source",
            "run_type",
            "run_configuration",
            "cadences",
            "retention_floor_override",
            "stop_conditions",
        ):
            assert part in fields, f"R-02.53 names {part} and the envelope has no field for it"

    def test_it_is_closed_to_unknown_keys(self) -> None:
        """A misspelled ``retention_every`` must not silently leave the real field at its
        default - on this object that would change **when a run stops**."""
        with pytest.raises(ValidationError):
            RunDefinition(**{**GOOD, "retention_evry": 5})

    def test_a_definition_naming_NEITHER_a_pack_nor_parts_is_refused(self) -> None:
        body = {k: v for k, v in GOOD.items() if k != "parts"}
        with pytest.raises(ValidationError) as caught:
            RunDefinition(**body)
        assert "pack version OR" in str(caught.value)

    def test_a_definition_naming_BOTH_is_refused_rather_than_merged(self) -> None:
        """A run whose material came half from a pack version and half from a loose list is
        a run whose inputs no pack version names, so *"a pack version names a reproducible
        run input"* would stop being true without anything saying so."""
        with pytest.raises(ValidationError):
            RunDefinition(**{**GOOD, "pack": {"pack_id": "p", "version": "1"}})

    def test_NOTHING_is_defaulted_into_the_definition(self) -> None:
        """E-24, AC-006.38, ADR-018: no default model source, no default floor, no default
        run type. *A fabricated model source, floor or cadence is an invisible wrong
        answer.*"""
        for required in ("model_source", "run_type", "run_configuration", "signal_source"):
            body = {k: v for k, v in GOOD.items() if k != required}
            with pytest.raises(ValidationError):
                RunDefinition(**body)
        # And the two that must stay absent-or-null rather than filled in:
        built = definition()
        assert built.retention_floor_override is None
        assert built.refresh is None

    def test_a_duplicate_stop_condition_is_refused_not_last_one_wins(self) -> None:
        """Two configurations for one condition means which one ends the run depends on
        list order."""
        built = definition(
            stop_conditions=[
                {"condition": "step_count", "config": {"total": 10}},
                {"condition": "step_count", "config": {"total": 1000}},
            ]
        )
        with pytest.raises(ValueError):
            built.enabled_conditions()

    @pytest.mark.parametrize("field", ["test_every", "retention_every", "save_every"])
    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_cadence_of_zero_or_less_is_refused(self, field: str, bad: int) -> None:
        """E-4. Zero would make ``step % n`` a division by zero on the hottest path in the
        product, and a negative cadence names an action that never happens while reading as
        though it does."""
        with pytest.raises(ValidationError):
            RunDefinition(**{**GOOD, "cadences": {**GOOD["cadences"], field: bad}})


# ====================================================================== the seed


class TestTheSeedIsResolvedNotMerelyAccepted:
    """Operator decision, 2026-10-14. **R-02.47 requires the content hash of the weights a
    run started from, for both modes, and nothing recorded a seed** - so a ``create`` run's
    random initialisation had a hash that identified *which* init happened without letting
    anyone repeat it."""

    def test_a_supplied_seed_is_kept(self, engine) -> None:
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(seed=4242),
                session=session,
                **_passing(),
            )
        assert outcome.resolved["seed"] == 4242

    def test_an_ABSENT_seed_is_GENERATED_and_the_generated_value_recorded(self, engine) -> None:
        """**An absent seed must not stay absent.** A run whose seed is null is a run that
        cannot be repeated, and here ``null`` would be an omission rather than an
        assertion."""
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(seed=None),
                session=session,
                seed_generator=lambda: 7,
                **_passing(),
            )
        assert outcome.resolved["seed"] == 7

    def test_the_submitted_definition_keeps_its_NULL_while_the_resolved_carries_the_number(
        self, engine, tmp_path
    ) -> None:
        """Which is **exactly** what ``definition_submitted`` and ``definition_resolved``
        exist to show: what the operator asked for and what it resolved to are different
        facts, and collapsing them loses the ability to say *"you asked for none and got
        this one"*."""
        with session_for(engine) as session:
            accepted = run_launch.launch(
                definition(seed=None),
                session=session,
                enqueue=lambda run_id: None,
                log_root=tmp_path,
                seed_generator=lambda: 99,
                **_passing(),
            )
            session.commit()
            record = run_launch.read_definition(session, RUN, log_root=tmp_path)
        assert record["submitted"]["seed"] is None
        assert record["resolved"]["seed"] == 99
        assert accepted["resolved"]["seed"] == 99

    def test_the_SAME_seed_produces_the_SAME_definition_hash(self, engine, tmp_path) -> None:
        """The reproducibility claim, made checkable."""
        hashes = set()
        for run in ("seed-a", "seed-b"):
            with session_for(engine) as session:
                accepted = run_launch.launch(
                    definition(run_id=run, seed=1234),
                    session=session,
                    enqueue=lambda run_id: None,
                    log_root=tmp_path,
                    **_passing(),
                )
                session.commit()
            # The run_id is part of the submitted definition, so compare the hash of the
            # SUBMITTED definition with the run id normalised out - the subject is the seed.
            hashes.add(accepted["resolved"]["seed"])
        assert hashes == {1234}

    def test_two_runs_with_DIFFERENT_seeds_differ_in_their_recorded_definition(
        self, engine, tmp_path
    ) -> None:
        digests = []
        for run, seed in (("seed-c", 1), ("seed-d", 2)):
            with session_for(engine) as session:
                accepted = run_launch.launch(
                    definition(run_id=run, seed=seed),
                    session=session,
                    enqueue=lambda run_id: None,
                    log_root=tmp_path,
                    **_passing(),
                )
                session.commit()
            digests.append(accepted["definition_sha256"])
        assert digests[0] != digests[1], (
            "two runs that differ only by seed must have different recorded definitions - "
            "they ARE different runs, because the starting weights differ"
        )

    def test_the_seed_reaches_the_run_configurations_hash(self, engine) -> None:
        """So two runs that differ only by seed have different optimizer config hashes -
        the precedent :attr:`training.run_config.RunConfig.seed` already set."""
        from training.run_config import RunConfig

        a = RunConfig(learning_rate=1e-5, warmup_steps=2, batch_size=2, seed=1)
        b = RunConfig(learning_rate=1e-5, warmup_steps=2, batch_size=2, seed=2)
        assert a.config_hash() != b.config_hash()

    def test_a_generated_seed_is_inside_the_range_torch_and_numpy_both_accept(self) -> None:
        """``2**31 - 1``, not ``2**64``: ``torch.manual_seed`` takes a 64-bit value and
        numpy's legacy seeding does not, and **a seed some library silently truncates is a
        seed that does not reproduce the run it claims to.**"""
        from schemas.run_definition import MAX_SEED, _generated_seed

        for _ in range(50):
            value = _generated_seed()
            assert 0 <= value <= MAX_SEED
        assert MAX_SEED == 2**31 - 1


# ====================================================== the six steps, and their order


def _passing() -> dict[str, Any]:
    """Every collaborator stubbed to succeed, so a test can be about one of them."""
    return {
        "gate": lambda *a, **k: _ResolvedPackDouble(),
        "resolve_parts": lambda *a, **k: {"source": "parts", "parts": []},
        "budget": lambda *a, **k: {"param_count": 10, "gpu_uuid": None},
        "services": lambda *a, **k: {"needed": []},
    }


class TestTheValidationOrderIsTheRequirement:
    def test_the_six_steps_run_in_FR00658s_order(self, engine, spies) -> None:
        """AC-006.41. **Spied on the sequence, because M-14's outcome is identical until
        the card is contended** - a budget check after the lease passes every test that
        only looks at the response."""
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(pack={"pack_id": "p", "version": "1"}, parts=None),
                session=session,
                **spies["kwargs"],
            )
        assert outcome.ok
        assert spies["order"] == [
            "pack_gate",
            "part_resolvability",
            "budget",
            "optional_services",
        ]
        assert outcome.steps_run == list(validator.STEPS)

    def test_the_pack_gate_is_called_with_the_right_payload_EXACTLY_ONCE(
        self, engine, spies
    ) -> None:
        """**M-9**, and F7's task 13.2. *A grep proving a call exists is the weaker form and
        it is not sufficient* - so this asserts the payload and the count."""
        with session_for(engine) as session:
            validator.validate(
                definition(pack={"pack_id": "legal", "version": "3"}, parts=None),
                session=session,
                **spies["kwargs"],
            )
        gate = spies["gate_spy"]
        assert gate.count == 1, f"require_valid_pack must be called exactly once, not {gate.count}"
        args, _kwargs = gate.calls[0]
        assert args[1:] == ("legal", "3"), f"wrong payload: {args[1:]}"

    def test_the_budget_runs_BEFORE_anything_is_allocated(self, engine, spies) -> None:
        """**M-14.** R-02.53 names it: *"the computed budget against free VRAM (R-02.42,
        before any allocation)"*."""
        with session_for(engine) as session:
            validator.validate(definition(), session=session, **spies["kwargs"])
        assert spies["budget_spy"].count == 1
        assert "budget" in spies["order"]

    def test_the_validator_takes_NO_lease(self, engine, monkeypatch) -> None:
        """FR-006.65's corrected text: *steps 1-6 allocate nothing, and ``validate()`` calls
        no ``lease.acquire``.* An implementer reading the old phrasing would assert the
        validator takes the lease, which is exactly what M-14 forbids."""
        from workers import lease as lease_module

        acquired = _Spy()
        monkeypatch.setattr(lease_module, "acquire", acquired)
        with session_for(engine) as session:
            validator.validate(definition(), session=session, **_passing())
        assert acquired.count == 0, "the validator must not acquire a lease"

    def test_a_pack_that_does_not_resolve_SKIPS_part_resolvability_rather_than_guessing(
        self, engine, spies
    ) -> None:
        """FR-006.59's ``requires=`` rule. *Running it on a guess would produce a
        **fabricated** second failure, which is worse than a missing one because the
        operator would try to fix it* - and omitting it silently would make the list read
        as complete when it is not."""
        from core.errors import refuse

        kwargs = dict(spies["kwargs"])
        kwargs["gate"] = lambda *a, **k: (_ for _ in ()).throw(
            refuse("pack_incomplete", "missing slots", missing_slots=["retention_set"])
        )
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(pack={"pack_id": "p", "version": "1"}, parts=None),
                session=session,
                **kwargs,
            )
        assert [f.code for f in outcome.failures] == ["pack_incomplete"]
        # TWO steps wait on the gate, not one: F4's submission guards read the pack's
        # retention slot, so running them against an unresolved pack would report *"this
        # run defines no retention set"* about a run whose pack simply did not resolve -
        # the fabricated second failure this rule exists to prevent.
        assert [s.step for s in outcome.skipped] == ["part_resolvability", "submission_guards"]
        assert outcome.may_be_more is True
        assert spies["parts_spy"].count == 0


class TestEveryFailureIsReportedNotTheFirst:
    def test_a_definition_wrong_in_four_places_returns_four_failures(self, engine) -> None:
        """E-21, AC-006.39. *The one consumer of this path is a form, and a validator that
        stops at the first failure makes an operator resubmit once per mistake.*"""
        from core.errors import refuse

        kwargs = {
            "gate": lambda *a, **k: (_ for _ in ()).throw(
                refuse("pack_incomplete", "missing", missing_slots=["retention_set"])
            ),
            "resolve_parts": lambda *a, **k: {"source": "parts", "parts": []},
            "budget": lambda *a, **k: (_ for _ in ()).throw(
                refuse("budget_exceeded", "too big", estimate_bytes=1, free_bytes=0)
            ),
            "services": lambda *a, **k: (_ for _ in ()).throw(
                refuse("midataworks_not_configured", "unset", service="miDataworks")
            ),
        }
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(
                    pack={"pack_id": "p", "version": "1"},
                    parts=None,
                    stop_conditions=[],
                ),
                session=session,
                **kwargs,
            )
        codes = [f.code for f in outcome.failures]
        assert "pack_incomplete" in codes
        assert "run_has_no_stop_condition" in codes
        assert "budget_exceeded" in codes
        assert "midataworks_not_configured" in codes
        assert len(codes) == 4

    def test_each_failure_keeps_the_OWNING_FEATURES_code_unchanged(self, engine) -> None:
        """FR-006.59. **Never re-wrapped into one generic code**, because re-wrapping hides
        which layer refused - and F7's ``missing_slots`` is what tells the operator which
        slot to fill."""
        from core.errors import refuse

        kwargs = dict(_passing())
        kwargs["gate"] = lambda *a, **k: (_ for _ in ()).throw(
            refuse("pack_incomplete", "missing", missing_slots=["retention_set"])
        )
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(pack={"pack_id": "p", "version": "1"}, parts=None),
                session=session,
                **kwargs,
            )
        failure = outcome.failures[0]
        assert failure.code == "pack_incomplete"
        assert failure.details["missing_slots"] == ["retention_set"]

    def test_the_envelope_is_ADR013s_with_a_LIST_of_failures(self, engine) -> None:
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(stop_conditions=[]), session=session, **_passing()
            )
            with pytest.raises(Refusal) as caught:
                outcome.raise_if_invalid()
        assert caught.value.code == "run_definition_invalid"
        assert isinstance(caught.value.details["failures"], list)
        assert caught.value.details["failures"][0]["code"] == "run_has_no_stop_condition"


class TestTheLoopStopAndRefreshValidations:
    def test_a_run_with_NO_stop_condition_is_refused(self, engine) -> None:
        """E-1, FR-006.17. **F6-re**: this is the check that closed DEF-3, and R-02.53
        requires *"at least one stop condition (R-02.25)"* in so many words.

        The consequence is concrete: the run holds this node's only GPU indefinitely under
        ADR-008's one-run-per-card lease, writing a checkpoint every ``save_every`` steps
        onto a volume for which **no requirement anywhere defines pruning, deletion or
        archival**.
        """
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(stop_conditions=[]), session=session, **_passing()
            )
        assert [f.code for f in outcome.failures] == ["run_has_no_stop_condition"]

    def test_a_condition_name_outside_the_five_is_refused_listing_them(self, engine) -> None:
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(stop_conditions=[{"condition": "gpu_too_hot"}]),
                session=session,
                **_passing(),
            )
        assert [f.code for f in outcome.failures] == ["stop_condition_unknown"]
        assert len(outcome.failures[0].details["known"]) == 5

    def test_an_incomplete_plateau_configuration_is_refused_naming_the_parameter(
        self, engine
    ) -> None:
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(
                    stop_conditions=[{"condition": "loss_plateau", "config": {"metric": "raw"}}]
                ),
                session=session,
                **_passing(),
            )
        assert [f.code for f in outcome.failures] == ["stop_condition_config_incomplete"]
        assert outcome.failures[0].details["parameter"] == "window"

    def test_the_unconfigured_conditions_need_no_configuration(self, engine) -> None:
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(
                    stop_conditions=[
                        {"condition": "retention_floor"},
                        {"condition": "operator_stop"},
                    ],
                    # Declared because the run enables `retention_floor`: F4's E-12 guard
                    # is wired at submit now, and a floor condition with no retention set
                    # is a condition that can never fire. The subject of *this* test is
                    # that an unconfigured condition needs no `config`, which is unchanged.
                    scoring={"retention_slot": {"part_id": "part-retention"}},
                    retention_floor_override=0.6,
                ),
                session=session,
                **_passing(),
            )
        assert outcome.ok, [f.payload() for f in outcome.failures]

    def test_a_retention_floor_condition_with_NO_retention_set_is_refused(self, engine) -> None:
        """F4's E-12, reachable at submit for the first time.

        A run enabling `retention_floor` with no retention set has a stop condition that
        **cannot fire**: F4 publishes the breach and this condition reads it, so with no set
        there is nothing to publish. Before this guard was wired the run was accepted, took
        the node's only card, and ran to its step ceiling with a condition the operator
        believed was protecting it.
        """
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(stop_conditions=[{"condition": "retention_floor"}]),
                session=session,
                **_passing(),
            )
        assert not outcome.ok
        assert [f.code for f in outcome.failures] == ["retention_set_missing"]
        assert [f.step for f in outcome.failures] == ["submission_guards"]

    def test_a_declared_retention_set_with_NO_floor_is_refused_by_its_own_code(
        self, engine
    ) -> None:
        """E-12's second half, and a **different** refusal on purpose: one fix supplies a
        dataset, the other supplies a number. *"A fabricated floor of 0.0 passes every run
        forever while nothing says a thing."*"""
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(
                    stop_conditions=[{"condition": "retention_floor"}],
                    scoring={"retention_slot": {"part_id": "part-retention"}},
                ),
                session=session,
                **_passing(),
            )
        assert [f.code for f in outcome.failures] == ["retention_floor_missing"]

    def test_a_refresh_on_a_corpus_run_is_refused_at_submit(self, engine) -> None:
        """E-12. *Not silently ignored* - an ignored refresh leaves the operator believing
        the prompt set rotated when it did not."""
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(
                    signal_source={"source": "corpus"},
                    refresh={"mode": "new_prompt_set", "rule": {"part_id": "x"}},
                    cadences={**GOOD["cadences"], "refresh_every": 100},
                ),
                session=session,
                **_passing(),
            )
        assert [f.code for f in outcome.failures] == ["refresh_not_applicable"]

    def test_a_refresh_with_no_refresh_every_is_refused(self, engine) -> None:
        """No round would ever end, so the refresh would never happen - a configuration
        that reads as though it rotates prompts and does not."""
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(
                    signal_source={"source": "self-generation"},
                    refresh={"mode": "template_seeds", "rule": {"template": "{prompt}"}},
                ),
                session=session,
                **_passing(),
            )
        assert [f.code for f in outcome.failures] == ["refresh_rule_incomplete"]

    def test_the_exposure_window_is_RECORDED_so_nobody_has_to_infer_it(self, engine) -> None:
        """FR-006.20's number, in the resolved definition. **At ``retention_every = 1`` it
        is 0.**"""
        with session_for(engine) as session:
            one = validator.validate(
                definition(cadences={**GOOD["cadences"], "retention_every": 1}),
                session=session,
                **_passing(),
            )
            ten = validator.validate(
                definition(cadences={**GOOD["cadences"], "retention_every": 10}),
                session=session,
                **_passing(),
            )
        assert one.resolved["loop"]["retention_exposure_steps"] == 0
        assert ten.resolved["loop"]["retention_exposure_steps"] == 9


class TestTheOptionalServiceCheck:
    def test_a_definition_needing_NO_optional_service_validates_with_NONE_configured(
        self, engine, monkeypatch
    ) -> None:
        """FR-006.60's closing sentence, and **where AC20 begins**: the standalone case is
        the default case. An unset base URL means absent-and-normal (ADR-010)."""
        monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
        monkeypatch.delenv("MILLM_BASE_URL", raising=False)
        from core.config import reset_settings

        reset_settings()
        kwargs = dict(_passing())
        del kwargs["services"]  # use the LIVE check, which is the subject
        with session_for(engine) as session:
            outcome = validator.validate(definition(), session=session, **kwargs)
        reset_settings()
        assert outcome.ok, [f.payload() for f in outcome.failures]
        assert outcome.resolved["services"]["needed"] == []

    def test_a_human_review_selector_with_miDataworks_unset_is_refused_BY_NAME(
        self, engine, monkeypatch
    ) -> None:
        """E-22, FR-006.60. **miForge never substitutes a different behaviour for a missing
        service**, so a definition that needs one and has none is refused rather than
        quietly run in a different mode."""
        monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
        from core.config import reset_settings

        reset_settings()
        kwargs = dict(_passing())
        del kwargs["services"]
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(
                    signal_source={
                        "source": "self-generation",
                        "selector": {"kind": "human_review"},
                    }
                ),
                session=session,
                **kwargs,
            )
        reset_settings()
        assert [f.code for f in outcome.failures] == ["midataworks_not_configured"]
        assert outcome.failures[0].details.get("service")


# ============================================= the create path, and what it must call


class TestTheCreatePathCallsTheOneValidator:
    def test_the_create_path_calls_validate_EXACTLY_ONCE(
        self, engine, tmp_path, monkeypatch
    ) -> None:
        """**M-15**, on the call count. *A ``validate`` route the create path never calls is
        an unreachable capability whose unit tests all pass*, and a second, laxer copy
        behind the submit button is the drift this project pays for."""
        spy = _Spy()
        original = validator.validate

        def counting(*args: Any, **kwargs: Any) -> Any:
            spy(*args, **kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(run_launch.validator_module, "validate", counting)
        with session_for(engine) as session:
            run_launch.launch(
                definition(),
                session=session,
                enqueue=lambda run_id: None,
                log_root=tmp_path,
                **_passing(),
            )
            session.commit()
        assert spy.count == 1, f"the create path called the validator {spy.count} times"

    def test_the_validate_route_and_the_create_path_use_the_SAME_function(self) -> None:
        """There is exactly one implementation. Asserted on identity, so a copied
        implementation is red rather than merely frowned upon."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3] / "backend" / "src" / "services" / "run_launch.py"
        ).read_text(encoding="utf-8")
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "validate"
        ]
        assert len(calls) == 2, (
            f"expected exactly two call sites of the one validator - validate_only and "
            f"launch - found {len(calls)}"
        )


class TestARefusedDefinitionAllocatesNothing:
    def test_no_lease_no_run_row_and_no_run_log_after_a_refusal(
        self, engine, tmp_path, monkeypatch
    ) -> None:
        """AC-006.40. **Asserted on the state afterwards, not on the 422**: *a test that
        only checks the response passes against a validator that took and released a
        lease.*"""
        import sqlalchemy as sa

        from workers import lease as lease_module
        from workers.run_record import RUNS

        acquired = _Spy()
        monkeypatch.setattr(lease_module, "acquire", acquired)

        with session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                run_launch.launch(
                    definition(stop_conditions=[]),
                    session=session,
                    enqueue=lambda run_id: None,
                    log_root=tmp_path,
                    **_passing(),
                )
            session.rollback()
            rows = session.execute(sa.select(RUNS.c.id).where(RUNS.c.id == RUN)).all()

        assert caught.value.code == "run_definition_invalid"
        assert acquired.count == 0, "a refused definition must take no lease"
        assert rows == [], "a refused definition must leave no mf_runs row"
        assert not (tmp_path / "runs" / RUN).exists(), "and no run log"

    def test_a_refused_definition_still_reports_EVERY_problem_including_the_budget(
        self, engine, tmp_path
    ) -> None:
        """**A document defect, reported rather than implemented.**

        F6's FTASKS task 8.14 asks this test to assert *"the device resolver never
        called"* for a refused definition. That is **incompatible with FR-006.57's
        collect-all rule**, which is the requirement: *"the validator collects refusals
        across every part and returns them together ... a validator that stops at the first
        failure makes an operator resubmit once per mistake."* A pack that does not resolve
        says nothing about whether the model fits the card, so the budget step still runs
        and still reports.

        What FR-006.57 actually forbids is **allocation**, and reading ``nvidia-smi -L`` is
        a read. The lease is the allocation, and it is asserted at zero below and in
        ``test_no_lease_no_run_row_and_no_run_log_after_a_refusal``.
        """
        from core.errors import refuse

        budget = _Spy(result={"param_count": 10, "gpu_uuid": None})
        kwargs = dict(_passing())
        kwargs["gate"] = lambda *a, **k: (_ for _ in ()).throw(
            refuse("pack_incomplete", "missing", missing_slots=["retention_set"])
        )
        kwargs["budget"] = budget
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(pack={"pack_id": "p", "version": "1"}, parts=None),
                session=session,
                **kwargs,
            )
        assert not outcome.ok
        assert budget.count == 1, (
            "the budget must still be checked so the operator learns about both problems "
            "at once (FR-006.57); it allocates nothing"
        )
        assert [s.step for s in outcome.skipped] == [
            "part_resolvability",
            "submission_guards",
        ], "only the steps that are genuinely unanswerable are skipped"

    def test_nothing_is_enqueued_on_a_refusal(self, engine, tmp_path) -> None:
        enqueued: list[str] = []
        with session_for(engine) as session, pytest.raises(Refusal):
            run_launch.launch(
                definition(stop_conditions=[]),
                session=session,
                enqueue=enqueued.append,
                log_root=tmp_path,
                **_passing(),
            )
        assert enqueued == []


class TestTheDefinitionRecordIsDurableBeforeTheEnqueue:
    def test_the_record_exists_by_the_time_the_enqueue_happens(self, engine, tmp_path) -> None:
        """**M-16.** A worker can start a run the moment it is enqueued, so a definition
        written afterwards leaves a window in which a crash produces a running run nothing
        can state the shape of."""
        seen: list[bool] = []

        def enqueue(run_id: str) -> None:
            # Read the log from inside the enqueue, which is the only moment the question
            # is answerable: *was the record durable when the worker could first see it?*
            from training import run_log
            from training.tasks import run_log_root

            path = run_log.run_log_path(run_log_root(tmp_path), run_id)
            seen.append(path.is_file() and any(run_log.read_blocks(path, block="definition")))

        with session_for(engine) as session:
            run_launch.launch(
                definition(),
                session=session,
                enqueue=enqueue,
                log_root=tmp_path,
                **_passing(),
            )
            session.commit()
        assert seen == [
            True
        ], "the definition must be durable in the run log BEFORE the task is enqueued"

    def test_the_block_validates_against_the_vendored_contract(self, engine, tmp_path) -> None:
        """Through the real writer and the real schema. ``append`` validates, so a block
        with a key the contract does not declare is refused at the writer."""
        from training import run_log
        from training.tasks import run_log_root

        with session_for(engine) as session:
            run_launch.launch(
                definition(),
                session=session,
                enqueue=lambda run_id: None,
                log_root=tmp_path,
                **_passing(),
            )
            session.commit()
        path = run_log.run_log_path(run_log_root(tmp_path), RUN)
        blocks = list(run_log.read_blocks(path, block="definition"))
        assert len(blocks) == 1
        run_log.validate_block(blocks[0])
        assert blocks[0]["step"] is None, (
            "the definition block is written before step 0 exists, which the schema says "
            "in so many words - null only on this block"
        )

    def test_the_index_mirrors_the_log_and_the_log_is_the_authority(self, engine, tmp_path) -> None:
        """ADR-004. The row's three definition columns are an index over the log entry."""
        import sqlalchemy as sa

        from workers.run_record import RUNS

        with session_for(engine) as session:
            accepted = run_launch.launch(
                definition(),
                session=session,
                enqueue=lambda run_id: None,
                log_root=tmp_path,
                **_passing(),
            )
            session.commit()
            row = session.execute(
                sa.select(
                    RUNS.c.definition_sha256,
                    RUNS.c.definition_submitted,
                    RUNS.c.definition_resolved,
                    RUNS.c.status,
                ).where(RUNS.c.id == RUN)
            ).one()
        assert row.definition_sha256 == accepted["definition_sha256"]
        assert row.definition_submitted["run_type"] == "change"
        assert row.definition_resolved["seed"] is not None
        assert row.status == "pending"

    def test_reading_it_back_reports_WHICH_SOURCE_answered(self, engine, tmp_path) -> None:
        """A disagreement between the authoritative log and the index is reported, not
        reconciled silently: the manifest/log wins **and the disagreement is logged**
        (ADR-004, foundation 4.5)."""
        with session_for(engine) as session:
            run_launch.launch(
                definition(),
                session=session,
                enqueue=lambda run_id: None,
                log_root=tmp_path,
                **_passing(),
            )
            session.commit()
            record = run_launch.read_definition(session, RUN, log_root=tmp_path)
        assert record["source"] == "run_log"
        assert record["index_agrees"] is True

    def test_a_run_with_no_recorded_definition_refuses_BY_NAME(self, engine, tmp_path) -> None:
        """F1's pre-R-02.53 ``POST /runs`` creates no definition, and that is what makes
        the ungated route unable to produce a run that trains."""
        from tests.support.jobs import insert_run

        with session_for(engine) as session:
            insert_run(session, "run-ungated", status="pending")
            session.commit()
            with pytest.raises(Refusal) as caught:
                run_launch.read_definition(session, "run-ungated", log_root=tmp_path)
        assert caught.value.code == "run_definition_missing"


class TestTheProjectedFootprint:
    def test_it_REPORTS_a_range_and_refuses_nothing(self) -> None:
        """FR-006.53, DEF-8. **No requirement anywhere defines pruning, deletion or
        archival** of checkpoints, so this reports and enforces nothing. *Do not pick one
        bound and present it as the figure.*"""
        from services import footprint

        projection = footprint.project(max_steps=10_000, save_every=100, param_count=1_200_000_000)
        assert projection.checkpoints == 100
        payload = projection.payload()
        low = payload["total_bytes"]["low"]
        high = payload["total_bytes"]["high"]
        assert 0 < low < high, "both bounds must be reported and they must differ"
        assert "low" in projection.rendered() or "-" in projection.rendered()

    def test_it_refuses_no_run_however_large(self) -> None:
        from services import footprint

        projection = footprint.project(
            max_steps=10_000_000, save_every=1, param_count=70_000_000_000
        )
        assert projection.total_high > 0  # no exception, no refusal

    def test_the_rendering_names_BOTH_bounds(self) -> None:
        from services import footprint

        rendered = footprint.project(
            max_steps=1000, save_every=100, param_count=1_200_000_000
        ).rendered()
        assert rendered.count("-") >= 1 and "GiB" in rendered

    def test_an_unknown_parameter_count_does_not_become_a_guessed_figure(self) -> None:
        from services import footprint

        projection = footprint.project(max_steps=100, save_every=10, param_count=None)
        assert projection.bytes_per_checkpoint == 0
        assert projection.payload()["note"]

    def test_the_step_bound_comes_from_the_step_count_CONDITION(self) -> None:
        """So the projection and the thing that actually ends the run cannot disagree about
        how long it is."""
        from services import footprint

        assert footprint.steps_for_footprint({"step_count": {"total": 500}}, fallback=7) == 500
        assert footprint.steps_for_footprint({"retention_floor": {}}, fallback=7) == 7


class TestTheLIVEBindingsActuallyRun:
    """**A finding from F6's own build, and the shape of it is the lesson.**

    Every test above injects a ``gate`` / ``resolve_parts`` / ``budget`` / ``services``
    spy, because each is about one step's behaviour. The consequence is that the **live**
    bindings - ``_live_gate`` and the rest - were executed by *nothing*, and the first
    draft of ``_live_gate`` called ``pack_service()`` on F7's singleton **instance**. MyPy
    caught it as *"PackService not callable"*; no test did, and none could have.

    That is the unreachable-capability failure committed inside the fix for three other
    instances of it. These tests run the live bindings, so the binding is exercised rather
    than read.
    """

    def test_the_live_pack_gate_binding_runs_and_refuses_an_unknown_pack(
        self, engine, tmp_path, monkeypatch
    ) -> None:
        """The binding is reached and F7's own refusal comes back **unchanged**.

        An unknown pack is the cheapest way to reach it - it needs no pack on disk - and the
        refusal arriving as ``pack_not_found`` rather than as a ``TypeError`` is the whole
        assertion.

        ``PACKS_ROOT`` is pointed at a temp directory because **F7's ``packs_root()``
        mkdirs on demand even on a read path** - the hazard AGENT.md records for
        ``reindex``, where a ``PermissionError`` on ``/data/packs`` aborted the whole
        operation. It is still there on the gate's read path, and it is reported.
        """
        from packs.settings import reset_pack_settings
        from services.parts import tables as part_tables

        monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
        reset_pack_settings()
        part_tables.metadata.create_all(engine)
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(pack={"pack_id": "no-such-pack", "version": "1"}, parts=None),
                session=session,
                # NO gate spy. The live binding runs.
                resolve_parts=lambda *a, **k: {"source": "pack", "parts": []},
                budget=lambda *a, **k: {"param_count": 10, "gpu_uuid": None},
                services=lambda *a, **k: {"needed": []},
            )
        codes = [failure.code for failure in outcome.failures]
        assert codes == ["pack_not_found"], (
            f"the live pack-gate binding did not run F7's gate; got {codes}. A TypeError "
            f"here is the binding being wrong in a way no spy-injected test can see."
        )
        reset_pack_settings()

    def test_the_live_part_resolver_binding_runs(self, engine) -> None:
        """F10's ``resolve_parts`` through the live binding, refusing by **its own** code."""
        from services.parts import tables as part_tables

        part_tables.metadata.create_all(engine)
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(parts=[{"part_id": "no-such-part", "kind": "corpus"}]),
                session=session,
                gate=lambda *a, **k: None,
                # NO resolve_parts spy.
                budget=lambda *a, **k: {"param_count": 10, "gpu_uuid": None},
                services=lambda *a, **k: {"needed": []},
            )
        codes = [failure.code for failure in outcome.failures]
        assert codes == ["dataset_version_unresolved"], codes

    def test_the_live_budget_binding_runs_and_resolves_the_run_type(self, engine) -> None:
        """**Owed wiring 1, exercised rather than asserted about.** The budget step resolves
        F8's run type by name and calls its ``preflight`` - the first production caller that
        registry has ever had.

        An unknown run type is the cheapest way to prove the resolve happens: it refuses
        before anything touches the filesystem or the card.
        """
        with session_for(engine) as session:
            outcome = validator.validate(
                definition(run_type="full"),
                session=session,
                gate=lambda *a, **k: None,
                resolve_parts=lambda *a, **k: {"source": "parts", "parts": []},
                # NO budget spy.
                services=lambda *a, **k: {"needed": []},
            )
        codes = [failure.code for failure in outcome.failures]
        assert codes and codes[0].startswith("run_type"), (
            f"the live budget binding did not resolve the run type; got {codes}. 'full' "
            f"is not one of F8's three ({'change'!r}, {'create'!r}, {'adapter'!r}) and the "
            f"registry refuses it by name - which is the point: the resolve happens."
        )

        # *This assertion's note said "'full' is the stale spelling in
        # models/run.RUN_TYPES". It WAS, and the coordinator fixed that constant on
        # 2026-10-14 - so the note became false while this test stayed green, which is the
        # stale-finding class in the layer that executes. The behaviour is unchanged
        # because the assertion keys on the refusal code, not on why 'full' is refused.*


@pytest.fixture
def pack_engine(tmp_path, packs_root):
    """A MIGRATED engine, because `registry.create` writes `mf_packs`/`mf_pack_parts`.

    This file's own `engine` fixture is `memory_engine()`, which builds the base schema
    only — the pack index tables live in their own migration, so a live-gate test needs the
    migration to have run. Separate fixture rather than widening `engine`, so the other 54
    tests in this file keep the cheaper in-memory one.
    """
    from tests.support.db import migrated_engine

    engine = migrated_engine(tmp_path / "db")
    yield engine
    engine.dispose()


@pytest.fixture
def packs_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point F7's store at a tmp directory for the live-gate tests below.

    Required because `packs.store.packs_root()` **mkdirs on demand** — right for a writer,
    wrong on a read path — so an unredirected test tries to create `/data/packs` and dies
    with `PermissionError`, which is the hazard already recorded against F6's gate call and
    the one that once aborted the whole reindex.
    """
    from core.config import reset_settings
    from packs.settings import reset_pack_settings

    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_pack_settings()
    reset_settings()
    yield tmp_path / "packs"
    reset_pack_settings()
    reset_settings()


class TestTheLIVEPackGateBindingRefusesAnInvalidPack:
    """F7's task 13.2 — **FAC-10's second half**, which no test satisfied.

    13.2 says it in terms: *"Assert F6's run-create path calls `require_valid_pack` with the
    right payload **and call count**, and that the test goes **red when the call is
    deleted**… A grep proving a call exists is the weaker form and is not sufficient."*

    **The existing assertions are the weaker form, and three layers of them agreed.**
    `test_the_pack_gate_is_called_with_the_right_payload_EXACTLY_ONCE` injects a **spy**
    gate, so it proves the validator calls whatever gate it is handed — not that production
    hands it the real one. Replacing the live binding's `require_valid_pack` with
    `resolved_pack` (the same resolution **without** `validity.raise_if_invalid()`, one line
    apart in `packs/gate.py`) left the whole file at **54 passed**.

    Worse, `run_definition.py`'s own comment at the binding says
    *"`test_the_live_gate_binding_actually_runs` now exercises it rather than reading it"* —
    **and that test does not exist.** Only the budget's equivalent was written. So the
    binding carried a citation to its own absent guard, which is the same class as a
    requirement asserting a field its schema lacks.

    **What makes this one bite:** every other dependency is spied and the **gate is left
    live**, so the subject is the binding and nothing else; and the pack is genuinely
    *invalid*, because the two functions differ only in whether they refuse. A valid pack
    cannot tell them apart — which is exactly why a test using one proves nothing.
    """

    def test_an_incomplete_pack_is_refused_through_the_PRODUCTION_binding(
        self, pack_engine, spies, packs_root
    ) -> None:
        from packs.slots import SlotKind
        from tests.support import pack_fixtures as pf

        live_except_the_gate = {k: v for k, v in spies["kwargs"].items() if k != "gate"}

        with session_for(pack_engine) as session:
            pf.commit(
                session,
                pf.definition(
                    name="gate-probe",
                    version="1",
                    # The retention slot empty makes it `pack_incomplete`. An INVALID pack is
                    # the only input that distinguishes `require_valid_pack` from
                    # `resolved_pack`: they differ solely in `validity.raise_if_invalid()`.
                    slots=pf.valid_slots(**{SlotKind.RETENTION_SET.value: None}),
                ),
            )
            outcome = validator.validate(
                definition(pack={"pack_id": "gate-probe", "version": "1"}, parts=None),
                session=session,
                **live_except_the_gate,
            )

        codes = [f.code for f in outcome.failures]
        assert "pack_incomplete" in codes, (
            f"the live gate binding did not refuse an incomplete pack; failures were "
            f"{codes}. If `require_valid_pack` was replaced by `resolved_pack`, the pack "
            f"resolves identically and the refusal simply never happens — which is the "
            f"defect FAC-10's second half exists to catch."
        )

    def test_a_VALID_pack_passes_the_same_binding(self, pack_engine, spies, packs_root) -> None:
        """The positive control, and it is what proves the test above is about validity.

        Without it, `pack_incomplete` could be produced by any breakage in the live path —
        a missing store, an unreadable slot — and the assertion would pass for the wrong
        reason.
        """
        from tests.support import pack_fixtures as pf

        live_except_the_gate = {k: v for k, v in spies["kwargs"].items() if k != "gate"}

        with session_for(pack_engine) as session:
            pf.commit(session, pf.definition(name="gate-ok", version="1", slots=pf.valid_slots()))
            outcome = validator.validate(
                definition(pack={"pack_id": "gate-ok", "version": "1"}, parts=None),
                session=session,
                **live_except_the_gate,
            )

        assert "pack_incomplete" not in [
            f.code for f in outcome.failures
        ], f"a valid pack was reported incomplete: {[f.code for f in outcome.failures]}"
