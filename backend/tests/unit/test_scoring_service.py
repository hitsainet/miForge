"""One pass, one implementation, no partial anything  (FR-004.7, 14a, 18, 21, 24; M-6, M-9).

The four properties this file exists for:

* **the paired measurement is one call** - test and retention at one step share the step, the
  provider and the decode parameters by construction, not by two calls being scheduled to
  coincide (FR-004.24);
* **no partial pass is ever recorded as a set score** (M-6) - the "robustness" change that
  silently reports a confident number over an unknown subset;
* **no provider fallback** (M-9) - a silent substitution rewrites the curve's meaning mid-run;
* **a cancelled pass records no set score** (E-18) - a prompt is scored and recorded, or not
  started.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from core.errors import Refusal
from providers.base import GenerationParams
from providers.fakes import FakeModelHandle, FakeProvider
from providers.in_process import InProcessProvider
from services.scoring import expectations, retention, service, sets
from services.scoring.tables import SCORES
from tests.support import scores as fixtures
from tests.support.db import migrated_engine, session_for

RULE = {
    "kind": "rule",
    "name": "len",
    "config_sha256": None,
    "expression": '{"min_length": 3}',
}


@pytest.fixture
def engine(tmp_path: Path):
    engine = migrated_engine(tmp_path / "db")
    fixtures.create_score_tables(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with session_for(engine) as session:
        yield session


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "run-1" / "run.jsonl"


def a_set(kind: str, count: int = 4) -> sets.ResolvedSet:
    reference = sets.SetReference(
        rendered=f"{kind}_set/general@1",
        structured={"type": f"{kind}_set", "name": "general", "version": "1"},
    )
    return sets.resolve(
        kind,
        reference,
        [{"prompt": f"p{i}", "group": "g"} for i in range(count)],
        markers=("evaluation_only",),
    )


def a_spec(**overrides) -> service.EvaluationSpec:
    expectation = expectations.resolve_expectation(RULE, source="pack")
    defaults = {
        "run_id": "run-1",
        "provider_name": "in_process",
        "aggregation": "mean",
        "decode": GenerationParams(max_tokens=16, temperature=0.0),
        "test_set": a_set("test"),
        "retention_set": a_set("retention"),
        "test_expectation": expectation,
        "retention_expectation": expectation,
        "floor": retention.ResolvedFloor(value=0.5, layer="pack_slot"),
        "batch_bound": 2,
    }
    defaults.update(overrides)
    return service.EvaluationSpec(**defaults)  # type: ignore[arg-type]


class TestThePairedMeasurement:
    def test_one_call_produces_both_scores(self, session, log_path: Path) -> None:
        """FR-004.24, and the contract agrees: one block, two entries, one provider, one
        decode parameter set. *Two calls will eventually be configured, or retried,
        differently*, and R-02.17's "scored the same way" is what the comparison rests on."""
        outcome = service.evaluate(
            a_spec(), 7, session=session, log_path=log_path, provider=FakeProvider(answers="ok")
        )
        assert {record.kind for record in outcome.records} == {"test", "retention"}
        assert len({record.provider for record in outcome.records}) == 1
        assert len({record.decode for record in outcome.records}) == 1
        assert len({record.created_at for record in outcome.records}) == 1

    def test_one_evaluation_block_carries_both(self, session, log_path: Path) -> None:
        outcome = service.evaluate(
            a_spec(), 7, session=session, log_path=log_path, provider=FakeProvider(answers="ok")
        )
        assert len(outcome.block["scores"]) == 2
        assert len(log_path.read_text().strip().splitlines()) == 1

    def test_step_zero_is_a_real_row_and_not_a_sentinel(self, session, log_path: Path) -> None:
        """Criterion 12 names the step-0 test score as a creation run's baseline; without it
        a run's first data point is already post-update with nothing to compare against."""
        service.evaluate(
            a_spec(), 0, session=session, log_path=log_path, provider=FakeProvider(answers="ok")
        )
        steps = session.execute(sa.select(SCORES.c.step).distinct()).scalars().all()
        assert steps == [0]

    def test_only_the_kinds_asked_for_are_scored(self, session, log_path: Path) -> None:
        outcome = service.evaluate(
            a_spec(),
            3,
            session=session,
            log_path=log_path,
            kinds=("test",),
            provider=FakeProvider(answers="ok"),
        )
        assert [record.kind for record in outcome.records] == ["test"]


class TestNoPartialPassIsEverRecorded:
    """M-6: *the plausible-looking "robustness" change.*"""

    def test_a_provider_failure_partway_records_NO_score(self, session, log_path: Path) -> None:
        handle = FakeModelHandle(
            answers="ok",
            device_uuid=None,
            fail_after=3,
            raise_error=Refusal("inference_provider_unreachable", "gone"),
        )
        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(),
                4,
                session=session,
                log_path=log_path,
                provider=InProcessProvider(handle, batch_bound=2),
            )
        assert caught.value.code == "inference_provider_unreachable"
        assert session.execute(sa.select(sa.func.count()).select_from(SCORES)).scalar_one() == 0
        assert not log_path.exists(), "a failed pass writes no evaluation block"

    def test_the_refusal_says_no_partial_score_was_recorded(self, session, log_path: Path) -> None:
        handle = FakeModelHandle(
            answers="ok",
            device_uuid=None,
            fail_after=1,
            raise_error=Refusal("judge_provider_unavailable", "endpoint down"),
        )
        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(),
                4,
                session=session,
                log_path=log_path,
                provider=InProcessProvider(handle, batch_bound=1),
            )
        assert "No partial score is recorded" in caught.value.message

    def test_f3s_original_code_travels_in_details_rather_than_being_discarded(
        self, session, log_path: Path
    ) -> None:
        """*A re-coded refusal loses its owner*, so the owner's code rides with it.

        The noun is corrected because a scoring pass is **not** a judge, and an operator who
        reads ``judge_provider_unavailable`` goes to the selector screen.
        """
        handle = FakeModelHandle(
            answers="ok",
            device_uuid=None,
            fail_after=0,
            raise_error=Refusal("judge_provider_unavailable", "endpoint down"),
        )
        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(),
                4,
                session=session,
                log_path=log_path,
                provider=InProcessProvider(handle, batch_bound=1),
            )
        assert caught.value.code == "inference_provider_unreachable"
        assert caught.value.details["provider_code"] == "judge_provider_unavailable"

    def test_a_short_reply_list_refuses_rather_than_scoring_what_came_back(
        self, session, log_path: Path
    ) -> None:
        class ShortProvider(FakeProvider):
            def generate(self, prompts, params, **options):
                return super().generate(prompts[:1], params, **options)

        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(batch_bound=99),
                4,
                session=session,
                log_path=log_path,
                provider=ShortProvider(answers="ok"),
            )
        assert caught.value.code == "inference_provider_reply_unparseable"


class TestCancellationAtAPromptBoundary:
    def test_a_cancelled_pass_records_no_set_score(self, session, log_path: Path) -> None:
        """E-18, AC-004.25. A prompt is scored and recorded, **or not started**."""
        calls = {"n": 0}

        def cancelled() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(),
                4,
                session=session,
                log_path=log_path,
                provider=FakeProvider(answers="ok"),
                is_cancelled=cancelled,
            )
        assert caught.value.code == "evaluation_cancelled"
        assert session.execute(sa.select(sa.func.count()).select_from(SCORES)).scalar_one() == 0

    def test_the_refusal_reports_how_far_it_got(self, session, log_path: Path) -> None:
        calls = {"n": 0}

        def cancelled() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(),
                4,
                session=session,
                log_path=log_path,
                provider=FakeProvider(answers="ok"),
                is_cancelled=cancelled,
            )
        assert caught.value.details["completed"] < caught.value.details["total"]


class TestTheProviderIsPinnedAndNeverSubstituted:
    """M-9: *a silent provider substitution rewrites the curve's meaning mid-run.*"""

    def test_an_unavailable_provider_refuses_rather_than_falling_back(
        self, session, log_path: Path
    ) -> None:
        from providers import registry

        registry.bind_model_handle(None)
        with pytest.raises(Refusal) as caught:
            service.evaluate(a_spec(), 1, session=session, log_path=log_path)
        assert caught.value.code == "inference_provider_unavailable"

    def test_an_undeclared_capability_refuses_the_evaluation(self, session, log_path: Path) -> None:
        """FR-004.8, after miStudio's verified finding that **miLLM ignores the OpenAI
        ``response_format`` parameter** - identical output with and without.

        *Nothing is sent hopefully*: a parameter the server silently ignores produces a
        number whose meaning depends on which server answered.
        """
        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(requires=frozenset({"response_format"})),
                1,
                session=session,
                log_path=log_path,
                provider=FakeProvider(answers="ok"),
            )
        assert caught.value.code == "inference_provider_capability_undeclared"
        assert caught.value.details["needed"] == ["response_format"]

    def test_a_provider_without_generation_refuses(self, session, log_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(),
                1,
                session=session,
                log_path=log_path,
                provider=FakeProvider(answers="ok", capabilities=frozenset()),
            )
        assert caught.value.code == "inference_provider_capability_undeclared"

    def test_a_recorded_discontinuity_rides_on_every_subsequent_score(
        self, session, log_path: Path
    ) -> None:
        """FR-004.10: F9 breaks the curve here rather than joining two instruments."""
        outcome = service.evaluate(
            a_spec(provider_discontinuity=True),
            4,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="ok"),
        )
        assert all(record.provider_discontinuity for record in outcome.records)


class TestReproducibilityIsMarkedWithAReason:
    """FR-004.18. Criterion 2 is exact **only** under deterministic decoding."""

    def test_greedy_decoding_is_reproducible(self, session, log_path: Path) -> None:
        outcome = service.evaluate(
            a_spec(), 1, session=session, log_path=log_path, provider=FakeProvider(answers="ok")
        )
        assert all(record.reproducible for record in outcome.records)
        assert all(record.not_reproducible_reason is None for record in outcome.records)

    def test_sampling_is_marked_not_reproducible_WITH_the_reason(
        self, session, log_path: Path
    ) -> None:
        outcome = service.evaluate(
            a_spec(decode=GenerationParams(max_tokens=16, temperature=0.8)),
            1,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="ok"),
        )
        record = outcome.records[0]
        assert record.reproducible is False
        assert "temperature 0.8" in (record.not_reproducible_reason or "")
        assert "no requirement defines a tolerance" in (record.not_reproducible_reason or "")

    def test_top_p_sampling_is_marked_too(self, session, log_path: Path) -> None:
        outcome = service.evaluate(
            a_spec(decode=GenerationParams(max_tokens=16, top_p=0.9)),
            1,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="ok"),
        )
        assert outcome.records[0].reproducible is False

    def test_a_not_reproducible_row_always_names_its_reason(self, session, log_path: Path) -> None:
        """The database refuses the pair, so the reason cannot be lost on the way in."""
        service.evaluate(
            a_spec(decode=GenerationParams(max_tokens=16, temperature=0.8)),
            1,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="ok"),
        )
        row = session.execute(sa.select(SCORES)).mappings().first()
        assert row is not None and row["not_reproducible_reason"]


class TestTheBreachIsDetectedAndPublishedButNotActedOn:
    def test_a_below_floor_retention_score_produces_a_signal(self, session, log_path: Path) -> None:
        outcome = service.evaluate(
            a_spec(floor=retention.ResolvedFloor(value=0.9, layer="pack_slot")),
            6,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="x"),  # too short for the rule: score 0.0
        )
        assert outcome.breach is not None
        assert outcome.breach.floor == 0.9
        assert outcome.block["retention_floor_breached"] is True

    def test_the_run_is_not_stopped_by_this_feature(self, session, log_path: Path) -> None:
        """R-02.25 is F6's. F4 detects, names and publishes; the run row is untouched."""
        service.evaluate(
            a_spec(floor=retention.ResolvedFloor(value=0.9, layer="pack_slot")),
            6,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="x"),
        )
        # Nothing in `mf_runs` was written - F4 holds no run row at all in this path.
        assert session.execute(sa.text("SELECT COUNT(*) FROM mf_runs")).scalar_one() == 0

    def test_at_the_floor_exactly_is_not_a_breach(self, session, log_path: Path) -> None:
        outcome = service.evaluate(
            a_spec(floor=retention.ResolvedFloor(value=1.0, layer="pack_slot")),
            6,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="long enough"),  # every prompt admitted: score 1.0
        )
        assert outcome.breach is None


class TestOneImplementationForEveryCaller:
    def test_no_configured_set_refuses_rather_than_recording_an_empty_pass(
        self, session, log_path: Path
    ) -> None:
        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(test_set=None, retention_set=None),
                1,
                session=session,
                log_path=log_path,
                provider=FakeProvider(answers="ok"),
            )
        assert caught.value.code == "evaluation_set_unresolvable"

    def test_an_unknown_kind_refuses(self, session, log_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            service.evaluate(
                a_spec(),
                1,
                session=session,
                log_path=log_path,
                kinds=("vibes",),
                provider=FakeProvider(answers="ok"),
            )
        assert caught.value.code == "evaluation_kind_unknown"

    def test_it_completes_synchronously_with_no_broker_running(
        self, session, log_path: Path
    ) -> None:
        """**The property F6's loop depends on**, and an AC20 case.

        ADR-009's in-process provider generates from parameters that exist only in the
        ``train`` worker's process, so the loop has to be able to await this call without a
        broker, a result backend or a second worker in the picture.
        """
        outcome = service.evaluate(
            a_spec(), 2, session=session, log_path=log_path, provider=FakeProvider(answers="ok")
        )
        assert outcome.records
