"""In-run human review  (AC-003.9, AC-003.10, AC-003.11; FR-003.7, E-9, E-10, E-11).

**This is the type that forces the third verdict value**, and the only path in F3 where
user text leaves the process.

Three obligations, each with a reason that is not procedural:

1. **There is no path from a deadline to an accept.** Defaulting silence to accept would
   turn an unattended queue into an unselected run; defaulting it to reject would silently
   discard every candidate a person had not reached yet.
2. **The outbound payload is asserted field by field, with the call count.** *A privacy
   guarantee verified by reading is a guarantee verified by nothing.* **Mutation control 6**
   drops a field from the payload, and the tests here must go red.
3. **With miDataworks unconfigured the type refuses by name** and the other four work -
   acceptance criterion 20, which pins the whole standalone reframe.

**It is in-run review, not R-02.39's authoring dry-run.** The dry-run is miForge's, uses no
queue and belongs to F7; conflating the two is an error this project has already made once.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from schemas.selector import VerdictValue, is_admitted
from services.selectors.plugins import human_review
from services.selectors.plugins.human_review import derive_external_id
from tests.support.selector_stubs import StubReviewQueue, human_config, resolved


@pytest.fixture
def queue():
    stub = StubReviewQueue()
    human_review.bind_review_queue(stub)
    yield stub
    human_review.bind_review_queue(None)


def selector_for(queue_name: str = "legal-review", **kwargs):
    return resolved(human_config(queue=queue_name, **kwargs))


class TestItIsInRunReviewNotTheAuthoringDryRun:
    def test_the_module_docstring_states_the_distinction(self) -> None:
        """Named in a test so the distinction cannot be quietly dropped from the module."""
        text = human_review.__doc__ or ""
        assert "R-02.39" in text
        assert "dry-run" in text
        assert "F7" in text

    def test_this_feature_keeps_no_queue_of_its_own(self, queue) -> None:
        """R-02.36: the queue is miDataworks'. F3 declares a port and calls it."""
        selector = selector_for()
        assert selector.bindings["review_queue"] is queue


class TestTheOutboundPayload:
    def test_every_field_of_every_candidate_is_asserted(self, queue) -> None:
        """AC-003.9. "Was called" passes against a call sending the wrong arguments."""
        selector = selector_for()
        selector.plugin.prefetch(["Q1", "Q2"], ["A1", "A2"], batch_id="b7")

        assert queue.post_count == 1, "two candidates go in one post, not two"
        posted = queue.all_posted
        assert len(posted) == 2
        assert [candidate.prompt for candidate in posted] == ["Q1", "Q2"]
        assert [candidate.completion for candidate in posted] == ["A1", "A2"]
        assert [candidate.external_id for candidate in posted] == [
            derive_external_id("b7", 0, "A1"),
            derive_external_id("b7", 1, "A2"),
        ]
        for candidate in posted:
            assert candidate.provenance == {"selector": selector.name, "batch_id": "b7"}

    def test_no_credential_and_no_host_path_leaves_in_the_payload(self, queue) -> None:
        """ADR-020. Provenance is a reference, not a filesystem tour."""
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        blob = str([candidate.model_dump() for candidate in queue.all_posted])
        for secret in ("Bearer", "token", "/home/", "password", "sk-"):
            assert secret not in blob

    def test_the_queue_is_created_with_at_least_two_labels(self, queue) -> None:
        """miDataworks requires a ``label_set`` of 2-64 free strings - verified there."""
        selector_for().plugin.prefetch(["Q"], ["A"], batch_id="b")
        assert len(queue.ensure_calls) == 1
        assert len(queue.ensure_calls[0]["label_set"]) >= 2
        assert queue.ensure_calls[0]["name"] == "legal-review"

    def test_decisions_are_read_back_by_external_id(self, queue) -> None:
        selector_for().plugin.prefetch(["Q"], ["A"], batch_id="b")
        assert queue.read_calls == [[derive_external_id("b", 0, "A")]]

    def test_a_batch_is_chunked_at_five_hundred(self, queue) -> None:
        """The stub raises above 500, so this would be red rather than merely unasserted."""
        selector = selector_for()
        prompts = [f"q{i}" for i in range(1201)]
        completions = [f"a{i}" for i in range(1201)]
        selector.plugin.prefetch(prompts, completions, batch_id="big")
        assert [len(batch) for batch in queue.posted] == [500, 500, 201]
        assert len(queue.all_posted) == 1201

    def test_a_cancelled_run_stops_posting_between_chunks(self, queue) -> None:
        """Suite tracker 4.28: the prefetch receives ``is_cancelled`` and checks it before
        each chunk, so a cancelled run does not keep putting candidates before a person."""
        selector = selector_for()
        prompts = [f"q{i}" for i in range(1201)]
        completions = [f"a{i}" for i in range(1201)]
        selector.plugin.prefetch(
            prompts, completions, batch_id="big", is_cancelled=lambda: len(queue.posted) >= 1
        )
        assert [len(batch) for batch in queue.posted] == [500]


class TestIdempotence:
    def test_the_external_id_is_deterministic(self) -> None:
        assert derive_external_id("b", 3, "answer") == derive_external_id("b", 3, "answer")

    def test_two_identical_completions_are_two_candidates(self) -> None:
        """Position is in the id, so a dict keyed on text cannot merge them."""
        assert derive_external_id("b", 0, "same") != derive_external_id("b", 1, "same")

    def test_a_reassembled_batch_does_not_collide_across_batches(self) -> None:
        assert derive_external_id("b1", 0, "a") != derive_external_id("b2", 0, "a")

    def test_a_retried_post_returns_the_existing_item(self, queue) -> None:
        """miDataworks' verified behaviour, so a network retry cannot ask a person twice."""
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        first = dict(selector.plugin.item_ids())
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        assert selector.plugin.item_ids() == first
        assert len(queue.items) == 1


class TestUndecidedUntilDecided:
    def test_an_undecided_candidate_carries_the_queue_item_id(self, queue) -> None:
        """E-10. Never converted to accept."""
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        verdict = selector.plugin.evaluate(
            prompt="Q", completion="A", context={"index": 0, "batch_id": "b"}
        )
        assert verdict.value is VerdictValue.UNDECIDED
        assert verdict.label in queue.items.values()
        assert "waiting on review item" in (verdict.reason or "")

    def test_an_undecided_verdict_is_not_admitted(self, queue) -> None:
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        verdict = selector.plugin.evaluate(
            prompt="Q", completion="A", context={"index": 0, "batch_id": "b"}
        )
        assert is_admitted(verdict) is False

    def test_an_undecided_verdict_carries_no_score(self, queue) -> None:
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        verdict = selector.plugin.evaluate(
            prompt="Q", completion="A", context={"index": 0, "batch_id": "b"}
        )
        assert verdict.score is None

    def test_a_passed_deadline_leaves_it_undecided_and_never_accepts(self, queue) -> None:
        """AC-003.11, E-11. ``timeout_seconds`` bounds the wait, never the verdict."""
        selector = resolved(human_config(timeout_seconds=0))
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        verdict = selector.plugin.evaluate(
            prompt="Q", completion="A", context={"index": 0, "batch_id": "b"}
        )
        assert verdict.value is VerdictValue.UNDECIDED
        assert is_admitted(verdict) is False

    def test_a_partially_decided_queue_yields_a_mix(self, queue) -> None:
        selector = selector_for()
        selector.plugin.prefetch(["Q1", "Q2"], ["A1", "A2"], batch_id="b")
        queue.decide(derive_external_id("b", 0, "A1"), "accept")
        selector.plugin.prefetch(["Q1", "Q2"], ["A1", "A2"], batch_id="b")

        first = selector.plugin.evaluate(
            prompt="Q1", completion="A1", context={"index": 0, "batch_id": "b"}
        )
        second = selector.plugin.evaluate(
            prompt="Q2", completion="A2", context={"index": 1, "batch_id": "b"}
        )
        assert first.value is VerdictValue.ACCEPT
        assert second.value is VerdictValue.UNDECIDED


class TestTheDecisionMappingIsTheContracts:
    @pytest.mark.parametrize(
        ("decision", "expected"),
        [
            ("accept", VerdictValue.ACCEPT),
            ("reject", VerdictValue.REJECT),
            ("override", VerdictValue.UNDECIDED),
            ("flag", VerdictValue.UNDECIDED),
        ],
    )
    def test_each_decision_maps_as_the_contract_says(self, queue, decision, expected) -> None:
        """**This plugin interprets no decision vocabulary of its own.** Two mappings of
        one decision set is how ``flag`` ends up meaning accept in one of them."""
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        queue.decide(derive_external_id("b", 0, "A"), decision, label="needs-citation")
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        verdict = selector.plugin.evaluate(
            prompt="Q", completion="A", context={"index": 0, "batch_id": "b"}
        )
        assert verdict.value is expected

    def test_an_override_preserves_the_label_beside_the_value(self, queue) -> None:
        """The label rides **beside** the value, never as a fourth enum member."""
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        queue.decide(derive_external_id("b", 0, "A"), "override", label="partially-correct")
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        verdict = selector.plugin.evaluate(
            prompt="Q", completion="A", context={"index": 0, "batch_id": "b"}
        )
        assert verdict.label == "partially-correct"
        assert verdict.value is VerdictValue.UNDECIDED

    def test_an_unrecognised_decision_kind_is_refused_not_defaulted(self, queue) -> None:
        """miDataworks may add kinds; the honest outcome is a named failure, not a verdict
        invented on its behalf."""
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        queue.decide(derive_external_id("b", 0, "A"), "escalate")
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(
                prompt="Q", completion="A", context={"index": 0, "batch_id": "b"}
            )
        assert exc.value.code == "review_decision_unrecognised"

    def test_decision_ids_are_returned_for_the_batch_record(self, queue) -> None:
        """R-02.36's last sentence, R-02.8."""
        selector = selector_for()
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        queue.decide(derive_external_id("b", 0, "A"), "accept")
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        assert selector.plugin.decision_ids() == {
            derive_external_id("b", 0, "A"): f"d-{derive_external_id('b', 0, 'A')}"
        }


class TestUnavailableByName:
    def test_with_no_port_bound_it_refuses_naming_midataworks(self) -> None:
        """E-9, AC-003.10, ADR-010."""
        human_review.bind_review_queue(None)
        with pytest.raises(Refusal) as exc:
            selector_for()
        assert exc.value.code == "midataworks_not_configured"
        assert exc.value.details["service"] == "midataworks"
        assert "human review" in exc.value.details["capabilities_unavailable"][0]

    def test_a_port_reporting_unavailable_refuses_the_same_way(self) -> None:
        human_review.bind_review_queue(StubReviewQueue(available=False))
        try:
            with pytest.raises(Refusal) as exc:
                selector_for()
            assert exc.value.code == "midataworks_not_configured"
        finally:
            human_review.bind_review_queue(None)

    def test_an_unreachable_configured_service_refuses_differently(self) -> None:
        """ "Absent" and "broken" are different facts; one refusal code for both would send
        the operator to the wrong place."""
        human_review.bind_review_queue(StubReviewQueue(unreachable=True))
        try:
            selector = selector_for()
            with pytest.raises(Refusal) as exc:
                selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
            assert exc.value.code == "midataworks_unreachable"
        finally:
            human_review.bind_review_queue(None)

    def test_no_verdict_is_defaulted_when_the_service_is_absent(self) -> None:
        """The refusal is the whole behaviour: no local queue, no fabricated accept."""
        human_review.bind_review_queue(None)
        with pytest.raises(Refusal):
            selector_for()


class TestBounds:
    def test_an_oversized_candidate_refuses_with_the_bound_named(self, queue) -> None:
        """A person asked to judge a truncated answer is judging a different answer."""
        selector = selector_for()
        with pytest.raises(Refusal) as exc:
            selector.plugin.prefetch(["Q"], ["x" * 70_000], batch_id="b")
        assert exc.value.code == "selector_text_too_long"
        assert exc.value.details["limit"] == 65_536
        assert queue.post_count == 0, "bounded BEFORE anything leaves the process"
