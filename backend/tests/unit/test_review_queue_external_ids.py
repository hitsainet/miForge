"""Deterministic ids, and batches that fit before they are sent  (FR-010.22; AC-010.8).

**Determinism is what makes a re-post after a crash idempotent.** miDataworks returns the
existing item for a repeated ``external_id``, so two processes that derive the same id for
the same candidate cannot create a duplicate - and nobody is asked to read the same candidate
twice. A non-deterministic id would make a resumed run re-queue everything it had already
posted, silently, and the symptom would be a reviewer's queue filling up with work they had
already done.

**Splitting happens before sending.** Never send-and-handle-413: a rejected batch costs the
whole payload's upload for an answer miForge already had.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from services.review.binding import (
    Candidate,
    candidate_payload,
    derive_external_id,
    payload_bytes,
    split_batches,
)


def ids_for(count: int, *, run_id: str = "run-1", step: int = 7) -> list[str]:
    return [
        derive_external_id(
            run_id=run_id,
            step=step,
            candidate_index=index,
            prompt="p",
            completion=f"c{index}",
        )
        for index in range(count)
    ]


class TestDeterminism:
    def test_the_same_candidate_gives_the_same_id(self):
        first = derive_external_id(
            run_id="r", step=3, candidate_index=0, prompt="p", completion="c"
        )
        second = derive_external_id(
            run_id="r", step=3, candidate_index=0, prompt="p", completion="c"
        )
        assert first == second

    def test_it_is_stable_across_processes(self):
        """Hashed through the **one** canonical-JSON function, so it does not depend on dict
        ordering, ``PYTHONHASHSEED`` or anything else a second process might differ on."""
        import subprocess
        import sys
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        script = (
            f"import sys; sys.path.insert(0, {str(src)!r});"
            "from services.review.binding import derive_external_id;"
            "print(derive_external_id(run_id='r', step=3, candidate_index=0,"
            " prompt='p', completion='c'))"
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert out == derive_external_id(
            run_id="r", step=3, candidate_index=0, prompt="p", completion="c"
        )

    @pytest.mark.parametrize(
        "changed",
        [
            {"run_id": "other"},
            {"step": 4},
            {"candidate_index": 1},
            {"prompt": "different"},
            {"completion": "different"},
        ],
    )
    def test_every_component_changes_the_id(self, changed: dict[str, object]):
        """The completion is in the material as well as the prompt: **the same prompt
        generated twice at one step is two different candidates**, and an id that collided
        would silently drop one of them by returning the other's queue item."""
        base = {"run_id": "r", "step": 3, "candidate_index": 0, "prompt": "p", "completion": "c"}
        assert derive_external_id(**base) != derive_external_id(**{**base, **changed})

    def test_the_id_fits_midataworks_limit_and_names_miforge(self):
        """Their ``external_id`` is 1-256 characters. The prefix makes a stray id
        recognisable in their UI as miForge's rather than as a mystery."""
        identifier = derive_external_id(
            run_id="a" * 64, step=999999, candidate_index=4, prompt="p", completion="c"
        )
        assert 1 <= len(identifier) <= 256
        assert identifier.startswith("miforge:")

    def test_ids_are_unique_within_a_step(self):
        assert len(set(ids_for(1000))) == 1000


class TestBatchSplitting:
    @pytest.mark.parametrize(
        ("count", "expected_batches"),
        [(0, 0), (1, 1), (499, 1), (500, 1), (501, 2), (1000, 2), (1001, 3)],
    )
    def test_the_boundaries(self, count: int, expected_batches: int):
        """0, 1, 500, 501 and 1,000 - the limit and both sides of it. ``0`` gives **no
        batches**: zero requests, not one empty request."""
        payloads = [
            candidate_payload(identifier, Candidate(prompt="p", completion="c"))
            for identifier in ids_for(count)
        ]
        batches = split_batches(payloads)
        assert len(batches) == expected_batches
        assert sum(len(batch) for batch in batches) == count
        assert all(len(batch) <= 500 for batch in batches)

    def test_no_candidate_is_lost_or_duplicated(self):
        payloads = [
            candidate_payload(identifier, Candidate(prompt="p", completion="c"))
            for identifier in ids_for(1001)
        ]
        flattened = [item["external_id"] for batch in split_batches(payloads) for item in batch]
        assert flattened == [item["external_id"] for item in payloads]

    def test_an_oversized_candidate_refuses_rather_than_being_truncated(self):
        """**A candidate cannot be split** - a person reads one prompt and one completion
        together, and sending half of each would ask them to judge something that was never
        generated. So it is refused by name rather than sent to be 413'd."""
        payload = candidate_payload("miforge:r:0:x", Candidate(prompt="p" * 70_000, completion="c"))
        with pytest.raises(Refusal) as caught:
            split_batches([payload])
        assert caught.value.code == "review_candidate_too_large"
        assert caught.value.details["ceiling"] == 65_536
        assert caught.value.details["bytes"] > 65_536

    def test_the_ceiling_is_measured_the_way_the_receiver_measures_it(self):
        """Their check is ``len(canonical_json(c.model_dump(mode="json")))`` - the
        **serialized candidate**, not the completion's string length. Measuring the string
        would pass a candidate whose ``provenance`` object is the thing over the limit."""
        big_provenance = candidate_payload(
            "miforge:r:0:x",
            Candidate(prompt="p", completion="c", provenance={"note": "x" * 70_000}),
        )
        assert payload_bytes(big_provenance) > 65_536
        with pytest.raises(Refusal):
            split_batches([big_provenance])


class TestTheCandidateBody:
    def test_it_is_their_candidate_model_key_for_key(self):
        """Their ``Candidate`` is ``extra="forbid"``, so an extra key here is a 422 rather
        than an ignored field - which is why one builder exists instead of each call site
        assembling a dict."""
        payload = candidate_payload("miforge:r:0:x", Candidate(prompt="p", completion="c"))
        assert sorted(payload) == ["completion", "external_id", "prompt", "provenance"]

    def test_model_output_is_included_only_when_there_is_one(self):
        """Their field is optional with a ``None`` default; sending an explicit ``null``
        where nothing was computed would record an absence as a measurement."""
        without = candidate_payload("x", Candidate(prompt="p", completion="c"))
        assert "model_output" not in without
        with_output = candidate_payload(
            "x", Candidate(prompt="p", completion="c", model_output={"logprob": -1.2})
        )
        assert with_output["model_output"] == {"logprob": -1.2}

    def test_provenance_defaults_to_an_empty_object_not_to_null(self):
        payload = candidate_payload("x", Candidate(prompt="p", completion="c"))
        assert payload["provenance"] == {}
