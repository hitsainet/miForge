"""Resolving a set, and the sample that is not a truncation.

FR-004.12, FR-004.16, FR-004.17, FR-004.19; E-5, E-6, E-17.

The sharp one here is **M-7**: ``rows[:n]`` takes the head of a file, and these sets are
written a group at a time. miStudio measured a judge on *"the first 200 rows of a sorted
file"* and one 15-source corpus contributed **2 of 15 sources** to the sample. A head-sliced
test score moves when the file's order changes and nothing else does - the curve reports a
change in the data layout as a change in the model.

So the sampling tests below are written to **fail against a head slice**, not merely to pass
against the real one. That is the difference between a test and a description.
"""

from __future__ import annotations

import json

import pytest

from core.errors import Refusal
from services.scoring import sets


def rows(count: int, *, groups: int = 1) -> list[dict[str, str]]:
    """``count`` prompt rows, grouped **in blocks**, which is how these files are written.

    Blocked rather than interleaved on purpose: a head slice of an interleaved file looks
    balanced, so an interleaved fixture would make the head-slice test pass against the
    defect. *Fixtures whose fields agree by construction are how a suite stays green over a
    real bug* - and so are fixtures whose ordering does.
    """
    out: list[dict[str, str]] = []
    per = count // groups
    for group in range(groups):
        for index in range(per):
            out.append(
                {"prompt": f"g{group}-p{index}", "reference": f"r{index}", "group": f"g{group}"}
            )
    return out


REFERENCE = sets.SetReference(
    rendered="test_set/general@1",
    structured={"type": "test_set", "name": "general", "version": "1"},
)


class TestAZeroPromptSetRefuses:
    def test_empty_refuses_by_name(self) -> None:
        """E-5. ``ScoreRecord.n`` has ``minimum: 1``, so a zero-prompt score has **no
        representation in the contract** - and ``score: 0, n: 0`` plots as a catastrophic
        regression that never happened."""
        with pytest.raises(Refusal) as caught:
            sets.resolve("test", REFERENCE, [])
        assert caught.value.code == "evaluation_set_empty"

    def test_a_row_with_no_prompt_refuses_rather_than_scoring_the_rest(self) -> None:
        """E-6. **Never score the subset that resolved.**"""
        with pytest.raises(Refusal) as caught:
            sets.resolve("test", REFERENCE, [{"prompt": "ok"}, {"notes": "no prompt here"}])
        assert caught.value.code == "evaluation_set_unresolvable"
        assert caught.value.details["row_index"] == 1

    def test_a_blank_prompt_counts_as_absent(self) -> None:
        with pytest.raises(Refusal):
            sets.resolve("test", REFERENCE, [{"prompt": "   "}])


class TestTheSampleIsStratifiedSeededAndRecorded:
    def test_the_same_seed_gives_the_same_sample(self) -> None:
        population = rows(60, groups=3)
        first = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(12, 7))
        second = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(12, 7))
        assert first.texts == second.texts

    def test_a_different_seed_gives_a_different_sample(self) -> None:
        """Otherwise "seeded" would be satisfied by ignoring the seed entirely."""
        population = rows(60, groups=3)
        first = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(12, 7))
        second = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(12, 99))
        assert first.texts != second.texts

    def test_the_seed_is_recorded_on_the_set(self) -> None:
        resolved = sets.resolve("test", REFERENCE, rows(30), sample=sets.SampleConfig(10, 3))
        assert resolved.subsample_seed == 3

    def test_a_null_seed_is_recorded_for_an_unsampled_set(self) -> None:
        """Null is an assertion that no seed was set - a fact worth recording, not hiding."""
        assert sets.resolve("test", REFERENCE, rows(5)).subsample_seed is None

    def test_the_population_is_recorded_beside_the_sample(self) -> None:
        """ADR-012 needs **both** numbers: a sample of 12 says nothing without "of 60"."""
        resolved = sets.resolve(
            "test", REFERENCE, rows(60, groups=3), sample=sets.SampleConfig(12, 7)
        )
        assert (resolved.n, resolved.population) == (12, 60)

    def test_every_group_is_represented(self) -> None:
        """**The property a head slice fails.**"""
        population = rows(60, groups=3)
        resolved = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(12, 7))
        assert {prompt.group for prompt in resolved.prompts} == {"g0", "g1", "g2"}

    def test_A_HEAD_SLICE_WOULD_FAIL_THIS(self) -> None:
        """M-7's assertion, written against the defect rather than against the fix.

        The 15-source corpus that contributed 2 sources is the recorded incident; this is
        the smallest fixture that reproduces its shape, and a ``rows[:n]`` implementation
        returns exactly one group for it.
        """
        population = rows(60, groups=3)
        head_slice = population[:12]
        assert len({row["group"] for row in head_slice}) == 1, (
            "the fixture no longer reproduces the head-slice failure, so the test below "
            "would pass against the defect too"
        )
        resolved = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(12, 7))
        assert len({prompt.group for prompt in resolved.prompts}) == 3

    def test_the_sample_is_proportional_across_groups(self) -> None:
        population = rows(90, groups=3)
        resolved = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(30, 5))
        counts = dict.fromkeys(("g0", "g1", "g2"), 0)
        for prompt in resolved.prompts:
            counts[prompt.group or ""] += 1
        assert all(8 <= value <= 12 for value in counts.values()), counts

    def test_the_sample_holds_only_real_rows(self) -> None:
        population = rows(40, groups=2)
        resolved = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(10, 1))
        assert {prompt.text for prompt in resolved.prompts} <= {row["prompt"] for row in population}

    def test_a_sample_at_or_above_the_population_is_the_population(self) -> None:
        population = rows(10)
        resolved = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(50, 1))
        assert resolved.texts == [row["prompt"] for row in population]

    def test_a_zero_size_sample_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            sets.resolve("test", REFERENCE, rows(10), sample=sets.SampleConfig(0, 1))
        assert caught.value.code == "evaluation_sample_empty"

    def test_an_ungrouped_set_still_samples_rather_than_slicing(self) -> None:
        population = [{"prompt": f"p{i}"} for i in range(50)]
        resolved = sets.resolve("test", REFERENCE, population, sample=sets.SampleConfig(10, 4))
        assert resolved.texts != [row["prompt"] for row in population[:10]]


class TestASetTooLargeRefusesRatherThanTruncating:
    def test_over_the_ceiling_refuses(self) -> None:
        """*A truncation is a head slice by another name.*"""
        with pytest.raises(Refusal) as caught:
            sets.resolve("test", REFERENCE, rows(50), max_prompts=10)
        assert caught.value.code == "evaluation_set_too_large"
        assert caught.value.details["population"] == 50

    def test_a_seeded_sample_passes_the_ceiling(self) -> None:
        resolved = sets.resolve(
            "test", REFERENCE, rows(50), sample=sets.SampleConfig(10, 2), max_prompts=10
        )
        assert resolved.n == 10


class TestTheMarkerNamesComeFromTheVendoredSchema:
    def test_they_are_the_boolean_properties_of_split(self) -> None:
        """Derived, not retyped - so a marker miDataworks adds is **carried**, not dropped.

        A dropped provenance flag is one of this project's three recorded defects of that
        class, and the derivation is what makes the next one impossible rather than unlikely.
        """
        assert sets.split_marker_names() == ("evaluation_only", "held_out")

    def test_they_agree_with_the_schema_read_independently(self) -> None:
        from pathlib import Path

        schema = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "docs"
                / "schemas"
                / sets.DATASET_SCHEMA_FILENAME
            ).read_text("utf-8")
        )
        properties = schema["$defs"]["Split"]["properties"]
        expected = tuple(
            sorted(name for name, body in properties.items() if body.get("type") == "boolean")
        )
        assert sets.split_marker_names() == expected

    def test_markers_are_carried_onto_the_set(self) -> None:
        resolved = sets.resolve("test", REFERENCE, rows(4), markers=("held_out",))
        assert resolved.split_markers == ["held_out"]

    def test_no_markers_is_different_from_an_external_part(self) -> None:
        """``()`` means *read and none set*; ``None`` means *this part declares no splits*."""
        assert sets.resolve("test", REFERENCE, rows(4), markers=()).markers == ()
        assert sets.resolve("test", REFERENCE, rows(4)).markers is None


class TestOneRendererForASetIdentity:
    def test_a_contract_reference_renders_as_midataworks_own_form(self) -> None:
        assert (
            sets.render_reference({"type": "test_set", "name": "general", "version": "3"})
            == "test_set/general@3"
        )

    def test_an_external_provenance_renders_distinguishably(self) -> None:
        rendered = sets.render_reference({"origin": "ops-export", "sha256": "ab" * 32})
        assert rendered.startswith("external:ops-export@")

    def test_neither_form_refuses_rather_than_synthesising_one(self) -> None:
        """**Nothing may synthesise a contract identity miDataworks never issued.**"""
        with pytest.raises(Refusal) as caught:
            sets.render_reference({"name": "general"})
        assert caught.value.code == "evaluation_set_unresolvable"

    def test_a_fallback_is_used_only_when_offered(self) -> None:
        assert sets.render_reference(None, fallback="pack/local-set") == "pack/local-set"

    def test_an_empty_rendering_refuses_at_the_type(self) -> None:
        with pytest.raises(Refusal):
            sets.SetReference(rendered="   ")


class TestReadingRowsFromAnImportedPart:
    def _manifest(self, tmp_path, *, held_out: bool = True) -> dict:
        split = tmp_path / "test.jsonl"
        split.write_text(
            "\n".join(
                json.dumps({"question": f"q{i}", "answer": f"a{i}", "source": f"s{i % 2}"})
                for i in range(6)
            ),
            encoding="utf-8",
        )
        return {
            "content": {
                "content_kind": "rows",
                "columns": [
                    {"name": "question", "semantic": "prompt"},
                    {"name": "answer", "semantic": "reference"},
                    {"name": "source", "semantic": "group"},
                ],
                "splits": [
                    {
                        "name": "test",
                        "path": "test.jsonl",
                        "held_out": held_out,
                        "evaluation_only": True,
                    }
                ],
            }
        }

    def test_the_prompt_reference_and_group_columns_are_the_parts_own(self, tmp_path) -> None:
        """Read from the manifest's ``semantic`` declarations, **not guessed from names**.

        The stratification key is therefore the dataset author's choice, which is the only
        way a stratified sample means what it says.
        """
        rows_read, markers = sets.prompt_rows_from_part(tmp_path, self._manifest(tmp_path))
        assert [row["prompt"] for row in rows_read] == [f"q{i}" for i in range(6)]
        assert rows_read[0]["reference"] == "a0"
        assert rows_read[0]["group"] == "s0"
        assert set(markers) == {"held_out", "evaluation_only"}

    def test_a_non_rows_part_refuses(self, tmp_path) -> None:
        with pytest.raises(Refusal) as caught:
            sets.prompt_rows_from_part(tmp_path, {"content": {"content_kind": "config"}})
        assert caught.value.code == "evaluation_set_unresolvable"

    def test_a_missing_split_file_refuses_rather_than_scoring_nothing(self, tmp_path) -> None:
        manifest = self._manifest(tmp_path)
        (tmp_path / "test.jsonl").unlink()
        with pytest.raises(Refusal) as caught:
            sets.prompt_rows_from_part(tmp_path, manifest)
        assert caught.value.code == "evaluation_set_unresolvable"

    def test_a_malformed_line_refuses_rather_than_being_skipped(self, tmp_path) -> None:
        """*A skipped line makes a truncated file read as a smaller set*, and the score
        would be over a sample size nobody chose."""
        manifest = self._manifest(tmp_path)
        (tmp_path / "test.jsonl").write_text('{"question": "q"}\nnot json\n', encoding="utf-8")
        with pytest.raises(Refusal) as caught:
            sets.prompt_rows_from_part(tmp_path, manifest)
        assert caught.value.details["line"] == 2

    def test_an_unknown_split_name_refuses(self, tmp_path) -> None:
        with pytest.raises(Refusal):
            sets.prompt_rows_from_part(tmp_path, self._manifest(tmp_path), split_name="nope")
