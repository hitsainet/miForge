"""Three refusals before a byte is read  (FR-002.5 - FR-002.9; AC-002.5, AC-002.16).

The admissibility case is the one whose violation is invisible: **a held-out split used as
training material makes every subsequent test score meaningless and nothing in the loss
curve says so.** It is mutation control M-3 - replace the call to F10's one rule with
``getattr(split, "held_out", False)`` and
:meth:`TestAdmissibility.test_a_held_out_split_is_refused_by_name` must go red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import Refusal
from training.signals import parts_reader
from training.signals.base import PartSpec


def part_payload(tmp_path: Path, **overrides: object) -> dict[str, object]:
    """A part row in the shape F10's index serves, with its manifest on disk.

    The manifest is written through the fixture builder and read back by **F10's own
    parser**, so this exercises the real read path rather than a dict that agrees with the
    reader by construction.
    """
    from tests.support import part_fixtures

    manifest, bodies = part_fixtures.miforge_set_manifest(
        miforge_set_kind="corpus", split_bodies={"train": b'{"body": "one"}\n'}
    )
    manifest["content"]["columns"] = [
        {
            "name": "body",
            "arrow_type": "string",
            "role": "content",
            "semantic": "text",
            "label_values": None,
            "extensions": {},
        },
    ]
    split = next(s for s in manifest["content"]["splits"] if s["name"] == "train")
    split["path"] = "train.jsonl"
    split["rows"] = 1
    held = overrides.pop("held_out", False)
    evaluation = overrides.pop("evaluation_only", False)
    split["held_out"] = bool(held)
    split["evaluation_only"] = bool(evaluation)
    manifest["content"]["splits"] = [split]
    directory = part_fixtures.write_export(tmp_path / "export", manifest, bodies)
    row: dict[str, object] = {
        "id": "part-1",
        "part_kind": "corpus",
        "source": "midataworks_contract",
        "contract_reference": {"type": "corpus", "name": "notes", "version": "1"},
        "external_provenance": None,
        "local_path": str(directory),
        "caveats": [],
        "licence_classes": [],
    }
    row.update(overrides)
    return row


class TestKind:
    def test_a_kind_mismatch_names_both_kinds(self, tmp_path: Path) -> None:
        """E-2: a test set offered to a corpus would train on the rows the run is scored
        against."""
        part = part_payload(tmp_path, part_kind="test_set")
        with pytest.raises(Refusal) as caught:
            parts_reader.resolve(part, PartSpec(kind="test_set", part_id="part-1"), source="corpus")
        assert caught.value.code == "part_kind_mismatch"
        assert caught.value.details["required_kind"] == "corpus"
        assert caught.value.details["supplied_kind"] == "test_set"


class TestAdmissibility:
    def test_a_training_split_resolves(self, tmp_path: Path) -> None:
        resolved = parts_reader.resolve(
            part_payload(tmp_path), PartSpec(kind="corpus", part_id="part-1"), source="corpus"
        )
        assert [split["name"] for split in resolved.splits] == ["train"]
        assert resolved.binding.as_record() == {"text": "text"}

    def test_a_held_out_split_is_refused_by_name(self, tmp_path: Path) -> None:
        """**Mutation control M-3.** F10's rule, called rather than re-implemented."""
        part = part_payload(tmp_path, held_out=True)
        with pytest.raises(Refusal) as caught:
            parts_reader.resolve(part, PartSpec(kind="corpus", part_id="part-1"), source="corpus")
        assert caught.value.code in ("no_training_splits", "split_not_available_for_training")
        assert "train" in json.dumps(caught.value.details)

    def test_an_evaluation_only_split_is_refused(self, tmp_path: Path) -> None:
        part = part_payload(tmp_path, evaluation_only=True)
        with pytest.raises(Refusal):
            parts_reader.resolve(part, PartSpec(kind="corpus", part_id="part-1"), source="corpus")

    def test_both_flags_together_are_refused(self, tmp_path: Path) -> None:
        part = part_payload(tmp_path, held_out=True, evaluation_only=True)
        with pytest.raises(Refusal):
            parts_reader.resolve(part, PartSpec(kind="corpus", part_id="part-1"), source="corpus")

    def test_a_named_held_out_split_is_refused_rather_than_filtered_out(
        self, tmp_path: Path
    ) -> None:
        """Returning an empty list would become a zero-row batch - a step that trains on
        nothing while every log line looks healthy."""
        part = part_payload(tmp_path, held_out=True)
        with pytest.raises(Refusal) as caught:
            parts_reader.resolve(
                part,
                PartSpec(kind="corpus", part_id="part-1", splits=("train",)),
                source="corpus",
            )
        assert caught.value.code == "split_not_available_for_training"
        assert caught.value.details["split"] == "train"

    def test_this_module_does_not_hold_the_two_flags_together(self) -> None:
        """F10's rule lives in **one** place, and this is the structural half of that.

        Asserted here as well as in F10's own guard, because the tempting edit is to copy
        the flags onto this feature's split payload "for display" - and a module that holds
        both is a module that can decide eligibility a second time.
        """
        import ast

        source = Path(parts_reader.__file__).read_text("utf-8")
        constants = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        attributes = {
            node.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute)
        }
        assert not {"held_out", "evaluation_only"} <= (constants | attributes)


class TestColumns:
    def test_a_part_with_no_usable_semantic_is_refused(self, tmp_path: Path) -> None:
        from tests.support import part_fixtures

        manifest, bodies = part_fixtures.miforge_set_manifest(
            miforge_set_kind="corpus", split_bodies={"train": b"{}\n"}
        )
        manifest["content"]["columns"] = [
            {
                "name": "text",
                "arrow_type": "string",
                "role": "content",
                "semantic": "other",
                "label_values": None,
                "extensions": {},
            }
        ]
        split = next(s for s in manifest["content"]["splits"] if s["name"] == "train")
        split["path"] = "train.jsonl"
        manifest["content"]["splits"] = [split]
        directory = part_fixtures.write_export(tmp_path / "e2", manifest, bodies)
        part = part_payload(tmp_path, local_path=str(directory))
        with pytest.raises(Refusal) as caught:
            parts_reader.resolve(part, PartSpec(kind="corpus", part_id="part-1"), source="corpus")
        assert caught.value.code == "corpus_no_content_column"


class TestProvenance:
    def test_a_contract_part_carries_its_reference_and_a_null_external(
        self, tmp_path: Path
    ) -> None:
        reference = parts_reader.part_reference(part_payload(tmp_path), kind="corpus")
        assert reference["contract_reference"] == {
            "type": "corpus",
            "name": "notes",
            "version": "1",
        }
        assert reference["external"] is None

    def test_an_external_part_carries_null_contract_fields(self, tmp_path: Path) -> None:
        """AC-002.16 / R-02.40: **null**, not an empty string and not a synthesised
        reference. A batch record never claims miDataworks provenance for anything
        miDataworks did not publish."""
        part = part_payload(
            tmp_path,
            contract_reference=None,
            external_provenance={
                "source": "file:///data/notes",
                "content_sha256": "a" * 64,
                "retrieved_at": "2026-10-14T00:00:00Z",
            },
        )
        reference = parts_reader.part_reference(part, kind="corpus")
        assert reference["contract_reference"] is None
        assert reference["external"]["source"] == "file:///data/notes"

    def test_a_part_with_neither_form_is_refused(self, tmp_path: Path) -> None:
        part = part_payload(tmp_path, contract_reference=None, external_provenance=None)
        with pytest.raises(Refusal) as caught:
            parts_reader.part_reference(part, kind="corpus")
        assert caught.value.code == "part_provenance_absent"


class TestReadingRows:
    def test_rows_come_back_with_resolvable_identifiers(self, tmp_path: Path) -> None:
        """``prompt_ids`` must be resolvable against the part they came from, which is what
        the contract asks of them."""
        resolved = parts_reader.resolve(
            part_payload(tmp_path), PartSpec(kind="corpus", part_id="part-1"), source="corpus"
        )
        rows = list(parts_reader.read_rows(resolved))
        assert rows == [("part-1:train:0", {"body": "one"})]

    def test_an_unsupported_suffix_refuses_rather_than_sniffing_the_bytes(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "split.txt"
        path.write_text("one\n", encoding="utf-8")
        with pytest.raises(Refusal) as caught:
            list(parts_reader._read_file(path, None))
        assert caught.value.code == "split_format_unsupported"
        assert ".parquet" in caught.value.details["supported"]

    def test_a_corrupt_json_line_is_refused_not_skipped(self, tmp_path: Path) -> None:
        """A skipped row makes a split read shorter than the manifest it was verified
        against."""
        path = tmp_path / "split.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(Refusal) as caught:
            list(parts_reader._read_file(path, None))
        assert caught.value.code == "split_corrupt"
        assert caught.value.details["line"] == 2

    def test_a_parquet_split_refuses_by_name_when_pyarrow_is_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal is **forced**, not waited for. It used to skip, and that was a hole.

        F2 imported ``pyarrow`` lazily and refused by name rather than raising an
        ``ImportError`` nobody can act on, because ``pyproject.toml`` has one owner and the
        dependency was owed. Right call. But the test it wrote was conditional - *"when the
        dependency lands this test's premise disappears with it, which is why it skips"* -
        and the dependency landed on 2026-10-14. On every machine where ``pyarrow`` is now
        installed, which is all of them, the test **skipped** and the refusal went untested.

        A refusal path does not stop mattering because the happy path works. A deployment
        can still be built without the wheel - it has no pure-CPU build on every platform,
        which is exactly why ``bitsandbytes`` is an extra - and then this is the only thing
        standing between an operator and a bare ``ImportError``. So the import is blocked
        here and the refusal asserted unconditionally.
        """
        import builtins

        real_import = builtins.__import__

        def no_pyarrow(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("pyarrow"):
                raise ImportError("blocked by the test: pyarrow is absent")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", no_pyarrow)

        path = tmp_path / "split.parquet"
        path.write_bytes(b"PAR1")
        with pytest.raises(Refusal) as caught:
            list(parts_reader._read_file(path, None))
        assert caught.value.code == "parquet_reader_unavailable"
        assert caught.value.details["dependency"] == "pyarrow"
        assert "pyarrow" in caught.value.message, (
            "the message must name the dependency, or the operator is told a format failed "
            "without being told what to install"
        )

    def test_a_parquet_split_actually_reads_now_that_pyarrow_is_declared(
        self, tmp_path: Path
    ) -> None:
        """The other half, and the one the declaration was made for.

        Parquet is **the format miDataworks' export service writes**, so this is the primary
        input path of miForge's main upstream - not an edge case. Asserted through the real
        reader and a real file written by ``pyarrow`` itself, rather than a fixture whose
        fields agree with the reader by construction.
        """
        parquet = pytest.importorskip("pyarrow.parquet")
        import pyarrow as pa

        path = tmp_path / "split.parquet"
        parquet.write_table(pa.table({"prompt": ["first", "second"], "extra": [1, 2]}), path)

        rows = list(parts_reader._read_file(path, None))
        assert [row["prompt"] for row in rows] == ["first", "second"]

        # RESOLVED COLUMNS ONLY, which is the reader's own contract: a column the caller did
        # not ask for must not arrive, or a split with a stray field changes what a batch
        # record says it drew from.
        narrowed = list(parts_reader._read_file(path, ["prompt"]))
        assert narrowed == [{"prompt": "first"}, {"prompt": "second"}]
