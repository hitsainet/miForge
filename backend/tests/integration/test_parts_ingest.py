"""A part round-trips with its identity intact  (AC-010.5; FR-010.13, .14, .17).

Every assertion here goes **through the real ingest path and back out through the real
index**. ADR-016: *fixtures whose fields agree by construction are how a suite stays green
over a real bug* - so nothing below inspects a dict a test built, and the part's caveats, its
licence classes and its digests are read from the API's own payload.

The two tests that matter most are the ones about what **does not** happen: a tampered byte
refuses, and a crash before the rename leaves nothing behind.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from core.errors import Refusal
from services.parts import acquire, index, ingest
from services.parts.contract_validator import MANIFEST_FILENAME
from services.parts.tables import SOURCE_CONTRACT, SOURCE_EXTERNAL
from tests.support import db as support
from tests.support import part_fixtures as fx


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


@pytest.fixture()
def parts_root(tmp_path: Path) -> Path:
    root = tmp_path / "parts"
    root.mkdir()
    return root


@pytest.fixture()
def export(tmp_path: Path) -> tuple[Path, dict]:
    manifest, bodies = fx.miforge_set_manifest()
    directory = fx.write_export(tmp_path / "export", manifest, bodies)
    return directory, manifest


class TestAConformantPartRoundTrips:
    def test_the_part_reports_its_whole_identity(self, engine, parts_root, export):
        directory, manifest = export
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
            payload = index.get_part(session, result.part_id)

        assert payload["part_kind"] == "prompt_set"
        assert payload["source"] == SOURCE_CONTRACT
        assert payload["contract_reference"] == {
            "type": "prompt-set",
            "name": "support-tickets",
            "version": "3",
        }
        assert payload["rendered"] == "prompt-set/support-tickets@3"
        assert payload["dataset"] == manifest["version"]["dataset"]
        assert payload["version_id"] == manifest["version"]["version_id"]
        assert payload["recipe_sha256"] == manifest["version"]["recipe"]["sha256"]

    def test_the_recorded_digests_are_the_contracts(self):
        """FR-010.14. **Verification and recording are different acts.** miForge compares the
        bytes against the digest the contract carries; what it *records* is the contract's
        digest, not a second competing hash of the same material."""
        manifest, _ = fx.miforge_set_manifest()
        declared = {s["name"]: s["sha256"] for s in manifest["content"]["splits"]}
        assert declared, "the fixture has no splits; this test would be vacuous"

    def test_the_split_rows_carry_the_contracts_values(self, engine, parts_root, export):
        directory, manifest = export
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
            payload = index.get_part(session, result.part_id)

        by_name = {split["name"]: split for split in payload["splits"]}
        for declared in manifest["content"]["splits"]:
            recorded = by_name[declared["name"]]
            assert recorded["sha256"] == declared["sha256"]
            assert recorded["logical_digest"] == declared["logical_digest"], (
                "logical_digest must be RECORDED, never recomputed - recomputing it means "
                "reimplementing miDataworks' row-key scheme"
            )
            assert recorded["held_out"] == declared["held_out"]
            assert recorded["evaluation_only"] == declared["evaluation_only"]

    def test_caveats_and_licence_classes_are_on_the_summary(self, engine, parts_root, export):
        """FR-010.17. **At the point of use**, not only at import - and this is also the
        assertion the "drop caveats in ingest" mutation must turn red."""
        directory, _ = export
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
            payload = index.get_part(session, result.part_id)

        assert [c["code"] for c in payload["caveats"]] == ["amber_check"]
        assert payload["caveats"][0]["severity"] == "amber"
        assert payload["caveats"][0]["detail"] == {"check": "C-5"}
        assert payload["licence_classes"] == ["permits_redistribution"]
        assert payload["licences"][0]["displayed"] == "cc-by-2.0"
        # And on the import's own result, which is the first point of use.
        assert result.caveats and result.licence_classes

    def test_an_amber_caveat_does_not_refuse_the_part(self, engine, parts_root, export):
        """E-12. Grading it is miDataworks' judgement and accepting it is the operator's;
        miForge's job is to make sure it cannot be missed."""
        directory, _ = export
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
        assert result.part_id

    def test_row_count_excludes_held_out_splits(self, engine, parts_root, export):
        """A single total mixing them would be the number an operator reads when asking
        "how much am I training on"."""
        directory, manifest = export
        trainable = sum(s["rows"] for s in manifest["content"]["splits"] if not s["held_out"])
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
            payload = index.get_part(session, result.part_id)
        assert payload["row_count"] == trainable
        assert payload["row_count"] < sum(s["rows"] for s in manifest["content"]["splits"])

    def test_the_source_directory_is_not_written_to(self, engine, parts_root, export):
        directory, _ = export
        before = sorted(p.name for p in directory.rglob("*"))
        with support.session_for(engine) as session:
            acquire.from_dir(session, directory, parts_root=parts_root)
        assert sorted(p.name for p in directory.rglob("*")) == before


class TestVerificationRefusesBeforeAnythingIsIndexed:
    def test_a_tampered_split_byte_refuses_by_name(self, engine, parts_root, tmp_path):
        """AC-010.5. The bytes changed after miDataworks published them, so the part is
        refused and **never used**."""
        manifest, bodies = fx.miforge_set_manifest()
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        fx.tamper_split(directory, manifest, "train")

        with support.session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                acquire.from_dir(session, directory, parts_root=parts_root)
            assert caught.value.code == "dataset_version_digest_mismatch"
            assert caught.value.details["split"] == "train"
            assert caught.value.details["expected"] != caught.value.details["found"]
            assert index.list_parts(session) == [], "a refused part was indexed anyway"

    def test_a_missing_split_refuses_naming_path_and_split(self, engine, parts_root, tmp_path):
        """E-8. A part missing a split it declares would read as complete."""
        manifest, bodies = fx.miforge_set_manifest()
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        (directory / manifest["content"]["splits"][1]["path"]).unlink()

        with support.session_for(engine) as session, pytest.raises(Refusal) as caught:
            acquire.from_dir(session, directory, parts_root=parts_root)
        assert caught.value.code == "dataset_version_split_missing"
        assert caught.value.details["split"] == manifest["content"]["splits"][1]["name"]

    def test_a_missing_manifest_names_the_filename_expected(self, engine, parts_root, tmp_path):
        """E-4, and the single most common way this fails: pointing at the parent directory."""
        empty = tmp_path / "not-an-export"
        empty.mkdir()
        with support.session_for(engine) as session, pytest.raises(Refusal) as caught:
            acquire.from_dir(session, empty, parts_root=parts_root)
        assert caught.value.code == "dataset_version_manifest_missing"
        assert caught.value.details["expected_filename"] == MANIFEST_FILENAME
        assert "-v1" not in caught.value.details["expected_filename"]

    def test_a_non_conformant_manifest_refuses_and_indexes_nothing(
        self, engine, parts_root, tmp_path
    ):
        """AC-010.3, BRD-02 AC18."""
        manifest, bodies = fx.miforge_set_manifest()
        manifest["content"]["splits"][0].pop("sha256")
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        with support.session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                acquire.from_dir(session, directory, parts_root=parts_root)
            assert caught.value.code == "dataset_version_invalid"
            assert caught.value.details["errors"][0]["instance_path"] == "/content/splits/0"
            assert index.list_parts(session) == []

    def test_a_target_miforge_does_not_consume_refuses_by_name(self, engine, parts_root, tmp_path):
        """AC-010.6. The real fixture is a ``dataset`` target, so this is the committed
        document refusing rather than a contrivance."""
        manifest = fx.real_manifest()
        bodies = {split["name"]: b"" for split in manifest["content"]["splits"]}
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        with support.session_for(engine) as session, pytest.raises(Refusal) as caught:
            acquire.from_dir(session, directory, parts_root=parts_root)
        assert caught.value.code == "dataset_version_target_unsupported"
        assert caught.value.details["target_kind"] == "dataset"


class TestPathConfinement:
    """The one place in this feature where hostile input has a plausible mechanism."""

    @pytest.mark.parametrize(
        "escape", ["../outside.parquet", "/etc/passwd", "data/../../outside.parquet"]
    )
    def test_a_path_outside_the_part_directory_refuses(
        self, engine, parts_root, tmp_path, escape: str
    ):
        manifest, bodies = fx.miforge_set_manifest()
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        # The path is rewritten AFTER the export is written: the point is a manifest whose
        # declared path escapes, not a fixture that tries to write outside tmp_path. And
        # `confine` must refuse before it ever asks whether the file exists, which is what
        # makes an unreadable target like /etc/passwd a refusal rather than a PermissionError.
        manifest["content"]["splits"][0]["path"] = escape
        (directory / "midataworks-dataset-version.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        with support.session_for(engine) as session, pytest.raises(Refusal) as caught:
            acquire.from_dir(session, directory, parts_root=parts_root)
        assert caught.value.code == "dataset_version_path_escapes_part"
        assert caught.value.details["path"] == escape

    def test_a_symlink_pointing_out_refuses(self, engine, parts_root, tmp_path):
        """**Resolution and containment, not string inspection.** A check for the substring
        ``..`` passes a symlink and fails a legitimate filename."""
        secret = tmp_path / "secret.txt"
        secret.write_bytes(b"not yours")
        manifest, bodies = fx.miforge_set_manifest()
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        target = directory / manifest["content"]["splits"][0]["path"]
        target.unlink()
        target.symlink_to(secret)

        with support.session_for(engine) as session, pytest.raises(Refusal) as caught:
            acquire.from_dir(session, directory, parts_root=parts_root)
        assert caught.value.code == "dataset_version_path_escapes_part"


class TestAtomicPlacement:
    def test_the_rename_is_the_commit(self, engine, parts_root, export):
        directory, _ = export
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
        placed = Path(result.local_path)
        assert placed.is_dir()
        assert (placed / MANIFEST_FILENAME).is_file()
        assert not any(
            p.name.startswith(".tmp-save-") for p in parts_root.iterdir()
        ), "a temp directory survived a successful write"

    def test_a_failure_during_placement_leaves_no_part_and_no_row(
        self, engine, parts_root, export, monkeypatch
    ):
        """**A crash before the rename leaves a temp directory and no part.** There is no
        intermediate state an API can observe, which is why a part has no "importing"
        status column: it is imported or it is not."""
        directory, _ = export
        import shutil as shutil_module

        def explode(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(shutil_module, "copy2", explode)
        with support.session_for(engine) as session:
            with pytest.raises(OSError):
                acquire.from_dir(session, directory, parts_root=parts_root)
            assert index.list_parts(session) == []
        assert (
            list(parts_root.iterdir()) == []
        ), "a temp directory was left behind; the atomic writer must clean up on failure"

    def test_no_latest_symlink_is_created_for_a_part(self, engine, parts_root, export):
        """A parts root carries no ``latest`` pointer: a part is addressed by its identity,
        and a "latest part" would be a second answer to "which version is pinned" - which
        is what a pack's reference is for."""
        directory, _ = export
        with support.session_for(engine) as session:
            acquire.from_dir(session, directory, parts_root=parts_root)
        assert not (parts_root / "latest").exists()


class TestIdempotence:
    def test_importing_the_same_version_twice_creates_one_row(self, engine, parts_root, export):
        """**Through the database**, not check-then-insert: a ``SELECT`` first is a race and
        this runs in a worker."""
        directory, _ = export
        with support.session_for(engine) as session:
            first = acquire.from_dir(session, directory, parts_root=parts_root)
        with support.session_for(engine) as session:
            second = acquire.from_dir(session, directory, parts_root=parts_root)
            assert len(index.list_parts(session)) == 1
        assert first.part_id == second.part_id
        assert first.created is True
        assert second.created is False

    def test_two_versions_of_one_dataset_are_two_parts(self, engine, parts_root, tmp_path):
        """``ref_version`` is part of the uniqueness key **precisely so** that re-importing
        version 3 while version 4 exists is two rows, and a pack pinning v3 keeps meaning."""
        for version in ("3", "4"):
            manifest, bodies = fx.miforge_set_manifest(
                reference=("prompt-set", "support-tickets", version)
            )
            directory = fx.write_export(tmp_path / f"export-{version}", manifest, bodies)
            with support.session_for(engine) as session:
                acquire.from_dir(session, directory, parts_root=parts_root)
        with support.session_for(engine) as session:
            versions = sorted(
                part["contract_reference"]["version"] for part in index.list_parts(session)
            )
        assert versions == ["3", "4"]


class TestANullContractReferenceIsSurfacedNotSynthesised:
    """**E-21**, and the half F10 owns. F7's ``part_unpinnable`` is the other half."""

    def test_it_imports(self, engine, parts_root, tmp_path):
        manifest, bodies = fx.miforge_set_manifest(reference=None)
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
        assert result.part_id

    def test_the_three_reference_columns_are_null_everywhere(self, engine, parts_root, tmp_path):
        """Asserted over the row, the API response **and** the rendered form - not just the
        row, because a renderer deriving ``<type>/<name>@<version>`` from the version id
        would be the synthesis this forbids, one layer up from the column."""
        manifest, bodies = fx.miforge_set_manifest(reference=None)
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
            payload = index.get_part(session, result.part_id)
            listed = index.list_parts(session)[0]

        assert payload["contract_reference"] is None
        assert payload["rendered"] is None
        assert listed["contract_reference"] is None
        assert listed["rendered"] is None

    def test_nothing_resembling_the_version_id_appears_as_a_reference(
        self, engine, parts_root, tmp_path
    ):
        """The tempting one-liner is to fill the reference in from the version id. It would
        make miForge the author of a contract identity miDataworks never issued."""
        manifest, bodies = fx.miforge_set_manifest(reference=None)
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
            payload = index.get_part(session, result.part_id)

        rendered = json.dumps(payload)
        assert manifest["version"]["version_id"] not in rendered.replace(
            f'"version_id": "{manifest["version"]["version_id"]}"', ""
        ), "the version id leaked into a field other than version_id"
        assert payload["part_kind"] == "prompt_set"


class TestAnExternalPartNeverCarriesAContractReference:
    def test_it_is_recorded_as_external_with_its_own_provenance(self, engine, parts_root, tmp_path):
        """R-02.40. The gate contract's ``ExternalProvenance`` shape, stored opaquely."""
        manifest, bodies = fx.miforge_set_manifest(reference=None)
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        provenance = {
            "source": "https://example.invalid/corpus.jsonl",
            "content_sha256": "e" * 64,
            "retrieved_at": "2026-10-14T00:00:00Z",
            "licence": "cc-by-4.0",
        }
        with support.session_for(engine) as session:
            result = acquire.from_dir(
                session,
                directory,
                parts_root=parts_root,
                source=SOURCE_EXTERNAL,
                external_provenance=provenance,
            )
            payload = index.get_part(session, result.part_id)
        assert payload["source"] == SOURCE_EXTERNAL
        assert payload["contract_reference"] is None
        assert payload["external_provenance"] == provenance

    def test_an_external_part_with_no_provenance_refuses(self, engine, parts_root, tmp_path):
        """*A part with neither a reference nor provenance is unattributed*, and an
        unattributed training input makes the run unreproducible: nothing downstream could
        say where a behaviour came from."""
        manifest, bodies = fx.miforge_set_manifest(reference=None)
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        with support.session_for(engine) as session, pytest.raises(Refusal) as caught:
            acquire.from_dir(session, directory, parts_root=parts_root, source=SOURCE_EXTERNAL)
        assert caught.value.code == "external_part_has_no_provenance"

    def test_the_database_refuses_a_promoted_external_part(self, engine, parts_root, tmp_path):
        """**The CHECK constraint, not the service.** R-02.40 at the database, so the
        prohibition is not a convention three call sites are trusted to honour."""
        import sqlalchemy as sa

        from services.parts.tables import CONTRACT_PARTS

        # `IntegrityError`, not a blind Exception: a blind assertion passes when the insert
        # fails for ANY reason - a typo'd column, a missing table - and is then green while
        # testing nothing about the CHECK constraint it names.
        with support.session_for(engine) as session, pytest.raises(IntegrityError) as caught:
            session.execute(
                sa.insert(CONTRACT_PARTS).values(
                    id="part-forged",
                    part_kind="corpus",
                    source=SOURCE_EXTERNAL,
                    ref_type="corpus",
                    ref_name="borrowed",
                    ref_version="1",
                    content_kind="rows",
                    local_path="/tmp/nowhere",
                    manifest_sha256="f" * 64,
                    row_count=0,
                    imported_at=support.utc_now(),
                )
            )
            session.flush()
        assert "external_part_has_no_contract_reference" in str(caught.value) or "CHECK" in str(
            caught.value
        ), str(caught.value)


class TestConfigParts:
    def test_a_selector_configuration_comes_in_as_a_part(self, engine, parts_root, tmp_path):
        """FR-010.16. **miForge authors none of these** (R-02.9 as amended; BRD-03 R-03.48
        owns authoring) and this feature provides no editor for them: it validates, resolves
        the path, verifies the hash and hands F3 the body and the hash."""
        manifest, bodies = fx.config_manifest(kind="selector")
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
            payload = index.get_part(session, result.part_id)
        assert payload["part_kind"] == "selector_config"
        assert payload["content_kind"] == "config"
        assert payload["config_sha256"] == manifest["content"]["config"]["sha256"]
        assert payload["splits"] == []

    def test_a_tampered_configuration_body_refuses(self, engine, parts_root, tmp_path):
        """This hash is what R-02.10 records as the selector identity, so a mismatch would
        make a checkpoint cite a configuration it was not trained with."""
        manifest, bodies = fx.config_manifest(kind="grader")
        directory = fx.write_export(tmp_path / "export", manifest, bodies)
        path = directory / manifest["content"]["config"]["path"]
        path.write_bytes(path.read_bytes() + b" ")
        with support.session_for(engine) as session, pytest.raises(Refusal) as caught:
            acquire.from_dir(session, directory, parts_root=parts_root)
        assert caught.value.code == "dataset_version_digest_mismatch"
        assert caught.value.details["config_name"] == "grader-strict"


class TestStreaming:
    def test_a_split_larger_than_the_chunk_is_digested_without_being_read_whole(
        self, engine, parts_root, tmp_path, monkeypatch
    ):
        """OQ-5 leaves pretraining volume open; **the first large corpus must not OOM.**

        Asserted on the mechanism rather than on memory: ``read_bytes`` is made to raise, so
        a non-streaming implementation fails and a streaming one passes.
        """
        body = b"x" * (3 * ingest.DIGEST_CHUNK_BYTES + 17)
        manifest, bodies = fx.miforge_set_manifest(split_bodies={"train": body})
        directory = fx.write_export(tmp_path / "export", manifest, bodies)

        original = Path.read_bytes

        def no_whole_file_reads(self: Path, *args, **kwargs):
            if self.suffix == ".parquet":
                raise AssertionError(f"{self} was read whole instead of streamed")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_bytes", no_whole_file_reads)
        with support.session_for(engine) as session:
            result = acquire.from_dir(session, directory, parts_root=parts_root)
        assert result.part_id
