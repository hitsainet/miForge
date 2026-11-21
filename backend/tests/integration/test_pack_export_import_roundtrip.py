"""Export, import, and the same run inputs  (FR-007.14, FR-007.15, FAC-2 = BRD AC10).

AC10's second half: *"a valid pack exported from one installation and imported into another
yields the same run inputs."* Operationally (FPRD §11): the same slot set, the same source
kinds, the same pinned versions, the same floor.

**Through the real writer and the real reader.** No literal fixture document appears here: a
hand-built document agrees with the reader by construction and would keep passing after the
writer changed - *fixtures whose fields agree by construction are how a suite stays green
over a real bug.*

"Another installation" is modelled as a **second pack root and a second database**, both
empty, which is what makes the test about the document that travelled rather than about rows
that happened to be shared.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.canonical import sha256_of
from packs import store
from packs.definition import PackVersionDefinition, run_input_projection
from packs.errors import PACK_DOCUMENT_INVALID, PACK_VERSION_COMMITTED, PackRefusal
from packs.settings import reset_pack_settings
from packs.slots import SlotKind
from packs.transfer import export_bytes, export_filename, import_document
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for

MIXED_SLOTS = {
    SlotKind.PROMPT_SET.value: fx.contract_slot(),
    SlotKind.CORPUS.value: fx.external_slot(
        origin="/data/operator/corpus.jsonl", content_sha256="1" * 64
    ),
    SlotKind.TEST_SET.value: fx.contract_slot("test-set", "notary-test", "1"),
    SlotKind.RETENTION_SET.value: fx.retention_slot(floor=0.72),
}


class Installation:
    """One installation: its own pack root, its own database, its own client."""

    def __init__(self, root: Path, db: Path) -> None:
        self.root = root
        self.engine = migrated_engine(db)

    def use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PACKS_ROOT", str(self.root))
        reset_pack_settings()

    def client(self) -> TestClient:
        app = create_app()

        def override():
            with session_for(self.engine) as session:
                yield session

        app.dependency_overrides[session_dependency] = override
        return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    installation = Installation(tmp_path / "here-packs", tmp_path / "here-db")
    installation.use(monkeypatch)
    yield installation
    installation.engine.dispose()
    reset_pack_settings()


@pytest.fixture
def there(tmp_path: Path):
    installation = Installation(tmp_path / "there-packs", tmp_path / "there-db")
    yield installation
    installation.engine.dispose()


class TestTheRoundTrip:
    def test_an_imported_pack_yields_the_same_run_inputs(
        self, here: Installation, there: Installation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FAC-2, through the real writer and the real reader, across two installations."""
        with session_for(here.engine) as session:
            original = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
        exported = export_bytes(original)

        there.use(monkeypatch)
        with session_for(there.engine) as session:
            imported = import_document(session, exported)

        assert run_input_projection(imported) == run_input_projection(original)
        # And the projection is not vacuous: it names every slot, both source kinds and the
        # floor. A test comparing two empty dicts would pass.
        projection = run_input_projection(imported)
        assert projection[SlotKind.PROMPT_SET.value]["source"] == "midataworks"
        assert projection[SlotKind.CORPUS.value]["source"] == "external"
        assert projection[SlotKind.RETENTION_SET.value]["floor"] == 0.72

    def test_the_document_read_back_off_the_other_installations_disk_is_identical(
        self, here: Installation, there: Installation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with session_for(here.engine) as session:
            original = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
        exported = export_bytes(original)

        there.use(monkeypatch)
        with session_for(there.engine) as session:
            import_document(session, exported)
        assert store.read(original.name, original.version).document() == original.document()

    def test_the_exported_bytes_are_the_bytes_the_digest_was_taken_over(
        self, here: Installation
    ) -> None:
        """So an export is **verifiable**: hash what arrived and compare (ADR-005)."""
        with session_for(here.engine) as session:
            pack = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
        document = json.loads(export_bytes(pack))
        body = {key: value for key, value in document.items() if key != "definition_sha256"}
        assert document["definition_sha256"] == sha256_of(body)

    def test_the_export_carries_no_pointer_back_to_this_installation(
        self, here: Installation
    ) -> None:
        """*By value* (FR-007.14): no database id, no API URL, no filesystem path of ours."""
        with session_for(here.engine) as session:
            pack = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
        text = export_bytes(pack).decode("utf-8")
        assert str(here.root) not in text
        assert "http://" not in text and "https://" not in text
        for forbidden in ("part_id", "local_path", "pack_version_id", "row_id"):
            assert forbidden not in text

    def test_the_export_carries_no_part_content(self, here: Installation) -> None:
        """FR-007.4: a pack is bytes in the hundreds, not megabytes. A reference travels."""
        with session_for(here.engine) as session:
            pack = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
        exported = export_bytes(pack)
        assert len(exported) < 4096
        assert b"prompt" not in exported.replace(b"prompt_set", b"").replace(b"prompt-set", b"")

    def test_the_filename_names_the_pack(self, here: Installation) -> None:
        with session_for(here.engine) as session:
            pack = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
        assert export_filename(pack) == "notary-md@1.pack.json"


class TestImportRefusesWhatItCannotMean:
    def test_a_file_that_is_not_json_is_refused_by_name(self, here: Installation) -> None:
        with session_for(here.engine) as session, pytest.raises(PackRefusal) as raised:
            import_document(session, b"not json at all")
        assert raised.value.code == PACK_DOCUMENT_INVALID

    def test_a_json_array_is_refused(self, here: Installation) -> None:
        with session_for(here.engine) as session, pytest.raises(PackRefusal) as raised:
            import_document(session, b"[]")
        assert raised.value.code == PACK_DOCUMENT_INVALID

    def test_a_future_schema_version_is_refused(self, here: Installation) -> None:
        with session_for(here.engine) as session:
            pack = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
            document = json.loads(export_bytes(pack))
            document["schema_version"] = "2"
            with pytest.raises(PackRefusal) as raised:
                import_document(session, document)
        assert raised.value.details["declared_version"] == "2"

    def test_importing_over_an_existing_version_is_refused(self, here: Installation) -> None:
        """An import that overwrote a version would make every run citing it mean something else."""
        with session_for(here.engine) as session:
            pack = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
            with pytest.raises(PackRefusal) as raised:
                import_document(session, export_bytes(pack))
        assert raised.value.code == PACK_VERSION_COMMITTED

    def test_an_unresolvable_import_is_stored_and_reported_never_rewritten(
        self, here: Installation, there: Installation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E8: import never rewrites a reference to make it resolve locally.

        Rewriting would make the imported pack a **different pack**, and AC10's claim is that
        the imported version yields the *same* run inputs.
        """
        with session_for(here.engine) as session:
            pack = fx.commit(session, fx.definition(slots=MIXED_SLOTS))
        exported = export_bytes(pack)

        there.use(monkeypatch)
        with session_for(there.engine) as session:
            imported = import_document(session, exported)
            from packs.resolver import resolve_slots
            from packs.validation import validate

            verdict = validate(imported, resolve_slots(session, imported))
        assert imported.document() == pack.document()
        assert verdict.state == "incomplete"
        # Reported, and reported as the right **kind** of problem: with no miDataworks
        # configured the contract parts are a service absence, not an authoring defect.
        assert verdict.unavailable_services == ("midataworks",)

    def test_a_failed_import_leaves_no_half_pack(
        self, here: Installation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FPRD §8: *an import is atomic - a pack version appears whole or not at all.*

        The index write is made to fail after the file is written, and the file must be gone.
        """
        from packs import registry, transfer

        with session_for(here.engine) as session:
            pack = fx.definition(slots=MIXED_SLOTS)
            exported = export_bytes(pack)

            def explode(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("index write failed")

            monkeypatch.setattr(transfer.registry, "index", explode)
            with pytest.raises(RuntimeError):
                transfer.import_document(session, exported)
            monkeypatch.setattr(transfer.registry, "index", registry.index)

        assert not store.exists(pack.name, pack.version)


class TestOverTheTransport:
    def test_export_then_import_through_the_api(
        self, here: Installation, there: Installation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with session_for(here.engine) as session:
            fx.commit(session, fx.definition(slots=MIXED_SLOTS))
            session.commit()
        with here.client() as client:
            response = client.get("/api/v1/packs/notary-md/versions/1/export")
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        assert "notary-md@1.pack.json" in response.headers["content-disposition"]
        document = response.json()

        there.use(monkeypatch)
        with there.client() as client:
            created = client.post("/api/v1/packs/import", json=document)
        assert created.status_code == 201, created.text
        assert created.json()["pack_id"] == "notary-md"

    def test_a_malformed_import_is_a_422_naming_the_schema(self, here: Installation) -> None:
        with here.client() as client:
            response = client.post("/api/v1/packs/import", json={"schema_version": "1"})
        assert response.status_code == 422, response.text
        error = response.json()["error"]
        assert error["code"] == "pack_document_invalid"
        assert error["details"]["schema"] == "pack-definition-v1.json"
        assert error["details"]["errors"]


def test_the_projection_excludes_the_retrieval_time(tmp_path: Path, monkeypatch) -> None:
    """``taken_in_at`` is when **this** installation saw the bytes, not part of the pin.

    The digest identifies the bytes; comparing the time would make the round trip compare two
    installations' histories rather than the document that travelled.
    """
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    reset_pack_settings()
    first = fx.definition(
        slots={
            **MIXED_SLOTS,
            SlotKind.CORPUS.value: fx.external_slot(
                origin="/data/operator/corpus.jsonl",
                content_sha256="1" * 64,
                retrieved_at="2026-01-01T00:00:00Z",
            ),
        }
    )
    second = fx.definition(
        slots={
            **MIXED_SLOTS,
            SlotKind.CORPUS.value: fx.external_slot(
                origin="/data/operator/corpus.jsonl",
                content_sha256="1" * 64,
                retrieved_at="2026-09-09T09:09:09Z",
            ),
        }
    )
    assert run_input_projection(first) == run_input_projection(second)
    # And the documents DO differ, so the projection is doing the work rather than the
    # documents happening to be equal.
    assert first.document() != second.document()
    reset_pack_settings()


def test_the_reader_is_the_one_the_store_uses(tmp_path: Path, monkeypatch) -> None:
    """One reader, so a pack read from disk and a pack imported are the same object."""
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    reset_pack_settings()
    pack = fx.definition(slots=MIXED_SLOTS)
    assert PackVersionDefinition.from_document(json.loads(export_bytes(pack))) == pack
    reset_pack_settings()
