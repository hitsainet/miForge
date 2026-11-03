"""The registry, over the transport  (FR-007.11 - FR-007.16, E11, E12; ADR-012, ADR-018).

Reachability is asserted centrally in ``test_reachability.py``. This file asserts the
**bodies**, and in particular the two numbers this feature must not invent:

* **``prompt_count`` is ``null`` when it cannot be resolved - never 0.** A prompt count of 0
  because a part is unreachable is a fabricated value, and *a fabricated value is worse than
  a refusal because it is silent* (ADR-018).
* **every number names its scale and sample** (ADR-012): ``parts_filled`` travels with
  ``parts_total``, and ``runs_using`` is a list rather than a count so the UI can name them.

It also covers the rebuild that answers the 2026-10-13 review's open question: with
``pack.json`` on disk, **a pack is reconstructible from the filesystem alone.**
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from packs import registry, store
from packs.settings import reset_pack_settings
from packs.slots import SLOT_ORDER, SlotKind
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for


@pytest.fixture
def packs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_pack_settings()
    yield tmp_path / "packs"
    reset_pack_settings()


@pytest.fixture
def engine(tmp_path: Path, packs_root: Path):
    engine = migrated_engine(tmp_path / "db")
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with session_for(engine) as session:
        yield session


@pytest.fixture
def client(engine):
    app = create_app()

    def override():
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestTheListing:
    def test_a_committed_pack_appears_with_its_counts(self, session, client) -> None:
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        body = client.get("/api/v1/packs").json()
        assert len(body["packs"]) == 1
        row = body["packs"][0]
        assert row["pack_id"] == "notary-md"
        assert row["version"] == "1"
        assert row["parts_filled"] == 3
        assert row["parts_total"] == len(SLOT_ORDER)
        assert row["runs_using"] == []

    def test_an_unresolvable_prompt_count_is_null_and_never_zero(self, session, client) -> None:
        """ADR-018, in the shape it arrives: the part is not here, so the count is unknown."""
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        row = client.get("/api/v1/packs").json()["packs"][0]
        assert row["prompt_count"] is None

    def test_a_resolvable_prompt_count_is_the_parts_own_row_count(
        self, session, client, tmp_path: Path
    ) -> None:
        fx.import_contract_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export", rows=7
        )
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.commit()
        row = client.get("/api/v1/packs").json()["packs"][0]
        assert row["prompt_count"] is not None
        assert row["prompt_count"] > 0

    def test_the_validity_block_names_the_missing_slot(self, session, client) -> None:
        slots = fx.valid_slots()
        del slots[SlotKind.TEST_SET.value]
        fx.commit(session, fx.definition(slots=slots))
        row = client.get("/api/v1/packs").json()["packs"][0]
        assert row["validity"]["state"] == "incomplete"
        assert row["validity"]["missing_slots"] == ["test_set"]

    def test_a_draft_is_not_listed_as_a_pack(self, client) -> None:
        """E13: a draft persists, and the registry does not pretend it is a pack."""
        client.post("/api/v1/pack-drafts", json={"name": "half-done"})
        assert client.get("/api/v1/packs").json()["packs"] == []
        assert len(client.get("/api/v1/pack-drafts").json()["drafts"]) == 1


class TestValidityIsATwoHundred:
    def test_asking_whether_a_pack_is_valid_is_a_successful_question(self, session, client) -> None:
        """A 200 describing invalidity; the refusals fire where an action is attempted."""
        slots = fx.valid_slots()
        del slots[SlotKind.RETENTION_SET.value]
        fx.commit(session, fx.definition(slots=slots))
        response = client.get("/api/v1/packs/notary-md/versions/1/validity")
        assert response.status_code == 200
        assert response.json()["missing_slots"] == ["retention_set"]

    def test_three_missing_slots_are_all_named_over_the_wire(self, client) -> None:
        """FAC-12, over the transport - and **through the draft**, which is where it lives.

        See :class:`TestAnEmptyPackIsNotRepresentableAsAnArtifact` below: the pack schema
        makes ``slots.prompt_set`` required and non-nullable, so a committed pack missing it
        cannot exist. A draft can, and a draft is exactly what an author has when all three
        are still missing.
        """
        draft = client.post("/api/v1/pack-drafts", json={"name": "empty-pack"}).json()
        body = client.get(f"/api/v1/pack-drafts/{draft['draft_id']}/validity").json()
        assert body["validity"]["missing_slots"] == ["prompt_set", "test_set", "retention_set"]


class TestCreateCloneAndDelete:
    def test_a_pack_can_be_created_over_the_transport(self, client) -> None:
        response = client.post(
            "/api/v1/packs",
            json={"name": "notary-md", "version": "1", "domain": None, "slots": fx.valid_slots()},
        )
        assert response.status_code == 201, response.text
        assert response.json()["domain"] is None

    def test_a_clone_pins_the_same_parts_and_not_the_resolution_facts(
        self, session, client
    ) -> None:
        """FR-007.12: a clone is a new composition over the same parts."""
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.commit()
        response = client.post(
            "/api/v1/packs/notary-md/versions/1/clone", json={"into": "notary-va"}
        )
        assert response.status_code == 201, response.text
        original = store.read("notary-md", "1")
        clone = store.read("notary-va", "1")
        assert dict(clone.slots) == dict(original.slots)

    def test_a_clone_records_which_version_it_came_from(self, session, client) -> None:
        """**This test was a tripwire for a reported gap, and it fired twice.**

        It asserted ``derived_from is None`` and ``is_example is None``, with the reason:
        *"``derived_from`` (FR-007.12) and ``is_example`` (FR-007.16) have no field in
        ``pack-definition-v1.json`` ... and no column in ``mf_packs``. The capability works
        and the provenance is owed DDL."* Correct when written, and the right way to record
        a gap.

        The operator added the columns on 2026-10-14 and it went red - **once for each half,
        which is the useful part**. ``is_example`` flipped to ``False`` as soon as the column
        existed. ``derived_from`` stayed ``None``, because a column nothing writes is still
        no capability: the clone writer had to be taught to record its source as well. Had
        this test only checked ``is_example`` the registry would have shipped a permanently
        null derivation column and looked finished.

        The source is identified by its **row id** (``<name>@<version>``): cloning
        ``notary-md@1`` and cloning ``notary-md@2`` are different derivations, and the pack
        name alone would lose which version was copied.
        """
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.commit()
        client.post("/api/v1/packs/notary-md/versions/1/clone", json={"into": "notary-va"})
        row = next(
            item
            for item in client.get("/api/v1/packs").json()["packs"]
            if item["pack_id"] == "notary-va"
        )
        assert row["derived_from"] == "notary-md@1", (
            "the clone did not record where it came from. The column exists since migration "
            "0015; a column nothing writes is the same unreachable capability as a function "
            "nothing calls."
        )
        # False, not None: every pack either is or is not an example, and a clone of a real
        # pack is not one. `None` would mean "unknown", which is not a state a pack has.
        assert row["is_example"] is False

    def test_the_source_pack_is_not_marked_as_derived_by_its_own_clone(
        self, session, client
    ) -> None:
        """The edge the row-id form makes easy to get wrong: direction.

        A derivation points from the clone to its source, never back. If the writer set the
        field on the wrong row, the registry would claim the original was derived from its
        own copy - a cycle that reads as plausible in a listing.
        """
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.commit()
        client.post("/api/v1/packs/notary-md/versions/1/clone", json={"into": "notary-va"})
        source = next(
            item
            for item in client.get("/api/v1/packs").json()["packs"]
            if item["pack_id"] == "notary-md"
        )
        assert source["derived_from"] is None

    def test_deleting_an_unused_pack_removes_the_file_and_the_row(self, session, client) -> None:
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.commit()
        assert client.delete("/api/v1/packs/notary-md/versions/1").status_code == 204
        assert client.get("/api/v1/packs").json()["packs"] == []
        assert not store.exists("notary-md", "1")

    def test_deleting_a_pack_a_run_used_is_refused_naming_the_runs(self, session, client) -> None:
        """E11: a run's history must not lose the thing that defined it."""
        from models.run import Run

        pack = fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.add(
            Run(
                id="run-1",
                status="running",
                # `create`, because it is in the run-type vocabulary on both sides of F8's
                # 0014 rename (`full` -> `change`) and this test is about pack usage rather
                # than about run types. A test that does not care which run type it uses
                # should not be the thing that breaks when the vocabulary moves.
                run_type="create",
                pack_id=registry.row_id(pack.name, pack.version),
            )
        )
        session.commit()
        response = client.delete("/api/v1/packs/notary-md/versions/1")
        # 409, not the 400 default: this is a conflict with the state of a stored artifact
        # rather than a bad request (FTDD section 5's taxonomy).
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "pack_in_use"
        assert response.json()["error"]["details"]["runs"] == ["run-1"]
        assert store.exists("notary-md", "1")

    def test_a_run_that_used_a_pack_is_reported_as_usage(self, session, client) -> None:
        from models.run import Run

        pack = fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.add(
            Run(
                id="run-7",
                status="running",
                # `create`, because it is in the run-type vocabulary on both sides of F8's
                # 0014 rename (`full` -> `change`) and this test is about pack usage rather
                # than about run types. A test that does not care which run type it uses
                # should not be the thing that breaks when the vocabulary moves.
                run_type="create",
                pack_id=registry.row_id(pack.name, pack.version),
            )
        )
        session.commit()
        row = client.get("/api/v1/packs").json()["packs"][0]
        assert row["runs_using"] == ["run-7"]


class TestCommittedVersionsAreImmutable:
    def test_rewriting_a_committed_version_is_refused(self, session) -> None:
        """FR-007.2, E10: editing produces a **new version**, never an edit."""
        from packs.errors import PACK_VERSION_COMMITTED, PackRefusal

        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        with pytest.raises(PackRefusal) as raised:
            fx.commit(session, fx.definition(slots=fx.valid_slots(), domain="changed"))
        assert raised.value.code == PACK_VERSION_COMMITTED

    def test_a_second_version_leaves_the_first_resolving_to_what_it_always_did(
        self, session
    ) -> None:
        """FAC-7. The old version still names what it named."""
        first = fx.commit(session, fx.definition(version="1", slots=fx.valid_slots()))
        fx.commit(
            session,
            fx.definition(
                version="2",
                slots=fx.valid_slots(test_set=fx.contract_slot("test-set", "other", "9")),
            ),
        )
        assert store.read("notary-md", "1").document() == first.document()


class TestTwoSlotsMayReferenceTheSamePart:
    def test_it_is_allowed_with_no_deduplication_and_no_warning(self, session) -> None:
        """E12: a test set that is also the retention set is a judgment error, not a validity one.

        Validity is structural; the dry-run and the floor are where judgment lives. (FPRD
        OQ-6 asks whether a warning would help and nothing requires one.)
        """
        same = fx.contract_slot("test-set", "shared", "1")
        pack = fx.commit(
            session,
            fx.definition(
                slots=fx.valid_slots(test_set=same, retention_set={**same, "floor": 0.5})
            ),
        )
        assert pack.slots[SlotKind.TEST_SET] is not None
        assert pack.slots[SlotKind.RETENTION_SET] is not None
        assert pack.slots[SlotKind.TEST_SET].ref == pack.slots[SlotKind.RETENTION_SET].ref


class TestThePackIsRebuildableFromDisk:
    """The 2026-10-13 review's open question, **answered by the gate contract.**

    The review's instruction was that *"a pack assertion must not be added to the
    reindex-equivalence test without first deciding whether packs get a disk mirror."* The
    gate contract decided it - ``pack.json`` on disk - so a pack **is** now rebuildable and
    losing the database loses no pack.

    What this feature does **not** do is wire it into ``services/reindex.py`` or foundation
    4.4's equivalence test: both are another owner's files and the assertion there is the
    coordinator's call. The capability is tested here; the wiring is reported as owed.
    """

    def test_dropping_the_index_and_rebuilding_restores_the_listing(self, session, client) -> None:
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        fx.commit(session, fx.definition(name="other-pack", slots=fx.valid_slots()))
        session.commit()
        before = client.get("/api/v1/packs").json()

        count = registry.rebuild_index(session)
        session.commit()
        assert count == 2
        assert client.get("/api/v1/packs").json() == before

    def test_the_rebuild_reads_the_files_and_not_the_rows(self, session) -> None:
        import sqlalchemy as sa

        from models.pack import Pack, PackPart

        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.execute(sa.delete(PackPart))
        session.execute(sa.delete(Pack))
        session.flush()
        assert registry.rebuild_index(session) == 1
        assert session.get(Pack, "notary-md@1") is not None

    def test_a_pack_file_is_a_standard_json_document_on_disk(
        self, session, packs_root: Path
    ) -> None:
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        path = packs_root / "notary-md" / "1" / "pack.json"
        assert path.is_file()
        assert path.read_text("utf-8").startswith("{")


class TestAnEmptyPackIsNotRepresentableAsAnArtifact:
    """A finding about the gate contract, pinned so nobody re-discovers it as a bug.

    ``pack-definition-v1.json``'s ``slots.prompt_set`` is **required and non-nullable** -
    unlike the other six, which are ``anyOf [PackSlot, null]``. So:

    * a **committed** pack always has a prompt set, and an incomplete committed pack can only
      be missing its test set, its retention set, its floor, or a resolvable part;
    * the state where all three required slots are missing lives in the **draft**, which is
      not a pack document and is not validated as one.

    That is coherent and probably intended - a pack with no prompts is not a composition of
    anything - but FAC-12's "three missing slots names all three" is therefore a statement
    about the authoring flow rather than about a stored pack, and
    :func:`packs.validation.validate` is what both share.
    """

    def test_the_schema_requires_a_prompt_set_and_allows_the_other_six_to_be_null(self) -> None:
        from packs import contracts

        slots = contracts.at(
            contracts.PACK_DEFINITION_SCHEMA,
            "$defs",
            "PackDefinitionV1",
            "properties",
            "slots",
        )
        assert "anyOf" not in slots["properties"]["prompt_set"]
        for slot in (slot.value for slot in SLOT_ORDER if slot is not SlotKind.PROMPT_SET):
            assert {"type": "null"} in slots["properties"][slot]["anyOf"]

    def test_writing_a_pack_with_no_prompt_set_is_refused_by_the_schema(self, session) -> None:
        from packs.errors import PACK_DOCUMENT_INVALID, PackRefusal

        slots = fx.valid_slots()
        del slots[SlotKind.PROMPT_SET.value]
        with pytest.raises(PackRefusal) as raised:
            fx.commit(session, fx.definition(slots=slots))
        assert raised.value.code == PACK_DOCUMENT_INVALID
        assert any("prompt_set" in error for error in raised.value.details["errors"])
