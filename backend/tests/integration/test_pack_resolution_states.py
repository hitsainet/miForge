"""The three resolution states, and the 503-vs-422 split  (FR-007.9, FAC-4, E2, E3, E4, E5).

**FAC-4 is the test this feature most needs to be right.** With miDataworks unconfigured, a
pack with contract parts reports ``midataworks_not_configured``, naming the service - **not**
invalid, **not** empty counts. The FTDD calls the distinction *"the single most likely thing
to be implemented carelessly"*, and collapsing it would tell every standalone operator that
their packs are broken.

**Stubbed at F10's interface, never at ``httpx``.** The contract path goes through
``services.parts.resolve.resolve_parts``, so the states are produced by importing a real part
(resolved), not importing it (not found / service absent), and importing a conformant version
that carries no contract reference (unpinnable).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from core.errors import Refusal
from packs import external as external_parts
from packs.definition import slot_from_document
from packs.resolver import bind_indexed_part, resolve_slots
from packs.settings import reset_pack_settings
from packs.slots import SlotKind
from packs.validation import validate
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for


@pytest.fixture
def packs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_pack_settings()
    reset_settings()
    yield tmp_path / "packs"
    reset_pack_settings()
    reset_settings()


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


class TestResolved:
    def test_an_imported_contract_part_resolves_with_its_facts(
        self, session, tmp_path: Path
    ) -> None:
        fx.import_contract_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export", rows=5
        )
        pack = fx.definition(slots={SlotKind.PROMPT_SET.value: fx.contract_slot()})
        resolution = resolve_slots(session, pack)[SlotKind.PROMPT_SET]
        assert resolution.state == "resolved"
        assert resolution.part_id
        assert resolution.local_path
        assert resolution.rows and resolution.rows > 0
        assert resolution.content_sha256

    def test_the_facts_come_from_f10s_payload_rather_than_being_re_derived(
        self, session, tmp_path: Path
    ) -> None:
        """So the registry listing and F10's own part listing cannot disagree about a count."""
        from services.parts import index

        part_id = fx.import_contract_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export"
        )
        payload = index.get_part(session, part_id)
        pack = fx.definition(slots={SlotKind.PROMPT_SET.value: fx.contract_slot()})
        resolution = resolve_slots(session, pack)[SlotKind.PROMPT_SET]
        assert resolution.rows == payload["row_count"]
        assert resolution.content_sha256 == payload["manifest_sha256"]

    def test_an_imported_external_part_resolves_by_its_digest(
        self, session, tmp_path: Path
    ) -> None:
        """The digest is the pin - the same file moved to a new path is the same part."""
        _part_id, provenance = fx.import_external_part(
            session,
            parts_root=tmp_path / "parts",
            export_root=tmp_path / "export",
            origin="/data/operator/prompts.jsonl",
            content_sha256="9" * 64,
        )
        pack = fx.definition(
            slots={
                SlotKind.PROMPT_SET.value: fx.external_slot(
                    origin=provenance["source"],
                    content_sha256=provenance["content_sha256"],
                    retrieved_at=provenance["retrieved_at"],
                )
            }
        )
        resolution = resolve_slots(session, pack)[SlotKind.PROMPT_SET]
        assert resolution.state == "resolved"
        assert resolution.local_path

    def test_the_digest_is_what_matches_and_not_merely_the_presence_of_a_part(
        self, session, tmp_path: Path
    ) -> None:
        """**A test finding from F7's own mutation sweep (M14), written because it survived.**

        The first version of this file asserted only that an external slot resolved - with
        **one** external part in the index, which made ``lookup_by_digest`` pass even when
        its comparison was replaced by ``if True``. The pin is the digest, so the test has to
        distinguish two parts and reject an unknown one.

        Re-run as a negative control after writing it: with the digest comparison mutated to
        ``if True``, this goes red.
        """
        _first, first_provenance = fx.import_external_part(
            session,
            parts_root=tmp_path / "parts",
            export_root=tmp_path / "export-one",
            origin="/data/one.jsonl",
            content_sha256="1" * 64,
        )
        second_id, second_provenance = fx.import_external_part(
            session,
            parts_root=tmp_path / "parts",
            export_root=tmp_path / "export-two",
            origin="/data/two.jsonl",
            content_sha256="2" * 64,
        )
        assert first_provenance["content_sha256"] != second_provenance["content_sha256"]

        # The SECOND part's digest must resolve to the SECOND part, not to whichever external
        # part the index happens to hold first.
        pack = fx.definition(
            slots={
                SlotKind.PROMPT_SET.value: fx.external_slot(
                    origin=second_provenance["source"],
                    content_sha256=second_provenance["content_sha256"],
                    retrieved_at=second_provenance["retrieved_at"],
                )
            }
        )
        resolution = resolve_slots(session, pack)[SlotKind.PROMPT_SET]
        assert resolution.state == "resolved"
        assert resolution.part_id == second_id

        # And a digest nothing carries is **not found** - not "the first external part".
        unknown = fx.definition(
            slots={
                SlotKind.PROMPT_SET.value: fx.external_slot(
                    origin=second_provenance["source"], content_sha256="f" * 64
                )
            }
        )
        missing = resolve_slots(session, unknown)[SlotKind.PROMPT_SET]
        assert missing.state == "not_found"
        assert missing.part_id is None

    def test_an_external_part_resolves_with_no_sibling_configured_at_all(
        self, session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FR-007.31's first half: **nothing** in this path reaches a sibling.

        Not merely "the client was stubbed" - the base URL is unset, so a client call would
        refuse by name, and this path never makes one.
        """
        monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
        _part_id, provenance = fx.import_external_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export"
        )
        pack = fx.definition(
            slots={
                SlotKind.PROMPT_SET.value: fx.external_slot(
                    origin=provenance["source"],
                    content_sha256=provenance["content_sha256"],
                    retrieved_at=provenance["retrieved_at"],
                )
            }
        )
        assert resolve_slots(session, pack)[SlotKind.PROMPT_SET].state == "resolved"


class TestServiceUnavailableIsNotNotFound:
    def test_an_unimported_contract_part_with_no_midataworks_names_the_service(
        self, session
    ) -> None:
        """FAC-4, and E3. **Nothing is wrong with the pack.**

        With nothing configured there is no way to tell *"this version was never published"*
        from *"this installation has not imported it yet"*, and only one of those is the
        pack's fault.
        """
        pack = fx.definition(slots={SlotKind.PROMPT_SET.value: fx.contract_slot()})
        resolution = resolve_slots(session, pack)[SlotKind.PROMPT_SET]
        assert resolution.state == "service_unavailable"
        assert resolution.service == "midataworks"
        assert resolution.rows is None

    def test_with_midataworks_configured_the_same_part_is_not_found(
        self, session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E2. Configured and still absent: the part is gone, and the pack must change."""
        monkeypatch.setenv("MIDATAWORKS_BASE_URL", "http://midataworks.test")
        # The settings are cached process-wide, so an env var set without this is an env var
        # the code never reads - which is how a test ends up asserting against a
        # configuration it did not set.
        reset_settings()
        pack = fx.definition(slots={SlotKind.PROMPT_SET.value: fx.contract_slot()})
        resolution = resolve_slots(session, pack)[SlotKind.PROMPT_SET]
        assert resolution.state == "not_found"
        assert resolution.service is None

    def test_the_counts_are_never_reported_as_zero(self, session, client) -> None:
        """ADR-018: 0 prompts because a service is absent is a fabricated measurement."""
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.commit()
        row = client.get("/api/v1/packs").json()["packs"][0]
        assert row["prompt_count"] is None
        assert row["prompt_count"] != 0

    def test_an_absent_service_is_a_503_over_the_transport(self, session, client) -> None:
        """A 503 naming the service, while a pack that is genuinely wrong is a 422."""
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        session.commit()
        response = client.post(
            "/api/v1/pack-drafts", json={"name": "x"}
        )  # a draft, to have a 2xx baseline
        assert response.status_code == 201

        from packs.errors import MIDATAWORKS_NOT_CONFIGURED, PackRefusal

        pack = fx.definition(slots=fx.valid_slots())
        with pytest.raises(PackRefusal) as raised:
            validate(pack, resolve_slots(session, pack)).raise_if_invalid()
        assert raised.value.code == MIDATAWORKS_NOT_CONFIGURED
        from api.app import status_for

        assert status_for(raised.value.code) == 503
        assert status_for("pack_part_unresolved") == 422
        assert status_for("pack_incomplete") == 422

    def test_a_pack_of_external_parts_is_unaffected_by_an_absent_service(
        self, session, tmp_path: Path
    ) -> None:
        """E4: nothing is degraded and nothing is reported missing."""
        _part_id, provenance = fx.import_external_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export"
        )
        slot = fx.external_slot(
            origin=provenance["source"],
            content_sha256=provenance["content_sha256"],
            retrieved_at=provenance["retrieved_at"],
        )
        pack = fx.definition(
            slots={
                SlotKind.PROMPT_SET.value: slot,
                SlotKind.TEST_SET.value: slot,
                SlotKind.RETENTION_SET.value: {**slot, "floor": 0.5},
            }
        )
        verdict = validate(pack, resolve_slots(session, pack))
        assert verdict.is_valid
        assert verdict.unavailable == ()


class TestUnpinnable:
    def test_a_version_with_no_contract_reference_cannot_pin_a_slot(
        self, session, tmp_path: Path
    ) -> None:
        """E5 / OQ-3, and the refusal's spelling is F10's.

        Such a version is **conformant**: ``Target.contract_reference`` is
        ``anyOf [ContractReference, null]`` *and* in ``required``, so a null is an assertion
        miDataworks deliberately permits, and F10 records it faithfully. It names nothing
        reproducible, so it cannot pin a pack slot - and **nothing synthesises a reference**
        from the version id or the dataset name.
        """
        part_id = fx.import_contract_part(
            session,
            parts_root=tmp_path / "parts",
            export_root=tmp_path / "export",
            reference=None,
        )
        with pytest.raises(Refusal) as raised:
            bind_indexed_part(session, part_id)
        assert raised.value.code == "part_unpinnable"
        assert part_id in raised.value.details["part_ids"]

    def test_binding_an_indexed_contract_part_yields_its_reference(
        self, session, tmp_path: Path
    ) -> None:
        part_id = fx.import_contract_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export"
        )
        reference = bind_indexed_part(session, part_id)
        assert reference.source == "midataworks"
        assert (reference.type, reference.name, reference.version) == (
            "prompt-set",
            "support-tickets",
            "3",
        )

    def test_an_external_part_is_bound_by_its_provenance_not_by_f10s_reference_lookup(
        self, session, tmp_path: Path
    ) -> None:
        """**A finding about F10's published entry point**, pinned here.

        F10's docstring says a pack slot may be pinned *"an external part by the id of the row
        that carries its provenance"* - and ``resolve_parts`` classifies on
        ``contract_reference is not None``, so **every** external part lands in
        ``.unpinnable`` and ``resolve_one`` refuses every such id. An external part therefore
        cannot be bound through that entry point at all.

        F7 reads the provenance instead (``packs.external.ref_for_part``) and reports the
        finding rather than editing F10's file. This test is what keeps the finding true: if
        F10 later adds a provenance-aware outcome, the first assertion starts failing and
        somebody re-reads this comment.
        """
        part_id, provenance = fx.import_external_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export"
        )
        with pytest.raises(Refusal) as raised:
            bind_indexed_part(session, part_id)
        assert raised.value.code == "part_unpinnable"

        reference = external_parts.ref_for_part(session, part_id)
        assert reference.source == "external"
        assert reference.origin == provenance["source"]
        assert reference.content_sha256 == provenance["content_sha256"]

    def test_the_part_ref_route_answers_for_both_source_kinds(
        self, session, client, tmp_path: Path
    ) -> None:
        contract_id = fx.import_contract_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export-a"
        )
        external_id, _provenance = fx.import_external_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export-b"
        )
        session.commit()

        contract = client.get(f"/api/v1/pack-part-refs/{contract_id}").json()
        assert contract["source"] == "midataworks"
        assert contract["inner_form"] == "prompt-set/support-tickets@3"

        external = client.get(f"/api/v1/pack-part-refs/{external_id}").json()
        assert external["source"] == "external"
        assert external["inner_form"].startswith("external/")
        # No display form in the response: the pack a part is addressed THROUGH is not known
        # here, and inventing one would be the layer confusion R-02.35 forbids.
        assert "display_form" not in external

        # The slot document each one fills (task 14.0's slot picker sends it unchanged), from
        # the one writer - and it reads back through the one reader as the same reference.
        assert contract["slot_document"] == {
            "contract_reference": {k: contract["ref"][k] for k in ("type", "name", "version")},
            "external": None,
            "floor": None,
        }
        assert external["slot_document"]["contract_reference"] is None
        assert external["slot_document"]["external"]["source"] == external["ref"]["origin"]
        for body in (contract, external):
            filled = slot_from_document(SlotKind.PROMPT_SET, body["slot_document"])
            assert filled is not None
            assert filled.ref.model_dump(mode="json") == body["ref"]

    def test_a_contract_part_is_not_offered_as_an_external_one(
        self, session, tmp_path: Path
    ) -> None:
        """The two source kinds do not convert, and the refusal says which it is."""
        part_id = fx.import_contract_part(
            session, parts_root=tmp_path / "parts", export_root=tmp_path / "export"
        )
        with pytest.raises(Refusal) as raised:
            external_parts.ref_for_part(session, part_id)
        assert raised.value.code == "part_not_external"


class TestResolutionIsBatched:
    def test_one_lookup_covers_every_contract_slot(
        self, session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§9's registry budget assumes one call per service, not one per slot.

        Asserted on the **call count**, because a per-slot resolver would still return the
        right answers - it would only be six chances for the answer to change half way
        through a validation that is supposed to be about one pack.
        """
        from packs import resolver

        calls: list[int] = []
        real = resolver.part_lookup.resolve_parts

        def counting(session_arg, targets):
            calls.append(len(targets))
            return real(session_arg, targets)

        monkeypatch.setattr(resolver.part_lookup, "resolve_parts", counting)
        pack = fx.definition(slots=fx.valid_slots())
        resolve_slots(session, pack)
        assert calls == [3]
