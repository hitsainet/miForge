"""The refusals, against a fake written independently of their repository  (AC-010.15).

**The happy path is the half that was going to work anyway.** Every expensive bug on this
seam is in a refusal: a misconfiguration reported as a network blip, a rejected credential
retried forever, a 413 discovered after uploading the payload. So the fake implements the
refusals (``tests/support/fake_midataworks.py``) and these tests exercise them.

**Written independently from miDataworks' route signatures and schemas**, not from a double
in that repository - because a shared double encodes its author's reading of its own API on
both sides of the test, and then a misunderstanding passes. miDataworks took exactly this
care in its own standalone test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clients.midataworks import MiDataworksClient
from clients.midataworks_errors import (
    MiDataworksRefused,
    MiDataworksUnauthorized,
    MiDataworksUnreachable,
)
from core.config import reset_settings
from core.errors import Refusal
from services.parts import acquire, index
from tests.support import db as support
from tests.support import part_fixtures as fx
from tests.support.fake_midataworks import FakeMiDataworks


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("MIDATAWORKS_BASE_URL", "http://dw.invalid:8000")
    reset_settings()
    yield
    reset_settings()


@pytest.fixture()
def fake() -> FakeMiDataworks:
    return FakeMiDataworks()


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


def client(fake: FakeMiDataworks, **kwargs) -> MiDataworksClient:
    return MiDataworksClient(
        action="talking to miDataworks",
        capability="part_import_http",
        transport=fake.transport(),
        sleep=lambda _s: None,
        **kwargs,
    )


class TestTheWireIsRight:
    def test_a_queue_is_created_with_miforge_as_its_origin_app(self, fake):
        """``origin_app`` comes from the **creating** caller's agent header, which is why
        miForge must create the queue it posts to."""
        with client(fake) as api:
            queue = api.create_external_queue(
                question="Is this good?",
                label_set=["good", "bad"],
                external_ref={"run_id": "run-1"},
            )
        assert queue["origin_app"] == "miforge"
        assert queue["kind"] == "external"
        assert queue["external_ref"] == {"run_id": "run-1"}

    def test_posting_to_another_apps_queue_is_refused_upstream(self, fake):
        """E-18. **miForge reports the refusal rather than creating a second queue** - a
        second queue would orphan the first and its decisions."""
        from tests.support.fake_midataworks import FakeQueue

        fake.queues["rq-theirs"] = FakeQueue(
            id="rq-theirs",
            kind="external",
            question="theirs",
            label_set=["a", "b"],
            origin_app="someone-else",
            external_ref={},
        )
        with pytest.raises(MiDataworksRefused) as caught, client(fake) as api:
            api.post_candidates(
                "rq-theirs",
                [{"external_id": "x1", "prompt": "p", "completion": "c", "provenance": {}}],
            )
        assert caught.value.details["midataworks_code"] == "ORIGIN_APP_MISMATCH"

    def test_a_repeated_external_id_returns_the_existing_item(self, fake):
        """**The idempotence a re-post after a crash relies on.**"""
        with client(fake) as api:
            queue = api.create_external_queue(question="q", label_set=["a", "b"], external_ref={})
            body = [{"external_id": "x1", "prompt": "p", "completion": "c", "provenance": {}}]
            first = api.post_candidates(queue["id"], body)
            second = api.post_candidates(queue["id"], body)
        assert first[0].created is True
        assert second[0].created is False
        assert first[0].item_id == second[0].item_id

    def test_the_read_back_watermark_excludes_its_own_boundary(self, fake):
        """Their filter is ``created_at > since``. A watermark that included its boundary
        would re-read the last decision forever, and every step would re-apply it."""
        from datetime import UTC, datetime

        with client(fake) as api:
            queue = api.create_external_queue(question="q", label_set=["a", "b"], external_ref={})
            api.post_candidates(
                queue["id"],
                [{"external_id": "x1", "prompt": "p", "completion": "c", "provenance": {}}],
            )
            at = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
            fake.decide(queue["id"], "x1", "accept", at=at)
            assert len(api.read_decisions(queue["id"])) == 1
            assert api.read_decisions(queue["id"], since=at) == []


class TestUnreachable:
    def test_connect_then_fail_retries_with_backoff_then_refuses(self, fake):
        waits: list[float] = []
        fake.fail_with = [503, 503, 503, 503]
        fake.retry_after = "2"
        with (
            pytest.raises(MiDataworksUnreachable) as caught,
            MiDataworksClient(
                action="listing versions",
                capability="available_parts_listing",
                transport=fake.transport(),
                sleep=waits.append,
                retry_max=3,
            ) as api,
        ):
            api.list_exports()
        assert caught.value.code == "midataworks_unreachable"
        assert caught.value.details["attempts"] == 4
        assert waits == [2.0, 2.0, 2.0], "Retry-After was not honoured"
        assert fake.call_count() == 4

    def test_it_recovers_when_the_sibling_comes_back(self, fake):
        """Transient really is transient: the retry exists to survive a restart, not to
        paper over an outage."""
        fake.fail_with = [503]
        with client(fake, retry_max=2) as api:
            assert api.list_exports() == []
        assert fake.call_count() == 2

    def test_unreachable_is_a_different_code_from_not_configured(self, fake, monkeypatch):
        """AC-010.15, end to end. **Collapsing them hides a misconfiguration behind a
        network message for as long as nobody looks.**"""
        fake.fail_with = [503]
        with pytest.raises(MiDataworksUnreachable), client(fake, retry_max=0) as api:
            api.list_exports()

        monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
        reset_settings()
        with pytest.raises(Refusal) as caught:
            MiDataworksClient(action="x", capability="part_import_http")
        assert caught.value.code == "midataworks_not_configured"


class TestUnauthorizedIsNeverRetried:
    def test_a_401_costs_exactly_one_call(self, fake):
        """**The row that is a policy** (ADR-020). Asserted on the call count."""
        fake.always_fail = 401
        with pytest.raises(MiDataworksUnauthorized), client(fake, retry_max=5) as api:
            api.list_exports()
        assert fake.call_count() == 1

    def test_the_refusal_names_the_identity_miforge_sent(self, fake):
        """Because the likely cause is that header or a new gate on their side, and the fix
        is a configuration change rather than another attempt."""
        fake.always_fail = 401
        with pytest.raises(MiDataworksUnauthorized) as caught, client(fake) as api:
            api.list_exports()
        assert caught.value.details["agent"] == "agent:miforge"
        assert caught.value.details["retried"] is False

    def test_a_403_that_names_its_own_reason_is_a_declined_action_not_a_credential(self, fake):
        """**The defect writing these tests found.**

        Mapping every 403 to ``midataworks_unauthorized`` sent the reader looking for a
        token that does not exist on this surface at all. miDataworks answers
        ``403 ORIGIN_APP_MISMATCH`` when miForge posts to a queue another application
        created - a miForge defect with a completely different fix. Their code is more
        specific than the status, so when they name one it is carried through.
        """
        from tests.support.fake_midataworks import FakeQueue

        fake.queues["rq-theirs"] = FakeQueue(
            id="rq-theirs",
            kind="external",
            question="theirs",
            label_set=["a", "b"],
            origin_app="someone-else",
            external_ref={},
        )
        with pytest.raises(MiDataworksRefused) as caught, client(fake, retry_max=3) as api:
            api.post_candidates(
                "rq-theirs",
                [{"external_id": "x1", "prompt": "p", "completion": "c", "provenance": {}}],
            )
        assert caught.value.code == "midataworks_refused"
        assert caught.value.details["midataworks_code"] == "ORIGIN_APP_MISMATCH"
        assert fake.call_count() == 1, "a 4xx must not be retried whichever code it carries"


class TestAMalformedAgentIdentityCannotBeSent:
    def test_settings_refuse_it_at_load(self, monkeypatch):
        """**At startup, not as a 400 on the first candidate post an hour into a run.**

        Their pattern is ``^agent:[a-z0-9][a-z0-9-]{0,47}$``, enforced by middleware on
        *every* request - so a malformed value fails everything, late, with a message about
        a header rather than about configuration.
        """
        monkeypatch.setenv("MIDATAWORKS_AGENT", "miforge")
        reset_settings()
        with pytest.raises(Exception) as caught:
            from core.config import get_settings

            get_settings()
        assert "agent:" in str(caught.value)

    def test_the_fake_would_have_refused_it(self, fake):
        """The fake implements their middleware, so this test proves the settings guard is
        protecting against something real rather than against an imagined rule."""
        import httpx

        response = fake.handle(
            httpx.Request(
                "GET",
                "http://dw.invalid:8000/api/v1/exports",
                headers={"X-Dataworks-Agent": "nope"},
            )
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_AGENT_IDENTITY"


class TestImportOverHttpProducesTheSameRowsAsADirectory:
    def test_the_two_acquisition_paths_agree(self, engine, fake, tmp_path):
        """AC-010.5. **Identical rows by construction**, because both paths hand the same
        staged directory to the same ingest function - not because two code paths agree."""
        manifest, bodies = fx.miforge_set_manifest()
        directory = fx.write_export(tmp_path / "export", manifest, bodies)

        # The manifest is servable whether or not it is listed in `files` - their
        # `export_file` route allows `{paths} | {MANIFEST_FILE}` - so it is registered here
        # under its own name and the client adds it to the download list.
        wire_files = {
            split["path"]: (directory / split["path"]).read_bytes()
            for split in manifest["content"]["splits"]
        }
        wire_files["midataworks-dataset-version.json"] = (
            directory / "midataworks-dataset-version.json"
        ).read_bytes()
        fake.add_export("exp-1", files=wire_files)

        parts_root_dir = tmp_path / "parts-dir"
        parts_root_http = tmp_path / "parts-http"
        parts_root_dir.mkdir()
        parts_root_http.mkdir()

        # Compared through the API's own payload, with the two fields that *must* differ
        # removed: the local path and the import time. Everything else - the kind, the
        # reference, the digests, the caveats, the licence classes - must be identical.
        with support.session_for(engine) as session:
            dir_result = acquire.from_dir(session, directory, parts_root=parts_root_dir)
            dir_payload = index.get_part(session, dir_result.part_id)

        with support.session_for(engine) as session:
            index.delete_part_index(session)

        with support.session_for(engine) as session:
            with client(fake) as api:
                http_result = acquire.from_http(
                    session, "exp-1", parts_root=parts_root_http, client=api
                )
            http_payload = index.get_part(session, http_result.part_id)

        for payload in (dir_payload, http_payload):
            payload.pop("local_path")
            payload.pop("imported_at")
        assert http_payload == dir_payload

    def test_an_export_with_no_files_refuses_by_name(self, engine, fake, tmp_path):
        fake.add_export("exp-empty", files={})
        parts_root = tmp_path / "parts"
        parts_root.mkdir()
        with (
            support.session_for(engine) as session,
            pytest.raises(Refusal) as caught,
            client(fake) as api,
        ):
            acquire.from_http(session, "exp-empty", parts_root=parts_root, client=api)
        assert caught.value.code == "export_has_no_files"

    def test_an_unknown_export_surfaces_their_refusal(self, engine, fake, tmp_path):
        parts_root = tmp_path / "parts"
        parts_root.mkdir()
        with (
            support.session_for(engine) as session,
            pytest.raises(MiDataworksRefused) as caught,
            client(fake) as api,
        ):
            acquire.from_http(session, "nope", parts_root=parts_root, client=api)
        assert caught.value.details["midataworks_code"] == "export_not_found"


class TestTheHubPathIsRecordedDebtAndRefusesByName:
    def test_it_names_the_paths_that_are_built(self):
        """OQ-1's unbuilt half. **Refused by name rather than missing**, so a caller
        discovers the gap as a message instead of an ``AttributeError``."""
        with pytest.raises(Refusal) as caught:
            acquire.from_hub()
        assert caught.value.code == "acquisition_path_not_built"
        assert caught.value.details["built"] == ["dir", "http"]
        assert "same path a Hub fetch would hand to" in caught.value.message
