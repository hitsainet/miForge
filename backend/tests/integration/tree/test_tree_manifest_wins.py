"""Where the index and the manifest disagree, the **manifest wins and it is logged**.

Task 3.11; ADR-004 verbatim - *"where they disagree, the manifest wins, and the
disagreement is logged rather than silently resolved"*; FR-009.29; foundation 4.5.

**Both halves, every time.** The resolution is not the interesting part - the log line is.
A divergence means something wrote the index without going through the manifest, or edited
a manifest in place; both are bugs, and a silent self-heal is how they stay unfound for
months. A test asserting only the rendered value would pass against a correct resolution
that said nothing, which is the going-quietly-dark failure ADR-013 names.

**The negative case is asserted first**, because without it a function that logged the
event unconditionally would satisfy every other test in this file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from models.checkpoint import Checkpoint
from services.reindex import DIVERGENCE_EVENT
from tests.support import db as support
from tests.support import tree as forest


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    reset_settings()

    engine = support.migrated_engine(tmp_path / "db")
    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with support.session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    client = TestClient(app, raise_server_exceptions=False)

    published = forest.root_checkpoint(checkpoints, run_id="run-a", step=100, domain="support")
    forest.indexed(engine, checkpoints, run_ids=("run-a",))

    try:
        yield client, checkpoints, engine, published
    finally:
        app.dependency_overrides.clear()
        reset_settings()


def divergences(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.message == DIVERGENCE_EVENT]


def node_for(client: TestClient, name: str) -> dict[str, Any]:
    body = client.get("/api/v1/tree").json()
    return next(node for node in body["nodes"] if node["name"] == name)


def plant(engine, name: str, **columns: Any) -> None:  # type: ignore[no-untyped-def]
    """Write values into the index **behind** the manifest's back.

    Which is what a divergence is: a row that did not come from the file beside the
    weights. Done with the ORM rather than by editing the manifest, so the manifest stays
    the thing that is right.
    """
    with support.session_for(engine) as session:
        row = session.get(Checkpoint, name)
        assert row is not None
        for key, value in columns.items():
            setattr(row, key, value)
        session.commit()


class TestAnAgreeingIndexIsSilent:
    def test_no_divergence_means_no_log_line(self, rig, caplog) -> None:  # type: ignore[no-untyped-def]
        """The negative case first. See the module docstring."""
        client, _, _, published = rig
        with caplog.at_level(logging.WARNING, logger="miforge.reindex"):
            node = node_for(client, published.name)
        assert divergences(caplog) == []
        assert node["index_stale"] is False

    def test_and_the_response_level_list_is_empty(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _, _ = rig
        assert client.get("/api/v1/tree").json()["index_stale"] == []


class TestTheManifestWinsOnARenderedValue:
    def test_the_DOMAIN_the_tree_renders_is_the_MANIFESTS(self, rig, caplog) -> None:  # type: ignore[no-untyped-def]
        """``domain`` is one of the four facts R-02.29 names, so a stale index row would
        put a wrong answer on the screen."""
        client, _, engine, published = rig
        plant(engine, published.name, domain="wrong-domain")
        with caplog.at_level(logging.WARNING, logger="miforge.reindex"):
            node = node_for(client, published.name)
        assert node["domain"] == "support"
        assert node["domain"] != "wrong-domain"
        assert node["index_stale"] is True

    def test_the_rendering_is_re_derived_and_not_left_stale_beside_the_value(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        """``domain_display`` is a *rendering* of ``domain`` and ``domain_state``. Letting
        the manifest win on the value while the sentence kept the row's answer would be the
        manifest winning on one field and losing on its own presentation."""
        client, _, engine, published = rig
        from models.domain_state import DomainState

        plant(
            engine,
            published.name,
            domain=None,
            domain_state=DomainState.ASSERTED_NULL.value,
        )
        node = node_for(client, published.name)
        assert node["domain"] == "support"
        assert node["domain_state"] == "recorded"
        assert node["domain_display"] == "support"

    def test_the_PARENT_edge_is_rebuilt_from_the_manifests_value(self, rig) -> None:  # type: ignore[no-untyped-def]
        """The edge, not merely the field: a stale ``parent`` would move the node in the
        tree, which is the shape of the defect ADR-004 exists to stop."""
        client, root, engine, published = rig
        other = forest.root_checkpoint(root, run_id="run-b", step=200)
        forest.indexed(engine, root, run_ids=("run-b",))
        plant(engine, published.name, parent=other.name)
        node = node_for(client, published.name)
        # The manifest says `parent: null`, so the node is a ROOT and not a child.
        assert node["parent"] == {"target": None, "state": "root"}

    def test_the_TEST_SCORE_comes_from_the_manifest(self, rig) -> None:  # type: ignore[no-untyped-def]
        """R-02.29's fourth fact, and the one an operator would act on."""
        client, root, engine, _ = rig
        from tests.support.checkpoints import score

        scored = forest.root_checkpoint(
            root, run_id="run-scored", step=10, test=score(value=0.81, n=120)
        )
        forest.indexed(engine, root, run_ids=("run-scored",))
        plant(engine, scored.name, test={"set": "contract-clauses-test", "score": 0.1, "n": 1})
        node = node_for(client, scored.name)
        assert node["test"]["score"] == 0.81
        assert node["test"]["n"] == 120


class TestTheDisagreementIsLOGGED:
    def test_one_line_per_divergent_artifact_naming_the_fields_and_BOTH_answers(
        self, rig, caplog
    ) -> None:  # type: ignore[no-untyped-def]
        client, _, engine, published = rig
        plant(engine, published.name, domain="wrong-domain")
        with caplog.at_level(logging.WARNING, logger="miforge.reindex"):
            node_for(client, published.name)

        records = divergences(caplog)
        assert len(records) == 1
        record = records[0]
        assert record.artifact == published.name  # type: ignore[attr-defined]
        assert record.divergence == "field_mismatch"  # type: ignore[attr-defined]
        assert record.resolution == "manifest_wins"  # type: ignore[attr-defined]
        assert "domain" in record.divergent_fields  # type: ignore[attr-defined]
        # BOTH answers, because a line saying only that something differed does not say
        # what was written behind the manifest's back.
        assert record.detail["domain"] == {  # type: ignore[attr-defined]
            "index": "wrong-domain",
            "manifest": "support",
        }

    def test_the_request_level_line_carries_the_stale_COUNT(self, rig, caplog) -> None:  # type: ignore[no-untyped-def]
        """FR-009.29's divergences have to be visible in aggregate and not only per node."""
        client, _, engine, published = rig
        plant(engine, published.name, domain="wrong-domain")
        with caplog.at_level(logging.INFO):
            client.get("/api/v1/tree")
        served = [r for r in caplog.records if r.message == "tree_served"]
        assert len(served) == 1
        assert served[0].index_stale == 1  # type: ignore[attr-defined]

    def test_the_response_names_which_nodes_are_stale(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, engine, published = rig
        plant(engine, published.name, domain="wrong-domain")
        assert client.get("/api/v1/tree").json()["index_stale"] == [published.name]


class TestTheForestDoesNotCorrectTheRowItReads:
    def test_the_same_request_twice_reports_the_divergence_twice(self, rig, caplog) -> None:  # type: ignore[no-untyped-def]
        """:func:`services.reindex.resolve_checkpoint` corrects the row, deliberately, so
        one artifact's divergence is not reported forever. **The forest must not**, and the
        reason is foundation 4.4: a read that rewrites rows makes the same request answer
        differently the second time, and the equivalence comparison could then not
        distinguish that from a faithful rebuild. ``reindex`` is what corrects the index.
        """
        client, _, engine, published = rig
        plant(engine, published.name, domain="wrong-domain")
        with caplog.at_level(logging.WARNING, logger="miforge.reindex"):
            first = node_for(client, published.name)
            second = node_for(client, published.name)
        assert first == second
        assert len(divergences(caplog)) == 2

        with support.session_for(engine) as session:
            row = session.get(Checkpoint, published.name)
            assert row is not None
            assert row.domain == "wrong-domain", "the forest read must not rewrite the index"


class TestAnUnreadableManifestIsNAMEDAndTheNodeSURVIVES:
    """The third outcome, and the one easiest to get wrong.

    The authority is gone, so there is nothing to overwrite the row with. Blanking the node
    would hide a rendered fact behind an empty one, and dropping it would hide the artifact
    - which is FR-009.28's reasoning applied to a second kind of absence. So the index
    values are kept, the node is named stale, and the line says the manifest could not be
    read rather than that a field mismatched.
    """

    def test_a_deleted_manifest_leaves_the_node_present_marked_and_logged(
        self, rig, caplog
    ) -> None:  # type: ignore[no-untyped-def]
        client, _, _, published = rig
        (published.directory / "manifest.json").unlink()
        with caplog.at_level(logging.WARNING, logger="miforge.reindex"):
            node = node_for(client, published.name)
        assert node["index_stale"] is True
        assert node["domain"] == "support", "the index row is kept; there is nothing to replace it"
        records = divergences(caplog)
        assert len(records) == 1
        assert records[0].divergence == "manifest_unreadable"  # type: ignore[attr-defined]

    def test_an_unparseable_manifest_is_the_same_outcome_and_not_a_500(self, rig, caplog) -> None:  # type: ignore[no-untyped-def]
        client, _, _, published = rig
        (published.directory / "manifest.json").write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="miforge.reindex"):
            response = client.get("/api/v1/tree")
        assert response.status_code == 200
        assert divergences(caplog)[0].divergence == "manifest_unreadable"  # type: ignore[attr-defined]

    def test_a_manifest_that_is_valid_json_but_the_wrong_artifact_is_also_named(
        self, rig, caplog
    ) -> None:  # type: ignore[no-untyped-def]
        """An adapter manifest where a checkpoint manifest should be: identifiable as a
        file and not usable as this artifact's record."""
        client, _, _, published = rig
        path = published.directory / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kind"] = "miforge.adapter-manifest"
        path.write_text(json.dumps(document), encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="miforge.reindex"):
            node = node_for(client, published.name)
        assert node["index_stale"] is True
        assert divergences(caplog)
