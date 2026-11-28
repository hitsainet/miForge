"""F8's four read/act routes over a real database and a real artifact root.

The preview is checked against the estimator it shares with the guard (*"a preview computed
differently from the gate is a preview of nothing"*), the adapter read is checked against the
**bytes on disk** rather than the index row (ADR-004), and the merge route is checked to
refuse AC17 **before** anything is enqueued.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.canonical import sha256_of
from core.config import reset_settings
from models.adapter import Adapter
from services import reindex as reindex_module
from services import runtypes_service
from tests.support.checkpoints import create_feature_tables, saved_checkpoint
from tests.support.jobs import insert_run, memory_engine, session_for
from tests.support.test_adapter_doubles import (
    TINY_CONFIG,
    FakePeftModel,
    adapter_config,
    checkpoint_record,
    hub_record,
    optimizer_block,
)
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import save as adapter_save

HUB_REPO = "LiquidAI/LFM2.5-1.2B-Instruct"
BASE_CONFIG: dict[str, Any] = {"learning_rate": 0.0001, "warmup_steps": 10, "batch_size": 1}


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A mounted app over an in-memory database and a real artifact root on disk."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    reset_settings()

    engine = memory_engine()
    create_feature_tables(engine)
    with session_for(engine) as session:
        insert_run(session, "run-7", status="running")

    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, checkpoints, engine
    finally:
        app.dependency_overrides.clear()
        reset_settings()


def write_adapter(
    root: Path,
    *,
    base: dict[str, Any],
    run_id: str = "run-7",
    step: int = 40,
) -> dict[str, Any]:
    doc = adapter_manifest.build(
        run_id=run_id,
        step=step,
        base=base,
        adapter=adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",)),
        optimizer=optimizer_block(),
    )
    return adapter_save.save(
        FakePeftModel(rank=8, alpha=16.0, target_modules=("q_proj",)),
        doc,
        adapter_paths.adapter_dir(run_id, step, root=root),
    )


class TestTheRunTypeList:
    def test_it_answers_from_the_live_registry(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        body = client.get("/api/v1/run-types").json()
        assert [entry["name"] for entry in body["run_types"]] == ["adapter", "change", "create"]

    def test_each_entry_carries_the_stored_spelling_F9_will_see_in_a_row(self, rig) -> None:  # type: ignore[no-untyped-def]
        """``stored_as`` now equals ``name`` for every entry, and that is the point.

        It asserted ``{"change": "full", ...}`` until 2026-10-14. F8 shipped that mapping
        because ``mf_runs.run_type``'s CHECK permitted ``full`` while the registry and
        BRD-02 §4 both say ``change``, and reported the divergence rather than editing a
        migration it did not own. Migration ``0014`` settled it on ``change`` and removed the
        map. The field is kept rather than dropped because it is a client-visible promise
        that no translation happens - which is exactly what was not true before.
        """
        client, _, _ = rig
        entries = client.get("/api/v1/run-types").json()["run_types"]
        stored = {e["name"]: e["stored_as"] for e in entries}
        assert stored == {"change": "change", "create": "create", "adapter": "adapter"}


class TestTheBudgetPreview:
    def test_a_creation_runs_estimate_comes_from_the_CONFIG(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        body = client.post(
            "/api/v1/budget/estimate",
            json={"run_type": "create", "config": BASE_CONFIG, "architecture_config": TINY_CONFIG},
        ).json()
        assert body["estimate"]["basis"] == "config"
        assert body["estimate"]["param_count"] > 0

    def test_an_adapter_runs_estimate_names_the_adapter_terms_separately(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        body = client.post(
            "/api/v1/budget/estimate",
            json={
                "run_type": "adapter",
                "config": BASE_CONFIG,
                "architecture_config": TINY_CONFIG,
                "adapter": {"rank": 8, "alpha": 16.0, "target_modules": ["q_proj"]},
            },
        ).json()
        assert body["estimate"]["run_type"] == "adapter"
        assert body["estimate"]["adapter_param_basis"] == "bound"
        assert body["estimate"]["adapter_param_count"] < body["estimate"]["base_param_count"]

    def test_an_adapter_run_with_no_adapter_block_is_refused(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        response = client.post(
            "/api/v1/budget/estimate",
            json={"run_type": "adapter", "config": BASE_CONFIG, "architecture_config": TINY_CONFIG},
        )
        assert response.json()["error"]["code"] == "adapter_config_required"

    def test_neither_input_is_refused_and_both_is_refused(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        neither = client.post(
            "/api/v1/budget/estimate", json={"run_type": "create", "config": BASE_CONFIG}
        ).json()
        assert neither["error"]["code"] == "budget_spec_incomplete"

        both = client.post(
            "/api/v1/budget/estimate",
            json={
                "run_type": "change",
                "config": BASE_CONFIG,
                "architecture_config": TINY_CONFIG,
                "param_count": 1000,
            },
        ).json()
        assert both["error"]["code"] == "budget_spec_ambiguous"

    def test_the_verdict_is_UNKNOWN_rather_than_FITS_when_no_card_is_present(self, rig) -> None:  # type: ignore[no-untyped-def]
        """*"I could not read the limit, so it fits"* is the pass-through failure wearing
        the estimator's name - and on a screen it would be the most misleading number
        there. CI has no card, so this is the real path here."""
        client, _, _ = rig
        body = client.post(
            "/api/v1/budget/estimate",
            json={"run_type": "create", "config": BASE_CONFIG, "architecture_config": TINY_CONFIG},
        ).json()
        if body["free_bytes"] is None:
            assert body["verdict"] == "unknown"
            assert body["claimable_bytes"] is None

    def test_an_unknown_key_in_the_request_is_refused_not_ignored(self, rig) -> None:  # type: ignore[no-untyped-def]
        """A misspelled ``architecture`` would otherwise leave the estimate derived from
        nothing, and a preview of nothing is worse than a refusal."""
        client, _, _ = rig
        response = client.post(
            "/api/v1/budget/estimate",
            json={"run_type": "create", "config": BASE_CONFIG, "architecture": TINY_CONFIG},
        )
        assert response.status_code == 422

    def test_the_preview_and_the_SERVICE_agree_for_identical_inputs(self, rig) -> None:  # type: ignore[no-untyped-def]
        """Asserted by calling both, not by reading both."""
        from training.run_config import RunConfig

        client, _, _ = rig
        over_http = client.post(
            "/api/v1/budget/estimate",
            json={"run_type": "create", "config": BASE_CONFIG, "architecture_config": TINY_CONFIG},
        ).json()
        direct = runtypes_service.estimate_budget(
            run_type="create",
            cfg=RunConfig(**BASE_CONFIG),
            architecture_config=TINY_CONFIG,
            free_bytes=over_http["free_bytes"],
        )
        assert over_http["estimate"] == direct["estimate"]

    def test_the_preview_and_the_GUARDS_estimator_are_the_same_code(self, rig) -> None:  # type: ignore[no-untyped-def]
        """One estimator, two callers (FR-008.18). The guard's own projection is compared
        field by field with what the preview served."""
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec
        from training.budget_guard import estimate as guard_estimate
        from training.run_config import RunConfig

        client, _, _ = rig
        served = client.post(
            "/api/v1/budget/estimate",
            json={"run_type": "create", "config": BASE_CONFIG, "architecture_config": TINY_CONFIG},
        ).json()["estimate"]
        projection = guard_estimate(
            AcquisitionSpec(mode=AcquisitionMode.CREATE, config=TINY_CONFIG),
            RunConfig(**BASE_CONFIG),
        ).as_record()
        assert {k: v for k, v in served.items() if k != "run_type"} == projection


class TestTheAdapterReadRoutes:
    def test_the_manifest_is_served_from_DISK(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, _ = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        body = client.get(f"/api/v1/adapters/{published['name']}").json()
        assert body["manifest"] == published
        assert Path(body["path"]).is_dir()

    def test_a_name_containing_a_SLASH_routes(self, rig) -> None:  # type: ignore[no-untyped-def]
        """The converter's whole reason. Without ``:path`` this would be a router 404."""
        client, root, _ = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        assert "/" in published["name"]
        assert client.get(f"/api/v1/adapters/{published['name']}").status_code == 200

    def test_the_response_reports_whether_the_INDEX_disagrees(self, rig) -> None:  # type: ignore[no-untyped-def]
        """ADR-004: where they disagree the manifest wins **and it is logged**. An endpoint
        serving the row would hide exactly the divergence an operator needs to see."""
        client, root, engine = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with session_for(engine) as session:
            reindex_module.reindex(session, root=root)
            session.commit()

        clean = client.get(f"/api/v1/adapters/{published['name']}").json()
        assert clean["indexed"] is True and clean["index_disagrees"] is False

        with session_for(engine) as session:
            row = session.get(Adapter, published["name"])
            assert row is not None
            row.weights_sha256 = "0" * 64
            session.commit()

        divergent = client.get(f"/api/v1/adapters/{published['name']}").json()
        assert divergent["index_disagrees"] is True
        assert divergent["manifest"] == published, "the manifest still wins"
        assert divergent["disagreements"]["weights_sha256"]["index"] == "0" * 64

    def test_an_absent_row_is_reported_as_STALE_INDEX_and_not_as_a_disagreement(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        """Otherwise every adapter would look divergent until the next rebuild."""
        client, root, _ = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        body = client.get(f"/api/v1/adapters/{published['name']}").json()
        assert body["indexed"] is False
        assert body["index_disagrees"] is False

    def test_the_listing_answers_from_the_index(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, engine = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with session_for(engine) as session:
            reindex_module.reindex(session, root=root)
            session.commit()
        body = client.get("/api/v1/adapters").json()
        assert body["count"] == 1
        assert body["adapters"][0]["name"] == published["name"]

    def test_an_unparseable_name_is_a_NAMED_refusal(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        body = client.get("/api/v1/adapters/not-a-qualified-name").json()
        assert body["error"]["code"] == "adapter_manifest_missing"

    def test_a_traversal_in_the_name_is_refused_rather_than_normalised(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        body = client.get("/api/v1/adapters/..%2F..%2Fetc:run:adapter-1").json()
        assert "error" in body


class TestTheMergeRouteRefusesBeforeAnythingIsEnqueued:
    """**Updated 2026-10-14 when F6 wired the enqueue** (F8 tasks 11.5 and 12.5, both
    marked ``[?]`` naming that feature).

    What this class protects is unchanged and is the important half: **AC17's
    unmerged-base check runs at the route, before anything is enqueued.** What changed is
    the admitted case - the route now returns ``accepted`` and sends
    ``miforge.train.merge_adapter`` to the ``train`` queue, where the base model is in
    hand, and ``mf_merges`` records the outcome. The old assertion that it *"enqueues
    NOTHING and says so"* pinned the unwired state rather than a property, and keeping it
    would have made F6's wiring look like a regression.

    ``run_id`` and ``step`` are now required: a merged checkpoint is a checkpoint of the
    run that asked for it, and defaulting them - or inventing them from the adapter's own
    run - would write a merged checkpoint into the lineage of the run that produced the
    adapter.
    """

    @pytest.fixture(autouse=True)
    def _no_broker(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, int]]:
        """Capture the enqueue instead of reaching a broker. **Records the payload.**

        A spy that recorded only "was called" would pass against a merge enqueued for the
        wrong run or the wrong step, which is a merged checkpoint written into another
        run's lineage (ADR-016 #1).
        """
        from api.v1.endpoints import runtypes as endpoint

        sent: list[tuple[str, str, int]] = []
        monkeypatch.setattr(
            endpoint,
            "enqueue_merge",
            lambda name, run_id, step: sent.append((name, run_id, step)),
        )
        self.sent = sent  # type: ignore[attr-defined]
        return sent

    def test_a_HUB_base_is_refused_with_AC17s_code(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, _ = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        body = client.post(
            f"/api/v1/adapters/{published['name']}/merge?run_id=run-merge&step=41"
        ).json()
        assert body["error"]["code"] == "merge_refused_no_unmerged_base"
        assert "Save a checkpoint of the base first" in body["error"]["message"]

    def test_a_surviving_unmerged_base_is_ADMITTED(self, rig) -> None:  # type: ignore[no-untyped-def]
        client, root, _ = rig
        base = saved_checkpoint(root, run_id="run-base", step=40)
        published = write_adapter(
            root,
            base=adapter_manifest.base_block(
                checkpoint_record(checkpoint_name=base.manifest["name"]),
                manifest_sha256=sha256_of(base.manifest),
            ),
        )
        body = client.post(
            f"/api/v1/adapters/{published['name']}/merge?run_id=run-merge&step=41"
        ).json()
        assert body["status"] == "accepted"
        assert body["adapter"]["base_checkpoint"] == base.manifest["name"]
        assert body["adapter"]["scaling"] == "alpha_over_r"

    def test_an_unrecognised_scaling_rule_is_refused_at_the_route(self, rig) -> None:  # type: ignore[no-untyped-def]
        """E-8 over the transport. Guessing the scale changes the served model silently."""
        client, root, _ = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        directory = adapter_paths.adapter_dir("run-7", 40, root=root)
        path = directory / adapter_paths.MANIFEST_NAME
        doc = json.loads(path.read_text())
        doc["adapter"]["scaling"] = "rslora"
        path.write_text(json.dumps(doc), encoding="utf-8")

        body = client.post(
            f"/api/v1/adapters/{published['name']}/merge?run_id=run-merge&step=41"
        ).json()
        # The schema catches it first, which is the stronger position - recorded as such.
        assert body["error"]["code"] in {
            "adapter_manifest_invalid",
            "adapter_scaling_unrecognised",
        }

    def test_the_route_enqueues_the_merge_with_its_payload_exactly_once(self, rig) -> None:  # type: ignore[no-untyped-def]
        """The reachability assertion: the call, **the count and the payload**.

        "Was called" passes against a merge enqueued for the wrong run, which writes a
        merged checkpoint into another run's lineage. Deleting the enqueue must make this
        red - a merge route that admits and sends nothing is a capability that does not
        exist for any caller, which is what it was until F6 landed.
        """
        client, root, _ = rig
        base = saved_checkpoint(root, run_id="run-base", step=40)
        published = write_adapter(
            root,
            base=adapter_manifest.base_block(
                checkpoint_record(checkpoint_name=base.manifest["name"]),
                manifest_sha256=sha256_of(base.manifest),
            ),
        )
        body = client.post(
            f"/api/v1/adapters/{published['name']}/merge?run_id=run-merge&step=41"
        ).json()
        assert body["status"] == "accepted"
        assert self.sent == [(published["name"], "run-merge", 41)], (  # type: ignore[attr-defined]
            "the merge must be enqueued exactly once, for the run and step that asked"
        )

    def test_a_refused_preflight_enqueues_nothing(self, rig) -> None:  # type: ignore[no-untyped-def]
        """The half that was always the point: AC17 refuses **before** anything is sent.

        And **no ``mf_merges`` row is written** for a preflight refusal - recording one
        would invent a semantic no requirement states, since R-02.45's three outcomes
        describe the merge and not a request for one.
        """
        client, root, _ = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        body = client.post(
            f"/api/v1/adapters/{published['name']}/merge?run_id=run-merge&step=41"
        ).json()
        assert body["error"]["code"] == "merge_refused_no_unmerged_base"
        assert self.sent == []  # type: ignore[attr-defined]


class TestEveryFailureUsesTheErrorEnvelope:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/api/v1/adapters/not-a-name"),
            ("post", "/api/v1/adapters/not-a-name/merge?run_id=run-merge&step=41"),
        ],
    )
    def test_the_envelope_carries_a_code_a_message_and_details(
        self, rig, method: str, path: str
    ) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = rig
        body = getattr(client, method)(path).json()
        assert set(body) == {"error"}
        assert {"code", "message"} <= set(body["error"])

    def test_a_refusals_message_says_what_to_do_next(self, rig) -> None:  # type: ignore[no-untyped-def]
        """ADR-021, and it is checkable: the merge refusal must contain an instruction."""
        client, root, _ = rig
        published = write_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        message = client.post(
            f"/api/v1/adapters/{published['name']}/merge?run_id=run-merge&step=41"
        ).json()["error"]["message"]
        assert "Save a checkpoint" in message


def _merge_tables(engine) -> None:  # type: ignore[no-untyped-def]
    """Create `mf_merges` on the rig's engine.

    It lives on `training.runtypes.tables`' own MetaData rather than `Base.metadata`, because
    `backend/src/models/` is not a feature agent's to edit - the same reason F10's part tables
    are created separately in `tests/support/checkpoints.create_feature_tables`. Without this
    the route answers **500 `no such table`**, which is how the first run of these tests
    failed: the router was mounted correctly and the schema was the thing missing.
    """
    from training.runtypes.tables import metadata as runtypes_metadata

    runtypes_metadata.create_all(engine)


class TestTheMergeRecordIsREADABLE:
    """``GET /api/v1/merges`` — the reader `mf_merges` did not have (added 2026-10-14).

    **Why this class exists.** The table was written, constrained by three CHECKs and covered
    by 28 unit tests, and **nothing read it**: `grep -rn "merge_record.listing" src/` outside
    the module itself returned nothing and there was no route. A table recording an
    irreversible operation that an operator can reach only by hand-querying PostgreSQL is the
    same unreachable capability as a function nothing calls — the sixth in this project's
    pattern, after four rebuilders in ``services/reindex.py`` and ``checkpoint_tree``.

    **The round trip is through the real writer and the real route**, never a hand-built row:
    *fixtures whose fields agree by construction are how a suite stays green over a real bug*,
    and a planted dict would agree with the reader about a shape neither shares with the
    writer.

    Reachability control: delete ``include_router(runtypes.merges_router)`` from
    ``api/v1/router.py`` and ``test_the_route_is_mounted_and_not_merely_defined`` goes red —
    as does ``test_reachability.py``, which names the path in ``EXPECTED_ROUTES``.
    """

    def test_the_route_is_mounted_and_not_merely_defined(self, rig) -> None:  # type: ignore[no-untyped-def]
        """A 200 with an empty list, not a 404. **The distinction is the whole test**: a path
        that did not match answers ``http_404``, and a handler that matched and found nothing
        answers by name — without it, an unmounted router looks exactly like no merges."""
        client, _, engine = rig
        _merge_tables(engine)
        response = client.get("/api/v1/merges")
        assert response.status_code == 200, "the router is defined but not mounted"
        assert response.json() == {"merges": [], "count": 0}

    def test_each_of_the_three_outcomes_round_trips_through_the_REAL_writer(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        from services import merge_record
        from training.runtypes.adapters.merge import OUTCOME_WRITTEN, MergeOutcome

        client, _, engine = rig
        _merge_tables(engine)
        with session_for(engine) as session:
            merge_record.record(
                session,
                MergeOutcome(
                    outcome=OUTCOME_WRITTEN,
                    adapter_name="LiquidAI/LFM2-1.2B:run-a:adapter-1",
                    adapter_step=100,
                    adapter_manifest_sha256="a" * 64,
                    base_checkpoint_name="LiquidAI/LFM2-1.2B:run-a:step-100",
                    result_checkpoint_name="LiquidAI/LFM2-1.2B:run-a:step-101",
                    refusal_code=None,
                ),
            )
            merge_record.record(
                session,
                merge_record.refused(
                    adapter_name="LiquidAI/LFM2-1.2B:run-b:adapter-1",
                    adapter_step=100,
                    adapter_manifest_sha256="b" * 64,
                    base_checkpoint_name="LiquidAI/LFM2-1.2B:run-b:step-100",
                    refusal_code="merge_base_not_checkpointed",
                ),
            )
            merge_record.record(
                session,
                merge_record.interrupted(
                    adapter_name="LiquidAI/LFM2-1.2B:run-c:adapter-1",
                    adapter_step=100,
                    adapter_manifest_sha256="c" * 64,
                    base_checkpoint_name="LiquidAI/LFM2-1.2B:run-c:step-100",
                ),
            )
            session.commit()

        body = client.get("/api/v1/merges").json()
        assert body["count"] == 3
        outcomes = {row["adapter_name"]: row["outcome"] for row in body["merges"]}
        assert outcomes == {
            "LiquidAI/LFM2-1.2B:run-a:adapter-1": "written",
            "LiquidAI/LFM2-1.2B:run-b:adapter-1": "refused",
            "LiquidAI/LFM2-1.2B:run-c:adapter-1": "interrupted",
        }

    def test_a_refused_merge_carries_its_CODE_because_nothing_else_records_it(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        """For ``written`` and ``interrupted`` a merged checkpoint may exist and its own
        manifest is authoritative (ADR-004). **A refused merge writes no checkpoint**, so this
        table is the whole of what is known about it — and the code is the answer to the only
        question an operator has, which is why it did not happen."""
        from services import merge_record

        client, _, engine = rig
        _merge_tables(engine)
        with session_for(engine) as session:
            merge_record.record(
                session,
                merge_record.refused(
                    adapter_name="LiquidAI/LFM2-1.2B:run-b:adapter-1",
                    adapter_step=100,
                    adapter_manifest_sha256="b" * 64,
                    base_checkpoint_name="LiquidAI/LFM2-1.2B:run-b:step-100",
                    refusal_code="merge_base_not_checkpointed",
                ),
            )
            session.commit()

        row = client.get("/api/v1/merges").json()["merges"][0]
        assert row["refusal_code"] == "merge_base_not_checkpointed"
        assert row["result_checkpoint_name"] is None

    def test_it_filters_by_adapter_which_is_one_of_the_two_real_lookups(
        self, rig
    ) -> None:  # type: ignore[no-untyped-def]
        from services import merge_record

        client, _, engine = rig
        _merge_tables(engine)
        with session_for(engine) as session:
            for run in ("run-a", "run-b"):
                merge_record.record(
                    session,
                    merge_record.refused(
                        adapter_name=f"LiquidAI/LFM2-1.2B:{run}:adapter-1",
                        adapter_step=100,
                        adapter_manifest_sha256="d" * 64,
                        base_checkpoint_name=f"LiquidAI/LFM2-1.2B:{run}:step-100",
                        refusal_code="merge_base_not_checkpointed",
                    ),
                )
            session.commit()

        wanted = "LiquidAI/LFM2-1.2B:run-b:adapter-1"
        body = client.get("/api/v1/merges", params={"adapter_name": wanted}).json()
        assert body["count"] == 1 and body["merges"][0]["adapter_name"] == wanted

    def test_the_limit_is_BOUNDED_rather_than_open(self, rig) -> None:  # type: ignore[no-untyped-def]
        """An endpoint that streams a whole table on request is a denial of service against
        the one database this installation has."""
        client, _, _ = rig
        assert client.get("/api/v1/merges", params={"limit": 100000}).status_code == 422
        assert client.get("/api/v1/merges", params={"limit": 0}).status_code == 422
