"""Reindex equivalence for the artifacts **F8's own writer** produces  (ADR-016 #4).

Drop the index, rebuild from manifests alone, assert identical API responses. Foundation
4.4's set already covers an adapter manifest written by hand; this adds the adapter and the
merged checkpoint written by the **real** writer, which is the half that catches a layout or
a field the hand-built fixture happens to agree with.

**And it catches the defect the TID's separate adapter root would have caused.** ``reindex``
walks ``settings.checkpoint_root`` and finds artifacts by manifest filename. An adapter under
a root of its own would be indexed at save and then **deleted from the index by the next
reindex and never rebuilt**, while a test that wrote and rebuilt from one configured root
would still pass. :class:`TestAnAdapterOutsideTheWalkedTreeWouldBeLost` makes that concrete
rather than leaving it as an argument in a docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.canonical import sha256_of
from models.domain_state import DomainState
from services.reindex import checkpoint_tree, list_adapters, list_checkpoints, reindex
from tests.support import db as support
from tests.support.checkpoints import saved_checkpoint
from tests.support.test_adapter_doubles import (
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


def f7_contract_element() -> dict[str, Any]:
    """One ``parts`` element **through F7's own producer**, never spelled here.

    F8's documents describe the element as ``{slot, pack, source, type, name, version}``;
    the shape F7 **shipped** - and the one the gate contract
    ``batch-record-v1.json#/$defs/PartReference`` settles - is
    ``{kind, contract_reference, external}``. Writing the documented shape into a fixture
    would have made every assertion here about a spelling that does not exist, which is
    exactly why the element is taken from the producer rather than typed out.
    """
    from packs.records import PackVersionIdentity, part_reference_record
    from packs.references import ContractPartRef
    from packs.slots import SlotKind

    return part_reference_record(
        SlotKind.PROMPT_SET,
        ContractPartRef(type="prompt_set", name="contract-clauses", version="4"),
        pack=PackVersionIdentity(pack_id="contracts", version="3"),
    ).element()


def f7_external_element() -> dict[str, Any]:
    """The other source kind, likewise from F7's producer."""
    from packs.records import PackVersionIdentity, part_reference_record
    from packs.references import ExternalPartRef
    from packs.slots import SlotKind

    return part_reference_record(
        SlotKind.CORPUS,
        ExternalPartRef(
            origin="https://example.invalid/corpus.jsonl",
            content_sha256="c" * 64,
            taken_in_at="2026-10-01T00:00:00Z",
        ),
        pack=PackVersionIdentity(pack_id="contracts", version="3"),
    ).element()


@pytest.fixture()
def engine(tmp_path: Path):  # type: ignore[no-untyped-def]
    return support.migrated_engine(tmp_path / "db")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoints"
    path.mkdir()
    return path


def write_real_adapter(
    root: Path, *, base: dict[str, Any], run_id: str = "run-7", step: int = 40
) -> dict[str, Any]:
    """One adapter through F8's **real** writer - layout, digest, validation and all."""
    doc = adapter_manifest.build(
        run_id=run_id,
        step=step,
        base=base,
        adapter=adapter_config(rank=8, alpha=16.0, target_modules=("q_proj", "v_proj")),
        optimizer=optimizer_block(),
        parts=[f7_contract_element()],
    )
    return adapter_save.save(
        FakePeftModel(rank=8, alpha=16.0, target_modules=("q_proj", "v_proj")),
        doc,
        adapter_paths.adapter_dir(run_id, step, root=root),
    )


class TestTheRealWritersAdapterSurvivesARebuild:
    def test_it_is_found_and_indexed(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        published = write_real_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with support.session_for(engine) as session:
            report = reindex(session, root=root)
            session.commit()
            assert report.adapters == 1
            assert [row["name"] for row in list_adapters(session)] == [published["name"]]

    def test_the_rebuilt_row_is_IDENTICAL_after_a_second_rebuild(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """Equivalence, not merely presence: the API response is the comparison."""
        write_real_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
            first = list_adapters(session)
        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
            second = list_adapters(session)
        assert first == second

    def test_every_indexed_field_came_from_the_MANIFEST(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        published = write_real_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
            row = list_adapters(session)[0]
        assert row["run_id"] == published["run_id"]
        assert row["step"] == published["step"]
        assert row["base_name"] == published["base"]["name"]
        assert row["weights_sha256"] == published["weights_sha256"]
        assert row["adapter_config"] == published["adapter"]
        assert row["parts"] == published["parts"]

    def test_an_adapters_domain_is_NOT_IN_KIND_and_never_borrowed_from_its_base(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """The adapter manifest declares no ``domain`` and closes its objects, so a
        conforming adapter manifest *cannot* carry one. Filling it from the base would
        label an adapter trained on a different pack with the base's domain, silently."""
        base = saved_checkpoint(root, run_id="run-base", step=40, domain="support")
        write_real_adapter(
            root,
            base=adapter_manifest.base_block(
                checkpoint_record(checkpoint_name=base.manifest["name"]),
                manifest_sha256=sha256_of(base.manifest),
            ),
        )
        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
            row = list_adapters(session)[0]
        assert row["domain"] is None
        assert row["domain_state"] == DomainState.NOT_IN_KIND.value

    def test_deleting_the_index_does_not_remove_the_DIRECTORY(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """ADR-004: the filesystem is the source of truth. Task 3.7's assertion."""
        published = write_real_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        directory = adapter_paths.adapter_dir("run-7", 40, root=root)
        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
        assert directory.is_dir()

        from models.adapter import Adapter

        with support.session_for(engine) as session:
            session.query(Adapter).delete()
            session.commit()
            assert list_adapters(session) == []
        assert directory.is_dir(), "an index row is not the artifact"

        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
            assert [row["name"] for row in list_adapters(session)] == [published["name"]]


class TestTheTreeEdgePointsAtTheBaseTheMANIFESTNames:
    def test_the_adapter_joins_its_own_base_and_not_the_runs_head(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """FR-008.34. An adapter's edge is to the base its own manifest names - and F9 must
        not infer the edge from the run, which is why the row carries ``base_name``."""
        base = saved_checkpoint(root, run_id="run-base", step=40)
        # A later checkpoint of the SAME run as the adapter, which would be the wrong edge.
        saved_checkpoint(root, run_id="run-7", step=90, payload=b"later")
        published = write_real_adapter(
            root,
            base=adapter_manifest.base_block(
                checkpoint_record(checkpoint_name=base.manifest["name"]),
                manifest_sha256=sha256_of(base.manifest),
            ),
        )
        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
            tree = checkpoint_tree(session, root=root)

        # `child`/`parent`/`edge`, which is reindex's own vocabulary - read from the
        # payload rather than guessed, because an edge test written against invented key
        # names is a test of the test.
        edges = {(edge["child"], edge["parent"], edge["edge"]) for edge in tree["edges"]}
        assert (published["name"], base.manifest["name"], "base") in edges
        assert not any(
            child == published["name"] and parent != base.manifest["name"]
            for child, parent, _ in edges
        )

    def test_an_adapter_and_a_checkpoint_of_the_same_run_and_step_are_distinguishable(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """R-02.29. The ``adapter-`` segment is what does it."""
        saved_checkpoint(root, run_id="run-7", step=40)
        published = write_real_adapter(root, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
            names = {
                node["name"]: node["kind"] for node in checkpoint_tree(session, root=root)["nodes"]
            }
        assert len(names) == 2
        assert names[published["name"]] == "adapter"

    def test_a_MERGED_checkpoint_carries_BOTH_edges(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        """FR-008.37: a merged checkpoint has two ancestors - ``parent`` to its base and
        ``merged_adapter`` to the adapter baked into it. Without the second field, two
        checkpoints at the same step with the same parent, one merged and one not, would be
        indistinguishable."""
        base = saved_checkpoint(root, run_id="run-base", step=40)
        adapter = write_real_adapter(
            root,
            base=adapter_manifest.base_block(
                checkpoint_record(checkpoint_name=base.manifest["name"]),
                manifest_sha256=sha256_of(base.manifest),
            ),
        )
        merged = saved_checkpoint(
            root,
            run_id="run-7",
            step=41,
            payload=b"merged",
            parent=base.manifest["name"],
            merged_adapter={
                "name": adapter["name"],
                "step": adapter["step"],
                "manifest_sha256": sha256_of(adapter),
            },
        )
        with support.session_for(engine) as session:
            reindex(session, root=root)
            session.commit()
            rows = list_checkpoints(session)
            tree = checkpoint_tree(session, root=root)

        row = next(r for r in rows if r["name"] == merged.manifest["name"])
        assert row["parent"] == base.manifest["name"]
        assert row["merged_adapter"]["name"] == adapter["name"]
        assert row["merged_adapter"]["manifest_sha256"] == sha256_of(adapter)

        # `child`/`parent`/`edge`, which is reindex's own vocabulary - read from the
        # payload rather than guessed, because an edge test written against invented key
        # names is a test of the test.
        edges = {(edge["child"], edge["parent"], edge["edge"]) for edge in tree["edges"]}
        assert (merged.manifest["name"], base.manifest["name"], "parent") in edges
        assert (merged.manifest["name"], adapter["name"], "merged_adapter") in edges


class TestAnAdapterOutsideTheWalkedTreeWouldBeLost:
    """Why the configurable ``/data/adapters`` root is a reported finding and not a feature.

    The failure is **invisible to a test that writes and rebuilds from one root**, which is
    exactly why it is demonstrated here rather than argued about.
    """

    def test_an_adapter_written_beside_the_root_is_not_found_by_reindex(
        self, engine, tmp_path: Path, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        elsewhere = tmp_path / "adapters-elsewhere"
        elsewhere.mkdir()
        write_real_adapter(elsewhere, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with support.session_for(engine) as session:
            report = reindex(session, root=root)
            session.commit()
        assert report.adapters == 0, (
            "an adapter outside the walked root is invisible to reindex; it would be "
            "indexed at save and deleted by the next rebuild, with nothing going red"
        )

    def test_the_configured_layout_IS_inside_the_walked_root(self, root: Path) -> None:
        assert root in adapter_paths.adapter_dir("run-7", 40, root=root).parents
