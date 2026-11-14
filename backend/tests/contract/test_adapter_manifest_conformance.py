"""Conformance: what the **real writer** produces validates, and omits no declared key.

Sync proves the vendored file matches the other repo's copy; **conformance proves our
writer matches the file.** Neither substitutes for the other, and the foundation's
parametrised sync test already covers ``adapter-manifest-v1.json``.

## And neither catches an OMITTED optional key - which is M17

Since the 2026-10-13 re-vendor every object in both schemas sets
``additionalProperties: false``, so an **invented** key is refused by ``jsonschema`` itself.
What a validator cannot see is a **declared, non-required key being left out**, and
``provider`` is exactly that: ``anyOf [string, null]``, defaulted ``null``, and in neither
``ScoreRecord``'s ``required`` list. A manifest missing it **validates clean**.

So the assertion that bites is a key-set comparison in the **missing** direction, with the
expected set **read from** ``docs/schemas/adapter-manifest-v1.json`` **at test time, never
retyped** - a retyped list agrees with itself, which is the fixture failure ADR-016 names.
**M17 goes red against this test alone**, and that single red is the only protection there
is: R-02.16's *"a score is only comparable against another taken the same way"* would
otherwise be breached silently in the one artifact built to travel without its run log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from models.domain_state import ArtifactKind
from services.reindex import load_schema, schema_violations
from services.scoring import manifest_payload
from tests.support.scores import score_record as f4_score
from tests.support.test_adapter_doubles import (
    FakePeftModel,
    adapter_config,
    checkpoint_record,
    hub_record,
    optimizer_block,
)
from training.runtypes.adapters import load as adapter_load
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import save as adapter_save
from training.runtypes.adapters import schema as adapter_schema

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


def score(set_name: str, **overrides: Any) -> dict[str, Any]:
    """A ``ScoreRecord`` **through F4's own producer**, not a hand-built dict.

    ``ScoreRecordData.score_record()`` is the projection under test on F4's side, so a key
    it stops emitting shows up here rather than being papered over by a fixture that
    happened to spell all five. F5's ``tests.support.checkpoints.score`` helper is *not*
    used: it omits ``floor`` when the floor is null, which F4's builder correctly refuses -
    and a fixture that cannot produce a conforming record would have made every assertion
    below about the fixture instead of about the writer.
    """
    return f4_score(set_rendered=set_name, **overrides).score_record()


def adapter_score_record_keys() -> frozenset[str]:
    """``ScoreRecord``'s declared keys, **from the adapter manifest's own schema file**."""
    return adapter_schema.declared_keys(adapter_schema.SCORE_RECORD_DEF)


def saved(root: Path, **overrides: Any) -> dict[str, Any]:
    """One adapter, written by the real writer and read back by the real reader."""
    values: dict[str, Any] = {
        "run_id": "run-7",
        "step": 40,
        "base": adapter_manifest.base_block(hub_record(HUB_REPO)),
        "adapter": adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",)),
        "optimizer": optimizer_block(),
        "test": score("contracts-test", score=0.81, n=120),
        "retention": score("retention-general", score=0.74, n=200, floor=0.70),
        "parts": [f7_contract_element()],
    }
    values.update(overrides)
    doc = adapter_manifest.build(**values)
    dest = adapter_paths.adapter_dir(str(values["run_id"]), int(values["step"]), root=root)
    adapter_save.save(FakePeftModel(rank=8, alpha=16.0, target_modules=("q_proj",)), doc, dest)
    return adapter_load.load_manifest(dest / adapter_paths.MANIFEST_NAME)


class TestTheRealWritersOutputConforms:
    def test_a_full_manifest_validates(self, tmp_path: Path) -> None:
        assert schema_violations(saved(tmp_path), ArtifactKind.ADAPTER) == []

    def test_a_manifest_with_no_scores_validates_with_BOTH_keys_present_and_null(
        self, tmp_path: Path
    ) -> None:
        """Null is an assertion that nothing ran at this step; an absent key is an
        omission. Both properties are nullable with a null default, and both are written."""
        doc = saved(tmp_path, test=None, retention=None)
        assert doc["test"] is None and doc["retention"] is None
        assert "test" in doc and "retention" in doc
        assert schema_violations(doc, ArtifactKind.ADAPTER) == []

    def test_a_checkpoint_base_manifest_validates(self, tmp_path: Path) -> None:
        doc = saved(
            tmp_path,
            base=adapter_manifest.base_block(checkpoint_record(), manifest_sha256="9" * 64),
        )
        assert schema_violations(doc, ArtifactKind.ADAPTER) == []
        assert doc["base"]["manifest_sha256"] == "9" * 64

    def test_the_round_trip_goes_through_the_real_writer_AND_the_real_reader(
        self, tmp_path: Path
    ) -> None:
        """Not a hand-built dict whose fields agree with the writer by construction."""
        doc = saved(tmp_path)
        on_disk = json.loads(
            (
                adapter_paths.adapter_dir("run-7", 40, root=tmp_path) / adapter_paths.MANIFEST_NAME
            ).read_text()
        )
        assert doc == on_disk

    def test_every_REQUIRED_key_is_present(self, tmp_path: Path) -> None:
        required = adapter_schema.required_keys("AdapterManifestV1")
        assert required <= set(saved(tmp_path))


class TestTheKeySetIsAssertedInTheMISSINGDirection:
    """M17's target. Validation cannot see this; only a key-set comparison can."""

    def test_every_score_record_carries_EVERY_key_the_schema_declares(self, tmp_path: Path) -> None:
        declared = adapter_score_record_keys()
        doc = saved(tmp_path)
        for kind in ("test", "retention"):
            assert (
                set(doc[kind]) == declared
            ), f"{kind}: expected {sorted(declared)}, got {sorted(doc[kind])}"

    def test_provider_is_among_them_and_carries_the_value_F4_REPORTED(self, tmp_path: Path) -> None:
        doc = saved(tmp_path)
        assert "provider" in adapter_score_record_keys()
        assert doc["test"]["provider"] == f4_score().provider.record()
        assert doc["test"]["provider"]

    def test_provider_is_NOT_in_the_schemas_required_list_which_is_WHY_this_test_exists(
        self,
    ) -> None:
        """Verified against the file. If a later re-vendor made it required, the schema
        would start catching the omission and this test's standing would change - so the
        premise is asserted rather than remembered."""
        assert "provider" not in adapter_schema.required_keys(adapter_schema.SCORE_RECORD_DEF)
        assert "provider" in adapter_score_record_keys()

    def test_a_manifest_with_provider_OMITTED_still_VALIDATES_clean(self, tmp_path: Path) -> None:
        """The whole reason M17 goes red against this file alone. Proved by validation, so
        nobody has to take it on trust."""
        doc = saved(tmp_path)
        doc["test"] = {k: v for k, v in doc["test"].items() if k != "provider"}
        assert schema_violations(doc, ArtifactKind.ADAPTER) == [], (
            "if this ever fails, the schema has started catching the omission and M17 is "
            "no longer the only protection"
        )
        assert set(doc["test"]) != adapter_score_record_keys(), "...but the key set does differ"

    def test_an_unrecorded_provider_is_written_as_NULL_and_never_synthesised(
        self, tmp_path: Path
    ) -> None:
        """Present-and-null is an assertion; absent is an omission. F8 writes what F4
        reports and synthesises nothing."""
        from providers.base import ProviderIdentity

        unrecorded = f4_score().score_record() | {"provider": None}
        doc = saved(tmp_path, test=unrecorded)
        assert isinstance(f4_score().provider, ProviderIdentity)
        assert doc["test"]["provider"] is None
        assert "provider" in doc["test"]

    def test_a_score_MISSING_a_declared_key_is_REFUSED_by_the_shared_builder(self) -> None:
        """F4's builder refuses rather than defaulting - *defaulting* ``provider`` *to null
        is M17 by another route*, because the record would validate and the obligation
        would be unmet."""
        from core.errors import Refusal

        with pytest.raises(Refusal) as caught:
            adapter_manifest.build(
                run_id="run-7",
                step=40,
                base=adapter_manifest.base_block(hub_record(HUB_REPO)),
                adapter=adapter_config(),
                optimizer=optimizer_block(),
                test={"set": "s", "score": 0.5, "n": 10},
            )
        assert caught.value.code == "manifest_score_payload_invalid"

    def test_the_expected_key_list_is_READ_FROM_THE_FILE_not_retyped(self) -> None:
        """A retyped list agrees with itself. This asserts the test's own discipline: the
        set must come from the vendored JSON, which is checked by reading it twice -
        through the helper, and directly."""
        direct = frozenset(load_schema(ArtifactKind.ADAPTER)["$defs"]["ScoreRecord"]["properties"])
        assert adapter_score_record_keys() == direct


class TestTheTwoSchemasScoreRecordsAGREE:
    """One builder serves both manifests, so their ``ScoreRecord``s must be the same shape.

    F4's :func:`services.scoring.manifest_payload.score_record_payload` reads its declared
    keys from the **checkpoint** manifest and F8 reuses it rather than writing a second
    builder. That reuse is only sound while the two definitions agree - so the agreement is
    a test rather than an assumption. If a future re-vendor adds a member to one and not the
    other, this goes red and names it, instead of F8 silently writing the wrong key set into
    the artifact that travels.
    """

    def test_the_declared_keys_are_identical(self) -> None:
        checkpoint = manifest_payload.declared_keys()
        assert adapter_score_record_keys() == checkpoint, {
            "adapter_only": sorted(adapter_score_record_keys() - checkpoint),
            "checkpoint_only": sorted(checkpoint - adapter_score_record_keys()),
        }

    def test_the_required_keys_are_identical(self) -> None:
        assert (
            adapter_schema.required_keys(adapter_schema.SCORE_RECORD_DEF)
            == manifest_payload.required_keys()
        )

    def test_the_optimizer_identity_is_the_same_shape_in_both(self) -> None:
        """The schema's own description: *"same shape as the checkpoint manifest's, so
        consumers can share code."*"""
        adapter_keys = adapter_schema.declared_keys("OptimizerIdentity")
        checkpoint_keys = frozenset(
            load_schema(ArtifactKind.CHECKPOINT)["$defs"]["OptimizerIdentity"]["properties"]
        )
        assert adapter_keys == checkpoint_keys

    def test_both_schemas_close_every_object(self) -> None:
        """FPRD D6, re-verified by walking the files rather than by reading about it."""
        for kind in (ArtifactKind.ADAPTER, ArtifactKind.CHECKPOINT):
            for name, definition in load_schema(kind)["$defs"].items():
                if definition.get("type") == "object":
                    assert definition.get("additionalProperties") is False, f"{kind}:{name}"


class TestTheWriterDefinesNoFieldOfItsOwn:
    def test_every_top_level_key_written_is_DECLARED_by_the_schema(self, tmp_path: Path) -> None:
        declared = adapter_schema.declared_keys("AdapterManifestV1")
        assert set(saved(tmp_path)) <= declared

    def test_the_writer_omits_only_OPTIONAL_keys(self, tmp_path: Path) -> None:
        """``parts`` is the one declared key a save may legitimately leave out, and only
        when the run had no pack at all - which is a different fact from a pack that
        resolved to no parts."""
        declared = adapter_schema.declared_keys("AdapterManifestV1")
        required = adapter_schema.required_keys("AdapterManifestV1")
        written = set(saved(tmp_path, parts=None))
        assert required <= written
        assert declared - written == {"parts"}

    def test_the_schema_file_is_the_one_the_sync_test_covers(self) -> None:
        """So conformance and sync are about the same bytes."""
        from services.reindex import SCHEMA_FILENAMES, schema_dir

        path = schema_dir() / SCHEMA_FILENAMES[ArtifactKind.ADAPTER]
        assert path.is_file()
        assert path.name == "adapter-manifest-v1.json"
