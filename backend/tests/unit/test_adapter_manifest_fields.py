"""Every manifest field's source, the two ``base`` cases, and the name/path rendering.

**This module defines no field** - ``adapter-manifest-v1.json`` is normative - so the tests
below check that each value came from the source the TDD's table names, and that the
document validates against the **real** file rather than against a description of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.errors import Refusal
from models.domain_state import ArtifactKind
from services.reindex import load_schema, schema_violations
from tests.support.test_adapter_doubles import (
    adapter_config,
    checkpoint_record,
    creation_record,
    hub_record,
    optimizer_block,
)
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths

HUB_REPO = "LiquidAI/LFM2.5-1.2B-Instruct"


def built(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "run_id": "run-7",
        "step": 40,
        "base": {"name": HUB_REPO, "manifest_sha256": None},
        "adapter": adapter_config(),
        "optimizer": optimizer_block(),
    }
    values.update(overrides)
    return adapter_manifest.build(**values)


class TestTheTwoBaseCasesTheSchemaDEFINES:
    """``AdapterBase.name``'s two cases, and the resolution order the schema fixes."""

    def test_a_HUB_base_names_the_repo_id_with_a_null_hash(self) -> None:
        """The case added to the schema on 2026-10-14, and *the ordinary case, not an edge
        one*: a standalone adapter trained on a Hub model."""
        block = adapter_manifest.base_block(hub_record(HUB_REPO))
        assert block == {"name": HUB_REPO, "manifest_sha256": None}
        assert "/" in block["name"], "a repo id contains a slash - the path converter's reason"

    def test_a_CHECKPOINT_base_names_the_checkpoints_manifest_name_and_its_hash(self) -> None:
        block = adapter_manifest.base_block(checkpoint_record(), manifest_sha256="9" * 64)
        assert block == {"name": "demo-base:run-1:step-40", "manifest_sha256": "9" * 64}
        # F5's own rendering, asserted rather than spelled: a plausible-looking name in the
        # wrong format would make the merge's base resolution refuse for the wrong reason.
        from services.checkpoints.paths import parse_qualified_name

        assert parse_qualified_name(block["name"]) == ("demo-base", "run-1", 40)

    def test_a_checkpoint_base_with_NO_HASH_is_refused_not_written_as_null(self) -> None:
        """``null`` asserts *the base was never a saved checkpoint*. Writing it for a base
        that **is** one would be a false assertion, and the exact base would stop being
        identifiable if its directory moved - which is what the field exists for."""
        with pytest.raises(Refusal) as caught:
            adapter_manifest.base_block(checkpoint_record())
        assert caught.value.code == "adapter_base_unresolved"
        assert "null" in caught.value.message

    def test_a_record_naming_no_base_at_all_is_refused(self) -> None:
        record = creation_record().model_copy(update={"base_model": ""})
        with pytest.raises(Refusal) as caught:
            adapter_manifest.base_block(record)
        assert caught.value.code == "adapter_base_unresolved"

    def test_the_schema_admits_both_cases_and_requires_only_the_name(self) -> None:
        """Read from the file: ``manifest_sha256`` is nullable with a null default, and
        ``name`` is the one required member."""
        definition = load_schema(ArtifactKind.ADAPTER)["$defs"]["AdapterBase"]
        assert definition["required"] == ["name"]
        assert {"type": "null"} in definition["properties"]["manifest_sha256"]["anyOf"]


class TestNameAndPathRenderOneIntegerTwoWays:
    """FR-008.33, BRD-01 R-01.24. ``step-<N>`` does not come back."""

    def test_the_name_renders_the_step_UNPADDED(self) -> None:
        assert adapter_paths.adapter_name(HUB_REPO, "run-7", 40) == f"{HUB_REPO}:run-7:adapter-40"

    def test_the_directory_renders_the_step_PADDED_to_six_digits(self) -> None:
        directory = adapter_paths.adapter_dir("run-7", 40, root=Path("/data/checkpoints"))
        assert directory.name == "step-000040"
        assert directory.parent.name == adapter_paths.ADAPTERS_SEGMENT

    def test_the_adapter_segment_distinguishes_it_from_a_checkpoint_of_the_same_step(
        self,
    ) -> None:
        from services.checkpoints.paths import qualified_name

        assert adapter_paths.adapter_name(HUB_REPO, "run-7", 40) != qualified_name(
            HUB_REPO, "run-7", 40
        )

    def test_the_padded_renderer_is_F5s_and_not_a_second_one(self) -> None:
        from services.checkpoints.paths import step_dir_name

        assert adapter_paths.adapter_dir("r", 7, root=Path("/x")).name == step_dir_name(7)

    @pytest.mark.parametrize("step", [0, 1, 40, 123456])
    def test_the_name_round_trips_including_a_base_that_contains_colons(self, step: int) -> None:
        """An adapter trained on a *checkpoint* carries that checkpoint's qualified name as
        its base, which contains colons - so the parser splits from the right."""
        base = "demo-base:run-1:40"
        parsed = adapter_paths.parse_adapter_name(adapter_paths.adapter_name(base, "run-9", step))
        assert parsed == (base, "run-9", step)

    @pytest.mark.parametrize(
        "name", ["", "nope", "base:run:40", "base:run:adapter-", "base:run:adapter-x", "adapter-1"]
    )
    def test_an_unparseable_name_returns_None_rather_than_a_guess(self, name: str) -> None:
        assert adapter_paths.parse_adapter_name(name) is None


class TestTheAdaptersDirectoryIsInsideTheWALKEDTree:
    """The correction to the TID's ``/data/adapters`` root, and why it matters.

    ``reindex`` walks ``settings.checkpoint_root`` and finds artifacts by manifest filename.
    An adapter under a root of its own would be indexed at save and then **deleted from the
    index by the next reindex and never rebuilt** - and the reindex-equivalence test would
    still pass, because it compares an index rebuilt from one root against responses served
    from that same root.
    """

    def test_an_adapter_lives_under_the_checkpoint_root(self, tmp_path: Path) -> None:
        directory = adapter_paths.adapter_dir("run-7", 40, root=tmp_path)
        assert tmp_path in directory.parents

    def test_reindex_would_find_its_manifest_by_the_filename_it_walks(self) -> None:
        from services.reindex import MANIFEST_FILENAMES

        assert MANIFEST_FILENAMES[ArtifactKind.ADAPTER] == adapter_paths.MANIFEST_NAME

    def test_the_adapter_latest_pointer_cannot_overwrite_the_checkpoint_one(
        self, tmp_path: Path
    ) -> None:
        """``atomic_dir`` swaps ``dest.parent / "latest"``. Were an adapter written directly
        inside the run directory, its ``latest`` would be the checkpoint ``latest``."""
        from services.checkpoints.paths import checkpoint_dir

        adapter = adapter_paths.adapter_dir("run-7", 40, root=tmp_path)
        checkpoint = checkpoint_dir("run-7", 40, root=tmp_path)
        assert adapter.parent != checkpoint.parent

    def test_a_run_id_called_adapters_cannot_collide(self, tmp_path: Path) -> None:
        """The level sits *inside* the run's directory, so there is nothing to collide with."""
        assert adapter_paths.adapters_dir("adapters", root=tmp_path) == (
            tmp_path / "adapters" / "adapters"
        )


class TestEveryFieldComesFromItsNamedSource:
    def test_the_constants_are_read_from_the_schema(self) -> None:
        doc = built()
        root = load_schema(ArtifactKind.ADAPTER)["$defs"]["AdapterManifestV1"]["properties"]
        assert doc["kind"] == root["kind"]["const"]
        assert doc["schema_version"] == root["schema_version"]["const"]

    def test_the_optimizer_block_is_F1s_shape_passed_through(self) -> None:
        assert built()["optimizer"] == optimizer_block()

    def test_the_adapter_block_comes_from_the_run_configuration(self) -> None:
        config = adapter_config(rank=16, alpha=32.0, target_modules=("q_proj", "v_proj"))
        assert built(adapter=config)["adapter"] == config.manifest_block()

    def test_created_at_is_RFC3339_and_generated_at_save(self) -> None:
        from datetime import datetime

        stamp = built()["created_at"]
        assert stamp.endswith("Z")
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))

    def test_weights_sha256_is_NOT_set_here(self) -> None:
        """The digest is of the file **as written**; hashing the in-memory state dict would
        describe bytes no consumer ever reads. So the saver adds it, and this document is
        deliberately not yet valid."""
        doc = built()
        assert "weights_sha256" not in doc
        assert schema_violations(doc, ArtifactKind.ADAPTER), (
            "a manifest with no digest must fail validation - BRD-01 R-01.27 verifies it "
            "and without the field a corrupted adapter loads silently (E-9)"
        )

    def test_the_document_validates_once_the_digest_is_added(self) -> None:
        doc = built()
        doc["weights_sha256"] = "a" * 64
        assert schema_violations(doc, ArtifactKind.ADAPTER) == []

    def test_the_four_fields_this_schema_LACKS_are_not_invented(self) -> None:
        """No ``domain``, no ``signal_source``, no ``selector_hash``, no ``grader_hash``
        (FPRD D7). The objects are closed, so inventing one would fail validation - and the
        asymmetry is reported rather than patched."""
        doc = built()
        doc["weights_sha256"] = "a" * 64
        for absent in ("domain", "signal_source", "selector_hash", "grader_hash"):
            assert absent not in doc
            assert schema_violations({**doc, absent: "x"}, ArtifactKind.ADAPTER), absent


class TestPartsAreF7sAndAreNeverConstructedHere:
    """FR-008.32, M16. **Against the shape F7 shipped, not the shape F8's documents
    describe** - and that difference is the finding.

    F8's FPRD D10 records the element as ``{slot, pack, source, type, name, version}`` with
    ``source`` as *"the member not to drop"*, taken from F7's **documents**. The element
    F7's producer actually emits, and the one the gate contract
    ``batch-record-v1.json#/$defs/PartReference`` settles, is
    ``{kind, contract_reference, external}`` with **exactly one of the latter two non-null**
    as the discriminator. A ``source`` check would have refused every element F7 produces.

    The guarantee D10 cares about is unchanged and is now enforced by the contract rather
    than by a remembered spelling: a part with **neither** reference is refused, because an
    unattributed training input makes the run unreproducible, and a part with **both** is
    refused because nothing says which provenance holds.
    """

    def _element(self, slot: Any, ref: Any) -> dict[str, Any]:
        from packs.records import PackVersionIdentity, part_reference_record

        return part_reference_record(
            slot, ref, pack=PackVersionIdentity(pack_id="contracts", version="3")
        ).element()

    def contract_element(self) -> dict[str, Any]:
        from packs.references import ContractPartRef
        from packs.slots import SlotKind

        return self._element(
            SlotKind.PROMPT_SET,
            ContractPartRef(type="prompt_set", name="contract-clauses", version="4"),
        )

    def external_element(self) -> dict[str, Any]:
        from packs.references import ExternalPartRef
        from packs.slots import SlotKind

        return self._element(
            SlotKind.CORPUS,
            ExternalPartRef(
                origin="https://example.invalid/corpus.jsonl",
                content_sha256="c" * 64,
                taken_in_at="2026-10-01T00:00:00Z",
            ),
        )

    def test_a_contract_part_from_F7s_PRODUCER_is_written_unreshaped(self) -> None:
        element = self.contract_element()
        assert built(parts=[element])["parts"] == [element]

    def test_an_external_part_keeps_its_own_provenance(self) -> None:
        element = self.external_element()
        written = built(parts=[element])["parts"][0]
        assert written == element
        assert written["contract_reference"] is None, (
            "an external part must not acquire a contract reference inside this artifact - "
            "that is F7's no-borrowed-provenance guarantee, inside F8's manifest"
        )
        assert written["external"]["content_sha256"] == "c" * 64

    def test_an_element_with_NEITHER_reference_is_refused(self) -> None:
        """The contract's own words: *"a part with neither is refused, because an
        unattributed training input makes the run unreproducible and nothing downstream can
        say where a behaviour came from."*"""
        with pytest.raises(Refusal) as caught:
            built(parts=[{"kind": "prompt_set", "contract_reference": None, "external": None}])
        assert caught.value.code == "adapter_parts_invalid"
        assert (caught.value.details or {})["references_present"] == []

    def test_the_SCHEMA_now_catches_BOTH_cases_and_the_code_still_diagnoses_them(self) -> None:
        """**This test was a tripwire and it fired.** It is the flipped half.

        F8 wrote it asserting the opposite - that ``PartReference`` had no ``oneOf``, ``not``
        or ``dependentSchemas``, so a neither-element and a both-element both validated clean
        - and said in its own docstring: *"If a later re-vendor adds the constraint, this
        goes red and says so."* The coordinator added the ``oneOf`` on 2026-10-14 and it went
        red, which is the whole value of asserting a fact about a contract instead of
        remembering one.

        So it now asserts the constraint **is** there and bites in both directions. The code
        check in :mod:`training.runtypes.adapters.manifest` is kept beside it, because a
        ``oneOf`` failure says only "is not valid under any of the given schemas" and names
        neither the element nor which direction; the refusal names both.
        """
        import jsonschema

        subschema = adapter_manifest.part_element_subschema()
        assert "oneOf" in subschema, (
            "the exactly-one-of constraint has been removed from batch-record-v1.json. An "
            "element naming NO part would validate clean again, and the adapter manifest is "
            "the one artifact built to travel without its run log - so an unattributed part "
            "there is unattributable for good."
        )
        validator = jsonschema.Draft202012Validator(subschema)

        neither = {"kind": "prompt_set", "contract_reference": None, "external": None}
        assert list(validator.iter_errors(neither)), "the schema accepts a part with no provenance"

        both = dict(self.contract_element())
        both["external"] = self.external_element()["external"]
        assert list(
            validator.iter_errors(both)
        ), "the schema accepts a part claiming two provenances"

        # And the two legitimate shapes still validate, or the constraint is too tight -
        # which would refuse every real part and is the more expensive direction to get wrong.
        assert not list(validator.iter_errors(self.contract_element()))
        assert not list(validator.iter_errors(self.external_element()))

    def test_an_element_with_BOTH_references_is_refused(self) -> None:
        """Exactly one. Nothing says which provenance would hold, so it is refused rather
        than resolved by precedence."""
        both = dict(self.contract_element())
        both["external"] = self.external_element()["external"]
        with pytest.raises(Refusal) as caught:
            built(parts=[both])
        assert caught.value.code == "adapter_parts_invalid"
        assert sorted((caught.value.details or {})["references_present"]) == [
            "contract_reference",
            "external",
        ]

    def test_an_element_with_the_shape_F8s_DOCUMENTS_describe_is_refused(self) -> None:
        """The finding, as a test. Had F8 written its documented spelling, this element
        would have been accepted and the project would have acquired a second answer."""
        with pytest.raises(Refusal) as caught:
            built(
                parts=[
                    {
                        "slot": "prompt_set",
                        "pack": "contracts/v3",
                        "source": "midataworks",
                        "type": "prompt-set",
                        "name": "contract-clauses",
                        "version": "4",
                    }
                ]
            )
        assert caught.value.code == "adapter_parts_invalid"

    def test_an_INVENTED_member_is_refused_because_the_contract_closes_the_object(
        self,
    ) -> None:
        extended = dict(self.contract_element())
        extended["licence_note"] = "CC-BY"
        with pytest.raises(Refusal) as caught:
            built(parts=[extended])
        assert caught.value.code == "adapter_parts_invalid"

    def test_a_non_object_element_is_refused(self) -> None:
        with pytest.raises(Refusal) as caught:
            built(parts=["prompt-set/contract-clauses@4"])
        assert caught.value.code == "adapter_parts_invalid"

    def test_the_schema_constrains_parts_items_to_objects_AND_NOTHING_MORE(self) -> None:
        """Verified against the file. This is *why* whoever spelled the element first would
        have defined it for the project, and why F8 reads the gate contract instead."""
        items = load_schema(ArtifactKind.ADAPTER)["$defs"]["AdapterManifestV1"]["properties"][
            "parts"
        ]["items"]
        assert items == {"type": "object"}

    def test_the_element_shape_is_READ_from_the_gate_contract(self) -> None:
        """One reader, through F7's own accessor - not a path spelled twice."""
        from packs.contracts import part_reference_subschema

        assert adapter_manifest.part_element_subschema() == part_reference_subschema()
        assert adapter_manifest.part_element_subschema()["additionalProperties"] is False

    def test_every_element_F7s_producer_emits_is_accepted(self) -> None:
        """The cross-seam assertion M16 exists for: F8 accepts what the one producer makes,
        for **both** source kinds, so the seam is checked from this side as well."""
        elements = [self.contract_element(), self.external_element()]
        assert built(parts=elements)["parts"] == elements

    def test_no_pack_omits_the_key_and_an_empty_pack_writes_an_empty_list(self) -> None:
        """Two different facts: *there was no pack* and *a pack resolved to no parts*."""
        assert "parts" not in built(parts=None)
        assert built(parts=[])["parts"] == []

    def test_this_module_does_not_CONSTRUCT_a_part_reference(self) -> None:
        """M16's static half, and it is about **construction, not mention**.

        The exactly-one-of check legitimately *names* ``contract_reference`` and ``external``
        in order to read them, so a text grep for those words is a check that cannot pass -
        the shape this project's lessons warn about twice over. So the assertion is on
        **dict literals**: no literal in this module may carry a part reference's members,
        which is what building an element here would look like.
        """
        import ast

        source = (
            Path(__file__).resolve().parents[3]
            / "backend/src/training/runtypes/adapters/manifest.py"
        ).read_text(encoding="utf-8")
        element_members = {"kind", "contract_reference", "external", "content_sha256"}
        offenders: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if len(keys & element_members) >= 2:
                offenders.append(f"line {node.lineno}: {sorted(keys)}")
        assert not offenders, (
            "a part reference is being constructed in the manifest assembler instead of "
            f"taken from F7's producer: {offenders}"
        )
