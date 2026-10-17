"""The validator refuses, and the refusal names what failed  (AC-010.3; BRD-02 AC18).

**The valid fixture is miDataworks' own document**, byte-for-byte from their fixture builder
(see ``tests/support/fixtures/midataworks_export/PROVENANCE.md``). Every invalid case is
derived from it by **one perturbation**, and the perturbations are driven by the **schema's
own ``required`` lists** - so a field that becomes required upstream produces a new case here
without anyone editing this file.

That is the whole of ADR-016's fixture rule applied to a contract: *fixtures whose fields
agree by construction are how a suite stays green over a real bug.* A manifest typed out here
would agree with a buggy validator, because both would encode the same misreading.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from services.parts import contract_validator as cv
from tests.support import part_fixtures as fx


class TestTheRealDocumentConforms:
    def test_the_upstream_fixture_validates(self):
        """**The one assertion in this repository that the two sides agree.**

        Not that miForge's reader agrees with miForge's fixture - that proves nothing - but
        that it agrees with a document miDataworks' own code produced.
        """
        assert cv.failures(fx.real_manifest()) == []

    def test_it_validates_from_its_committed_bytes_too(self):
        """Parsed from the file as committed, so a reformatting of the fixture is caught."""
        version, digest = cv.parse_and_validate(fx.real_manifest_bytes(), source="fixture")
        assert version.dataset == "humor-format-balanced"
        assert len(digest) == 64

    def test_the_digest_is_of_the_bytes_as_received(self):
        """FR-010.14. Re-serialising first would digest **miForge's** serialisation rather
        than what miDataworks sent, which makes the value useless as a cross-repo check -
        and that value is the only thing that lets two repositories agree they are looking
        at one manifest."""
        import hashlib

        raw = fx.real_manifest_bytes()
        _, digest = cv.parse_and_validate(raw, source="fixture")
        assert digest == hashlib.sha256(raw).hexdigest()

        from core.canonical import sha256_of

        assert digest != sha256_of(fx.real_manifest()), (
            "the digest matches a canonical re-serialisation, so it is NOT the digest of "
            "the bytes as received - the mutation FR-010.14 exists to forbid"
        )


class TestEveryRequiredFieldIsRefusedByName:
    @pytest.mark.parametrize(
        "field",
        sorted(cv.load_schema()["required"]),
    )
    def test_a_missing_top_level_field_refuses_naming_it(self, field: str):
        """Parametrised over the **schema's** ``required`` list, read from the vendored file."""
        broken = fx.without(fx.real_manifest(), field)
        found = cv.failures(broken)
        assert found, f"removing {field} validated clean"
        assert any(
            field in (f.expected or []) or field in f.message for f in found
        ), f"nothing in the refusal names {field}: {[f.message for f in found]}"

    def test_the_refusal_carries_pointer_keyword_and_expected(self):
        """FR-010.12. The operator must be able to fix the input from the refusal alone."""
        broken = fx.without(fx.real_manifest(), "content", "splits")
        with pytest.raises(Refusal) as caught:
            cv.validate(broken, source="unit test")
        assert caught.value.code == "dataset_version_invalid"
        errors = caught.value.details["errors"]
        assert errors
        first = errors[0]
        assert set(first) == {
            "instance_path",
            "schema_keyword",
            "schema_path",
            "expected",
            "message",
        }
        assert first["instance_path"] == "/content", "a JSON Pointer, not a $.jq path"
        assert first["schema_keyword"] == "oneOf" or "splits" in str(first["expected"])

    def test_all_failures_are_collected_not_just_the_first(self):
        """An operator fixing an export wants every problem at once."""
        broken = fx.real_manifest()
        broken["content"]["splits"][0].pop("sha256")
        broken["content"]["splits"][1].pop("logical_digest")
        found = cv.failures(broken)
        assert len(found) >= 2, [f.message for f in found]

    def test_a_failure_inside_the_content_union_names_the_split_and_the_field(self):
        """**The defect this test found.** ``content`` is a discriminated ``oneOf``, so
        without :func:`services.parts.contract_validator._descend` a missing ``sha256`` on
        one split reported as *"{the entire content object} is not valid under any of the
        given schemas"* at ``/content`` - naming neither the split nor the field and dumping
        a page of JSON into the message. An operator cannot act on that, which is exactly
        what FR-010.12 forbids.
        """
        broken = fx.real_manifest()
        broken["content"]["splits"][1].pop("sha256")
        found = cv.failures(broken)
        assert [f.instance_path for f in found] == ["/content/splits/1"], [
            f.instance_path for f in found
        ]
        assert "sha256" in found[0].message
        assert found[0].schema_keyword == "required"

    def test_a_content_kind_the_contract_does_not_define_keeps_the_union_error(self):
        """When **no** branch matches the discriminator, the original message is the useful
        one: the document names a content kind that does not exist, and descending into a
        branch it never meant would report a page of irrelevant field errors."""
        broken = fx.real_manifest()
        broken["content"]["content_kind"] = "nonsense"
        found = cv.failures(broken)
        assert [f.instance_path for f in found] == ["/content"]
        assert found[0].schema_keyword == "oneOf"

    def test_the_discriminator_list_matches_the_schemas_own(self):
        """The list exists because ``discriminator`` is an OpenAPI extension ``jsonschema``
        does not interpret - so there is nothing to read *through*. This is what stops it
        going stale: if the contract adds a union on another property, this goes red.
        """
        schema = cv.load_schema()
        names: set[str] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                discriminator = node.get("discriminator")
                if isinstance(discriminator, dict) and discriminator.get("propertyName"):
                    names.add(str(discriminator["propertyName"]))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(schema)
        assert names, "the schema declares no discriminators; this test would be vacuous"
        assert names <= cv._DISCRIMINATORS, sorted(names - cv._DISCRIMINATORS)

    def test_the_failures_are_ordered_stably(self):
        """A refusal whose ``details`` reorder between calls is one nobody can diff."""
        broken = fx.without(fx.real_manifest(), "caveats")
        assert [f.message for f in cv.failures(broken)] == [f.message for f in cv.failures(broken)]


class TestTheConstDiscriminators:
    def test_a_wrong_kind_is_refused_by_the_const(self):
        """E-6. The schema's ``const`` catches it and the refusal names the field."""
        broken = fx.real_manifest()
        broken["kind"] = "midataworks.something-else"
        found = cv.failures(broken)
        assert any(f.instance_path == "/kind" for f in found), [f.instance_path for f in found]

    def test_a_schema_version_of_2_is_refused(self):
        """E-7. **A v2 is a new file**, which the schema's own description says, not a
        widened reader. Accepting ``"2"`` here would mean miForge validating a contract it
        has never seen against the rules of the one it has."""
        broken = fx.real_manifest()
        broken["schema_version"] = "2"
        found = cv.failures(broken)
        assert any(f.instance_path == "/schema_version" for f in found)

    def test_an_unknown_top_level_key_is_refused(self):
        """``additionalProperties: false`` at the root, so R-02.35's "exactly the schema's
        fields" is enforced by the schema rather than by good intentions."""
        broken = fx.real_manifest()
        broken["invented_key"] = "whatever"
        assert cv.failures(broken)


class TestNotJsonIsNamedRatherThanCrashing:
    def test_bytes_that_are_not_json_refuse_by_name(self):
        """A ``JSONDecodeError`` reaching the application's last-resort handler is a 500,
        which turns a refusal the operator can act on into one they cannot."""
        with pytest.raises(Refusal) as caught:
            cv.parse_and_validate(b"<html>not a manifest</html>", source="an upload")
        assert caught.value.code == "dataset_version_not_json"
        assert caught.value.details["source"] == "an upload"


class TestTheReadViewCannotRejectWhatTheSchemaAccepts:
    def test_a_null_contract_reference_reads_as_none(self):
        """**E-21.** ``Target.contract_reference`` is ``anyOf [ContractReference, null]``
        *and* is in ``Target.required`` - so null is conformant and must not be refused."""
        document = fx.real_manifest()
        assert document["target"]["contract_reference"] is None
        version = cv.validate(document, source="unit test")
        assert version.contract_reference is None

    def test_the_schema_really_permits_it(self):
        """Parsed, not taken on trust. This is the assertion that makes the one above a
        statement about the **contract** rather than about this fixture."""
        target = cv.load_schema()["$defs"]["Target"]
        assert "contract_reference" in target["required"]
        options = target["properties"]["contract_reference"]["anyOf"]
        assert {"type": "null"} in options

    def test_a_present_reference_reads_and_renders_the_inner_layer_only(self):
        document, _ = fx.miforge_set_manifest(reference=("prompt-set", "support-tickets", "3"))
        version = cv.validate(document, source="unit test")
        reference = version.contract_reference
        assert reference is not None
        assert reference.as_dict() == {
            "type": "prompt-set",
            "name": "support-tickets",
            "version": "3",
        }
        assert reference.render() == "prompt-set/support-tickets@3"
        assert "/" not in reference.render().split("/")[0], "no pack prefix - that is F7's"

    def test_the_reference_shape_is_the_gate_contracts_shape(self):
        """``batch-record-v1.json``'s ``ContractReference``. **Their shape, not restated.**"""
        import json
        from pathlib import Path

        schema = json.loads((cv.schema_dir() / "batch-record-v1.json").read_text("utf-8"))
        assert sorted(schema["$defs"]["ContractReference"]["required"]) == [
            "name",
            "type",
            "version",
        ]
        document, _ = fx.miforge_set_manifest()
        reference = cv.validate(document, source="t").contract_reference
        assert reference is not None
        assert sorted(reference.as_dict()) == ["name", "type", "version"]
        assert Path(cv.schema_dir()).is_dir()

    def test_splits_carry_the_two_required_booleans(self):
        """FR-010.18 needs something to read rather than a field it has to default."""
        version = cv.validate(fx.real_manifest(), source="unit test")
        for split in version.splits:
            assert isinstance(split.held_out, bool)
            assert isinstance(split.evaluation_only, bool)
        assert [s.training_eligible for s in version.splits] == [True, False]

    def test_caveats_and_licences_are_carried_verbatim(self):
        """FR-010.17. **A dropped licence provenance is one of this project's three recorded
        defects of this class**, which is why this is a test and not a convention."""
        version = cv.validate(fx.real_manifest(), source="unit test")
        assert [c.code for c in version.caveats] == ["amber_check"]
        assert version.caveats[0].severity == "amber"
        assert version.caveats[0].detail == {"check": "C-5"}
        assert version.licence_classes == ("permits_redistribution",)
        assert version.licences[0]["displayed"] == "cc-by-2.0"
        assert version.licences[0]["table_version"] == 1

    def test_extensions_are_passed_through_unparsed(self):
        """miDataworks' extension point. Reading it would be miForge building on an
        unversioned surface."""
        document = fx.real_manifest()
        document["extensions"] = {"anything": {"they": ["like"]}}
        version = cv.validate(document, source="unit test")
        assert version.extensions == {"anything": {"they": ["like"]}}


class TestTheSchemaIsReadNotDescribed:
    def test_the_enums_come_from_the_file(self):
        assert cv.enum_at("$defs", "Target", "properties", "kind") == {
            "dataset",
            "trl_export",
            "detector_role",
            "miforge_set",
            "selector_config",
            "grader_config",
            "reward_bundle",
        }

    def test_an_optional_enum_is_found_through_its_any_of(self):
        """Their generator renders an optional enum as ``anyOf [{enum}, {null}]``; a reader
        that only understood the plain shape would see no enum and silently return nothing."""
        assert cv.enum_at("$defs", "Target", "properties", "miforge_set_kind") == {
            "prompt_set",
            "corpus",
            "preference_pairs",
            "test_set",
            "retention_set",
        }

    def test_a_missing_pointer_refuses_rather_than_returning_an_empty_set(self):
        """An empty set would make every exhaustiveness check pass **vacuously** - the
        construction-green failure this project keeps finding."""
        with pytest.raises(Refusal) as caught:
            cv.enum_at("$defs", "NoSuchDefinition", "properties", "kind")
        assert caught.value.code == "schema_shape_unexpected"

    def test_a_node_that_is_not_an_enum_refuses(self):
        with pytest.raises(Refusal) as caught:
            cv.enum_at("$defs", "Split", "properties", "path")
        assert caught.value.code == "schema_shape_unexpected"
