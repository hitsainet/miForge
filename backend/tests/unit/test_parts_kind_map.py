"""The kind map is exhaustive over the **schema's own** enums  (AC-010.6; FR-010.15).

Parametrised over the values read from ``docs/schemas/midataworks-dataset-version-v1.json``,
not over a list typed here. The difference is the whole mechanism: *a single-case test passes
while a new upstream value falls through a default*, and a hard-coded list is a third
description of someone else's vocabulary.

**The import-time assertion is the real guard** and these tests only prove it is not
vacuous. When miDataworks adds a target kind, :mod:`services.parts.kind_map` fails to import
- loudly, at startup - rather than treating the new value as something it is not.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from services.parts import contract_validator as cv
from services.parts import kind_map
from tests.support import part_fixtures as fx

SCHEMA_TARGET_KINDS = sorted(cv.enum_at("$defs", "Target", "properties", "kind"))
SCHEMA_SET_KINDS = sorted(cv.enum_at("$defs", "Target", "properties", "miforge_set_kind"))


#: What each ``target.kind`` must additionally carry, because the contract's ``Target`` has
#: an ``allOf`` of ``if/then`` branches that constrain one field per kind to ``type:
#: "string"`` - and ``type: "string"`` **excludes null**. Discovered by writing the tests
#: below rather than by reading the schema, which is the asymmetry this project keeps
#: rediscovering.
BRANCH_FIELDS: dict[str, dict[str, object]] = {
    "trl_export": {"trl_type": "sft", "trl_version": "1.14.1"},
    "detector_role": {"detector_role": "train"},
    "miforge_set": {"miforge_set_kind": "prompt_set"},
}


def target_for(kind: str, **overrides: object) -> dict[str, object]:
    """A conformant ``target`` object for ``kind``, with its branch fields filled in."""
    return {
        "kind": kind,
        "dataset_target_type": None,
        "trl_type": None,
        "trl_version": None,
        "detector_role": None,
        "miforge_set_kind": None,
        "contract_reference": None,
        **BRANCH_FIELDS.get(kind, {}),
        **overrides,
    }


def version_with(target: dict[str, object]) -> cv.DatasetVersion:
    """A **validated** document carrying ``target``. Validation first, always.

    Deliberately not a way to build an unvalidated document: every caller of
    :func:`services.parts.kind_map.resolve` in production has validated first, and a test
    helper that skipped it would be testing a path nothing takes.
    """
    document = fx.real_manifest()
    document["target"] = target
    return cv.validate(document, source="unit test")


class TestEveryValueIsMappedOrRefusedByName:
    @pytest.mark.parametrize("target_kind", SCHEMA_TARGET_KINDS)
    def test_every_schema_target_kind_has_an_explicit_outcome(self, target_kind: str):
        """**No fall-through.** Either a part kind, or a refusal that names the kind."""
        handled = kind_map.CONSUMED_TARGET_KINDS | set(kind_map.REFUSED_TARGET_KINDS)
        assert target_kind in handled, (
            f"{target_kind!r} is in the vendored schema and is neither mapped nor refused. "
            f"That is the value a default would swallow."
        )

    @pytest.mark.parametrize("set_kind", SCHEMA_SET_KINDS)
    def test_every_schema_set_kind_maps_to_a_part_kind(self, set_kind: str):
        assert set_kind in kind_map.MIFORGE_SET_KIND_TO_PART_KIND

    @pytest.mark.parametrize("target_kind", sorted(kind_map.REFUSED_TARGET_KINDS))
    def test_a_refused_kind_refuses_naming_itself_and_why(self, target_kind: str):
        """E-10. ``detector_role`` is a miStudio target and ``trl_export`` is a TRL
        trainer's: real artifacts for other consumers, declined rather than unknown."""
        if target_kind == "reward_bundle":
            # Its content is a `reward_bundle` union branch, so a conformant document needs
            # a whole different `content`. Covered by the mapping assertion above; building
            # one here would test the schema's union, not the map.
            pytest.skip("needs a reward_bundle content branch; the mapping is asserted above")
        with pytest.raises(Refusal) as caught:
            kind_map.resolve(version_with(target_for(target_kind)))
        assert caught.value.code == "dataset_version_target_unsupported"
        assert caught.value.details["target_kind"] == target_kind
        assert caught.value.details["reason"]

    @pytest.mark.parametrize("set_kind", SCHEMA_SET_KINDS)
    def test_a_miforge_set_resolves_to_its_part_kind(self, set_kind: str):
        resolved = kind_map.resolve(
            version_with(target_for("miforge_set", miforge_set_kind=set_kind))
        )
        assert resolved.part_kind == set_kind
        assert resolved.miforge_set_kind == set_kind
        assert resolved.is_config is False

    @pytest.mark.parametrize("config_kind", ["selector_config", "grader_config"])
    def test_a_config_target_resolves_to_its_own_part_kind(self, config_kind: str):
        """**The two axes.** BRD-03 R-03.48 lists six bullets of sets miForge consumes and
        the sixth is "selector and grader configurations" - two kinds in one bullet. In the
        schema they are not ``miforge_set_kind`` values at all; they are their own
        ``target.kind`` values with ``content_kind: "config"``. Both statements are true and
        describe different axes."""
        document, _ = fx.config_manifest(kind=config_kind.removesuffix("_config"))
        resolved = kind_map.resolve(cv.validate(document, source="unit test"))
        assert resolved.part_kind == config_kind
        assert resolved.is_config is True

    def test_the_schema_itself_refuses_a_null_set_kind_and_the_documents_say_otherwise(self):
        """E-11, **corrected against the schema rather than against the feature documents.**

        The FPRD's E-11, the FTID's section 3.5 and the FTASKS' 3.5 all say the schema
        *"requires the key present"* while *"a null value on this target is an omission in
        substance"* - which reads as *the schema permits it, so this layer must catch it.*
        **It does not permit it.** ``Target``'s third ``allOf`` branch is
        ``if kind == "miforge_set" then {"miforge_set_kind": {"type": "string"}}``, and
        ``type: "string"`` excludes ``null``.

        So E-11 is enforced one layer earlier than three documents say, and the refusal in
        ``kind_map`` is defence in depth for an unvalidated caller. Both halves are asserted
        here, because the interesting fact is **where** the guard is.
        """
        document = fx.real_manifest()
        document["target"] = target_for("miforge_set", miforge_set_kind=None)
        found = cv.failures(document)
        assert [f.instance_path for f in found] == ["/target/miforge_set_kind"], [
            f.instance_path for f in found
        ]
        assert found[0].expected == "string"

    def test_kind_maps_own_refusal_still_fires_for_an_unvalidated_document(self):
        """Defence in depth: ``resolve`` is public and nothing stops an unvalidated call."""
        document = fx.real_manifest()
        document["target"] = target_for("miforge_set", miforge_set_kind=None)
        unvalidated = cv.DatasetVersion(raw=document)
        with pytest.raises(Refusal) as caught:
            kind_map.resolve(unvalidated)
        assert caught.value.code == "dataset_version_miforge_set_kind_missing"
        assert sorted(caught.value.details["expected_one_of"]) == SCHEMA_SET_KINDS


class TestTheMapCannotDriftFromEitherSide:
    def test_it_names_no_value_the_schema_does_not_define(self):
        """A ghost is as bad as a gap: it means the schema moved and the map did not."""
        handled = kind_map.CONSUMED_TARGET_KINDS | set(kind_map.REFUSED_TARGET_KINDS)
        assert handled <= set(SCHEMA_TARGET_KINDS), sorted(handled - set(SCHEMA_TARGET_KINDS))

    def test_the_part_kinds_are_the_gate_contracts_vocabulary(self):
        """``PartReference.kind`` in ``batch-record-v1.json``. A part kind nothing downstream
        can record is a part that cannot appear in a batch's provenance."""
        import json

        schema = json.loads((cv.schema_dir() / "batch-record-v1.json").read_text("utf-8"))
        enum = set(schema["$defs"]["PartReference"]["properties"]["kind"]["enum"])
        assert set(kind_map.PART_KINDS) == enum

    def test_the_import_time_assertion_is_not_vacuous(self):
        """It must actually fire. Proved by calling it with a value removed from the map -
        the mutation control for this file, run inline because the function is pure."""
        original = dict(kind_map.REFUSED_TARGET_KINDS)
        try:
            kind_map.REFUSED_TARGET_KINDS.pop("detector_role")
            with pytest.raises(Refusal) as caught:
                kind_map._assert_exhaustive()
            assert caught.value.code == "kind_map_incomplete"
            assert "detector_role" in caught.value.details["unhandled"]
        finally:
            kind_map.REFUSED_TARGET_KINDS.clear()
            kind_map.REFUSED_TARGET_KINDS.update(original)
        # Restored, and the restore is confirmed rather than assumed.
        kind_map._assert_exhaustive()

    def test_it_also_fires_on_a_ghost(self):
        original = dict(kind_map.REFUSED_TARGET_KINDS)
        try:
            kind_map.REFUSED_TARGET_KINDS["invented_kind"] = "not in the schema"
            with pytest.raises(Refusal) as caught:
                kind_map._assert_exhaustive()
            assert caught.value.code == "kind_map_names_a_ghost"
        finally:
            kind_map.REFUSED_TARGET_KINDS.clear()
            kind_map.REFUSED_TARGET_KINDS.update(original)
        kind_map._assert_exhaustive()
