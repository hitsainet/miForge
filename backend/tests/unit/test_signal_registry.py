"""The four sources resolve from one registry  (FR-002.1, AC-002.1; ADR-016 #1).

**The negative control for this file:** delete one ``registry.register(...)`` line from
``training/signals/__init__.py``'s ``_REGISTRATIONS`` tuple and run this file - the first
two tests must go red. That control was run, not assumed; it is recorded in the report.

The second test is the one that keeps the vocabulary honest: the registry's key set is
asserted equal to ``checkpoint-manifest-v1.json``'s ``signal_source`` enum, **read from the
file rather than retyped**, because a source whose name cannot be written into a manifest
cannot be saved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import Refusal
from training.signals import ensure_registered, registry
from training.signals.base import RunContext, SourceCapability

SCHEMA_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas"


@pytest.fixture(autouse=True)
def _registered() -> None:
    ensure_registered()


class TestTheLiveRegistry:
    def test_all_four_sources_are_present_as_the_step_resolves_them(self) -> None:
        """Resolved through ``registry.get``, which is the one path every consumer uses."""
        assert registry.names() == ("corpus", "preference-pairs", "reward", "self-generation")
        for name in registry.names():
            source = registry.get(name)(RunContext())
            assert source.capability().name == name

    def test_the_registry_keys_equal_the_manifest_enum(self) -> None:
        """A source that cannot be written into a manifest cannot be saved (R-02.20, R-02.35)."""
        manifest = json.loads((SCHEMA_DIR / "checkpoint-manifest-v1.json").read_text("utf-8"))
        enum = manifest["$defs"]["CheckpointManifestV1"]["properties"]["signal_source"]["enum"]
        assert set(registry.names()) == set(enum)
        # The field is REQUIRED, which is what makes the enum closed in practice: this is
        # why the first release registers in tree only.
        assert "signal_source" in manifest["$defs"]["CheckpointManifestV1"]["required"]

    def test_an_unknown_name_refuses_listing_the_four(self) -> None:
        """E-1."""
        with pytest.raises(Refusal) as caught:
            registry.get("rlhf")
        assert caught.value.code == "signal_source_unknown"
        assert sorted(caught.value.details["registered"]) == list(registry.names())

    def test_registering_a_name_twice_raises_rather_than_winning(self) -> None:
        """Last-one-wins would make which source trains a run depend on import order."""
        existing = registry.capability("corpus")
        with pytest.raises(Refusal) as caught:
            registry.register("corpus", lambda ctx: None, existing)  # type: ignore[arg-type]
        assert caught.value.code == "signal_source_already_registered"

    def test_a_capability_naming_a_different_source_is_refused(self) -> None:
        """The listing and the registry must agree, or the UI offers a source that is not
        the one that would run."""
        capability = SourceCapability(
            name="not-corpus",
            part_kind="corpus",
            needs_selector=False,
            needs_grader=False,
            needs_generation=False,
            objective="sft",
            summary="x",
        )
        with pytest.raises(Refusal) as caught:
            registry.register("brand-new", lambda ctx: None, capability)  # type: ignore[arg-type]
        assert caught.value.code == "signal_source_capability_mismatch"

    def test_ensure_registered_is_idempotent(self) -> None:
        """A repeated import is normal; a second plugin claiming one name is not."""
        ensure_registered()
        ensure_registered()
        assert len(registry.names()) == 4


class TestEverySourceDeclaresWhatItNeeds:
    @pytest.mark.parametrize(
        "name,part_kind,selector,grader,generation,objective",
        [
            ("corpus", "corpus", False, False, False, "sft"),
            ("self-generation", "prompt_set", True, False, True, "sft"),
            ("preference-pairs", "preference_pairs", False, False, False, "dpo"),
            ("reward", "prompt_set", False, True, True, "grpo"),
        ],
    )
    def test_capabilities(
        self,
        name: str,
        part_kind: str,
        selector: bool,
        grader: bool,
        generation: bool,
        objective: str,
    ) -> None:
        """The table a listing is built from. ``reward`` reads a **prompt set**: the
        contract's ``miforge_set_kind`` has no reward value, verified by parsing it."""
        capability = registry.capability(name)
        assert capability.part_kind == part_kind
        assert capability.needs_selector is selector
        assert capability.needs_grader is grader
        assert capability.needs_generation is generation
        assert capability.objective == objective
