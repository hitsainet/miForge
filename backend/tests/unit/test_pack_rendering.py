"""Two rendering layers, one direction, no parser  (FR-007.27, FAC-9).

R-02.35: ``<type>/<name>@<version>`` is **miDataworks' contract form** and
``<pack>/<type>/<name>@<version>`` is **miForge prefixing its own pack identity for
display**. *"The two are different layers and neither is a rival spelling of the other."*

FAC-9 is asserted as an **absence**: the display form appears in no stored reference, no
outbound payload and no contract field, and there is no parser to read one back. An absence
is what the claim actually is, and reading the module would not establish it.

Mutation control (task 14.3b) for this file: make ``display_form`` return a plain ``str``, or
add a ``parse_display_form``. ``test_the_display_form_is_a_distinct_type`` and
``test_no_parser_exists`` must go red.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from core.errors import Refusal
from packs import rendering
from packs.references import ContractPartRef, ExternalPartRef
from packs.rendering import (
    DisplayRef,
    contract_form,
    display_form,
    external_form,
    unowned_form,
)

SOURCE_DIR = Path(rendering.__file__).resolve().parent

CONTRACT = ContractPartRef(type="retention-set", name="general", version="2")
EXTERNAL = ExternalPartRef(
    origin="/data/operator/prompts.jsonl",
    content_sha256="c" * 64,
    taken_in_at=datetime(2026, 10, 14, tzinfo=UTC),
)


class TestTheTwoLayers:
    def test_the_contract_form_is_theirs_and_carries_no_pack_segment(self) -> None:
        rendered = contract_form(CONTRACT)
        assert rendered == "retention-set/general@2"
        assert "/" in rendered
        # The inner layer has exactly two segments' worth of structure: a `type/name@version`
        # and no pack prefix. F10's own test asserts the mirror of this from its side.
        assert rendered.count("/") == 1

    def test_the_display_form_prefixes_the_pack_it_is_addressed_through(self) -> None:
        assert display_form(CONTRACT, "notary-md") == "notary-md/retention-set/general@2"

    def test_a_part_may_be_addressed_through_another_pack(self) -> None:
        """FR-007.25: the mockup's own case - a notary pack using a general retention set.

        The pack segment is a **parameter**, not a field of the reference, because addressing
        is a property of the composition rather than of the part.
        """
        assert display_form(CONTRACT, "general") == "general/retention-set/general@2"

    def test_the_superseded_dataset_at_version_rendering_appears_nowhere(self) -> None:
        """P-25's interim ``<dataset>@<version>`` is superseded (R-02.35).

        A grep over this feature's source, because *"we did not write it"* is a claim and a
        grep is a check.
        """
        for path in SOURCE_DIR.glob("*.py"):
            text = path.read_text("utf-8")
            assert "<dataset>@<version>" not in text.replace(
                "``<dataset>@<version>``", ""
            ), f"{path.name} renders the superseded form"


class TestAnExternalPartNeverRendersContractShaped:
    def test_the_external_marker_sits_where_a_type_would(self) -> None:
        """FR-007.24: the two can never be read as the same shape."""
        rendered = external_form(EXTERNAL)
        assert rendered.startswith("external/")
        assert rendered.endswith("@" + "c" * rendering.DIGEST_PREFIX)

    def test_the_display_form_of_an_external_part_carries_the_marker(self) -> None:
        rendered = display_form(EXTERNAL, "notary-md")
        assert rendered.startswith("notary-md/external/")

    def test_the_contract_renderer_cannot_be_handed_an_external_part(self) -> None:
        """Typed out, not checked at runtime: the signature is the guard MyPy enforces."""
        with pytest.raises(AttributeError):
            contract_form(EXTERNAL)  # type: ignore[arg-type]


class TestTheDisplayFormIsDisplayOnly:
    def test_the_display_form_is_a_distinct_type(self) -> None:
        assert isinstance(display_form(CONTRACT, "p"), DisplayRef)
        assert type(display_form(CONTRACT, "p")) is not str

    def test_no_parser_exists(self) -> None:
        """No round trip through the display string - that round trip IS the layer confusion."""
        for name in ("parse_display_form", "parse", "from_display", "split_display"):
            assert not hasattr(rendering, name), (
                f"rendering.{name} exists. A parser would let a display form be read back "
                f"into a contract reference, which R-02.35 forbids."
            )

    def test_the_display_form_never_enters_a_stored_reference(self) -> None:
        """The stored reference is the typed one; a ``DisplayRef`` is not accepted anywhere.

        Asserted by construction: ``ContractPartRef``'s fields are the contract's three, and
        a slot holds a reference rather than a string, so there is no field a display form
        could be assigned to. The negative control for task 14.3b writes one into a reference
        column and this test plus the registry's must go red.
        """
        reference = ContractPartRef(type="t", name="n", version="1")
        assert not any(
            isinstance(getattr(reference, field), DisplayRef)
            for field in type(reference).model_fields
        )


class TestAPartAddressedThroughNoPackIsRefused:
    def test_it_refuses_rather_than_inventing_a_segment(self) -> None:
        """FPRD OQ-9, surfaced as a question rather than resolved by a guess.

        The mockup offers ``general/retention/mixed@2`` and ``general`` is not one of its
        packs. miForge has no built-in domain - a banner this feature renders - so inventing
        an unowned-parts namespace here would answer a product question in a renderer.
        """
        with pytest.raises(Refusal) as raised:
            display_form(CONTRACT, None)
        assert raised.value.code == "pack_part_unresolved"
        assert "no pack" in raised.value.message

    def test_the_refusal_carries_the_inner_form_not_a_display_form(self) -> None:
        with pytest.raises(Refusal) as raised:
            display_form(EXTERNAL, None)
        assert raised.value.details["reference"] == external_form(EXTERNAL)
        assert "/" in str(raised.value.details["reference"])


class TestAnUnownedPartRendersItsInnerFormAndNothingMore:
    """FPRD OQ-9, **settled 2026-10-14**: no unowned-parts namespace, and no invented segment.

    The refusal in the class above is kept. What is added here is the named remedy the refusal
    message points at, because a refusal naming a remedy that does not exist leaves the caller
    with the private ``_inner`` or with a hand-built f-string - and a hand-built one is how the
    two layers get conflated.

    Mutation control for this class: make ``unowned_form`` prefix anything - ``"general/"``, or
    ``f"{ref.type}/"`` - and ``test_no_segment_is_invented_for_either_source_kind`` goes red on
    both kinds. Make ``display_form(ref, None)`` fall back to ``unowned_form`` instead of
    refusing, and ``test_a_lost_pack_still_refuses_rather_than_rendering`` goes red.
    """

    def test_no_segment_is_invented_for_either_source_kind(self) -> None:
        assert unowned_form(CONTRACT) == "retention-set/general@2"
        assert unowned_form(EXTERNAL) == "external//data/operator/prompts.jsonl@cccccccccccc"

    def test_it_is_exactly_the_inner_form_of_each_kind(self) -> None:
        """Derived from the two inner renderers, so the three can never drift apart."""
        assert unowned_form(CONTRACT) == contract_form(CONTRACT)
        assert unowned_form(EXTERNAL) == external_form(EXTERNAL)

    def test_the_mockups_general_namespace_is_not_resurrected(self) -> None:
        """``general`` here is the retention set's *name*, not a pack. The mockup's
        ``general/retention/mixed@2`` read it as a pack segment; this asserts it is not one."""
        rendered = unowned_form(CONTRACT)
        assert not rendered.startswith("general/")
        assert rendered.count("/") == 1

    def test_it_is_display_only_like_the_pack_prefixed_form(self) -> None:
        """A ``DisplayRef``, so MyPy strict rejects it where a stored ``str`` is required.

        The unowned rendering is still a rendering. A caller that wants the contract string
        *as data* calls ``contract_form`` and gets a ``str``.
        """
        assert isinstance(unowned_form(CONTRACT), DisplayRef)
        assert isinstance(unowned_form(EXTERNAL), DisplayRef)

    def test_a_lost_pack_still_refuses_rather_than_rendering(self) -> None:
        """**The reason this is a second function and not a lenient first one.**

        ``pack_id`` arrives from nullable storage, so a ``None`` is more often a pack that was
        lost than a part that genuinely has none. ``display_form`` must still refuse it; only a
        caller that names the unowned case gets a rendering.
        """
        with pytest.raises(Refusal):
            display_form(CONTRACT, None)
        assert unowned_form(CONTRACT) == contract_form(CONTRACT)

    def test_the_refusal_names_the_function_that_renders_it(self) -> None:
        """A remedy a caller can act on. The message named one before it existed."""
        with pytest.raises(Refusal) as raised:
            display_form(CONTRACT, None)
        assert "unowned_form" in raised.value.message
        assert hasattr(rendering, "unowned_form")

    def test_there_is_still_no_parser_in_the_other_direction(self) -> None:
        """Adding a rendering must not add a reader. Rule 3 of the module, re-asserted here
        because this class is the one that adds a public renderer."""
        assert not any(
            name.startswith("parse") or name.startswith("from_") for name in dir(rendering)
        )
