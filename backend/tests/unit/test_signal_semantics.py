"""A column is bound by its declared semantic, never by its name  (FR-002.7, AC-002.6).

**The decisive test in this file is** :meth:`TestNothingIsInferredFromAName.
test_columns_are_not_chosen_by_name`: a part whose columns are *called* ``text`` and
``chosen`` while their declared semantics are ``other`` must refuse. It is the one that
proves the rule, and it is mutation control M-4 - fall back to the first string column when
no semantic matches and it must go red.

Why it matters more than it reads: a corpus bound to the wrong column trains perfectly
happily on the wrong field. The loss falls, the run completes, every checkpoint saves, and
nothing anywhere says the model was fitted to a label column.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from training.signals import semantics


def column(name: str, semantic: str) -> dict[str, str]:
    return {"name": name, "semantic": semantic, "role": "content", "arrow_type": "string"}


class TestTheEnumComesFromTheVendoredSchema:
    def test_every_schema_semantic_is_classified(self) -> None:
        """The import-time exhaustiveness check, asserted rather than assumed to have run.

        A semantic added upstream must produce a loud startup failure, not fall through a
        default - which is F10's ``kind_map`` arrangement, cited rather than copied.
        """
        classified = {s for members in semantics.ROLE_SEMANTICS.values() for s in members}
        classified |= set(semantics.UNUSED_SEMANTICS)
        assert semantics.SCHEMA_SEMANTICS <= classified
        assert classified <= semantics.SCHEMA_SEMANTICS

    def test_every_unused_semantic_carries_its_reason(self) -> None:
        """ "Unused" with no reason is indistinguishable from "nobody looked at it"."""
        assert all(reason for reason in semantics.UNUSED_SEMANTICS.values())


class TestEachSourceBindsWhatItNeeds:
    def test_corpus_accepts_plain_text(self) -> None:
        binding = semantics.resolve_columns([column("body", "text")], "corpus")
        assert binding.as_record() == {"text": "text"}
        assert binding.column("text") == "body"

    def test_corpus_accepts_chat_messages(self) -> None:
        binding = semantics.resolve_columns([column("turns", "messages")], "corpus")
        assert binding.as_record() == {"messages": "messages"}

    def test_corpus_accepts_a_prompt_completion_pair(self) -> None:
        binding = semantics.resolve_columns(
            [column("q", "prompt"), column("a", "completion")], "corpus"
        )
        assert binding.as_record() == {"prompt": "prompt", "completion": "completion"}

    def test_a_prompt_set_binds_the_prompt(self) -> None:
        for source in ("self-generation", "reward"):
            binding = semantics.resolve_columns([column("q", "prompt")], source)
            assert binding.column("prompt") == "q"

    def test_pairs_bind_chosen_and_rejected(self) -> None:
        binding = semantics.resolve_columns(
            [column("good", "chosen"), column("bad", "rejected")], "preference-pairs"
        )
        assert binding.column("chosen") == "good"
        assert binding.column("rejected") == "bad"


class TestNothingIsInferredFromAName:
    def test_columns_are_not_chosen_by_name(self) -> None:
        """**The test this module exists for** (mutation control M-4).

        The columns are *called* ``text`` and ``chosen``; their declared semantics are
        ``other``. "The first string column" is a default that always fires, and ADR-018
        calls a default that always fires a fabricated value.
        """
        columns = [column("text", "other"), column("chosen", "other")]
        with pytest.raises(Refusal) as caught:
            semantics.resolve_columns(columns, "corpus")
        assert caught.value.code == "corpus_no_content_column"
        assert caught.value.details["declared"] == ["other"]

    def test_a_part_declaring_nothing_usable_lists_what_it_does_declare(self) -> None:
        """E-4: the refusal says what to do next, not merely that something failed."""
        columns = [column("label", "label"), column("p", "probability")]
        with pytest.raises(Refusal) as caught:
            semantics.resolve_columns(columns, "corpus")
        assert caught.value.details["declared"] == ["label", "probability"]
        assert "label, probability" in caught.value.message

    def test_pairs_without_preference_columns_refuse_by_their_own_code(self) -> None:
        """E-12."""
        with pytest.raises(Refusal) as caught:
            semantics.resolve_columns([column("body", "text")], "preference-pairs")
        assert caught.value.code == "pairs_no_preference_columns"

    def test_a_prompt_set_without_a_prompt_column_refuses_by_its_own_code(self) -> None:
        with pytest.raises(Refusal) as caught:
            semantics.resolve_columns([column("body", "text")], "self-generation")
        assert caught.value.code == "prompt_set_no_prompt_column"


class TestAmbiguityIsRefusedRatherThanResolved:
    def test_two_usable_candidates_with_no_configuration_refuse_and_list_them(self) -> None:
        columns = [column("short", "text"), column("long", "text")]
        with pytest.raises(Refusal) as caught:
            semantics.resolve_columns(columns, "corpus")
        assert caught.value.code == "column_semantic_ambiguous"
        assert caught.value.details["candidates"] == ["short", "long"]

    def test_configuration_resolves_the_ambiguity(self) -> None:
        columns = [column("short", "text"), column("long", "text")]
        binding = semantics.resolve_columns(columns, "corpus", configured={"text": "long"})
        assert binding.column("text") == "long"

    def test_a_configured_column_that_is_not_usable_is_refused(self) -> None:
        """Naming a column whose semantic cannot fill the role is a configuration error,
        not a licence to use it."""
        columns = [column("short", "text"), column("label", "label")]
        with pytest.raises(Refusal) as caught:
            semantics.resolve_columns(columns, "corpus", configured={"text": "label"})
        assert caught.value.code == "column_binding_unknown"


class TestTheSourcePolicyIsOneTable:
    def test_an_unknown_source_refuses_rather_than_returning_no_roles(self) -> None:
        """An empty role set would make every part "usable" by accident."""
        with pytest.raises(Refusal) as caught:
            semantics.usable_role_sets("rlhf")
        assert caught.value.code == "signal_source_unknown"
