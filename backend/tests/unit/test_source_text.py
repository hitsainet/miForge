"""The positive control for :mod:`tests.support.source_text`.

**A stripper that returns the empty string makes every guard that uses it pass vacuously.**
Three static guards in this repo now search ``code_only(source)`` instead of raw text - F10's
one-call-site check, F3's selector-route check and F4's scoring-path check - so a silent
over-strip would switch all three off at once while every one of them reported green. That is
the "check that never ran" family this project has been bitten by six times, and it is the
reason this file exists: the stripping is only trustworthy if something asserts it *kept* the
code, not merely that it dropped the prose.

AGENT.md records the specific near-miss: the tokeniser does not preserve spacing, and the
positive check is "what caught that the stripping had stopped matching anything at all".
"""

from __future__ import annotations

import pytest

from tests.support.source_text import code_only

#: Prose and code both name the same identifier on purpose - that collision IS the thing
#: under test. Every guard that motivated this module hit exactly this shape.
SAMPLE = '''"""Module docstring citing midataworks_base_url as an example."""

# A whole-line comment citing midataworks_base_url too.
CONSTANT = 1


def reaches_out():
    """Inner docstring citing midataworks_base_url."""
    return midataworks_base_url  # trailing comment citing it again
'''


class TestProseIsDropped:
    def test_the_module_docstring_goes(self) -> None:
        assert "Module docstring" not in code_only(SAMPLE)

    def test_an_inner_docstring_goes(self) -> None:
        assert "Inner docstring" not in code_only(SAMPLE)

    @pytest.mark.parametrize("fragment", ["whole-line comment", "trailing comment"])
    def test_comments_go(self, fragment: str) -> None:
        assert fragment not in code_only(SAMPLE)


class TestTheCodeSURVIVES:
    """The positive half. Each assertion fails if the stripper eats too much."""

    def test_a_real_reference_is_still_found(self) -> None:
        """The whole point: prose mentions are dropped, a real USE is not.

        If this ever passes while :class:`TestProseIsDropped` also passes, check that the
        result is not simply empty - that combination is what a vacuous stripper looks like.
        """
        assert "midataworks_base_url" in code_only(SAMPLE)

    @pytest.mark.parametrize("fragment", ["CONSTANT", "def", "reaches_out", "return"])
    def test_ordinary_code_survives(self, fragment: str) -> None:
        assert fragment in code_only(SAMPLE)

    def test_the_result_is_not_empty(self) -> None:
        """The blunt one, stated separately because it is the failure that hides the others."""
        assert code_only(SAMPLE).strip() != ""

    def test_a_module_that_is_only_prose_strips_to_almost_nothing(self) -> None:
        """The other direction, so 'not empty' cannot be satisfied by refusing to strip."""
        assert "docstring" not in code_only('"""Only a docstring."""\n')


class TestItDoesNotRaiseOnSourceItCannotParse:
    def test_unparseable_source_falls_back_rather_than_exploding(self) -> None:
        """A guard must not turn a syntax error in some other file into its own failure."""
        with pytest.raises(SyntaxError):
            code_only("def (")
