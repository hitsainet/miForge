"""miForge contains no call to the stale-clearing endpoint  (FR-011.34; **MUT-7**).

**Two layers, and which is which matters.** The *absence of the code* is **policy**, pinned
by this file. The *role gate* is **protocol**, enforced by miLLM: BRD-01 R-01.29 makes
``clear-stale`` operator-only and miForge holds ``trainer``, so the call is unreachable even
if the code tried. *A constraint the other side makes unreachable is worth more than one a
document merely states* - but **only the policy half is miForge's to keep**, so only the
policy half gets a mutation control.

Why it matters at all: clearing another process's attachment is how **two trainers end up
writing the same tensors**, and the holder may be mid-commit.

**The scanner has a self-test.** An absence test that cannot see what it is looking for is
always green, which is the construction-green failure this project keeps finding - including
in its own review harness.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
FORBIDDEN = "clear-stale"

#: The modules that may *name* the string. Naming it is not calling it: the client's op
#: table lists it as part of R-01.17's cited gated set, and asserts it is never sent;
#: ``identity.py`` holds the constant. **Every other mention is a finding.**
ALLOWED_MENTIONS = frozenset(
    {
        "training/hotupdate/client.py",
        "training/hotupdate/identity.py",
        "training/hotupdate/refusals.py",
    }
)


def modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def relative(path: Path) -> str:
    return str(path.relative_to(SRC))


def call_sites(tree: ast.AST) -> list[int]:
    """Lines where something is **called** with the forbidden op as an argument.

    By AST rather than by text, for the reason F10's own guard gives: *"a comment or
    docstring naming a function neither satisfies nor trips a check."* This file's docstring
    names ``clear-stale`` six times over, and a text matcher could not tell that from a call.
    """
    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            if isinstance(argument, ast.Constant) and argument.value == FORBIDDEN:
                found.append(node.lineno)
    return found


class TestTheCallIsAbsentFromTheSourceTree:
    def test_no_module_calls_anything_with_the_clear_stale_op(self) -> None:
        """**MUT-7 adds such a call and this is what catches it.** The protocol layer would
        still refuse it, which is exactly why the *code* must be caught: a call that is
        refused by the other side still means miForge tried."""
        offenders = {
            relative(path): lines
            for path in modules()
            if (lines := call_sites(ast.parse(path.read_text("utf-8"))))
            # The client's own guard raises *on* the op, which is a call carrying the
            # string. Excluded by module, not by heuristic, so the exclusion is auditable.
            and relative(path) not in ALLOWED_MENTIONS
        }
        assert offenders == {}, (
            f"{offenders} calls something with the {FORBIDDEN!r} op. R-02.34: miForge never "
            f"clears another process's attachment on its own initiative - that is how two "
            f"trainers end up writing the same tensors, and the holder may be mid-commit."
        )

    def test_the_client_refuses_the_op_rather_than_merely_omitting_it(self) -> None:
        """The second inside-miForge layer: an explicit assertion, so a *dynamic* op name
        reaching ``call()`` is caught too. The source scan above cannot see that form."""
        from training.hotupdate import identity
        from training.hotupdate.client import DELIBERATELY_ABSENT

        assert identity.FORBIDDEN_OP == FORBIDDEN
        assert FORBIDDEN in DELIBERATELY_ABSENT
        assert "OPERATOR-ONLY" in DELIBERATELY_ABSENT[FORBIDDEN]

    def test_no_module_names_the_endpoint_path(self) -> None:
        offenders = [
            relative(path)
            for path in modules()
            if "/api/weights/clear-stale" in path.read_text("utf-8")
            and relative(path) not in ALLOWED_MENTIONS
        ]
        assert offenders == []


class TestTheScannerSeesWhatItLooksFor:
    """**The self-test.** A guard that cannot see one form is a guard with a hole, and the
    hole is invisible until someone writes code through it."""

    @pytest.mark.parametrize(
        "form",
        [
            'client.call("clear-stale")',
            'client.call(op="clear-stale")',
            'self._post("clear-stale", {})',
            'httpx.post("clear-stale")',
        ],
    )
    def test_each_call_form_is_seen(self, form: str) -> None:
        assert call_sites(ast.parse(form)) != [], f"the scanner is blind to {form!r}"

    def test_a_mere_mention_is_not_a_call(self) -> None:
        """So the guard does not fail on this feature's own prose, which is how a correct
        document gets "fixed" to satisfy a checker."""
        assert call_sites(ast.parse('OPS = {"clear-stale": "never called"}')) == []
        assert call_sites(ast.parse('"""clear-stale is operator-only."""')) == []

    def test_the_allowlist_is_not_stale(self) -> None:
        """Each allowed module must exist **and** actually mention the op - otherwise the
        allowlist is excluding something that is no longer there and the exclusion reads as
        a reviewed decision."""
        for allowed in ALLOWED_MENTIONS:
            path = SRC / allowed
            assert path.is_file(), f"{allowed} does not exist; the allowlist is stale"
            assert FORBIDDEN in path.read_text(
                "utf-8"
            ), f"{allowed} no longer mentions {FORBIDDEN!r}; remove it from the allowlist"
