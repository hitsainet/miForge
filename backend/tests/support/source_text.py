"""Reading a module's **code** without its prose.

**A citation is not an operation.** A static guard that greps raw source text cannot tell a
module that *calls* something from one that *explains* it, and this project has now paid for
that distinction eight times:

* F4's migration guard flagged `0016` for a **comment** citing F4's own precedent;
* F11's fence test tripped on its own explanatory comment (``mode == "attach"``);
* ``test_midataworks_one_call_site.py`` swept in ``clients/activations.py`` for one line of
  prose naming ``clients/midataworks.py`` as the shape it was built to;
* and on 2026-10-16 two more, both from docstrings written to *record* a defect: F3's
  ``selectors/settings.py`` cites ``midataworks_base_url`` while explaining the split-module
  blindness, and F4's ``scoring/settings.py`` cites ``test_activations_client.py``.

Each time the checker was wrong and the work was right. *Every single time a fresh checker in
this project disagreed with careful work, the work was right and the checker was wrong* - so
the fix is always to compare against the code, never to edit the prose into silence.

This module exists so there is **one** implementation rather than a fourth copy. It was
promoted here from ``tests/unit/test_midataworks_one_call_site.py`` on 2026-10-16, where it
had been retuned on 2026-10-14; that file now delegates.

``tests/unit/test_source_text.py`` carries the **positive control** that this does not strip
everything - without it, a stripper that returned ``""`` would make every guard in the repo
pass vacuously, which is the failure mode that matters most here.
"""

from __future__ import annotations

import ast
import io
import tokenize

__all__ = ["code_only"]


def code_only(source: str) -> str:
    """``source`` with every docstring and whole-line comment removed.

    Comments are dropped by the tokeniser; docstrings are dropped by walking the tree,
    because a module docstring is an ``Expr`` holding a ``Constant`` and the tokeniser cannot
    tell it from a string the code uses.

    Note the tokeniser does **not** preserve spacing - the result is token-joined and is fit
    for substring and membership checks, not for display or re-parsing.
    """
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            end = first.end_lineno or first.lineno
            docstring_lines.update(range(first.lineno, end + 1))

    kept: list[str] = []
    for index, line in enumerate(source.splitlines(), start=1):
        if index in docstring_lines:
            continue
        kept.append(line)
    without_docstrings = "\n".join(kept)

    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(without_docstrings).readline):
            if token.type == tokenize.COMMENT:
                continue
            out.append(token.string)
    except (tokenize.TokenError, IndentationError):  # pragma: no cover - defensive
        return without_docstrings
    return " ".join(out)
