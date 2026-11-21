"""Every ``mf_`` table declared under ``src/`` is created by some Alembic revision.

**This exists because one was not, and the suite could not see it.**
``training/signals/batch_index.py`` declared ``mf_batches``, F2 wrote a row per assembled
batch, F9's run screen read it — and **no revision created it**, so on a database built the
way production builds one the table was absent and the first batch of the first run would
have failed with ``relation "mf_batches" does not exist``. Six integration test files
called ``batch_index.metadata.create_all(engine)`` in their own fixtures, which made the
table exist for the tests and for nothing else. Green suite, 33 exercised FRs, no table.

That is ADR-016 #1 in its database form: *a capability is not shipped until a test fails
when its wiring is removed.* ``test_db_migration.py`` cannot catch it, by construction —
it compares the migrated schema with ``Base.metadata``, and these feature-local tables are
deliberately **off** ``Base.metadata`` per foundation 4.2's ratchet. So the comparison that
covers them has to be this one, over the declarations themselves.

The check is **mechanical and bidirectional**: a declared table with no revision is the
defect above; a created table nothing declares is a migration left behind by deleted code.
Both are reported, and the positive controls below require the two sides to be non-empty,
because a sweep that found no declarations would pass while asserting nothing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2]
SRC = BACKEND / "src"
VERSIONS = BACKEND / "alembic" / "versions"

#: Matches a Core ``sa.Table("mf_x", ...)`` and an ORM ``__tablename__ = "mf_x"``. Both
#: spellings are in use: ADR-017 puts an ORM row in ``models/`` and a feature-local index
#: table on its own ``MetaData``.
_DECLARATIONS = (
    re.compile(r'sa\.Table\(\s*"(mf_[a-z0-9_]+)"'),
    re.compile(r'__tablename__\s*=\s*"(mf_[a-z0-9_]+)"'),
)


def _declared() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in _DECLARATIONS:
            for match in pattern.finditer(text):
                found.setdefault(match.group(1), []).append(str(path.relative_to(BACKEND)))
    return found


def _created() -> dict[str, list[str]]:
    """Tables created by a revision, found by AST rather than substring.

    By AST because a docstring or a comment mentioning a table is not a ``create_table``,
    and a substring sweep that counted those would report a table as migrated when the
    revision only talks about it.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(VERSIONS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "create_table" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value, []).append(path.name)
    return found


@pytest.fixture(scope="module")
def declared() -> dict[str, list[str]]:
    return _declared()


@pytest.fixture(scope="module")
def created() -> dict[str, list[str]]:
    return _created()


class TestTheSweepItselfMeasuresSomething:
    """Positive controls. Both assertions below pass vacuously against empty sets."""

    def test_declarations_were_found(self, declared: dict[str, list[str]]) -> None:
        assert len(declared) >= 15, (
            f"found only {sorted(declared)} declared tables. If the declaration spellings "
            f"changed, this file's regexes stop seeing them and the assertion below holds "
            f"over nothing."
        )

    def test_revisions_were_parsed(self, created: dict[str, list[str]]) -> None:
        assert len(created) >= 15, (
            f"found only {sorted(created)} tables created by revisions - the AST walk is "
            f"not seeing create_table calls"
        )

    def test_the_two_known_spellings_are_both_represented(
        self, declared: dict[str, list[str]]
    ) -> None:
        """One regex matching nothing would silently halve the coverage."""
        core = [t for t, where in declared.items() if any("tables.py" in w for w in where)]
        orm = [t for t, where in declared.items() if any("/models/" in w for w in where)]
        assert core, "no Core sa.Table declarations matched"
        assert orm, "no ORM __tablename__ declarations matched"


class TestEveryDeclaredTableIsMigrated:
    def test_no_table_is_declared_without_a_revision_creating_it(
        self, declared: dict[str, list[str]], created: dict[str, list[str]]
    ) -> None:
        missing = {name: declared[name] for name in sorted(set(declared) - set(created))}
        assert not missing, (
            f"these tables are declared in code and created by no migration: {missing}. On "
            f"a database built by `alembic upgrade head` they do not exist, so the first "
            f"write to one fails with `relation ... does not exist` - while a test that "
            f"calls `metadata.create_all(engine)` in its own fixture passes. That is how "
            f"`mf_batches` reached two review rounds unmigrated."
        )

    def test_no_revision_creates_a_table_nothing_declares(
        self, declared: dict[str, list[str]], created: dict[str, list[str]]
    ) -> None:
        """The other direction: a migration outliving the code that used it.

        Reported rather than tolerated because an undeclared table is schema nothing reads,
        and the next person cannot tell whether it is dead or whether its reader is missing.
        """
        orphans = {
            name: created[name]
            for name in sorted(set(created) - set(declared))
            if name.startswith("mf_")
        }
        assert not orphans, (
            f"these tables are created by a migration and declared nowhere in src/: " f"{orphans}"
        )

    def test_no_table_is_created_by_two_revisions(self, created: dict[str, list[str]]) -> None:
        """Two revisions creating one table is how the revision graph forks."""
        duplicated = {name: where for name, where in created.items() if len(where) > 1}
        assert not duplicated, f"created more than once: {duplicated}"
