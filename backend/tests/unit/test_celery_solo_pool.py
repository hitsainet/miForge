"""The solo pool, and the guard that keeps its consequence honoured  (foundation 6.2).

ADR-006 fixes every worker at ``--pool=solo -c 1``. The configuration is asserted here
because **a command-line flag is a thing an operator can omit**, and the two consequences
of the pool are not optional:

* ``revoke(terminate=True)`` signals a pool child and solo has none, so the call returns
  successfully and nothing happens;
* a busy solo worker never reads the control queue.

The second half of this file is the **guard**: no module under ``backend/src/`` may reach
for Celery's terminating revoke. It is a scan rather than a review note because this is the
exact shape of mistake that looks correct - the call succeeds, and the operator sees a
cancellation that was accepted.

**This guard's first version was wrong, and its own negative case could not have found it.**
It matched the file as text, so it flagged the three modules whose docstrings *name* the
prohibition - including :mod:`workers.cancellation`, whose entire purpose is to forbid it.
Its negative case was ``# cancellation is cooperative; never terminate=True``, which
contains no ``revoke(`` at all and therefore passed against a scanner with no comment
handling whatsoever: **a negative case that agrees with the guard by construction**, the
same shape as a fixture whose fields agree by construction, and for the same reason the
usual cause of a green suite over a real defect.

So the scan now walks the **AST**, which ignores docstrings, comments and string literals
for free and needs no exclusion logic to get wrong. The negative cases below include the
three spellings that defeated the regex, each of which must be judged innocent - so a
future edit back to a text scan goes red rather than re-acquiring the defect. The shape is
copied from ``test_no_sibling_imports.py``: the scanner is a plain function over a string,
because a scanner that can only be run over the real tree cannot be proven to see anything.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from workers.celery_app import celery_app

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "backend" / "src"

#: Deliberately empty, for the reason foundation 1.5 states: an allowlist is where a guard
#: goes to die. There is no legitimate terminating revoke under the solo pool - the call
#: cannot work, so an exception would only be a way of pretending it does.
ALLOWLIST: frozenset[str] = frozenset()


def find_terminating_revokes(source: str, filename: str = "<string>") -> list[tuple[int, str]]:
    """Return ``(lineno, rendered_call)`` for every terminating revoke in ``source``.

    Catches a call to anything **named** ``revoke`` - bare, or as the attribute of any
    expression, so ``app.control.revoke(...)``, ``celery_app.control.revoke(...)`` and
    ``AsyncResult(tid).revoke(...)`` are all seen - carrying a ``terminate`` keyword whose
    value is a **truthy constant**. ``terminate=True``, ``terminate=1`` and
    ``terminate="yes"`` are hits; ``terminate=False`` and a bare ``revoke(tid)`` are not,
    because a non-terminating revoke is a legitimate thing to send to a queue.

    **Two limits, stated rather than guessed at** (ADR-018: refuse with a reason rather
    than guess):

    * A call renamed at import - ``from celery.app.control import revoke as kill`` - is
      **not** seen, because the name is gone by the time this walks the tree. The
      no-sibling-import guard forbids no such alias, so the only way to reach Celery's
      control API here is through ``celery_app.control``, which keeps the attribute name;
      and a module that went to the trouble of aliasing it would be a deliberate act, not
      the accident this guard exists to catch.
    * ``terminate=some_variable`` is **not** flagged. Its value cannot be judged
      statically, and flagging it would be the fabricated verdict ADR-018 forbids - the
      same decision ``test_no_sibling_imports.py`` makes for ``import_module(name)``.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise AssertionError(f"{filename} does not parse: {exc}") from exc

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None
        )
        if name != "revoke":
            continue
        for keyword in node.keywords:
            if keyword.arg != "terminate":
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and bool(value.value):
                hits.append((node.lineno, f"revoke(..., terminate={value.value!r})"))
    return hits


def python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py"))


class TestTheConfiguration:
    def test_the_pool_is_solo_in_the_configuration_not_only_on_the_command_line(self):
        assert celery_app.conf.worker_pool == "solo"

    def test_concurrency_is_one(self):
        assert celery_app.conf.worker_concurrency == 1

    def test_nothing_is_prefetched(self):
        """Prefetching on a queue whose tasks run for hours parks a task in one worker's
        buffer while another worker is idle."""
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_acks_are_early_so_an_interrupted_step_is_never_redelivered(self):
        """R-02.31: an interrupted step is **never retried**, because re-applying a
        non-idempotent AdamW update to a half-written parameter set compounds the damage.
        ``task_acks_late=True`` would redeliver the whole run."""
        assert celery_app.conf.task_acks_late is False
        assert celery_app.conf.task_reject_on_worker_lost is False

    def test_results_expire_so_redis_is_never_a_store_of_record(self):
        """ADR-003: Redis is the broker and the progress channel, never a store of record."""
        assert celery_app.conf.result_expires == 3600


# --------------------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------------------


def test_source_tree_is_not_empty() -> None:
    """A guard that scans nothing is green - the construction-green failure this project
    has already shipped once, in a review harness that passed against zero documents."""
    assert python_files(), f"no Python files found under {SRC}; the guard would be vacuous"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_module_does_not_call_a_terminating_revoke(path: Path) -> None:
    relative = str(path.relative_to(SRC))
    if relative in ALLOWLIST:  # pragma: no cover - the allowlist is empty
        pytest.fail(
            f"{relative} is in ALLOWLIST, but there is no legitimate terminating revoke "
            f"under the solo pool: the call cannot work. Remove the entry and the call."
        )
    hits = find_terminating_revokes(path.read_text(encoding="utf-8"), relative)
    assert not hits, "\n".join(
        f"{relative}:{line} calls {rendered} - it signals a pool child and the solo pool "
        f"has none, so the call succeeds and NOTHING HAPPENS. Reach a running job through "
        f"the mf_runs row instead (workers/cancellation.py)."
        for line, rendered in hits
    )


# --------------------------------------------------------------------------------------
# The self-test. Task 10.2's discipline: a guard that has never been seen fail is not
# known to work, and every spelling it cannot see is a way to break ADR-006 with a green
# suite.
# --------------------------------------------------------------------------------------

SHOULD_CATCH: list[tuple[str, str]] = [
    ("bare call", "revoke(task_id, terminate=True)\n"),
    ("through control", "app.control.revoke(task_id, terminate=True)\n"),
    ("full app path", "celery_app.control.revoke(tid, terminate=True)\n"),
    ("on a result object", "AsyncResult(tid).revoke(terminate=True)\n"),
    ("spaced keyword", "app.control.revoke(tid, terminate = True)\n"),
    ("split over lines", "revoke(\n    task_id,\n    terminate=True,\n)\n"),
    ("truthy int", "app.control.revoke(tid, terminate=1)\n"),
    ("inside a function", "def stop(t):\n    app.control.revoke(t, terminate=True)\n"),
    ("inside a try", "try:\n    revoke(t, terminate=True)\nexcept Exception:\n    pass\n"),
]


@pytest.mark.parametrize(("label", "source"), SHOULD_CATCH, ids=[c[0] for c in SHOULD_CATCH])
def test_scanner_sees_every_spelling(label: str, source: str) -> None:
    """The scanner must catch this form. If it does not, the guard above is decorative."""
    assert find_terminating_revokes(
        source, f"<self-test {label}>"
    ), f"the scanner did NOT see a terminating revoke in the {label!r} form:\n{source}"


SHOULD_IGNORE: list[tuple[str, str]] = [
    # ---- the three spellings that defeated this guard's first, text-based version. Each
    # ---- is real prose from one of the three modules it wrongly flagged.
    (
        "module docstring",
        '"""``revoke(terminate=True)`` signals a pool child and solo has none."""\n',
    ),
    ("hash comment", "# Never revoke(terminate=True): solo has no pool child.\n"),
    ("string literal", "MESSAGE = 'do not call revoke(task_id, terminate=True)'\n"),
    (
        "function docstring",
        'def stop(t):\n    """Never revoke(t, terminate=True) - cooperative only."""\n'
        "    return t\n",
    ),
    # ---- genuinely innocent code
    ("non-terminating revoke", "app.control.revoke(task_id)\n"),
    ("terminate false", "app.control.revoke(task_id, terminate=False)\n"),
    ("terminate zero", "app.control.revoke(task_id, terminate=0)\n"),
    # ---- not judgeable statically; see the scanner's docstring
    ("variable terminate", "app.control.revoke(t, terminate=hard)\n"),
    # ---- a different function that happens to share a keyword
    ("unrelated call", "pool.shutdown(terminate=True)\n"),
]


@pytest.mark.parametrize(("label", "source"), SHOULD_IGNORE, ids=[c[0] for c in SHOULD_IGNORE])
def test_scanner_does_not_cry_wolf(label: str, source: str) -> None:
    """False positives matter as much as misses.

    **The first four cases are the defect.** A guard that fires on the documentation of
    the rule it enforces gets an allowlist entry, and then it protects nothing - and the
    three files it flagged were the three that explain why the rule exists. A guard that
    flags ``pool.shutdown(terminate=True)`` gets switched off for the same reason.
    """
    assert not find_terminating_revokes(
        source, f"<negative {label}>"
    ), f"false positive on the {label!r} case:\n{source}"
