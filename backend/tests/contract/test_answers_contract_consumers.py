# Origin: miDataworks (Onegaishimas/miDataworks)
#   backend/tests/unit/contract/test_answers_contract_consumers.py @ d7c28abb
# Mode: adapt (docs/REUSE.md). Kept: the parity run over the shared pairs and its non-vacuity check.
# Changed: miForge's paths and sibling lookup; miServer's file runs in THIS interpreter (it imports
# only the standard library), never in another checkout's venv. The consumer table and its compare
# are miDataworks 023's (@ e32610fa), with miForge's consumers and paths.
"""The answer contract's cross-repo checks (021 FR-021.19 to .21, MF8; FTASKS 1.8).

**Classifier parity (MF8).** miForge's classifier (``backend/src/schemas/contract_tools.py``) keeps
miServer 044's row names and must answer every shared pair exactly as miServer's does. The pairs are
``tests/support/fixtures/answer_classifier_pairs.json``. miServer's ``contract_tools.py`` is read
from its checkout's ``origin/main`` with ``git show`` and run in a subprocess through
``tests/support/classifier_pairs.py``. Nothing is imported from miServer into this process.

The sibling is found through ``tests/support/siblings.py``: absent, or without the file on its
``main``, the test skips loudly, and fails under the require switches.

**Consumers (FR-021.20, .21; FTASKS 8.3, 8.4).** ``CONSUMERS`` names each app that vendors
``docs/schemas/miforge-answers-v1.json``, where (FTASKS 0.6, 0.7), and its state:

- ``pending``: it has not vendored the file yet; an absent copy passes (push 021-1);
- ``vendored``: an absent copy fails (021-2 for miAssist after its 006-4; miWorkbench when its
  ledger row C.2 lands).

A present copy is compared with the committed contract by the writer's own classifier, the copy as
the old side: equal passes; an additive difference passes and says the consumer must re-vendor (it
is behind); a breaking difference under the same major fails (it holds what miForge no longer
serves, which needs ``miforge-answers-v2.json``). A copy under another ``$id`` fails. A sibling
that is not checked out skips loudly, and fails under the require switches.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

from schemas import answers_contract, contract_tools
from tests.support import classifier_pairs, siblings

OURS = Path(__file__).resolve().parents[2] / "src" / "schemas" / "contract_tools.py"
THEIRS = "miserver/api/schemas/contract_tools.py"
REF = "origin/main"


def _run(classifier: Path) -> dict[str, list[list[object]]]:
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(classifier_pairs.__file__)),
            str(classifier),
            str(classifier_pairs.PAIRS_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    loaded: dict[str, list[list[object]]] = json.loads(done.stdout)
    return loaded


def test_the_classifier_answers_every_pair_as_miservers_does(tmp_path: Path) -> None:
    located = siblings.locate(siblings.MISERVER)
    repo = located.path
    if not (repo / ".git").exists():
        siblings.absent(f"miServer is not checked out (tried {', '.join(located.tried)})")
    shown = subprocess.run(
        ["git", "show", f"{REF}:{THEIRS}"],
        cwd=repo,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if shown.returncode != 0:
        siblings.absent(f"miServer {REF} has no {THEIRS}; fetch the checkout at {repo}")
    theirs = tmp_path / "miserver_contract_tools.py"
    theirs.write_bytes(shown.stdout)
    mine = _run(OURS)
    other = _run(theirs)
    differ = {pid: (mine[pid], other.get(pid)) for pid in mine if mine[pid] != other.get(pid)}
    assert not differ, f"the classifiers differ on these pairs (ours, miServer {REF}): {differ}"
    assert set(mine) == set(other) == {p["id"] for p in classifier_pairs.load()["pairs"]}


def test_the_parity_run_reports_what_this_process_classifies() -> None:
    """The subprocess path measures the same classifier as the in-process tests: a broken load
    would make the parity vacuous."""
    from schemas import contract_tools

    assert OURS.is_file(), OURS
    assert _run(OURS) == classifier_pairs.run(contract_tools.classify, classifier_pairs.load())


# -- the consumers (FR-021.20, .21) ---------------------------------------------------------------

COMMITTED = answers_contract.CONTRACT_PATH
#: app -> (the sibling, its vendored path, named by this major's file, and its state).
CONSUMERS: dict[str, tuple[siblings.Sibling, str, str]] = {
    "miAssist": (
        siblings.MIASSIST,
        "docs/schemas/" + answers_contract.CONTRACT_FILENAME,
        "pending",
    ),
    "miWorkbench": (
        siblings.MIWORKBENCH,
        "backend/src/mcp_server/vendored/" + answers_contract.CONTRACT_FILENAME,
        "pending",
    ),
}


def compare(committed: dict[str, Any], copy: dict[str, Any] | None, state: str) -> tuple[bool, str]:
    """(passes, why) for one consumer's copy."""
    if copy is None:
        if state == "pending":
            return True, "not vendored yet (pending)"
        return False, "the copy is missing, and the consumer is vendored"
    if copy.get("$id") != committed["$id"]:
        return False, f"the copy's $id is {copy.get('$id')!r}, not {committed['$id']!r}"
    old, new = answers_contract.body_of(copy), answers_contract.body_of(committed)
    changes = contract_tools.classify(old, new, answers_contract.resolver(copy, committed))
    if not changes:
        return True, "equal"
    bad = contract_tools.breaking(changes)
    if bad:
        return False, (
            f"breaking under the same major (needs {answers_contract.NEXT_MAJOR}):\n"
            + "\n".join(f"  {c}" for c in bad)
        )
    return True, (
        f"additive: the consumer must re-vendor (its revision {copy.get('revision')}, "
        f"committed {committed.get('revision')})"
    )


@pytest.mark.parametrize("app", sorted(CONSUMERS))
def test_each_consumers_copy_is_the_committed_contract_or_additively_behind(app: str) -> None:
    sibling, path, state = CONSUMERS[app]
    located = siblings.locate(sibling)
    if not located.path.is_dir():
        siblings.absent(f"{app} is not checked out (tried {', '.join(located.tried)})")
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    vendored = located.path / path
    copy = json.loads(vendored.read_text(encoding="utf-8")) if vendored.is_file() else None
    ok, why = compare(committed, copy, state)
    assert ok, f"{app} ({vendored}): {why}"
    if why.startswith("additive"):
        warnings.warn(f"{app}: {why}", stacklevel=1)


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(COMMITTED.read_text(encoding="utf-8"))
    return loaded


def _planted(tmp_path: Path, doc: dict[str, Any]) -> dict[str, Any]:
    path = tmp_path / answers_contract.CONTRACT_FILENAME
    path.write_text(answers_contract.render(doc), encoding="utf-8")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def test_an_equal_copy_passes(committed: dict[str, Any], tmp_path: Path) -> None:
    assert compare(committed, _planted(tmp_path, committed), "vendored") == (True, "equal")


def test_a_copy_behind_by_an_additive_revision_passes_and_says_re_vendor(
    committed: dict[str, Any], tmp_path: Path
) -> None:
    behind = json.loads(json.dumps(committed))
    behind["responses"].pop(sorted(behind["responses"])[0])  # the committed file ADDS this route
    behind["revision"] = max(committed["revision"] - 1, 0)
    ok, why = compare(committed, _planted(tmp_path, behind), "vendored")
    assert ok and "re-vendor" in why, why


def test_a_copy_holding_a_route_miforge_dropped_fails(
    committed: dict[str, Any], tmp_path: Path
) -> None:
    ahead = json.loads(json.dumps(committed))
    ahead["responses"]["GET /api/v1/a-route-miforge-dropped"] = {"200": {"type": "object"}}
    ok, why = compare(committed, _planted(tmp_path, ahead), "vendored")
    assert not ok and "breaking" in why, why


def test_a_copy_under_another_id_fails(committed: dict[str, Any], tmp_path: Path) -> None:
    other = {
        **json.loads(json.dumps(committed)),
        "$id": "https://hitsai.net/schemas/elsewhere.json",
    }
    ok, why = compare(committed, _planted(tmp_path, other), "vendored")
    assert not ok and "$id" in why, why


def test_an_absent_copy_passes_only_while_pending(committed: dict[str, Any]) -> None:
    assert compare(committed, None, "pending")[0] is True
    assert compare(committed, None, "vendored")[0] is False


def test_the_states_are_this_pushs() -> None:
    """021-1 ships both consumers pending (FR-021.20); 021-2 (miAssist) and a later push
    (miWorkbench) flip them."""
    assert {app: state for app, (_, _, state) in CONSUMERS.items()} == {
        "miAssist": "pending",
        "miWorkbench": "pending",
    }
    for app, (_, path, _) in CONSUMERS.items():
        assert Path(path).name == answers_contract.CONTRACT_FILENAME, app
