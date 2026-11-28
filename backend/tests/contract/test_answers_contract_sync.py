# Origin: miDataworks (Onegaishimas/miDataworks)
#   backend/tests/unit/contract/test_answers_contract_sync.py @ d7c28abb
# Mode: adapt (docs/REUSE.md). Kept: the five checks. Changed: miForge's imports, identity and
# sibling lookup (tests/support/siblings.py); the subprocess writes its rendering to a file, because
# create_app logs to stdout.
"""The published answer contract is the live app's, deterministic and digested (FR-021.1, .2, .19).

- Two interpreters with different hash seeds render byte-identical text.
- The committed ``docs/schemas/miforge-answers-v1.json`` equals the writer's rendering of the live
  app (the pin). Regenerate with
  ``PYTHONPATH=backend/src:backend python backend/scripts/generate_output_schemas.py --write``.
- ``content_sha256`` recomputes from the file itself, so a hand edit without the writer is red.
- Its top-level keys are miServer's (MF7): one reader serves every app's contract. Cross-repo: skips
  loudly without miServer, and fails under the require switches.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from schemas import answers_contract
from schemas import contract_tools as tools
from tests.support import siblings

REPO = Path(__file__).resolve().parents[3]
RENDER = (
    "import logging, sys\n"
    "logging.disable(logging.CRITICAL)\n"
    "from schemas import answers_contract as a\n"
    "open(sys.argv[1], 'w', encoding='utf-8').write(a.render(a.answers_contract()))\n"
)
MISERVER_CONTRACT = "docs/schemas/miserver-answers-v1.json"


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8"))
    return loaded


def _render_with_seed(seed: str, out: Path) -> str:
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "PYTHONPATH": f"{REPO}/backend/src{os.pathsep}{REPO}/backend",
    }
    done = subprocess.run(
        [sys.executable, "-c", RENDER, str(out)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return out.read_text(encoding="utf-8")


def test_two_hash_seeds_render_identical_bytes(tmp_path: Path) -> None:
    first = _render_with_seed("1", tmp_path / "one.json")
    second = _render_with_seed("2", tmp_path / "two.json")
    assert first, "the writer rendered nothing"
    assert first == second, "the contract depends on the hash seed (or a clock): FR-021.1"


def test_the_committed_contract_is_the_live_apps() -> None:
    text = answers_contract.CONTRACT_PATH.read_text(encoding="utf-8")
    failures = answers_contract.stale(text, answers_contract.answers_contract())
    assert not failures, "\n".join(failures)


def test_the_file_states_its_identity(committed: dict[str, Any]) -> None:
    assert answers_contract.CONTRACT_PATH.is_file()
    assert answers_contract.CONTRACT_PATH.parent == REPO / "docs" / "schemas"
    assert committed["$id"] == "https://hitsai.net/schemas/miforge-answers-v1.json"
    assert committed["kind"] == "miforge.answer-contract/v1"
    assert "do not edit by hand" in committed["description"]


def test_the_digest_recomputes(committed: dict[str, Any]) -> None:
    assert answers_contract.problems(committed) == []
    assert committed["content_sha256"] == tools.content_digest(committed)


def test_a_hand_edit_without_the_writer_is_red(committed: dict[str, Any]) -> None:
    edited = copy.deepcopy(committed)
    first = sorted(edited["responses"])[0]
    status = sorted(k for k in edited["responses"][first] if k[:1].isdigit())[0]
    edited["responses"][first][status] = {"type": "string"}
    found = answers_contract.problems(edited)
    assert any("content_sha256" in line for line in found), found


def test_a_bad_revision_is_red(committed: dict[str, Any]) -> None:
    edited = dict(committed, revision=0)
    assert any("revision" in line for line in answers_contract.problems(edited))


def test_a_stale_file_names_the_file_and_the_command(committed: dict[str, Any]) -> None:
    body = answers_contract.body_of(committed)
    grown = copy.deepcopy(body)
    grown["responses"]["GET /api/v1/__a_route_nobody_generated"] = {"200": {"type": "object"}}
    found = answers_contract.stale(answers_contract.render(committed), grown)
    assert any(
        answers_contract.CONTRACT_FILENAME in line and answers_contract.REGENERATE in line
        for line in found
    ), found


def test_the_top_level_keys_are_miservers(committed: dict[str, Any]) -> None:
    """Read from miServer's ``origin/main`` (``git show``), not its working tree."""
    located = siblings.locate(siblings.MISERVER)
    repo = located.path
    if not (repo / ".git").exists():
        siblings.absent(f"miServer is not checked out (tried {', '.join(located.tried)})")
    shown = subprocess.run(
        ["git", "show", f"origin/main:{MISERVER_CONTRACT}"],
        cwd=repo,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if shown.returncode != 0:
        siblings.absent(
            f"miServer origin/main has no {MISERVER_CONTRACT}; fetch the checkout at {repo}"
        )
    assert set(committed) == set(json.loads(shown.stdout))
