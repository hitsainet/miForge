"""The lint gate exists, runs every declared tool over the whole tree, and gates images.

ADR-018 declares Black, Ruff and MyPy strict; until 2026-10-16 no workflow ran any of them
(suite tracker 4.27). This pins the wiring: the job is in ``backend-tests.yml`` - which
``sync-to-clean.yml``'s gate requires before anything reaches the mirror, whose image build
trusts that verdict (2026-10-20) - and each tool runs over the paths it must, with no
``--exit-zero``, ``|| true`` or ``continue-on-error`` to make a red quiet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "backend-tests.yml"
SYNC = ROOT / ".github" / "workflows" / "sync-to-clean.yml"


def _workflow() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text("utf-8"))
    return loaded


def _lint_runs() -> list[str]:
    job = _workflow()["jobs"]["lint"]
    assert not job.get("continue-on-error"), "a lint job that may fail is not a gate"
    return [str(step["run"]) for step in job["steps"] if "run" in step]


class TestTheLintGate:
    def test_it_is_in_the_workflow_the_image_build_gates_on(self) -> None:
        gate = yaml.safe_load(SYNC.read_text("utf-8"))["jobs"]["gate"]
        assert "backend-tests.yml" in gate["env"]["REQUIRED_WORKFLOWS"].split()
        assert "lint" in _workflow()["jobs"]

    @pytest.mark.parametrize(
        "command",
        [
            "ruff check backend/src backend/tests",
            "black --check backend/src backend/tests",
            "mypy",
        ],
    )
    def test_each_tool_runs_over_the_whole_tree(self, command: str) -> None:
        runs = _lint_runs()
        assert command in runs, f"{command!r} is not a step of the lint job: {runs}"

    def test_nothing_quiets_a_red(self) -> None:
        for run in _lint_runs():
            for softener in ("--exit-zero", "|| true", "|| :", "--fix"):
                assert softener not in run, f"{softener!r} in a lint step: {run}"

    def test_the_linters_are_pinned(self) -> None:
        """A floating linter turns the gate red on a release, with no commit to blame."""
        import tomllib

        dev = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"][
            "optional-dependencies"
        ]["dev"]
        for tool in ("black", "ruff", "mypy"):
            pins = [spec for spec in dev if spec.split("==")[0] == tool]
            assert len(pins) == 1 and "==" in pins[0], f"{tool} is not pinned exactly: {dev}"

    def test_tests_alone_may_shout(self) -> None:
        """N802/N801 are off for ``backend/tests`` only; ``backend/src`` keeps them."""
        import tomllib

        lint = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["tool"]["ruff"]["lint"]
        assert lint["per-file-ignores"] == {"backend/tests/**": ["N802", "N801"]}
        assert "N802" not in lint.get("ignore", [])
