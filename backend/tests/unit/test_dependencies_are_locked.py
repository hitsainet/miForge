"""Every Python dependency is pinned exactly, once, and the image and CI install the same set.

Deps sync, 2026-10-21 (operator request; suite baseline ``miSuite/0xcc/
dependency_baseline_2026-10-21.md``, review record ``0xcc/reviews/deps_sync_2026-10-21.md``).

Until then ``pyproject.toml`` carried ranges (``fastapi>=0.115,<0.116``, ``torch>=2.4,<3``,
``pyarrow>=17,<22`` ...) and nothing pinned a transitive dependency, so two installs of the
same commit could differ. They did: the image resolved torch's own dependencies from PyPI
(filelock 4.2.4, fsspec 2026.9.0, networkx 3.7) while CI took the PyTorch CPU index's
(filelock 3.32.3, fsspec 2026.7.0, networkx 3.6.1) - the shape of the suite's
``dev-venv-vs-image-version-drift`` failure, where a changed API passed every local test and
failed only in production.

What is pinned here, each read from the files rather than restated:

* every requirement in ``pyproject.toml`` (core and every extra) is ``==`` - a range is red;
* ``constraints.txt`` (the full lock) agrees with every one of those pins;
* every install of the project - ``backend/Dockerfile`` and each ``backend-tests.yml`` step -
  passes ``-c constraints.txt``, and CI's CPU-torch pre-install is ``--no-deps`` at the
  pyproject pin (without ``--no-deps`` the CPU index's copies of torch's dependencies would
  come back);
* the RUNNING environment holds only distributions the lock pins, at the locked version. CI
  and the image both run this suite, so a red here in either is the two disagreeing.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
PYPROJECT = REPO / "pyproject.toml"
CONSTRAINTS = REPO / "constraints.txt"
DOCKERFILE = REPO / "backend" / "Dockerfile"
BACKEND_TESTS = REPO / ".github" / "workflows" / "backend-tests.yml"

#: The post-install check both install paths run. Loaded from its file, not restated, so the
#: suite and the build judge drift with ONE function.
CHECK_SCRIPT = REPO / "backend" / "scripts" / "check_installed_matches_lock.py"
CHECK_COMMAND = "python backend/scripts/check_installed_matches_lock.py constraints.txt"


def _load_check() -> Any:
    spec = importlib.util.spec_from_file_location("check_installed_matches_lock", CHECK_SCRIPT)
    assert spec and spec.loader, CHECK_SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load_check()
env_drift = check.drift

EXACT = re.compile(r"==[0-9][0-9A-Za-z.]*")


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def split(requirement: str) -> tuple[str, str]:
    """``"uvicorn[standard]==0.54.0"`` -> ``("uvicorn", "==0.54.0")``."""
    head = requirement.split(";", 1)[0]
    head = re.sub(r"\[[^\]]*\]", "", head).strip()
    match = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", head)
    assert match, requirement
    return canon(match.group(1)), match.group(2).replace(" ", "")


def declared(text: str | None = None) -> dict[str, str]:
    project = tomllib.loads(text if text is not None else PYPROJECT.read_text("utf-8"))["project"]
    requirements = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    return dict(split(r) for r in requirements)


def locked(text: str | None = None) -> dict[str, str]:
    return check.read_lock(text if text is not None else CONSTRAINTS.read_text("utf-8"))  # type: ignore[no-any-return]


def pip_installs(text: str) -> list[str]:
    """Every ``pip install`` command, with shell line continuations joined and comments cut."""
    joined = re.sub(r"\\\n", " ", text)
    commands = []
    for line in joined.splitlines():
        code = line.split("#", 1)[0]
        for part in re.split(r"&&|;", code):
            if re.search(r"\bpip install\b", part):
                commands.append(" ".join(part.split()))
    return commands


class TestTheParsers:
    def test_split_reads_extras_markers_and_ranges(self) -> None:
        assert split("uvicorn[standard]==0.54.0") == ("uvicorn", "==0.54.0")
        assert split('torch>=2.4,<3; platform_system == "Linux"') == ("torch", ">=2.4,<3")
        assert split("huggingface_hub==2.2.0") == ("huggingface-hub", "==2.2.0")

    def test_locked_refuses_a_range_and_a_duplicate(self) -> None:
        with pytest.raises(ValueError, match="not an exact pin"):
            locked("fastapi>=0.143\n")
        with pytest.raises(ValueError, match="twice"):
            locked("fastapi==0.143.3\nFastAPI==0.143.3\n")

    def test_pip_installs_joins_continuations_and_skips_comments(self) -> None:
        text = (
            "# pip install not-this\n"
            'RUN mkdir x \\\n    && pip install -c constraints.txt ".[bnb]" \\\n'
            "    && pip uninstall -y miforge\n"
        )
        assert pip_installs(text) == ['pip install -c constraints.txt ".[bnb]"']

    def test_env_drift_names_an_unlocked_and_a_different_version(self) -> None:
        lock = {"torch": "2.14.1", "filelock": "4.2.4"}
        assert env_drift({"torch": "2.14.1+cpu", "miforge": "0.1.0", "pip": "25"}, lock) == {}
        drift = env_drift({"filelock": "3.32.3", "greenlet": "3.5.6"}, lock)
        assert set(drift) == {"filelock", "greenlet"}


class TestPyprojectPinsExactly:
    def test_every_requirement_is_an_exact_pin(self) -> None:
        loose = {n: s for n, s in declared().items() if not EXACT.fullmatch(s)}
        assert not loose, f"pin with == (suite baseline 2026-10-21): {loose}"

    def test_the_lock_agrees_with_every_pin(self) -> None:
        lock = locked()
        disagree = {
            name: {"pyproject": spec, "constraints.txt": lock.get(name)}
            for name, spec in declared().items()
            if lock.get(name) != spec.removeprefix("==")
        }
        assert not disagree, f"regenerate constraints.txt (its header says how): {disagree}"

    def test_the_image_torch_arg_is_the_pyproject_pin(self) -> None:
        arg = re.search(r"^ARG TORCH_VERSION=(\S+)$", DOCKERFILE.read_text("utf-8"), re.M)
        assert arg, "backend/Dockerfile carries no ARG TORCH_VERSION"
        assert f"=={arg.group(1)}" == declared()["torch"]


class TestEveryInstallUsesTheLock:
    def test_the_image_copies_and_installs_with_the_lock(self) -> None:
        text = DOCKERFILE.read_text("utf-8")
        assert re.search(
            r"^COPY [^\n]*\bconstraints\.txt\b", text, re.M
        ), "backend/Dockerfile does not COPY constraints.txt into the build"
        installs = pip_installs(text)
        assert installs, "no pip install found in backend/Dockerfile"
        unlocked = [c for c in installs if "--no-deps" not in c and "-c constraints.txt" not in c]
        assert not unlocked, f"an image install resolves without the lock: {unlocked}"

    def test_every_ci_install_uses_the_lock(self) -> None:
        installs = pip_installs(BACKEND_TESTS.read_text("utf-8"))
        project = [c for c in installs if re.search(r"(-e\s+)?['\"]?\.(\[|['\"]|\s|$)", c)]
        assert len(project) >= 3, f"expected the lint, SQLite and PostgreSQL installs: {installs}"
        unlocked = [c for c in project if "-c constraints.txt" not in c]
        assert not unlocked, f"a CI install resolves without the lock: {unlocked}"

    def test_ci_takes_only_torch_from_the_cpu_index(self) -> None:
        torch_pin = declared()["torch"]
        installs = [c for c in pip_installs(BACKEND_TESTS.read_text("utf-8")) if "torch" in c]
        assert len(installs) >= 3, installs
        for command in installs:
            assert "--no-deps" in command, (
                f"{command!r}: without --no-deps the CPU index supplies torch's dependencies "
                "at versions the image never installs"
            )
            assert f"torch{torch_pin}" in command.replace('"', "").replace(
                "'", ""
            ), f"{command!r} is not pyproject's torch{torch_pin}"


class TestTheBuildsRunThePostInstallCheck:
    """A test that notices drift is later than a build that refuses it. Both install paths run
    the script straight after installing, so a drifted image never gets pushed."""

    def test_the_image_runs_it_after_installing_dependencies(self) -> None:
        text = re.sub(r"\\\n", " ", DOCKERFILE.read_text("utf-8"))
        runs = [line for line in text.splitlines() if line.startswith("RUN ")]
        deps = next((i for i, r in enumerate(runs) if "-c constraints.txt" in r), None)
        assert deps is not None, "no RUN installs with -c constraints.txt"
        assert (
            "check_installed_matches_lock.py constraints.txt" in runs[deps]
        ), "the dependency RUN does not end with the post-install lock check"
        assert re.search(
            r"^COPY [^\n]*backend/scripts/check_installed_matches_lock\.py", text, re.M
        ), "the check is not copied into the build"

    def test_every_ci_job_runs_it_after_installing(self) -> None:
        text = BACKEND_TESTS.read_text("utf-8")
        installs = text.count('pip install -c constraints.txt -e ".[dev]"')
        assert installs >= 3, installs
        assert (
            text.count(CHECK_COMMAND) == installs
        ), f"{installs} locked installs but {text.count(CHECK_COMMAND)} post-install checks"

    def test_the_script_exits_red_on_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(check, "installed_now", lambda: {"greenlet": "3.5.6"})
        assert check.main(["check", str(CONSTRAINTS)]) == 1
        monkeypatch.setattr(check, "installed_now", lambda: {"torch": "2.14.1+cpu"})
        assert check.main(["check", str(CONSTRAINTS)]) == 0


class TestTheRunningEnvironmentIsTheLock:
    def test_every_installed_distribution_is_locked_at_its_version(self) -> None:
        drift = env_drift(check.installed_now(), locked())
        assert not drift, (
            "this environment is not the locked set - install with "
            f"`pip install -c constraints.txt ...`: {drift}"
        )
