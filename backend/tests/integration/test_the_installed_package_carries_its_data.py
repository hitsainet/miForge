"""The INSTALLED package carries every data file the code reads at run time (017 OR2-017-1).

The image runs ``pip install --no-deps .`` (``backend/Dockerfile``) and the MCP pod starts
``python -m mcp_server`` from WORKDIR ``/app``, so it imports the **site-packages copy**, never
``backend/src``. Until 2026-11-16 ``mcp_server/output_schemas.json`` was not package data: the
wheel left it out, the server logged a warning and published all 90 tools untyped, and every
test stayed green because every test imports from the source tree, where the file sits beside
the module. The Dockerfile's own ``/usr/local/lib/docs/schemas`` note is the same class.

This builds the wheel the way the image does (``pyproject.toml`` and ``backend/src`` only, no
build isolation, no index) and installs it into a THROWAWAY venv with that venv's own ``pip``. The
running interpreter's ``pip`` is never run: a ``pip install --prefix/--target`` from a venv's
own ``pip`` uninstalls a same-named distribution from THAT venv, and did so to a developer venv's
editable install on 2026-11-16. The install sees only the throwaway venv; only afterwards does a
``.pth`` line give the import its dependencies (the running interpreter's site-packages
directories; their own ``.pth`` files, an editable install of this repository among them, are not
processed). The import runs in a process started outside the repository. Nothing here reaches
the network.

The data-file audit counts the files the wheel is BUILT FROM - the copied ``backend/src`` - not
the working directory, so a git-ignored ``*.egg-info`` or ``build/`` in a developer checkout is
neither copied nor counted (R3 review fix OR3-017-2, 2026-11-16).

R2 review fix (OR2-017-1), 2026-11-16. Negative control: delete the
``[tool.setuptools.package-data]`` table in ``pyproject.toml``; both cases go red.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "backend" / "src"

#: Files under ``backend/src`` that are not run-time data and need not ship.
NOT_RUNTIME = {"py.typed"}

#: Never copied: build and install by-products a developer checkout may hold (git-ignored).
NOT_SOURCE = ("__pycache__", "*.pyc", "*.egg-info", "build", "dist")


def _data_files(src: Path) -> list[str]:
    """Every non-Python file under the copied ``src``, as a wheel path (``pkg/name``)."""
    found = []
    for path in src.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix in {".py", ".pyc"} or path.name in NOT_RUNTIME:
            continue
        found.append(path.relative_to(src).as_posix())
    return sorted(found)


def _run(
    args: list[str], cwd: Path, pythonpath: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cwd),
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


def _dependency_dirs() -> list[str]:
    """The running interpreter's site-packages directories, a layered venv's included."""
    dirs = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
    dirs.update(entry for entry in sys.path if entry.endswith("site-packages"))
    return sorted(entry for entry in dirs if Path(entry).is_dir())


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("wheel")
    project = root / "project"
    project.mkdir()
    # What the image's `COPY pyproject.toml` and `COPY backend/src/` put beside each other.
    shutil.copy2(REPO / "pyproject.toml", project / "pyproject.toml")
    shutil.copytree(SRC, project / "backend" / "src", ignore=shutil.ignore_patterns(*NOT_SOURCE))
    # Counted NOW, from what the wheel is built from: the build writes its own egg-info here.
    data = _data_files(project / "backend" / "src")

    venv = root / "venv"
    made = _run([sys.executable, "-m", "venv", str(venv)], cwd=root)
    assert made.returncode == 0, made.stderr[-2000:]
    python = venv / "bin" / "python"

    # Build with setuptools' own backend (no pip): setuptools comes from the running
    # interpreter's site-packages on PYTHONPATH, read only.
    out = root / "dist"
    out.mkdir()
    build = (
        "import sys; from setuptools import build_meta; print(build_meta.build_wheel(sys.argv[1]))"
    )
    built = _run(
        [str(python), "-c", build, str(out)], cwd=project, pythonpath=":".join(_dependency_dirs())
    )
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]
    (wheel,) = out.glob("miforge-*.whl")

    # Install with the THROWAWAY venv's own pip; nothing else is on its path yet.
    installed_ok = _run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(wheel)], cwd=root
    )
    assert installed_ok.returncode == 0, installed_ok.stdout[-2000:] + installed_ok.stderr[-2000:]
    site = Path(
        _run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            cwd=root,
        ).stdout.strip()
    )
    (site / "_miforge_test_dependencies.pth").write_text("\n".join(_dependency_dirs()) + "\n")
    return {
        "wheel": wheel,
        "python": python,
        "site": site,
        "root": root,
        "data": data,
    }


def test_the_wheel_carries_every_data_file_under_backend_src(installed: dict[str, Any]) -> None:
    data = installed["data"]
    assert "mcp_server/output_schemas.json" in data  # the audit sees the file it exists for
    with zipfile.ZipFile(installed["wheel"]) as wheel:
        names = set(wheel.namelist())
    missing = [name for name in data if name not in names]
    assert missing == [], (
        f"not in the wheel the image installs: {missing}. Add them to "
        "[tool.setuptools.package-data] in pyproject.toml."
    )


def test_the_installed_mcp_server_publishes_the_typed_tools(installed: dict[str, Any]) -> None:
    """From the installed copy, in a process started outside the repository."""
    from mcp_server.server import load_output_schemas

    expected = sorted(load_output_schemas())
    assert len(expected) >= 5, expected
    probe = (
        "import json, mcp_server.server as s; "
        "print(json.dumps({'file': s.__file__, 'path': str(s.OUTPUT_SCHEMAS_PATH), "
        "'typed': sorted(s.load_output_schemas())}))"
    )
    result = _run([str(installed["python"]), "-c", probe], cwd=installed["root"])
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(seen["file"]).is_relative_to(installed["site"]), seen["file"]
    assert not Path(seen["file"]).is_relative_to(REPO), seen["file"]
    assert seen["typed"] == expected
