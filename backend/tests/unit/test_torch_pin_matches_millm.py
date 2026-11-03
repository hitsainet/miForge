"""miForge's image pins the torch build miLLM serves, so CUDA IPC can attach at all.

``check_build_identity`` refuses an attach when torch or CUDA differ between the trainer
and the serving process (a CUDA IPC handle is a contract between matching builds). Both
images used to float on PyPI (``torch>=…,<3``) and matched only by build date.

**miLLM's authority, in order:** an exact ``torch==`` pin in its ``pyproject.toml`` or
``Dockerfile`` if it ever has one; otherwise the build its hardware acceptance MEASURED in
the serving pod (``0xcc/reviews/attachable_weights_hardware_2026-10-16.md`` §1). The
second is weaker - miLLM is unpinned and a rebuild can move - and the docstring of the
skip/assert says so rather than hiding it.

Cross-repo convention: the sibling is found beside this checkout or at
``MIFORGE_MILLM_ROOT``; it skips LOUDLY when absent (the public mirror), and fails under
``MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
HARDWARE_RECORD = Path("0xcc/reviews/attachable_weights_hardware_2026-10-16.md")


def _millm_root() -> Path:
    override = os.environ.get("MIFORGE_MILLM_ROOT")
    return Path(override).expanduser() if override else REPO_ROOT.parent / "miLLM"


def _absent(reason: str) -> None:
    if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def miforge_pin() -> str:
    match = re.search(r"^ARG TORCH_VERSION=(\S+)$", DOCKERFILE.read_text("utf-8"), re.M)
    assert match, "backend/Dockerfile carries no `ARG TORCH_VERSION=` pin"
    return match.group(1)


def millm_served(root: Path) -> tuple[str, str]:
    """``(version, source)``: miLLM's exact pin if it has one, else its measured build."""
    for name in ("pyproject.toml", "Dockerfile"):
        path = root / name
        if path.is_file():
            pinned = re.search(r"""torch==([0-9][^"'\s,;]*)""", path.read_text("utf-8"))
            if pinned:
                return pinned.group(1), f"miLLM {name} pin"
    record = root / HARDWARE_RECORD
    if not record.is_file():
        _absent(f"miLLM has no torch pin and no hardware record at {record}")
    measured = re.search(
        r"\|\s*torch / CUDA / cuDNN\s*\|\s*([0-9][^\s/|]*)", record.read_text("utf-8")
    )
    assert measured, f"{record} no longer states the measured torch build"
    return measured.group(1), f"miLLM measured build ({HARDWARE_RECORD})"


def _base(version: str) -> str:
    return version.split("+", 1)[0]


class TestTheTorchPin:
    def test_the_image_pins_an_exact_torch(self) -> None:
        pin = miforge_pin()
        assert re.fullmatch(r"\d+\.\d+\.\d+", pin), pin
        text = DOCKERFILE.read_text("utf-8")
        assert '"torch==${TORCH_VERSION}"' in text, "the ARG is declared but not installed"

    def test_it_equals_the_build_millm_serves(self) -> None:
        root = _millm_root()
        if not root.is_dir():
            _absent(f"miLLM is not checked out at {root}; the pin cannot be compared")
        theirs, source = millm_served(root)
        assert _base(theirs) == miforge_pin(), (
            f"miForge's image pins torch {miforge_pin()} but {source} is {theirs}. CUDA IPC "
            f"attach will refuse incompatible_torch_build: move backend/Dockerfile's "
            f"TORCH_VERSION (and coordinate miLLM's) so both images serve one build."
        )

    def test_the_parser_reads_a_pin_before_a_record(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('deps = ["torch==2.99.0"]\n')
        assert millm_served(tmp_path) == ("2.99.0", "miLLM pyproject.toml pin")

    def test_the_parser_reads_the_measured_record(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('deps = ["torch>=2.10.0,<3"]\n')
        record = tmp_path / HARDWARE_RECORD
        record.parent.mkdir(parents=True)
        record.write_text("| torch / CUDA / cuDNN | 2.14.1+cu130 / 13.0 / 92400 |\n")
        assert millm_served(tmp_path)[0] == "2.14.1+cu130"
