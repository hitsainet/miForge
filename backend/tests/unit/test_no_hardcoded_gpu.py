"""No module may name a GPU by index or by bus address  (task 1.6).

ADR-008: **pin by UUID, resolved at submit from live ``nvidia-smi -L``, never index and
never bus address.** Both halves of that rule were written after an incident:

* **Indices inverted.** The enumeration order of two cards changed between boots, so
  ``cuda:0`` silently became the other card. Nothing failed; the work just ran on the
  wrong device, and a benchmark taken against it was wrong by the ratio of the two cards.
* **A copied bus address unbound the root NVMe.** A PCI address was copied out of a
  hardware table into a different context and applied to the wrong device class. This is
  why miStudio's hardware table is **cited, never copied**, and why a bus-address literal
  is banned outright here rather than merely discouraged.

Modelled on miStudio's ``test_no_hard_coded_gpu.py`` (see ``docs/REUSE.md``). A ratchet:
the allowlist starts empty and only ever shrinks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "backend" / "src"

#: Empty, and it is a *ratchet*: entries may be removed, never added. A hard-coded device
#: is always replaceable by a resolved UUID, so there is no case to grandfather.
ALLOWLIST: frozenset[str] = frozenset()

# A device string carrying an ordinal: "cuda:0", "cuda:1", 'cuda:12'.
RE_CUDA_INDEX = re.compile(r"""['"]cuda:\s*\d+['"]""")

# A PCI bus address, in both the short and the domain-qualified spellings that
# `nvidia-smi` and `lspci` print: 01:00.0 and 00000000:01:00.0.
#
# The fields are VALIDATED, not merely shaped, because `BB:DD.F` is ambiguous with a
# time: "12:30.5" matches the shape exactly. PCI device numbers are five bits (0x00-0x1f)
# and function numbers three (0-7), so a real address is distinguishable from a clock
# reading - 0x30 is not a device number. Caught by this file's own negative case before
# the suite was ever run, which is the argument for having negative cases.
RE_BUS_CANDIDATE = re.compile(
    r"""['"](?:(?P<domain>[0-9A-Fa-f]{4,8}):)?"""
    r"""(?P<bus>[0-9A-Fa-f]{2}):(?P<device>[0-9A-Fa-f]{2})\.(?P<function>\d)['"]"""
)


def _is_pci_address(match: re.Match[str]) -> bool:
    """True when the captured fields are a *valid* PCI address, not just address-shaped."""
    device = int(match.group("device"), 16)
    function = int(match.group("function"))
    # A domain-qualified literal is unambiguous; a bare BB:DD.F is only an address if the
    # device and function are in range.
    return not (device > 0x1F or function > 7)


# CUDA_VISIBLE_DEVICES assigned a literal ordinal list rather than a resolved value.
# `CUDA_VISIBLE_DEVICES = resolved_uuid` is correct and is not matched.
# The `['"]?\s*\]?` between the name and the operator covers all three ways this is
# written: `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`, `CUDA_VISIBLE_DEVICES = "0"` and
# `{"CUDA_VISIBLE_DEVICES": "0"}`. The first form was MISSED by the first draft and the
# self-test below is what found it - the subscript closing bracket sits between the name
# and the `=`, so a pattern expecting them adjacent sees nothing.
RE_VISIBLE_LITERAL = re.compile(
    r"""CUDA_VISIBLE_DEVICES['"]?\s*\]?\s*[:=]\s*['"]\s*\d+(\s*,\s*\d+)*\s*['"]"""
)

# `.cuda(0)` / `.cuda(device=1)` - the ordinal form of `.to(device)`.
RE_DOT_CUDA_ORDINAL = re.compile(r"\.cuda\(\s*(?:device\s*=\s*)?\d+\s*\)")

# `torch.cuda.set_device(0)` - sets the process-wide default by ordinal.
RE_SET_DEVICE_ORDINAL = re.compile(r"set_device\(\s*\d+\s*\)")

BUS_ADDRESS_WHY = (
    "a PCI bus address literal - a copied bus address once unbound the root NVMe; "
    "cite miStudio's hardware table, never copy an address out of it"
)

CHECKS: tuple[tuple[re.Pattern[str], str], ...] = (
    (RE_CUDA_INDEX, "a 'cuda:N' device string - resolve a UUID instead (ADR-008)"),
    (
        RE_VISIBLE_LITERAL,
        "CUDA_VISIBLE_DEVICES set to a literal ordinal - set it from a resolved UUID",
    ),
    (RE_DOT_CUDA_ORDINAL, "`.cuda(N)` with an ordinal - use the resolved device"),
    (RE_SET_DEVICE_ORDINAL, "`set_device(N)` with an ordinal - use the resolved device"),
)


def find_hardcoded_gpu(source: str) -> list[tuple[int, str, str]]:
    """Return ``(lineno, matched_text, why)`` for every hard-coded device reference.

    Line-based rather than AST-based on purpose: these appear in f-strings, in comments
    that get copied into code, in YAML embedded in docstrings and in subprocess argument
    lists, none of which an AST import walk reaches.
    """
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        # A line may opt out ONLY by explaining itself; the marker names the rule so the
        # exemption is greppable and reviewable. It is not an allowlist of files.
        if "noqa: gpu-literal" in line:
            continue
        for pattern, why in CHECKS:
            match = pattern.search(line)
            if match:
                hits.append((lineno, match.group(0), why))
        for candidate in RE_BUS_CANDIDATE.finditer(line):
            if _is_pci_address(candidate):
                hits.append((lineno, candidate.group(0), BUS_ADDRESS_WHY))
    return hits


def python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py"))


def test_source_tree_is_not_empty() -> None:
    """A ratchet that scans nothing is green."""
    assert python_files(), f"no Python files under {SRC}; the ratchet would be vacuous"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_module_names_no_gpu_by_index_or_address(path: Path) -> None:
    relative = str(path.relative_to(SRC))
    if relative in ALLOWLIST:  # pragma: no cover - empty by design
        pytest.fail(f"{relative} is allowlisted; this is a ratchet, remove the entry")
    hits = find_hardcoded_gpu(path.read_text(encoding="utf-8"))
    assert not hits, "\n".join(
        f"{relative}:{line} contains {text!r}: {why}" for line, text, why in hits
    )


# --------------------------------------------------------------------------------------
# Self-test: the ratchet must be able to see each form. Same reasoning as the
# sibling-import guard - an absence test blind to what it looks for is always green.
# --------------------------------------------------------------------------------------

POSITIVE_CASES: list[tuple[str, str]] = [
    ("cuda index double quoted", 'DEVICE = "cuda:0"\n'),
    ("cuda index single quoted", "DEVICE = 'cuda:1'\n"),
    ("cuda index in a call", 'model.to("cuda:0")\n'),
    ("bus address short", 'GPU = "01:00.0"\n'),
    ("bus address domain qualified", 'GPU = "00000000:01:00.0"\n'),
    ("bus address at the field limits", 'GPU = "ff:1f.7"\n'),  # device 0x1f, function 7
    ("visible devices literal", 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'),
    ("visible devices literal list", 'env = {"CUDA_VISIBLE_DEVICES": "0,1"}\n'),
    ("dot cuda ordinal", "tensor = tensor.cuda(0)\n"),
    ("dot cuda keyword ordinal", "tensor = tensor.cuda(device=1)\n"),
    ("set_device ordinal", "torch.cuda.set_device(0)\n"),
]


@pytest.mark.parametrize("label,source", POSITIVE_CASES, ids=[c[0] for c in POSITIVE_CASES])
def test_ratchet_sees_every_form(label: str, source: str) -> None:
    assert find_hardcoded_gpu(source), (
        f"the ratchet did NOT see the {label!r} form:\n{source}\n"
        f"Each form it misses is a way to pin the wrong card with a green suite."
    )


NEGATIVE_CASES: list[tuple[str, str]] = [
    ("bare cuda", 'DEVICE = "cuda"\n'),
    ("resolved index interpolated", 'DEVICE = f"cuda:{resolved_index}"\n'),
    ("uuid string", 'GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"\n'),
    ("visible from variable", 'os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid\n'),
    ("cpu device", 'DEVICE = "cpu"\n'),
    ("to with a variable", "model.to(device)\n"),
    ("device order", 'os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"\n'),
    ("a version number", 'TORCH = "2.4.0"\n'),
    # Address-SHAPED but not a valid address: 0x30 is not a 5-bit device number. This is
    # the case that found the false positive in the first draft of the pattern.
    ("a time", 'STAMP = "12:30.5"\n'),
    ("a duration", 'ELAPSED = "01:45.9"\n'),  # function 9 > 7
]


@pytest.mark.parametrize("label,source", NEGATIVE_CASES, ids=[c[0] for c in NEGATIVE_CASES])
def test_ratchet_does_not_cry_wolf(label: str, source: str) -> None:
    """``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is *required* by foundation task 8.7, and a
    version string is not a bus address. A ratchet with false positives gets disabled."""
    assert not find_hardcoded_gpu(source), f"false positive on the {label!r} case: {source!r}"
