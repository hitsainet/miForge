"""Byte-identity of every vendored contract in ``docs/schemas/``  (foundation task 1.4).

ADR-014. This closes three obligations that were owed for weeks and could not be met
because miForge had no test workflow: ``checkpoint-manifest-v1.json`` (which had no sync
test in *either* repo), ``midataworks-dataset-version-v1.json`` (R-02.37) and
``adapter-manifest-v1.json`` (R-02.44 / BRD-01 R-01.28).

**Why a test and not a recorded md5.** AGENT.md carried a recorded hash for the checkpoint
manifest and it went stale three times in one hour, once recording a value that matched no
file in either repo. A reader checking a correct pair of files against a wrong recorded
value concludes the contract has been tampered with, and "fixes" the file that was right.
The hash is a courtesy to a human reader; *this* is the authority.

**Scanning plus an explicit map, not one or the other.** The parametrisation is driven by
scanning the directory, so adding a schema adds a case without editing this file
(task 3.6). But every scanned file must also appear in ``OWNERSHIP`` or the test FAILS -
refuse rather than guess (ADR-018). A new shared contract that nobody declared an owner
for is exactly the drift this file exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support import siblings

#: Repository root, found by walking up from this file rather than from the CWD, so the
#: suite behaves the same under `pytest`, `pytest backend/tests` and an IDE runner.
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"


class Vendor:
    """Where a sibling repository's copy of a shared contract lives.

    The checkout is found by ``tests/support/siblings.py``, the one lookup (FR-015.8): its
    variables first, then the folder beside this checkout, new name before old (miLLM is
    becoming miServer). A sync test that hard-codes ``/home/<someone>/app/...`` passes only
    on one workstation; the lookup follows the checkout wherever it moves.
    """

    def __init__(self, sibling: siblings.Sibling, relative: str, obligation: str):
        self.sibling = sibling
        self.repo = sibling.name
        self.relative = relative
        self.obligation = obligation

    def located(self) -> siblings.Located:
        return siblings.locate(self.sibling)

    def root(self) -> Path:
        return self.located().path

    def overrides(self) -> str:
        """Every spelling tried, for a skip or failure message."""
        return ", ".join(self.located().tried)

    def copy_path(self, filename: str) -> Path:
        return self.root() / self.relative / filename


MILLM = Vendor(
    sibling=siblings.MISERVER,
    relative="docs/schemas",
    obligation="BRD-01 R-01.12 / R-01.28",
)
MIDATAWORKS = Vendor(
    sibling=siblings.MIDATAWORKS,
    # R-03.47 makes the docs/schemas copy normative; the backend/src/schemas/data copy in
    # that repo is a packaged duplicate and must NOT be used as the reference.
    relative="docs/schemas",
    obligation="BRD-02 R-02.37 / BRD-03 R-03.47",
)


class Owned:
    """miForge owns this contract and is normative for it; listed vendors copy it."""

    def __init__(self, *vendors: Vendor, requirement: str):
        self.vendors = vendors
        self.requirement = requirement
        self.normative_here = True


class Vendored:
    """A sibling owns this contract; miForge holds a byte-identical copy."""

    def __init__(self, source: Vendor, requirement: str):
        self.source = source
        self.requirement = requirement
        self.normative_here = False


class Internal:
    """miForge owns it and **no sibling vendors it**, so there is no copy to compare.

    A distinct class rather than ``Owned()`` with an empty vendor list, because an empty
    list would make the byte-identity assertion pass *vacuously* and the suite would
    report a check it did not perform. Named instead, so a reader can see that nothing is
    compared and why.

    These still **publish to the public mirror** - ``docs/schemas/`` is the one ``docs/``
    path the sync keeps - which is intended: they are the contracts a future consumer
    would need. If one of them ever is vendored, move it to :class:`Owned` and the
    byte-identity assertion starts applying.
    """

    def __init__(self, requirement: str, note: str = ""):
        self.requirement = requirement
        self.note = note
        self.normative_here = True


#: The ownership table. ``docs/schemas/README.md`` (task 9.3) states the same thing in
#: prose for a human; this is the machine-checked copy.
OWNERSHIP: dict[str, Owned | Vendored | Internal] = {
    "checkpoint-manifest-v1.json": Owned(MILLM, requirement="R-02.20 / BRD-01 R-01.12"),
    "adapter-manifest-v1.json": Owned(MILLM, requirement="R-02.44 / BRD-01 R-01.28"),
    "midataworks-dataset-version-v1.json": Vendored(MIDATAWORKS, requirement="R-02.37"),
    # GENERATED in miLLM from its registered /api/weights/* routes (suite decision D39
    # follow-up, 2026-10-17). F11's fake weights surface is validated against this copy by
    # `test_fake_weights_surface_conforms.py`, so the fake cannot drift from the real surface
    # again without a red - which it did on four counts while every F11 test stayed green.
    "millm-weights-api-v1.json": Vendored(MILLM, requirement="BRD-01 R-01.1 / R-01.20 (D39)"),
    # GENERATED in miLLM from its registered POST /v1/completions route and error table
    # (FR-25.15, suite D61, miLLM main ce8046b). F11 14.3b's float32 verification depends on
    # its `scoring_precision` request field, echo and 409 refusal; the fake serving side in
    # `test_hotupdate_verify_float32.py` is validated against this copy.
    "millm-scoring-api-v1.json": Vendored(MILLM, requirement="miLLM FR-25.15 / F11 14.3b (D61)"),
    # The four contract-gate schemas (foundation task 3.0). Internal today: nothing
    # outside miForge consumes them.
    "batch-record-v1.json": Internal(
        requirement="R-02.8",
        note="Written by F2, read by F7 and carried as the run log's 'batch' block.",
    ),
    "step-record-v1.json": Internal(
        requirement="R-02.14",
        note=(
            "The run log, five discriminated blocks. Its 'step' block's commit_duration_ms "
            "is an input to miLLM Feature 31's acceptance gate, so if that measurement is "
            "ever exchanged rather than reported, this becomes a vendored contract."
        ),
    ),
    "selector-config-v1.json": Internal(
        requirement="R-02.9",
        note=(
            "The configuration BODY is authored in miDataworks and cited by hash; this "
            "schema describes the citation and the five types, not their contents."
        ),
    ),
    "pack-definition-v1.json": Internal(
        requirement="R-02.38",
        note=(
            "'pack.json' on disk. miForge-internal for now; if packs ever become portable "
            "it needs the vendoring discipline and moves to Owned."
        ),
    ),
}


def discovered_schemas() -> list[str]:
    """Every JSON contract in ``docs/schemas/``, sorted for a stable test order."""
    if not SCHEMA_DIR.is_dir():  # pragma: no cover - the directory is tracked
        return []
    return sorted(p.name for p in SCHEMA_DIR.glob("*.json"))


def test_schema_directory_is_not_empty() -> None:
    """A parametrised suite over an empty list is green and proves nothing.

    This project shipped a review harness that was green against a tree containing zero
    documents. The same failure here would make every assertion below vacuous.
    """
    assert discovered_schemas(), f"no *.json contracts found under {SCHEMA_DIR}"


@pytest.mark.parametrize("filename", discovered_schemas())
def test_every_schema_has_a_declared_owner(filename: str) -> None:
    """A shared contract with no declared owner is refused, not guessed (ADR-018).

    Task 3.6 says to add each new schema to this test. Driving the parametrisation from
    the directory means the case appears automatically; this assertion means it appears
    as a *failure* until somebody says who owns the file.
    """
    assert filename in OWNERSHIP, (
        f"{filename} is in docs/schemas/ with no entry in OWNERSHIP. Add one naming the "
        f"normative repo and the requirement, and record it in docs/schemas/README.md."
    )


@pytest.mark.parametrize("filename", discovered_schemas())
def test_schema_is_byte_identical_in_every_repo(filename: str) -> None:
    """The vendored copies agree byte for byte, or the test says exactly where they differ.

    Byte-identity rather than semantic equivalence is deliberate: two JSON documents that
    parse to the same object can still disagree on key order and formatting, and the
    suite's whole purpose is to make a drifted copy impossible to miss in review.
    """
    ours = SCHEMA_DIR / filename
    assert ours.is_file(), f"{ours} is missing"
    our_bytes = ours.read_bytes()

    entry = OWNERSHIP.get(filename)
    if entry is None:
        pytest.fail(f"{filename} has no OWNERSHIP entry; see the companion test")

    if isinstance(entry, Internal):
        pytest.skip(
            f"{filename} is miForge-internal ({entry.requirement}); no sibling vendors it, "
            f"so there is no copy to compare. {entry.note}"
        )

    if isinstance(entry, Vendored):
        source = entry.source.copy_path(filename)
        if not entry.source.root().is_dir():
            siblings.absent(
                f"{entry.source.repo} is not checked out at {entry.source.root()} "
                f"(tried {entry.source.overrides()}); "
                f"cannot verify {filename} ({entry.requirement})"
            )
        assert source.is_file(), (
            f"{entry.source.repo} owns {filename} ({entry.requirement}) but its normative "
            f"copy is missing at {source}"
        )
        assert our_bytes == source.read_bytes(), (
            f"{filename} differs from {entry.source.repo}'s normative copy at {source}. "
            f"{entry.source.repo} owns this contract: re-vendor from there, do not edit "
            f"the copy here."
        )
        return

    check_vendors(filename, our_bytes, entry)


def check_vendors(filename: str, our_bytes: bytes, entry: Owned) -> None:
    """miForge owns ``filename``: every vendor's copy must match OUR bytes.

    An absent vendor checkout, or a vendor that has not copied the file yet, skips loudly -
    and FAILS under a require switch (``siblings.absent``). Before feature 015 both cases
    called ``pytest.skip`` directly, so a required cross-repo run reported success while
    comparing nothing (FTDD §10.6, defect 2).
    """
    for vendor in entry.vendors:
        their = vendor.copy_path(filename)
        if not vendor.root().is_dir():
            siblings.absent(
                f"{vendor.repo} is not checked out at {vendor.root()} "
                f"(tried {vendor.overrides()}); "
                f"cannot verify that it vendors {filename} ({entry.requirement})"
            )
        if not their.is_file():
            # A missing vendor copy is the OTHER repo's unmet obligation, not a defect
            # here, so it does not fail miForge's ordinary suite - but it is reported by
            # name, and a required cross-repo run fails on it.
            siblings.absent(
                f"{vendor.repo} has not vendored {filename} yet ({vendor.obligation}). "
                f"miForge is normative for it; nothing is owed on this side."
            )
        assert our_bytes == their.read_bytes(), (
            f"{filename} differs between miForge (normative, {entry.requirement}) and "
            f"{vendor.repo}'s vendored copy at {their}. Change it HERE, then copy it "
            f"there; the figure in AGENT.md is informational and this test is the authority."
        )
