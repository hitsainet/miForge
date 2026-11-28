"""The predicate every mirror-only skip rests on (mirror CI, 2026-11-08).

The public mirror's Backend Tests went red after the 015 P0 push: tests that need history or a
stripped file ran on the one-commit snapshot. They now skip there - and only there. These tests
hold the predicate to real commits of every shape, tie its marker to the sync workflow that
writes it, and pin that THIS checkout is recognised correctly, so no mirror-only skip can fire
on the private CI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from tests.support import legacy_identifiers as li
from tests.support import mirror_snapshot as ms

SYNC = li.REPO_ROOT / ".github" / "workflows" / "sync-to-clean.yml"


@pytest.mark.parametrize("shape", sorted(ms.EXPECTED))
def test_only_the_synced_orphan_is_the_mirror(tmp_path: Path, shape: str) -> None:
    tree = ms.make_repo(tmp_path / "t", shape)
    assert ms.mirror_snapshot(tree) is ms.EXPECTED[shape], shape


def test_the_sync_writes_the_marker_the_predicate_reads() -> None:
    """If the sync stopped writing either trailer the mirror would stop being recognised and its
    runs would FAIL loudly - never the reverse."""
    steps = yaml.safe_load(SYNC.read_text())["jobs"]["sync"]["steps"]
    build = next(s["run"] for s in steps if s.get("name") == "Build the orphan snapshot")
    assert "git checkout --orphan clean" in build
    assert '--trailer "Source-Commit: ${SOURCE_SHA}"' in build
    assert '--trailer "Private-CI: passed"' in build
    assert ms.SYNC_TRAILERS[3] == "Private-CI: passed"


def test_this_checkout_is_the_mirror_only_without_the_full_export() -> None:
    """Live: the tree under test is the mirror snapshot exactly when the private-only full
    export is absent. On the private CI the predicate is therefore False, so nothing may skip
    there as "the mirror"; on the mirror it proves the snapshot was recognised."""
    full = (li.REPO_ROOT / li.FULL).is_file()
    assert ms.mirror_snapshot() is (not full)


@pytest.mark.parametrize("shape", ["none", "private", "shallow-private", "orphan-plain", "mirror"])
def test_the_public_fallback_needs_the_mirror_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """This repository's own tree reads the public export in place of the full one only as the
    mirror snapshot; anywhere else the missing full export fails, so no withheld-record skip or
    ``variant == "public"`` branch can fire on the private CI."""
    tree = ms.make_repo(tmp_path / "t", shape)
    (tree / "docs").mkdir()
    shutil.copy2(li.REPO_ROOT / li.PUBLIC, tree / li.PUBLIC)
    monkeypatch.setattr(li, "REPO_ROOT", tree)
    monkeypatch.delenv(li.OVERRIDE_ENV, raising=False)
    if ms.EXPECTED[shape]:
        assert li.load(tree).variant == "public"
    else:
        with pytest.raises(li.ExportError, match="not the mirror snapshot"):
            li.load(tree)


@pytest.mark.parametrize("shape", ["private", "shallow-private", "orphan-plain", "mirror"])
@pytest.mark.parametrize(
    ("module", "test", "missing"),
    [
        ("tests.unit.test_origin_headers", "test_every_header_has_an_index_row", "docs/REUSE.md"),
        (
            "tests.contract.test_legacy_identifiers",
            "test_public_equals_full_minus_withheld",
            "docs/legacy-identifiers.md",
        ),
    ],
)
def test_a_stripped_file_skips_only_on_the_mirror_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    module: str,
    test: str,
    missing: str,
) -> None:
    """The real tests whose file the sync strips: on a tree without it they skip as the mirror
    snapshot and FAIL as anything else (both used to skip on any tree missing the file)."""
    import importlib

    mod = importlib.import_module(module)
    tree = ms.make_repo(tmp_path / "t", shape)
    assert not (tree / missing).exists()
    monkeypatch.setattr(mod, "REPO_ROOT", tree)
    try:
        getattr(mod, test)()
    except pytest.skip.Exception as skip:
        assert shape == "mirror", f"{test} skipped on a {shape!r} tree: {skip}"
        assert "MIRROR" in str(skip)
    except AssertionError as failure:
        assert shape != "mirror", f"{test} failed on the mirror snapshot: {failure}"
        assert "not the mirror snapshot" in str(failure)
    else:
        pytest.fail(f"{test} passed on a tree without {missing}")
