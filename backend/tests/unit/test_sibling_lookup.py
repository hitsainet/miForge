"""The one sibling lookup (feature 015, FR-015.8; FTID §3.6).

Five test modules each found miLLM their own way, and three pre-existing defects lived in the
differences: one honoured only ``MISTUDIO_REQUIRE_CROSS_REPO_CHECKS``, one read ``MILLM_REPO``
before ``MIFORGE_MILLM_ROOT`` (reversed precedence), and ``test_schema_sync.py``'s ``Owned``
branch skipped an absent vendor even when cross-repo checks were required. ``siblings.py`` is
now the only implementation; these tests pin its rules with both folder spellings present at
once and DIFFERENT contents, so a lookup that picks the wrong tree cannot pass by agreement.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.contract import test_schema_sync as sync
from tests.support import siblings

#: The miServer folder (a name, so the inline-lookup guard sees no path join with a literal).
NEW_FOLDER = siblings.MISERVER.folders[0]
ALL_VARIABLES = (
    *siblings.MISERVER.env,
    *siblings.RETIRED_ENV,
    *siblings.MIDATAWORKS.env,
    *siblings.REQUIRE_SWITCHES,
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every case starts with no sibling variable and no require switch set."""
    for name in ALL_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _tree(base: Path, folder: str, marker: str) -> Path:
    root = base / folder
    (root / "docs").mkdir(parents=True)
    (root / "marker.txt").write_text(marker)
    return root


def test_folder_spelling(tmp_path: Path) -> None:
    """The miServer folder is found beside the checkout, also with the old folder beside it."""
    _tree(tmp_path, "miLLM", "miLLM")
    _tree(tmp_path, "miServer", "miServer")
    located = siblings.locate(siblings.MISERVER, base=tmp_path)
    assert located.path == tmp_path / NEW_FOLDER
    assert located.spelling == "../miServer"
    assert (located.path / "marker.txt").read_text() == "miServer"


def test_retired_spellings_find_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P7 (FTID §9.7): the old folder and the old variables are retired. With only they present
    the sibling is ABSENT, reported under its new folder name, so a stale checkout or a
    leftover variable cannot quietly stand in for miServer."""
    _tree(tmp_path, "miLLM", "folder-old")
    for name in siblings.RETIRED_ENV:
        monkeypatch.setenv(name, str(_tree(tmp_path / "env", name, name)))
    located = siblings.locate(siblings.MISERVER, base=tmp_path)
    assert located.path == tmp_path / NEW_FOLDER
    assert not located.path.exists()
    assert located.spelling == "../miServer"
    assert not set(siblings.RETIRED_ENV) & set(located.tried)
    assert "../miLLM" not in located.tried


@pytest.mark.parametrize("winner", range(len(siblings.MISERVER.env)))
def test_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winner: int) -> None:
    """The variables are read in order: the first SET one wins over every later one."""
    _tree(tmp_path, "miServer", "folder-new")
    _tree(tmp_path, "miLLM", "folder-old")
    for index, name in enumerate(siblings.MISERVER.env):
        if index >= winner:
            root = _tree(tmp_path / "env", name, name)
            monkeypatch.setenv(name, str(root))
    located = siblings.locate(siblings.MISERVER, base=tmp_path)
    expected = siblings.MISERVER.env[winner]
    assert located.spelling == expected
    assert (located.path / "marker.txt").read_text() == expected


def test_set_variable_pointing_nowhere_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A set variable is the operator's statement of where the sibling is: a missing path
    there must surface as "absent", never fall through silently to a folder beside us."""
    _tree(tmp_path, "miServer", "beside")
    nowhere = tmp_path / "does-not-exist"
    monkeypatch.setenv("MIFORGE_MISERVER_ROOT", str(nowhere))
    located = siblings.locate(siblings.MISERVER, base=tmp_path)
    assert located.path == nowhere
    assert located.spelling == "MIFORGE_MISERVER_ROOT"


@pytest.mark.parametrize("prefix", ["MIFORGE", "MIWORKBENCH", "MISTUDIO"])
def test_require_switch_spellings(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """Every switch spelling turns an absent sibling into a failure, not a skip."""
    assert siblings.required() is False
    assert _outcome(lambda: siblings.absent("miServer is not checked out"))[0] == "skip"
    monkeypatch.setenv(f"{prefix}_REQUIRE_CROSS_REPO_CHECKS", "1")
    assert siblings.required() is True
    kind, message = _outcome(lambda: siblings.absent("miServer is not checked out"))
    assert kind == "fail", f"{prefix}_REQUIRE_CROSS_REPO_CHECKS=1 must turn absent into a failure"
    assert f"{prefix}_REQUIRE_CROSS_REPO_CHECKS" in message


def test_messages_name_every_spelling_tried(tmp_path: Path) -> None:
    located = siblings.locate(siblings.MISERVER, base=tmp_path)
    assert located.tried == ("MIFORGE_MISERVER_ROOT", "MISERVER_REPO", "../miServer")
    message = siblings.not_checked_out(siblings.MISERVER, located, "the torch pin is unchecked")
    for spelling in located.tried:
        assert spelling in message
    assert "miServer is not checked out" in message


def test_miserver_config_py_reads_only_the_miserver_package(tmp_path: Path) -> None:
    """Since P8 only ``miserver/core/config.py`` is read (015 FTASKS 14.5, DR-20 retired).

    A ``core/config.py`` under any other top-level directory - the shape the pre-P8 package had
    - reads as absent, so a stale checkout skips (or fails under the switch) rather than being
    checked against the wrong file.
    """
    for other in ("server", "legacy", "app"):
        planted = tmp_path / other / "core" / "config.py"
        planted.parent.mkdir(parents=True)
        planted.write_text("OTHER = 1\n")
    assert siblings.miserver_config_py(tmp_path) is None
    new = tmp_path / "miserver" / "core" / "config.py"
    new.parent.mkdir(parents=True)
    new.write_text("NEW = 1\n")
    assert siblings.miserver_config_py(tmp_path) == new
    assert siblings.miserver_config_py(tmp_path / "nothing") is None


def test_miserver_config_py_looks_nowhere_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one path probed is ``miserver/core/config.py``: a reinstated fallback to any other
    package directory (the pre-P8 one included) probes a second path and turns this red."""
    looked: list[Path] = []
    for probe in ("is_file", "exists", "is_dir"):
        real = getattr(Path, probe)

        def spy(self: Path, *args: Any, _real: Callable[..., bool] = real, **kw: Any) -> bool:
            looked.append(self)
            return _real(self, *args, **kw)

        monkeypatch.setattr(Path, probe, spy)
    assert siblings.miserver_config_py(tmp_path) is None
    assert looked == [tmp_path / "miserver" / "core" / "config.py"], looked


def _outcome(call: Callable[[], object]) -> tuple[str, str]:
    """Run ``call`` and name its pytest outcome: ``("skip"|"fail"|"none", message)``.

    Caught explicitly rather than with ``pytest.raises(pytest.fail.Exception)`` alone: a skip
    escaping that block marks THIS test skipped, which exits 0, so a regression back to
    ``pytest.skip`` would read as green (control N52a found exactly that).
    """
    try:
        call()
    except pytest.skip.Exception as exc:
        return "skip", str(exc)
    except pytest.fail.Exception as exc:
        return "fail", str(exc)
    return "none", ""


def _owned_against(root: Path) -> sync.Owned:
    vendor = sync.Vendor(
        sibling=siblings.MISERVER, relative="docs/schemas", obligation="test obligation"
    )
    assert vendor.root() == root, "the test must point the vendor at its tmp root"
    return sync.Owned(vendor, requirement="test requirement")


def test_owned_absent_root_fails_under_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect 2 (FTDD §10.6): an absent vendor ROOT skipped even under the switch."""
    root = tmp_path / "no-such-checkout"
    monkeypatch.setenv("MIFORGE_MISERVER_ROOT", str(root))
    entry = _owned_against(root)
    check = lambda: sync.check_vendors("checkpoint-manifest-v1.json", b"{}", entry)  # noqa: E731
    assert _outcome(check)[0] == "skip"
    monkeypatch.setenv("MIFORGE_REQUIRE_CROSS_REPO_CHECKS", "1")
    kind, message = _outcome(check)
    assert kind == "fail", f"an absent vendor root must FAIL under the switch, got {kind}"
    assert "not checked out" in message


def test_owned_absent_copy_fails_under_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect 2, second half: a checked-out vendor with no COPY skipped even under the switch."""
    root = _tree(tmp_path, "miServer", "present")
    monkeypatch.setenv("MIFORGE_MISERVER_ROOT", str(root))
    entry = _owned_against(root)
    check = lambda: sync.check_vendors("checkpoint-manifest-v1.json", b"{}", entry)  # noqa: E731
    assert _outcome(check)[0] == "skip"
    monkeypatch.setenv("MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS", "1")
    kind, message = _outcome(check)
    assert kind == "fail", f"an absent vendor copy must FAIL under the switch, got {kind}"
    assert "has not vendored" in message
