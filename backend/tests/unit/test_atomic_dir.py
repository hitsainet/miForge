"""``core.atomic_dir`` - the primitive that replaced the torn-weights apparatus.

The order test asserts the **observed syscall order**, not the outcome, because the
outcome is identical under every ordering until the power fails. A test that checked only
"the directory is there and the files are right" passes against a writer that renames
before it syncs, which is the one failure this primitive exists to prevent.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from core import atomic_dir as atomic_dir_module
from core.atomic_dir import LATEST, TMP_PREFIX, atomic_dir, is_temp_dir
from core.errors import Refusal


def test_clean_exit_publishes(tmp_path: Path) -> None:
    dest = tmp_path / "step-000100"
    with atomic_dir(dest) as staging:
        (staging / "model.safetensors").write_bytes(b"weights")
    assert (dest / "model.safetensors").read_bytes() == b"weights"


def test_exception_discards_and_leaves_no_trace(tmp_path: Path) -> None:
    dest = tmp_path / "step-000100"
    with pytest.raises(RuntimeError), atomic_dir(dest) as staging:
        (staging / "model.safetensors").write_bytes(b"partial")
        raise RuntimeError("the step failed after writing one tensor")
    assert not dest.exists()
    assert not (tmp_path / LATEST).exists()
    # And nothing left behind to be discovered or to fill the volume.
    assert list(tmp_path.iterdir()) == []


def test_cancellation_discards_too(tmp_path: Path) -> None:
    """Cooperative cancellation raises a ``BaseException`` subclass (ADR-007).

    If the context manager caught only ``Exception``, a cancelled save would fall through
    to the commit path and publish a partial artifact - the opposite of what cancelling
    means.
    """

    class Cancelled(BaseException):
        pass

    dest = tmp_path / "step-000100"
    with pytest.raises(Cancelled), atomic_dir(dest) as staging:
        (staging / "model.safetensors").write_bytes(b"partial")
        raise Cancelled()
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_syscall_order_is_fsync_files_then_dir_then_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fsync`` every file -> ``fsync`` the temp dir -> ``rename``.

    MUT-1 moves the rename before the fsyncs and MUT-2 drops the directory fsync; both
    must turn this test red.
    """
    calls: list[str] = []
    real_fsync, real_rename = os.fsync, os.rename

    def spy_fsync(fd: int) -> None:
        try:
            kind = "dir" if os.path.isdir(f"/proc/self/fd/{fd}") else "file"
        except OSError:  # pragma: no cover
            kind = "file"
        calls.append(f"fsync:{kind}")
        real_fsync(fd)

    def spy_rename(src: object, dst: object) -> None:
        calls.append("rename")
        real_rename(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "rename", spy_rename)

    dest = tmp_path / "step-000100"
    with atomic_dir(dest, latest_link=False) as staging:
        (staging / "a.bin").write_bytes(b"a")
        (staging / "b.bin").write_bytes(b"b")

    rename_at = calls.index("rename")
    assert calls[:rename_at].count("fsync:file") == 2, f"both files must be fsynced: {calls}"
    assert calls[rename_at - 1] == "fsync:dir", (
        f"the temp DIRECTORY must be fsynced immediately before the rename, else the "
        f"rename itself may not survive a power loss: {calls}"
    )


def test_latest_is_a_relative_symlink(tmp_path: Path) -> None:
    """Relative, so a copied checkpoint tree keeps its meaning (ADR-004).

    An absolute symlink silently resolves to the *source* host's path when that path
    happens to exist on the destination - a wrong answer that looks right.
    """
    dest = tmp_path / "step-000100"
    with atomic_dir(dest) as staging:
        (staging / "model.safetensors").write_bytes(b"w")
    link = tmp_path / LATEST
    assert link.is_symlink()
    assert os.readlink(link) == "step-000100", "must be the bare directory name"
    assert not os.path.isabs(os.readlink(link))


def test_latest_link_false_writes_no_pointer(tmp_path: Path) -> None:
    """The trainer-state layout carries no ``latest`` (R-02.15)."""
    dest = tmp_path / "step-000100"
    with atomic_dir(dest, latest_link=False) as staging:
        (staging / "optimizer.pt").write_bytes(b"o")
    assert not (tmp_path / LATEST).exists()


def test_latest_is_swapped_not_recreated(tmp_path: Path) -> None:
    """A second save repoints ``latest`` without a window where it is absent."""
    for step in ("step-000100", "step-000200"):
        with atomic_dir(tmp_path / step) as staging:
            (staging / "model.safetensors").write_bytes(b"w")
    assert os.readlink(tmp_path / LATEST) == "step-000200"


def test_temp_directory_is_a_dot_prefixed_sibling(tmp_path: Path) -> None:
    """Dot-prefixed **and** a sibling. Both halves are load-bearing.

    A sibling because ``rename`` is atomic only within a filesystem; dot-prefixed because
    a sibling necessarily sits inside the walked tree, and AC8 requires that no partial
    checkpoint be *discoverable* - which atomicity alone does not give.
    """
    dest = tmp_path / "step-000100"
    seen: list[Path] = []
    with atomic_dir(dest) as staging:
        seen.append(staging)
        (staging / "model.safetensors").write_bytes(b"w")
    assert seen[0].parent == dest.parent, "the temp dir must be a SIBLING of the destination"
    assert seen[0].name.startswith(TMP_PREFIX)
    assert is_temp_dir(seen[0].name), "every discovery path excludes it via is_temp_dir"


def test_refuses_cross_filesystem_checkpoint_root(tmp_path: Path) -> None:
    """E-3, and it is only testable because ``tmp_parent`` is a parameter.

    With the temp parent hardcoded to ``dest.parent`` this refusal is dead code: the guard
    compares a path with itself and is always true, so nothing a test controls can reach
    it. ``/dev/shm`` is a different device (tmpfs) from the pytest tmp directory, which is
    what makes a real cross-device case constructible without mounting anything.
    """
    other_device = Path("/dev/shm")
    if not other_device.is_dir() or other_device.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("/dev/shm is unavailable or on the same device; no second filesystem")

    with (
        pytest.raises(Refusal) as caught,
        atomic_dir(tmp_path / "step-000100", tmp_parent=other_device),
    ):
        pass  # pragma: no cover - the refusal happens before the body runs
    assert caught.value.code == "checkpoint_root_not_atomic"
    assert not (tmp_path / "step-000100").exists()
    assert not (tmp_path / LATEST).exists()


def test_default_temp_parent_does_not_refuse(tmp_path: Path) -> None:
    """The negative half of the guard.

    Without this, a mutation that **inverted** the comparison would still pass the
    refusal test above - it would refuse everything, including every real save.
    """
    dest = tmp_path / "step-000100"
    with atomic_dir(dest) as staging:
        (staging / "model.safetensors").write_bytes(b"w")
    assert dest.is_dir()


def test_refuses_to_overwrite_a_published_artifact(tmp_path: Path) -> None:
    """The step is part of the identity, so an overwrite means two parameter sets, one name."""
    dest = tmp_path / "step-000100"
    with atomic_dir(dest) as staging:
        (staging / "model.safetensors").write_bytes(b"first")
    with pytest.raises(Refusal) as caught, atomic_dir(dest):
        pass  # pragma: no cover
    assert caught.value.code == "artifact_already_exists"
    assert (dest / "model.safetensors").read_bytes() == b"first", "the original is untouched"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (f"{TMP_PREFIX}abc123", True),
        (".latest.tmp-abc123", True),
        ("step-000100", False),
        ("latest", False),
        ("manifest.json", False),
    ],
)
def test_is_temp_dir(name: str, expected: bool) -> None:
    assert is_temp_dir(name) is expected


class TestAPublishedArtifactIsReadableByAnotherUid:
    """The modes are set explicitly at the commit, not left to the umask.

    **The hazard was cross-repo and silent.** `tmp.mkdir()` took no mode, so a published
    checkpoint's permissions were `0o777 & ~umask` — fine at the common 022, `0o750` or
    `0o700` under 027 or 077. miForge's backend runs as **uid 10001** and miLLM's as **uid
    1000**, so cross-reading a checkpoint worked *by umask accident*; the failure would have
    appeared as miLLM seeing a checkpoint it cannot open, on a different machine from the one
    that wrote it, long after the save reported success.

    It is a requirement too, not a preference: R-02.21 (every saved checkpoint is a loadable
    model) and R-02.51 (loadable by **any** inference server) are both unmet by an artifact
    the serving uid cannot read, however standard the layout is.
    """

    def test_the_published_directory_is_world_readable(self, tmp_path: Path) -> None:
        dest = tmp_path / "step-000001"
        with atomic_dir(dest) as work:
            (work / "model.safetensors").write_bytes(b"\x00")
        assert stat.S_IMODE(dest.stat().st_mode) == atomic_dir_module.ARTIFACT_DIR_MODE

    def test_the_files_inside_are_world_readable(self, tmp_path: Path) -> None:
        dest = tmp_path / "step-000002"
        with atomic_dir(dest) as work:
            (work / "model.safetensors").write_bytes(b"\x00")
            (work / "nested").mkdir()
            (work / "nested" / "manifest.json").write_text("{}")
        assert stat.S_IMODE((dest / "model.safetensors").stat().st_mode) == (
            atomic_dir_module.ARTIFACT_FILE_MODE
        )
        assert stat.S_IMODE((dest / "nested").stat().st_mode) == (
            atomic_dir_module.ARTIFACT_DIR_MODE
        )
        assert stat.S_IMODE((dest / "nested" / "manifest.json").stat().st_mode) == (
            atomic_dir_module.ARTIFACT_FILE_MODE
        )

    def test_a_HOSTILE_UMASK_does_not_change_the_result(self, tmp_path: Path) -> None:
        """The actual regression. Under 0o077 the old code published `0o700`.

        This is what makes the fix a fix rather than a restatement of the default: with the
        umask that breaks it, the published mode is still correct.
        """
        previous = os.umask(0o077)
        try:
            dest = tmp_path / "step-000003"
            with atomic_dir(dest) as work:
                (work / "model.safetensors").write_bytes(b"\x00")
        finally:
            os.umask(previous)
        mode = stat.S_IMODE(dest.stat().st_mode)
        assert mode == atomic_dir_module.ARTIFACT_DIR_MODE, (
            f"published {oct(mode)} under a 0o077 umask; miLLM's uid 1000 cannot read a "
            f"checkpoint miForge's uid 10001 wrote"
        )
        assert stat.S_IMODE((dest / "model.safetensors").stat().st_mode) == (
            atomic_dir_module.ARTIFACT_FILE_MODE
        )

    def test_nobody_but_the_owner_may_WRITE(self, tmp_path: Path) -> None:
        """Read, never write: another process must load an artifact and must not alter one."""
        dest = tmp_path / "step-000004"
        with atomic_dir(dest) as work:
            (work / "model.safetensors").write_bytes(b"\x00")
        for path in (dest, dest / "model.safetensors"):
            mode = stat.S_IMODE(path.stat().st_mode)
            assert not mode & stat.S_IWGRP, f"{path} is group-writable"
            assert not mode & stat.S_IWOTH, f"{path} is world-writable"

    def test_the_TEMP_directory_is_private_while_it_is_being_written(self, tmp_path: Path) -> None:
        """A half-written checkpoint must not look like a published one to another process."""
        dest = tmp_path / "step-000005"
        seen: list[int] = []
        with atomic_dir(dest) as work:
            seen.append(stat.S_IMODE(work.stat().st_mode))
            (work / "model.safetensors").write_bytes(b"\x00")
        assert seen == [0o700], f"the temp directory was {oct(seen[0])} while incomplete"
