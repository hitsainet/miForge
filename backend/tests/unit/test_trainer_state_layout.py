"""The trainer-state layout contract itself  (foundation task 3.7).

Three features read this layout - F1 writes it, F5 reads it for a continue, F11 reads it
for recovery - so its shape is a contract and not an implementation detail.

**This file exists because the contract was wrong and nothing noticed.** It rendered a bare
``000500`` while BRD-01 R-01.24 says, in as many words, that *"checkpoint and trainer-state
paths use ``step-`` plus the integer zero-padded to six digits (``step-000500``)"*, and
R-02.15 requires the trainer-state path to use the same form as the weights checkpoint. The
contract's own docstring asserted the opposite - *"``step-<N>`` appears in no contract"* -
which confused R-01.24's removal of the **unpadded** form from *paths* with removal of the
prefix altogether. F1 then built against it correctly, because its brief said to import the
contract rather than re-derive it.

The compounding half is the one worth a test of its own: :func:`latest_complete` filtered
candidates on ``name.isdigit()``, which **no conforming directory would ever match**. It
would have returned ``None`` every time, silently, and a continue would have started fresh
with a complete trainer state sitting beside it - producing exactly the unexplained loss
discontinuity R-02.22 exists to prevent. A wrong path form is visible the first time anyone
looks in the directory; a reader that silently matches nothing is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import Refusal
from schemas.trainer_state import (
    REQUIRED_FILES,
    STEP_DIR_PREFIX,
    TrainerStateDir,
    is_complete,
    latest_complete,
    require_complete,
)


def _write_complete(root: Path, step: int) -> Path:
    directory = TrainerStateDir(root=root, step=step).path
    directory.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (directory / name).write_bytes(b"x")
    return directory


class TestThePathForm:
    """BRD-01 R-01.24, quoted: ``step-`` plus the integer zero-padded to six digits."""

    def test_the_prefix_is_present(self) -> None:
        assert TrainerStateDir(Path("/s/run-7"), 500).path.name == "step-000500"

    def test_six_digit_padding_so_directories_sort_lexicographically(self) -> None:
        names = [TrainerStateDir(Path("/s"), n).path.name for n in (2, 10, 100, 1000)]
        assert names == sorted(names), (
            "the padding exists so a plain directory listing is in step order; without it "
            "'step-2' sorts after 'step-1000'"
        )

    def test_padding_is_a_minimum_not_a_truncation(self) -> None:
        """A run past 999,999 steps must not be silently mangled into six digits."""
        assert TrainerStateDir(Path("/s"), 1_234_567).path.name == "step-1234567"

    def test_the_prefix_is_the_one_in_the_contract(self) -> None:
        assert STEP_DIR_PREFIX == "step-"


class TestCompleteness:
    """A reader must never trust that the writer was atomic.

    The writer may have been killed before its own cleanup ran, so completeness is decided
    by what is on disk - and the manifest is written last, which is what makes the check
    meaningful rather than a formality.
    """

    def test_all_required_files_present_is_complete(self, tmp_path: Path) -> None:
        assert is_complete(_write_complete(tmp_path, 100))

    @pytest.mark.parametrize("missing", REQUIRED_FILES)
    def test_any_missing_file_is_incomplete(self, tmp_path: Path, missing: str) -> None:
        directory = _write_complete(tmp_path, 100)
        (directory / missing).unlink()
        assert not is_complete(directory), f"a state missing {missing} read as complete"

    def test_require_complete_refuses_by_name_and_names_what_is_missing(
        self, tmp_path: Path
    ) -> None:
        directory = _write_complete(tmp_path, 100)
        (directory / "rng.pt").unlink()
        with pytest.raises(Refusal) as caught:
            require_complete(directory)
        assert caught.value.code == "trainer_state_incomplete"
        assert "rng.pt" in caught.value.details["missing"], (
            "the refusal must name the missing file: 'incomplete' alone sends the reader "
            "looking at the whole directory"
        )

    def test_an_absent_directory_refuses_rather_than_starting_fresh(self, tmp_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            require_complete(tmp_path / "step-000100")
        assert caught.value.code == "trainer_state_absent"


class TestLatestComplete:
    """The reader whose filter matched nothing. This is the defect's expensive half."""

    def test_it_finds_a_conforming_directory_at_all(self, tmp_path: Path) -> None:
        """The regression, stated at its bluntest.

        With the old ``name.isdigit()`` filter this returned ``None`` against a perfectly
        good state, and a continue would have started fresh - silently.
        """
        _write_complete(tmp_path, 500)
        found = latest_complete(tmp_path)
        assert found is not None, (
            "latest_complete found nothing beside a complete trainer state. That is the "
            "silent failure: a continue starts fresh and the loss discontinuity looks like "
            "a real regression."
        )
        assert found.step == 500

    def test_it_returns_the_highest_step(self, tmp_path: Path) -> None:
        for step in (100, 900, 200):
            _write_complete(tmp_path, step)
        found = latest_complete(tmp_path)
        assert found is not None and found.step == 900

    def test_it_skips_an_incomplete_newer_state_rather_than_stopping_at_it(
        self, tmp_path: Path
    ) -> None:
        """A crash leaves the NEWEST directory torn; the useful answer is the newest usable one."""
        _write_complete(tmp_path, 100)
        torn = _write_complete(tmp_path, 200)
        (torn / "trainer_state.json").unlink()
        found = latest_complete(tmp_path)
        assert found is not None and found.step == 100

    def test_it_ignores_an_in_flight_write(self, tmp_path: Path) -> None:
        """A save happening right now must never be mistaken for a candidate."""
        _write_complete(tmp_path, 100)
        in_flight = tmp_path / ".tmp-save-deadbeef"
        in_flight.mkdir()
        for name in REQUIRED_FILES:
            (in_flight / name).write_bytes(b"x")
        found = latest_complete(tmp_path)
        assert found is not None and found.step == 100

    def test_it_ignores_a_directory_that_is_not_a_step(self, tmp_path: Path) -> None:
        _write_complete(tmp_path, 100)
        (tmp_path / "latest").mkdir()
        (tmp_path / "notes").mkdir()
        found = latest_complete(tmp_path)
        assert found is not None and found.step == 100

    def test_an_empty_root_is_None_not_an_error(self, tmp_path: Path) -> None:
        assert latest_complete(tmp_path) is None

    def test_a_missing_root_is_None_not_an_error(self, tmp_path: Path) -> None:
        assert latest_complete(tmp_path / "never-created") is None


def test_an_unknown_file_in_the_layout_is_refused(tmp_path: Path) -> None:
    """A fourth file only one reader knows about is how three readers get three layouts."""
    with pytest.raises(Refusal) as caught:
        TrainerStateDir(tmp_path, 100).file("something_new.pt")
    assert caught.value.code == "trainer_state_file_unknown"


def test_there_is_no_latest_pointer_in_this_layout() -> None:
    """Checkpoints have one; trainer state does not (R-02.15, and F1's layout).

    Asserted because the shared atomicity primitive defaults ``latest_link=True``, so a
    "behaviour-preserving" refactor that left the default on would add a symlink F1 never
    wrote - and nothing else would complain.
    """
    assert "latest" not in REQUIRED_FILES
