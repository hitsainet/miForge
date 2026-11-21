"""Path containment for caller-supplied run identifiers (CodeQL py/path-injection, alerts 5-12).

A ``run_id`` comes from URL paths and submitted run definitions and is joined onto a root
that is then read, appended to or created. ``core.paths`` is the one check; these tests pin
the check **and its wiring** at the two path builders every flagged site goes through
(``run_log.run_log_path`` and ``checkpoints.paths.run_dir``) and through the mounted API.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.errors import Refusal
from core.paths import contained_child, single_segment
from services.checkpoints import paths as checkpoint_paths
from training import run_log

TRAVERSALS = ["../escape", "..", ".", "a/b", "/etc", "a\x00b"]


@pytest.fixture
def escape(tmp_path: Path) -> tuple[Path, str]:
    """A root holding one LEGAL single-segment name that is a symlink pointing outside it."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    return root, "linked"


class TestTheHelper:
    @pytest.mark.parametrize("name", TRAVERSALS)
    def test_a_name_that_is_not_one_segment_is_refused(self, name: str) -> None:
        with pytest.raises(Refusal) as caught:
            single_segment(name, code="x_invalid", what="a thing")
        assert caught.value.code == "x_invalid"

    def test_a_plain_name_passes_and_resolves_under_the_root(self, tmp_path: Path) -> None:
        assert single_segment("run-1", code="x", what="t") == "run-1"
        assert contained_child(tmp_path, "run-1", code="x", what="t") == (
            tmp_path.resolve() / "run-1"
        )

    def test_a_symlink_escape_is_refused(self, escape: tuple[Path, str]) -> None:
        root, name = escape
        with pytest.raises(Refusal) as caught:
            contained_child(root, name, code="x_invalid", what="a thing")
        assert caught.value.code == "x_invalid"
        assert "resolves outside" in str(caught.value)

    def test_a_link_to_a_sibling_sharing_the_roots_prefix_is_refused(self, tmp_path: Path) -> None:
        """``/x/root-evil`` starts with ``/x/root`` as a string; the check is on a path
        boundary, so a bare ``startswith(root)`` would let this through."""
        root = tmp_path / "root"
        sibling = tmp_path / "root-evil"
        root.mkdir()
        sibling.mkdir()
        (root / "link").symlink_to(sibling, target_is_directory=True)
        with pytest.raises(Refusal):
            contained_child(root, "link", code="x_invalid", what="a thing")

    def test_a_link_to_the_root_itself_is_not_a_child(self, tmp_path: Path) -> None:
        (tmp_path / "self").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(Refusal):
            contained_child(tmp_path, "self", code="x_invalid", what="a thing")

    def test_a_symlink_that_stays_inside_is_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "real").mkdir()
        (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
        assert contained_child(tmp_path, "alias", code="x", what="t") == (
            (tmp_path / "real").resolve()
        )


class TestTheRunLogPathIsContained:
    @pytest.mark.parametrize("name", TRAVERSALS[:-1])
    def test_traversal_is_refused_by_name(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(Refusal) as caught:
            run_log.run_log_path(tmp_path, name)
        assert caught.value.code == "run_log_run_id_invalid"

    def test_empty_keeps_its_own_code(self, tmp_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            run_log.run_log_path(tmp_path, "")
        assert caught.value.code == "run_log_run_id_required"

    def test_a_symlinked_run_directory_is_refused(self, escape: tuple[Path, str]) -> None:
        root, name = escape
        with pytest.raises(Refusal) as caught:
            run_log.run_log_path(root, name)
        assert caught.value.code == "run_log_run_id_invalid"

    def test_append_cannot_write_outside_the_root(self, escape: tuple[Path, str]) -> None:
        root, name = escape
        with pytest.raises(Refusal):
            run_log.run_log_path(root, name)
        assert not any((root.parent / "outside").iterdir())


class TestTheCheckpointRunDirIsContained:
    @pytest.mark.parametrize("name", TRAVERSALS)
    def test_traversal_is_refused_by_name(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(Refusal) as caught:
            checkpoint_paths.run_dir(name, root=tmp_path)
        assert caught.value.code in {"checkpoint_run_id_invalid", "checkpoint_run_id_required"}

    def test_a_symlinked_run_directory_is_refused(self, escape: tuple[Path, str]) -> None:
        root, name = escape
        with pytest.raises(Refusal) as caught:
            checkpoint_paths.run_dir(name, root=root)
        assert caught.value.code == "checkpoint_run_id_invalid"


class TestTheCheckpointStepDirIsContained:
    def test_a_step_directory_linked_out_of_the_run_is_refused(self, tmp_path: Path) -> None:
        """The step arrives from a URL path; a ``step-NNNNNN`` that is a link out of its run
        is refused, not followed (CodeQL py/path-injection, the checkpoint endpoints)."""
        run = checkpoint_paths.run_dir("run-1", root=tmp_path)
        run.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (run / checkpoint_paths.step_dir_name(5)).symlink_to(outside, target_is_directory=True)
        with pytest.raises(Refusal) as caught:
            checkpoint_paths.checkpoint_dir("run-1", 5, root=tmp_path)
        assert caught.value.code == "checkpoint_step_dir_invalid"

    def test_an_ordinary_step_directory_is_under_its_run(self, tmp_path: Path) -> None:
        path = checkpoint_paths.checkpoint_dir("run-1", 5, root=tmp_path)
        assert path == tmp_path.resolve() / "run-1" / "step-000005"


class TestTheMountedRouteRefusesAnEscape:
    def test_boundaries_refuses_a_symlinked_run_rather_than_reading_through_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Through the live app: a run directory that is a link out of the log root is
        refused ``run_log_run_id_invalid`` - not read, and not reported as merely absent."""
        from api.app import create_app
        from training import tasks

        logs = tmp_path / "logs"
        outside = tmp_path / "outside"
        logs.mkdir()
        (outside / "evil").mkdir(parents=True)
        (outside / "evil" / run_log.RUN_LOG_NAME).write_text("")
        (logs / "evil").symlink_to(outside / "evil", target_is_directory=True)
        monkeypatch.setattr(tasks, "run_log_root", lambda root=None: logs)

        response = TestClient(create_app()).get("/api/v1/runs/evil/boundaries")
        assert response.status_code in {400, 403, 404, 409, 422}
        assert response.json()["error"]["code"] == "run_log_run_id_invalid"
