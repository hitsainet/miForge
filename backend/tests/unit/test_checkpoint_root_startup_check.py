"""The artifact root is proved writable and atomic **at startup**  (F5 FR-005.5, task 1.10).

`atomic_dir` has always refused `checkpoint_root_not_atomic` when it publishes. That refusal
is correct and it is not enough: **a save point arrives hours into a run**, so a misconfigured
root was discovered at the moment the first checkpoint would have been written, and the run
had nothing to show for the time. F5 recorded this as the half it could not build, because
`core/config.py` is not a feature agent's file: *"Startup validation is a five-line addition
whenever someone owns that file."*

**Two decisions worth not re-litigating.**

1. **The check performs the real operation rather than predicting it.** It creates a sibling
   directory and compares devices — exactly what `atomic_dir` does at every save. Asking
   `os.statvfs` or parsing `/proc/mounts` would be a second implementation of the question,
   and the two would disagree first on a **bind mount**, which is how this deployment mounts
   every one of its roots.
2. **It logs and does not raise.** Refusing to construct the app would make an unwritable
   artifact root take down `/api/health`, the pack screens and every surface that touches no
   artifact — and an operator would be diagnosing a dead API rather than a misconfigured
   mount. The capability actually unavailable is *saving*, and that still refuses by name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.atomic_dir import verify_root_is_atomic
from core.errors import Refusal


class TestTheProbeIsTheRealOperation:
    def test_a_usable_root_passes_and_is_created_if_absent(self, tmp_path: Path) -> None:
        """mkdir-on-demand is right **here** because this is a writer's path, not a reader's.

        `packs.store.packs_root()` doing the same thing on a READ path is what aborted a whole
        reindex with `PermissionError`. The distinction is the lesson, not the call.
        """
        root = tmp_path / "checkpoints"
        assert not root.exists()
        assert verify_root_is_atomic(root) == root
        assert root.is_dir()

    def test_it_leaves_nothing_behind(self, tmp_path: Path) -> None:
        """A probe directory left in the artifact root would be indexed as an artifact with
        no manifest, and `reindex` would log a nonconforming artifact on every rebuild."""
        root = tmp_path / "checkpoints"
        verify_root_is_atomic(root)
        assert list(root.iterdir()) == []

    def test_it_is_idempotent_across_restarts(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        for _ in range(3):
            verify_root_is_atomic(root)
        assert list(root.iterdir()) == []


class TestAnUnwritableRootIsADIFFERENTFaultWithADifferentFix:
    def test_it_refuses_by_its_own_name_and_not_as_non_atomic(self, tmp_path: Path) -> None:
        """Collapsing the two would send someone to repartition a disk over a `chown`.

        This is the live deployment's actual failure: `/data` is root-owned and the process
        runs as uid 10001, which is how `/data/packs` surfaced as a 500 rather than silently
        discarding packs.
        """
        root = tmp_path / "checkpoints"
        root.mkdir()
        # The ROOT, not its parent: the probe is created INSIDE the root, so making the
        # parent read-only proves nothing. The first version of this test did that and
        # reported DID NOT RAISE - a test that could not reach the condition it named.
        root.chmod(0o500)
        try:
            with pytest.raises(Refusal) as caught:
                verify_root_is_atomic(root)
        finally:
            root.chmod(0o700)
        assert caught.value.code == "checkpoint_root_unwritable"
        assert str(root) in caught.value.message

    def test_the_refusal_names_the_root_so_an_operator_knows_which_mount(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "checkpoints"
        root.mkdir()
        root.chmod(0o500)
        try:
            with pytest.raises(Refusal) as caught:
                verify_root_is_atomic(root)
        finally:
            root.chmod(0o700)
        assert caught.value.details["root"] == str(root)


class TestTheCheckIsWIREDIntoStartup:
    """ADR-016 #1: a capability is not shipped until a test fails when its wiring is removed.

    Control: delete the `atomic_dir.verify_root_is_atomic(...)` call from `api/app.create_app`
    and both tests below go red — the first because nothing is called, the second because the
    source no longer mentions it.
    """

    def test_create_app_calls_it_with_the_CONFIGURED_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The payload, not just "was called".** A check run against a hard-coded path or
        against `Path.cwd()` would pass a was-called assertion while proving nothing about the
        root this installation actually publishes to.
        """
        import api.app as app_module
        from core.config import Settings

        seen: list[Path] = []
        root = tmp_path / "configured-checkpoints"

        monkeypatch.setattr(
            app_module, "get_settings", lambda: Settings(_env_file=None, checkpoint_root=root)
        )
        monkeypatch.setattr(
            app_module.atomic_dir,
            "verify_root_is_atomic",
            lambda r: (seen.append(r), r)[1],
        )
        app_module.create_app()
        assert seen == [root], "the startup check did not run against the configured root"

    def test_a_refusal_is_REPORTED_and_does_not_take_the_api_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unwritable artifact root must not make `/api/health` and the pack screens
        unreachable. The app still builds; the log names what failed and its consequence."""
        import api.app as app_module
        from core.config import Settings
        from core.errors import refuse

        monkeypatch.setattr(
            app_module,
            "get_settings",
            lambda: Settings(_env_file=None, checkpoint_root=tmp_path / "nope"),
        )

        def _refuse(_root: Path) -> Path:
            raise refuse("checkpoint_root_unwritable", "planted", root=str(_root))

        monkeypatch.setattr(app_module.atomic_dir, "verify_root_is_atomic", _refuse)

        # The module's own logger is recorded directly rather than through `caplog`:
        # `create_app` calls `configure_logging`, which installs this project's structured
        # handlers, so the root-handler capture `caplog` relies on sees nothing. A test that
        # read an empty `caplog.records` as "it did not log" would have been measuring the
        # logging configuration, not the behaviour.
        recorded: list[tuple[str, dict]] = []

        class _Recorder:
            def error(self, message: str, **kwargs: object) -> None:
                recorded.append((message, dict(kwargs.get("extra") or {})))  # type: ignore[arg-type]

        monkeypatch.setattr(app_module, "log", _Recorder())
        app = app_module.create_app()

        assert app is not None, "a bad artifact root must not take the whole API down"
        assert [extra["code"] for _, extra in recorded] == [
            "checkpoint_root_unwritable"
        ], "the refusal must be reported by name exactly once, not swallowed"
        assert (
            "consequence" in recorded[0][1]
        ), "the log must say what is unavailable, not only that something failed"
