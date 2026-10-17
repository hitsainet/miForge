"""Loading from a Hugging Face repo  (FR-001.4; E-3, E-4, AC-001.2).

**The Hub is faked at the ``huggingface_hub`` boundary**, not by patching ``httpx``. That is
ADR-010's single-call-site discipline applied to an external service: there is one seam to
replace, and the fake exercises the real refusal translation rather than a transport error
this code never sees.

Two facts this file pins, both of which are about honesty rather than behaviour:

* **The resolved commit is recorded beside the requested revision**, never instead of it.
  A branch moves; ``main`` is what was asked for and the commit is what was trained, and a
  record carrying only one of them cannot say which.
* **An unreachable Hub is reported as the Hub**, never as a sibling. BRD-02 §2a puts the
  Hub under *External*; reporting it as ``millm_not_configured`` or as a generic "sibling
  unavailable" sends the operator to the wrong machine entirely (E-4).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.config import Settings, reset_settings
from core.errors import Refusal
from tests.unit.test_acquisition_local_dir import build_model_dir
from training.acquisition import hub as hub_module
from training.acquisition.base import AcquisitionSpec, SourceKind
from training.acquisition.hub import DEFAULT_REVISION, HubAcquirer, resolve_revision

COMMIT = "a" * 40


class FakeHub:
    """The ``huggingface_hub`` surface this feature uses. Two functions, nothing else."""

    def __init__(self, *, snapshot: Path, sha: str | None = COMMIT, error: Exception | None = None):
        self.snapshot = snapshot
        self.sha = sha
        self.error = error
        self.repo_info_calls: list[dict[str, Any]] = []
        self.download_calls: list[dict[str, Any]] = []

    def repo_info(self, repo_id: str, revision: str) -> Any:
        self.repo_info_calls.append({"repo_id": repo_id, "revision": revision})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(sha=self.sha)

    def snapshot_download(self, repo_id: str, revision: str) -> str:
        self.download_calls.append({"repo_id": repo_id, "revision": revision})
        return str(self.snapshot)


class RevisionNotFoundError(Exception):
    """Named to match what the real client raises; the translation is by type name."""


class RepositoryNotFoundError(Exception):
    pass


@pytest.fixture
def rooted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkpoints = tmp_path / "checkpoints"
    parts = tmp_path / "parts"
    checkpoints.mkdir()
    parts.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(_env_file=None, checkpoint_root=checkpoints, parts_root=parts),  # type: ignore[call-arg]
    )
    yield checkpoints
    reset_settings()


@pytest.fixture
def snapshot(rooted: Path) -> Path:
    return build_model_dir(rooted, name="snapshot")


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeHub) -> FakeHub:
    monkeypatch.setattr(hub_module, "_hub", lambda: fake)
    return fake


class TestRevisionResolution:
    def test_the_revision_is_resolved_to_a_commit_FIRST(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        """Before anything is downloaded, so a branch that moves between resolution and
        download cannot change what is trained."""
        fake = install(monkeypatch, FakeHub(snapshot=snapshot))
        assert resolve_revision("org/model", "main") == COMMIT
        assert fake.repo_info_calls == [{"repo_id": "org/model", "revision": "main"}]
        assert fake.download_calls == [], "nothing may be fetched before the commit is known"

    def test_the_download_is_pinned_to_the_commit_not_the_branch(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        fake = install(monkeypatch, FakeHub(snapshot=snapshot))
        HubAcquirer().preflight(AcquisitionSpec(repo="org/model", revision="main"))
        assert fake.download_calls == [{"repo_id": "org/model", "revision": COMMIT}]

    def test_a_missing_revision_refuses_naming_the_repo_AND_the_revision(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        """E-3's requirement. Naming both is what lets the operator tell a typo'd branch
        from a typo'd repo, which have different fixes."""
        install(
            monkeypatch,
            FakeHub(snapshot=snapshot, error=RevisionNotFoundError("no such revision")),
        )
        with pytest.raises(Refusal) as caught:
            resolve_revision("org/model", "v9")
        assert caught.value.code == "model_revision_not_found"
        assert caught.value.details == {"repo": "org/model", "revision": "v9"}
        assert "org/model" in caught.value.message and "v9" in caught.value.message

    def test_a_missing_repo_refuses_and_mentions_a_token(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        install(monkeypatch, FakeHub(snapshot=snapshot, error=RepositoryNotFoundError("gone")))
        with pytest.raises(Refusal) as caught:
            resolve_revision("org/private", "main")
        assert caught.value.code == "model_revision_not_found"
        assert "token" in caught.value.message

    def test_an_unreachable_hub_is_reported_as_THE_HUB_not_a_sibling(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        """E-4. The Hub is an external dependency (BRD-02 §2a *External*), so the refusal
        must not look like a miForge integration problem - nothing in miForge is
        misconfigured and no sibling is involved."""
        install(monkeypatch, FakeHub(snapshot=snapshot, error=TimeoutError("dns")))
        with pytest.raises(Refusal) as caught:
            resolve_revision("org/model", "main")
        assert caught.value.code == "model_fetch_failed"
        assert "Hub, not a sibling" in caught.value.message
        assert "not_configured" not in caught.value.code

    def test_a_hub_that_returns_no_commit_refuses_rather_than_inventing_one(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        """**No default.** Recording the requested name as though it were a commit would
        make the record read as a pinned revision while describing a moving branch."""
        install(monkeypatch, FakeHub(snapshot=snapshot, sha=None))
        with pytest.raises(Refusal) as caught:
            resolve_revision("org/model", "main")
        assert caught.value.code == "model_revision_not_found"
        assert "as though it were a commit" in caught.value.message


class TestBothHalvesOfTheRevisionSurvive:
    """AC-001.2, and the mutation that drops ``source_revision_resolved``."""

    def test_preflight_leaves_the_REQUEST_in_place_and_adds_the_commit_beside_it(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        """Two fields, deliberately - the same reason the run log's definition block keeps
        ``submitted`` and ``resolved`` apart. Overwriting the request would make the record
        claim a commit was asked for when a branch was."""
        install(monkeypatch, FakeHub(snapshot=snapshot))
        enriched = HubAcquirer().preflight(AcquisitionSpec(repo="org/model", revision="main"))
        assert enriched.revision == "main"
        assert enriched.revision_resolved == COMMIT

    def test_an_unnamed_revision_records_the_default_explicitly(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        """A blank "requested" field would read as though no revision was asked for, which
        is a different fact from asking for ``main``."""
        install(monkeypatch, FakeHub(snapshot=snapshot))
        acquirer = HubAcquirer()
        enriched = acquirer.preflight(AcquisitionSpec(repo="org/model"))
        record = _record_via_fake_load(monkeypatch, acquirer, enriched)
        assert record.source_revision_requested == DEFAULT_REVISION
        assert record.source_revision_resolved == COMMIT

    def test_the_record_carries_the_repo_the_commit_and_the_hash(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        install(monkeypatch, FakeHub(snapshot=snapshot))
        acquirer = HubAcquirer()
        enriched = acquirer.preflight(AcquisitionSpec(repo="org/model", revision="main"))
        record = _record_via_fake_load(monkeypatch, acquirer, enriched)
        assert record.source_kind is SourceKind.HUB
        assert record.source_repo == "org/model"
        assert record.base_model == f"hf:org/model@{COMMIT}"
        assert record.parent is None, "a Hub repo at a revision IS the lineage root"
        assert len(record.starting_weights_sha256) == 64

    def test_the_same_repo_and_revision_yields_an_IDENTICAL_starting_hash(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        """AC-001.2's second half. The hash is over relative names and file content, so a
        cache that moved between runs cannot change it."""
        install(monkeypatch, FakeHub(snapshot=snapshot))
        acquirer = HubAcquirer()
        spec = AcquisitionSpec(repo="org/model", revision="main")
        first = _record_via_fake_load(monkeypatch, acquirer, acquirer.preflight(spec))
        second = _record_via_fake_load(monkeypatch, acquirer, acquirer.preflight(spec))
        assert first.starting_weights_sha256 == second.starting_weights_sha256

    def test_acquire_without_a_resolved_commit_refuses(self, snapshot: Path) -> None:
        with pytest.raises(Refusal) as caught:
            HubAcquirer().acquire(AcquisitionSpec(repo="org/model", revision="main"))
        assert caught.value.code == "acquisition_preflight_skipped"
        assert "only honest thing to record" in caught.value.message


class TestTheHubPathRunsTheWholeLocalGate:
    """**Two test findings from the mutation sweep (M27, M25), written up as regressions.**

    ``require_tokenizer_at_root`` and the dtype gate were both fully unit-tested, and
    **nothing asserted they were wired into the Hub acquirer's pre-flight**: deleting the
    tokenizer call from ``hub.preflight`` left the whole suite green while
    ``local_dir.preflight`` kept its own test. That is the weaker form of ADR-016 #1 - *a
    grep proves a call exists, not that anything depends on it* - and it is worse on this
    path than on the local one, because a fetched snapshot is the case where the operator
    has no chance to inspect the layout first.
    """

    def test_a_fetched_snapshot_with_a_nested_tokenizer_is_REFUSED(
        self, monkeypatch: pytest.MonkeyPatch, rooted: Path
    ) -> None:
        from tests.unit.test_acquisition_local_dir import build_model_dir

        snapshot = build_model_dir(rooted, name="nested", tokenizer=None)
        nested = snapshot / "tokenizer"
        nested.mkdir()
        (nested / "tokenizer.json").write_text("{}")
        install(monkeypatch, FakeHub(snapshot=snapshot))

        with pytest.raises(Refusal) as caught:
            HubAcquirer().preflight(AcquisitionSpec(repo="org/model", revision="main"))
        assert caught.value.code == "model_tokenizer_not_at_root"

    def test_a_fetched_snapshot_of_quantized_weights_is_REFUSED(
        self, monkeypatch: pytest.MonkeyPatch, rooted: Path
    ) -> None:
        """The dtype gate's wiring on the same path, for the same reason."""
        from tests.unit.test_acquisition_local_dir import build_model_dir

        snapshot = build_model_dir(rooted, name="int8", dtype="I8")
        install(monkeypatch, FakeHub(snapshot=snapshot))

        with pytest.raises(Refusal) as caught:
            HubAcquirer().preflight(AcquisitionSpec(repo="org/model", revision="main"))
        assert caught.value.code == "model_quantized"
        assert "FP16 or BF16" in caught.value.message
        # And the refusal names the repo AT THE RESOLVED COMMIT, not the branch asked for:
        # a quantized model found at a moving branch is a fact about one commit.
        assert COMMIT in caught.value.details["source"]

    def test_preflight_runs_the_tokenizer_check_once_on_the_fetched_directory(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: Path
    ) -> None:
        """The payload and the call count, not "was called" (ADR-016 #1)."""
        from training.acquisition import hub as hub_mod

        seen: list[Path] = []
        monkeypatch.setattr(hub_mod, "require_tokenizer_at_root", seen.append)
        install(monkeypatch, FakeHub(snapshot=snapshot))
        HubAcquirer().preflight(AcquisitionSpec(repo="org/model", revision="main"))
        assert seen == [snapshot]


class TestAvailability:
    """``available()`` answers from the *build*, never from the network."""

    def test_an_absent_hub_client_is_reported_BY_NAME_with_its_lost_capability(self) -> None:
        """On a machine without the ML wheels - which is this suite's own machine, and the
        state ADR-023 requires to be workable - the mode is reported unavailable **by name**
        rather than omitted or crashing. An option missing from a response is
        indistinguishable from one that is broken."""
        state = HubAcquirer().available()
        try:
            import huggingface_hub  # noqa: F401

            installed = True
        except ImportError:
            installed = False

        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            runtime = True
        except ImportError:
            runtime = False

        assert state.available is (installed and runtime)
        if not state.available:
            assert state.reason == "ml_runtime_unavailable"
            assert (
                state.unavailable_capabilities
            ), "the refusal must name the capabilities lost, not only that something failed"
            assert any(
                "Hugging Face repo" in capability or "training step" in capability
                for capability in state.unavailable_capabilities
            )

    def test_availability_does_not_reach_the_network_or_shell_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreachable Hub is a *run* failure (E-4), not a build-level unavailability, and
        ``GET /models/sources`` answers with no run in hand and must answer instantly.

        Enforced by making both expensive things explode: a probe of either fails the test
        rather than merely slowing it down.
        """
        import subprocess

        def explode(*_args, **_kwargs):  # pragma: no cover - the point is it is not hit
            raise AssertionError("available() must not shell out or probe the network")

        monkeypatch.setattr(subprocess, "run", explode)
        monkeypatch.setattr(
            hub_module,
            "_hub",
            lambda: (_ for _ in ()).throw(AssertionError("available() must not reach out")),
        )
        HubAcquirer().available()


def _record_via_fake_load(
    monkeypatch: pytest.MonkeyPatch, acquirer: HubAcquirer, spec: AcquisitionSpec
):
    """Run ``acquire`` with the torch/transformers layer faked, and return its record.

    Only the ML import is faked. Everything the assertions are about - the revision fields,
    the lineage, the digest - runs through the real code, which is the point: a fake that
    reached further up would make the record agree with the test by construction.
    """
    from training.acquisition import local_dir

    monkeypatch.setattr(local_dir, "_load_pretrained", lambda directory, dtype: (object(), None))
    return acquirer.acquire(spec).record


def test_the_fake_matches_the_real_clients_surface() -> None:
    """The fake must not drift from the two functions this feature actually calls.

    Checked by reading the production module's own call sites rather than by importing
    ``huggingface_hub``, which is not installed here (ADR-023) - so this asserts the fake
    covers what the code uses, which is the property that matters.
    """
    source = Path(hub_module.__file__).read_text(encoding="utf-8")
    for call in ("hub.repo_info(", "hub.snapshot_download("):
        assert call in source, f"{call} is no longer how the Hub is reached; update the fake"
    assert hasattr(FakeHub, "repo_info") and hasattr(FakeHub, "snapshot_download")


def test_the_fixture_snapshot_is_a_real_loadable_layout(snapshot: Path) -> None:
    """Guards the fixture itself: a snapshot missing its tokenizer would make every refusal
    test above pass for the wrong reason."""
    assert (snapshot / "config.json").is_file()
    assert json.loads((snapshot / "config.json").read_text())["model_type"] == "llama"
    assert (snapshot / "tokenizer.json").is_file()
    assert list(snapshot.glob("*.safetensors"))
