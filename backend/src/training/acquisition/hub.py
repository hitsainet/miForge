"""Load a model from a Hugging Face repo  (R-02.46 case 1; FR-001.4).

**The resolved revision is recorded, not only the requested one.** A branch name moves;
the commit it pointed at when the run started is what was actually trained. Dropping the
resolved value makes FR-001.4's criterion - *a second run against the same repo and
revision records an identical starting hash* - unverifiable, and it makes a moved branch a
silent change of what is being trained. That is why the revision is resolved to a commit
**first**, before anything is downloaded.

## One call site, and the Hub is not a sibling

Every outbound Hub call in miForge is in this module. ADR-010's one-outbound-call-site
discipline is written for optional *siblings*, and the Hub is not one - it is an external
dependency (BRD-02 section 2a). The discipline is borrowed anyway for two practical
reasons: the fake in the tests has one seam to replace, and **a Hub failure must be
reported as a Hub failure** (E-4). Reporting an unreachable Hub as ``millm_not_configured``
or as a generic "sibling unavailable" sends the operator to the wrong machine entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import refuse
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    Availability,
    SourceKind,
    lineage_for_first_load,
    ml_runtime_availability,
)
from training.acquisition.digest import weights_sha256
from training.acquisition.dtype_gate import assert_trainable, collect_weight_dtypes
from training.acquisition.local_dir import (
    param_count_from_headers,
    read_config,
    require_tokenizer_at_root,
    weight_files,
)
from training.acquisition.placement import place

#: The default when the operator names no revision. Recorded as the *requested* value so
#: the record never implies a precision the request did not have, while the resolved commit
#: carries the precision that matters.
DEFAULT_REVISION = "main"


def _hub() -> Any:
    """Import ``huggingface_hub``, refusing by name when it is absent.

    Lazy for the same reason as torch (ADR-023): the whole non-GPU surface of this feature,
    including every refusal path, must import on a machine without the ML wheels.
    """
    try:
        import huggingface_hub
    except ImportError as exc:
        raise refuse(
            "ml_runtime_unavailable",
            "huggingface_hub is not installed in this process, so a Hub model cannot be "
            "fetched. A local directory load and everything else in miForge work without "
            "it (ADR-023).",
            detail=str(exc),
        ) from exc
    return huggingface_hub


def resolve_revision(repo: str, revision: str) -> str:
    """Resolve a branch, tag or commit to **the commit** (FR-001.4, E-3).

    Raises:
        Refusal: ``model_revision_not_found`` naming the repo **and** the revision asked
            for - E-3's requirement - so the operator can tell a typo'd branch from a
            typo'd repo; ``model_fetch_failed`` naming **the Hub** when it cannot be
            reached at all (E-4).
    """
    hub = _hub()
    try:
        info = hub.repo_info(repo_id=repo, revision=revision)
    except Exception as exc:  # the hub raises a family of errors; all are refusals here
        name = type(exc).__name__
        if "RevisionNotFound" in name:
            raise refuse(
                "model_revision_not_found",
                f"The repo {repo!r} exists but has no revision {revision!r}. Name a branch, "
                f"tag or commit that exists in it.",
                repo=repo,
                revision=revision,
            ) from exc
        if "RepositoryNotFound" in name or "GatedRepo" in name or "EntryNotFound" in name:
            raise refuse(
                "model_revision_not_found",
                f"The Hugging Face Hub has no accessible repo {repo!r} (asked at revision "
                f"{revision!r}). Check the repo id, and whether it needs a token.",
                repo=repo,
                revision=revision,
            ) from exc
        raise refuse(
            "model_fetch_failed",
            f"The Hugging Face Hub could not be reached while resolving {repo!r} at "
            f"{revision!r}: {exc}. **This is the Hub, not a sibling application** - no "
            f"miForge integration is involved and nothing in miForge is misconfigured.",
            repo=repo,
            revision=revision,
        ) from exc

    commit = getattr(info, "sha", None)
    if not commit:
        # No default. A record whose resolved revision was fabricated from the requested
        # one would read as a pinned commit while describing a moving branch.
        raise refuse(
            "model_revision_not_found",
            f"The Hub returned no commit sha for {repo!r} at {revision!r}, so the revision "
            f"actually trained could not be recorded. Refusing rather than recording the "
            f"requested name as though it were a commit.",
            repo=repo,
            revision=revision,
        )
    return str(commit)


def download_snapshot(repo: str, commit: str) -> Path:
    """Fetch the repo at an exact commit and return the local directory.

    Pinned to the resolved commit rather than the requested branch, so a branch that moves
    between resolution and download cannot change what is trained.
    """
    hub = _hub()
    try:
        local = hub.snapshot_download(repo_id=repo, revision=commit)
    except Exception as exc:
        raise refuse(
            "model_fetch_failed",
            f"Fetching {repo!r} at {commit} from the Hugging Face Hub failed: {exc}. "
            f"**The Hub is an external dependency, not a sibling application.**",
            repo=repo,
            revision=commit,
        ) from exc
    return Path(local)


class HubAcquirer:
    """``load`` from a Hugging Face repo id and revision (FR-001.4)."""

    @property
    def mode(self) -> AcquisitionMode:
        return AcquisitionMode.LOAD

    def available(self) -> Availability:
        """Whether the Hub client is installed. **Does not probe the network.**

        The protocol forbids a network call here: ``GET /api/v1/models/sources`` answers
        with no run in hand and must answer instantly, and an unreachable Hub is a *run*
        failure (E-4), not a build-level unavailability.
        """
        runtime = ml_runtime_availability("load a model from a Hugging Face repo")
        if not runtime.available:
            return runtime
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return Availability(
                available=False,
                reason="ml_runtime_unavailable",
                message=(
                    "huggingface_hub is not installed in this process, so a model cannot "
                    "be fetched from the Hub. A local directory load is unaffected."
                ),
                unavailable_capabilities=("load from a Hugging Face repo",),
            )
        return Availability(available=True)

    def preflight(self, spec: AcquisitionSpec) -> AcquisitionSpec:
        """Resolve the revision, fetch, then run the local directory's whole gate.

        The fetch happens in preflight because the dtype gate reads the weight files'
        headers and there is nothing to read until they are local. It is still before the
        GPU lease and before any VRAM: a download is bytes on a disk, not an allocation.
        """
        if spec.repo is None:
            raise refuse(
                "model_source_missing",
                "A Hub load needs a repo id. Refusing rather than defaulting to a repo the "
                "operator did not name.",
            )
        # The default is named explicitly rather than left blank: a blank "requested"
        # field reads as though no revision was asked for, which is a different fact.
        requested = spec.revision or DEFAULT_REVISION
        commit = resolve_revision(spec.repo, requested)
        directory = download_snapshot(spec.repo, commit)

        config = read_config(directory)
        shards = weight_files(directory)
        require_tokenizer_at_root(directory)
        assert_trainable(config, collect_weight_dtypes(shards), source=f"{spec.repo}@{commit}")

        return spec.model_copy(
            update={
                # The request is left where it was; the commit lands beside it. See
                # AcquisitionSpec.revision_resolved for why these are two fields.
                "revision_resolved": commit,
                "param_count": param_count_from_headers(shards),
            }
        )

    def acquire(self, spec: AcquisitionSpec) -> AcquiredModel:
        """Load the fetched snapshot into this process.

        ``spec.revision`` is the **resolved commit** by the time this runs - preflight
        replaced it - so the snapshot is addressed by commit and the record carries both
        the request and the resolution.
        """
        from training.acquisition.local_dir import _load_pretrained

        if spec.repo is None or spec.revision_resolved is None:
            raise refuse(
                "acquisition_preflight_skipped",
                "acquire() was called without a resolved commit. Call preflight() first: "
                "it is where the revision becomes a commit, and the commit is the only "
                "honest thing to record as what was trained.",
            )

        commit = spec.revision_resolved
        directory = download_snapshot(spec.repo, commit)
        config = read_config(directory)
        shards = weight_files(directory)
        dtype_name = assert_trainable(
            config, collect_weight_dtypes(shards), source=f"{spec.repo}@{commit}"
        )
        module, tokenizer = _load_pretrained(directory, dtype_name)
        # ONTO THE LEASED CARD, for the same reason as the local directory it shares
        # `_load_pretrained` with: the loader leaves every tensor in host memory.
        placed = place(module, spec.gpu_request)

        identity = f"hf:{spec.repo}@{commit}"
        base_model, parent = lineage_for_first_load(model_identity=identity)
        return AcquiredModel(
            module=module,
            tokenizer=tokenizer,
            record=AcquisitionRecord(
                mode=AcquisitionMode.LOAD,
                source_kind=SourceKind.HUB,
                source_repo=spec.repo,
                # Both halves. The requested name may have been a branch; the resolved
                # commit is what was trained, and only recording both makes a repeated run
                # comparable (FR-001.4, and the mutation that drops this field).
                source_revision_requested=spec.revision or DEFAULT_REVISION,
                source_revision_resolved=commit,
                starting_weights_sha256=weights_sha256(shards, relative_to=directory),
                base_model=base_model,
                parent=parent,
                param_count=spec.param_count or param_count_from_headers(shards),
                device_kind=placed.device_kind,
                device_uuid=placed.device_uuid,
                device_placement_reason=placed.reason,
            ),
            device_uuid=placed.device_uuid,
        )
