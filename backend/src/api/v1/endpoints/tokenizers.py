"""The tokenizer a ``create`` run ships, as lists the New run form can offer (2026-10-20).

A run that creates a model from an architecture config is **given** a tokenizer - miForge
never trains one - and is refused ``tokenizer_required`` at submit when it names none
(``training.acquisition.scratch``). The backend has accepted the two spellings since
2026-10-16 (``tokenizer_path`` or ``tokenizer_repo`` with ``tokenizer_revision`` on the model
source), but the form offered neither, so every create run started from it was refused.
These two read-only routes are what the form's tokenizer picker reads:

* ``GET /tokenizers/local`` - the directories under the model roots that carry tokenizer
  files **at their root**: a tokenizer-only directory, a model directory, or a checkpoint.
  The roots and the confinement are exactly ``local_dir.resolve_within_roots``'s, and the
  file names are ``local_dir.TOKENIZER_CANDIDATES``, so this list and the create path cannot
  disagree about what a tokenizer directory is. ``GET /models/local`` is not reused because
  it lists only directories holding ``config.json``, and a tokenizer-only directory has none.
* ``GET /hub/revision`` - a Hub repository's revision resolved to the commit it points at
  now, and whether that commit carries a tokenizer, a config and weights at its root. It goes
  through ``training.acquisition.hub`` (every outbound Hub call lives there) and downloads
  nothing.

Neither route writes anything, and neither is a permission: the create path resolves the
revision and reads the files again at submit.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query

from api.v1.endpoints.models_local import (
    DEFAULT_LIMIT,
    MAX_DEPTH,
    MAX_LIMIT,
    ROOT_MISSING,
    ROOT_UNREADABLE,
)
from services import acquisition as acquisition_service
from services.checkpoints.manifest import MANIFEST_NAME as CHECKPOINT_MANIFEST
from training.acquisition.local_dir import CONFIG_FILE, TOKENIZER_CANDIDATES, WEIGHTS_SUFFIX

router = APIRouter(tags=["acquisition"])
log = logging.getLogger("miforge.api.tokenizers")

#: A Hub repository id: ``name`` or ``org/name``. Checked here so a malformed id is a 422
#: naming the field rather than a Hub round trip that fails with the Hub's own wording.
HUB_REPO_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$"


def _kind(names: set[str]) -> str:
    """``checkpoint`` (miForge's manifest), ``model`` (a config.json) or ``tokenizer``."""
    if CHECKPOINT_MANIFEST in names:
        return "checkpoint"
    if CONFIG_FILE in names:
        return "model"
    return "tokenizer"


def _describe(directory: Path, root: Path, files: list[str]) -> dict[str, Any]:
    names = set(files)
    present = [name for name in TOKENIZER_CANDIDATES if name in names]
    newest = max((directory / name).stat().st_mtime for name in present)
    return {
        "path": str(directory),
        "root": str(root),
        "relative_path": str(directory.relative_to(root)),
        "kind": _kind(names),
        "tokenizer_files": present,
        "modified_at": datetime.fromtimestamp(newest, tz=UTC).isoformat(),
    }


def scan_root(root: Path) -> list[dict[str, Any]]:
    """Every directory under one root with tokenizer files at its own root, bounded in depth.

    A hit's subdirectories are not walked: a model directory's ``tokenizer/`` subfolder is a
    layout the create path refuses for the parent, and listing it beside the parent would
    offer two answers for one model. A directory whose tokenizer is *only* in a subfolder is
    walked into, so the subfolder itself - which the create path accepts - is listed.
    """
    resolved_root = root.expanduser().resolve()
    found: list[dict[str, Any]] = []
    for current, subdirs, files in os.walk(resolved_root, followlinks=False):
        here = Path(current)
        depth = len(here.relative_to(resolved_root).parts)
        if here != resolved_root and any(name in files for name in TOKENIZER_CANDIDATES):
            found.append(_describe(here, resolved_root, files))
            subdirs[:] = []
            continue
        if depth >= MAX_DEPTH:
            subdirs[:] = []
            continue
        subdirs[:] = [name for name in subdirs if not name.startswith(".")]
    return found


@router.get(
    "/tokenizers/local",
    summary="The tokenizer directories a create run can use, under the model roots",
)
def list_local_tokenizers(
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Directories holding tokenizer files at their root, most recently written first."""
    roots = acquisition_service.model_roots()
    rows: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            unreadable.append(
                {"root": str(root), "code": ROOT_MISSING, "reason": "not a directory on this host"}
            )
            continue
        try:
            for entry in scan_root(root):
                if entry["path"] not in seen:
                    seen.add(entry["path"])
                    rows.append(entry)
        except OSError:
            log.warning(
                "model root could not be read",
                exc_info=True,
                extra={"code": ROOT_UNREADABLE, "root": str(root)},
            )
            unreadable.append(
                {
                    "root": str(root),
                    "code": ROOT_UNREADABLE,
                    "reason": "could not be read (the server log has the error)",
                }
            )

    rows.sort(key=lambda row: (row["modified_at"], row["path"]), reverse=True)
    return {
        "tokenizers": rows[offset : offset + limit],
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "roots": [str(root) for root in roots],
        "unreadable_roots": unreadable,
        "max_depth": MAX_DEPTH,
    }


@router.get(
    "/hub/revision",
    summary="Resolve a Hub repository's revision to a commit and say what is at its root",
)
def check_hub_revision(
    repo: Annotated[
        str,
        Query(
            min_length=1,
            max_length=200,
            pattern=HUB_REPO_PATTERN,
            description="The repository, such as org/name.",
        ),
    ],
    revision: Annotated[
        str | None,
        Query(
            min_length=1,
            max_length=200,
            description="A branch, tag or commit. Empty means main.",
        ),
    ] = None,
) -> dict[str, Any]:
    """The commit the revision points at now, and the tokenizer, config and weights files
    at that commit's root. Nothing is downloaded and nothing is recorded."""
    from training.acquisition import hub

    requested = revision or hub.DEFAULT_REVISION
    commit = hub.resolve_revision(repo, requested)
    root_files = {name for name in hub.remote_files(repo, commit) if "/" not in name}
    tokenizer_files = [name for name in TOKENIZER_CANDIDATES if name in root_files]
    return {
        "repo": repo,
        "revision": requested,
        "commit": commit,
        "is_commit": requested == commit,
        "tokenizer_files": tokenizer_files,
        "has_tokenizer": bool(tokenizer_files),
        "has_config": CONFIG_FILE in root_files,
        "has_weights": any(name.endswith(WEIGHTS_SUFFIX) for name in root_files),
    }
