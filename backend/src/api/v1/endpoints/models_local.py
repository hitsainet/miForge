"""``GET /api/v1/models/local`` - the model directories a ``load`` can read  (task 14.0, AC-009.41).

The New run form's **Path** was a text box. The operator had to know, and type, a directory
on a host they were not logged into, and ``model_source_outside_root`` told them afterwards
where they should have looked. This route lists what is there.

## Exactly the places a load may read, and nothing else

The roots are :func:`services.acquisition.model_roots` - the same two directories
``local_dir.resolve_within_roots`` confines a submitted path to. Listing anything outside them
would offer the operator a choice preflight then refuses. A **symlink that leads out of a
root is not listed**, for the same reason: confinement is checked against the *resolved*
path, so such a directory is one the load would refuse.

A directory is listed when it holds ``config.json`` - the file without which "miForge cannot
tell what model it is" (``local_dir.read_config``). Whether it also holds safetensors shards
and a tokenizer **at its root** is reported rather than filtered, except under
``has_weights=true``: a create run wants the config of a directory that may have no weights,
and a directory missing its tokenizer is one the operator should *see* is incomplete rather
than wonder why it is absent. The file names are ``local_dir``'s own constants, so this list
and the loader cannot disagree about what a model directory is.

## Bounded, and saying so

The walk stops at :data:`MAX_DEPTH` levels below each root and does not descend into a
model directory once found (a checkpoint's own subdirectories are not further models). That
bound is why the form keeps a typed escape hatch for this field, and the response states the
depth so the screen can say so. A root that does not exist or cannot be read is **named** in
``unreadable_roots``, never silently skipped - an empty list from an unmounted volume reads
as "there are no models", which is false.

Read-only. Reads directory entries and, for each model directory, at most
:data:`CONFIG_READ_LIMIT` bytes of ``config.json`` to name its architecture.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query

from services import acquisition as acquisition_service
from services.checkpoints.manifest import MANIFEST_NAME as CHECKPOINT_MANIFEST
from training.acquisition.local_dir import CONFIG_FILE, TOKENIZER_CANDIDATES, WEIGHTS_SUFFIX

router = APIRouter(prefix="/models", tags=["acquisition"])
log = logging.getLogger("miforge.api.models_local")

#: The stable codes an ``unreadable_roots`` row carries. The OS error's own text stays in the
#: server log (CodeQL py/stack-trace-exposure, alert #28): it can name paths and host details
#: the client has no business reading, and a code is what a screen can match on.
ROOT_MISSING = "model_root_missing"
ROOT_UNREADABLE = "model_root_unreadable"

#: How far below a root the walk looks. A checkpoint is ``<root>/<run_id>/step-<n>``, a
#: placed base model is usually ``<root>/<org>/<name>``; four leaves room for one more level.
MAX_DEPTH = 4
#: A config larger than this is not read for its architecture (it is still listed).
CONFIG_READ_LIMIT = 1_000_000
DEFAULT_LIMIT = 100
MAX_LIMIT = 500


def _architecture(config_path: Path) -> str | None:
    """``architectures[0]`` or ``model_type``, as ``local_dir`` names one. ``None`` if unread."""
    try:
        if config_path.stat().st_size > CONFIG_READ_LIMIT:
            return None
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    architectures = parsed.get("architectures")
    if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
        return architectures[0]
    model_type = parsed.get("model_type")
    return model_type if isinstance(model_type, str) else None


def _describe(directory: Path, root: Path) -> dict[str, Any]:
    names = set(os.listdir(directory))
    config_path = directory / CONFIG_FILE
    modified = datetime.fromtimestamp(config_path.stat().st_mtime, tz=UTC)
    return {
        "path": str(directory),
        "root": str(root),
        "relative_path": str(directory.relative_to(root)),
        # A checkpoint carries miForge's manifest; anything else is a placed directory.
        "kind": "checkpoint" if CHECKPOINT_MANIFEST in names else "directory",
        "architecture": _architecture(config_path),
        "has_weights": any(name.endswith(WEIGHTS_SUFFIX) for name in names),
        "has_tokenizer": any(name in names for name in TOKENIZER_CANDIDATES),
        "config_path": str(config_path),
        "modified_at": modified.isoformat(),
    }


def scan_root(root: Path) -> list[dict[str, Any]]:
    """Every model directory under one root, confined to it and bounded in depth."""
    resolved_root = root.expanduser().resolve()
    found: list[dict[str, Any]] = []
    for current, subdirs, files in os.walk(resolved_root, followlinks=False):
        here = Path(current)
        depth = len(here.relative_to(resolved_root).parts)
        if CONFIG_FILE in files and here != resolved_root:
            found.append(_describe(here, resolved_root))
            subdirs[:] = []  # a model directory's subdirectories are not further models
            continue
        if depth >= MAX_DEPTH:
            subdirs[:] = []
            continue
        # Hidden directories (`.cache`, `.git`) hold no model an operator placed. A linked
        # directory is never walked INTO (`followlinks=False`): one pointing out of the root
        # is one a load would refuse, and one pointing in is a second name for a directory
        # the walk reaches by its own path.
        subdirs[:] = [name for name in subdirs if not name.startswith(".")]
    return found


@router.get("/local", summary="The model directories a load can read, under the model roots")
def list_local_models(
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
    has_weights: Annotated[
        bool | None, Query(description="true: only directories with safetensors shards")
    ] = None,
) -> dict[str, Any]:
    """Model directories, most recently written first, with the roots that were read."""
    roots = acquisition_service.model_roots()
    models: list[dict[str, Any]] = []
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
                # Two roots may nest (a parts root inside the checkpoint root); one row each.
                if entry["path"] not in seen:
                    seen.add(entry["path"])
                    models.append(entry)
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

    if has_weights is not None:
        models = [row for row in models if row["has_weights"] is has_weights]
    models.sort(key=lambda row: (row["modified_at"], row["path"]), reverse=True)

    return {
        "models": models[offset : offset + limit],
        "total": len(models),
        "limit": limit,
        "offset": offset,
        "roots": [str(root) for root in roots],
        "unreadable_roots": unreadable,
        "max_depth": MAX_DEPTH,
    }
