"""Forest fixtures, built by the **real writers** and read back from disk  (task 3.6).

*"Fixtures whose fields agree by construction are how a suite stays green over a real
bug"* (ADR-016). A hand-built index row agrees with :func:`services.tree.compose`'s
expectations by construction and proves nothing, so every artifact here is produced by

* **F5's real writer** - :func:`services.checkpoints.writer.write`, through
  :func:`tests.support.checkpoints.saved_checkpoint`; and
* **F8's real writer** - :func:`training.runtypes.adapters.save.save`, over its real
  manifest builder and its real config model,

and then **indexed by ``reindex`` from the manifests on disk**. Nothing in this module
writes an ORM row. That is what makes a renamed manifest field, a changed layout or a
dropped key show up here rather than in production.

## The one artifact no writer can produce, and why it is written by hand anyway

A checkpoint manifest **missing its required ``domain`` key**. F5's
:func:`services.checkpoints.manifest.build` takes ``domain`` keyword-only with no default
and validates against the vendored schema, deliberately - *"an absent key is
unrepresentable"*. So :func:`checkpoint_without_domain_key` writes through the real writer
and then deletes the key from the published file. The artifact is nonconforming by
construction, which is the point: ``reindex`` must record it with ``domain_state`` of
``absent`` rather than refuse it (a checkpoint that exists on disk and is invisible is the
worse failure), and the tree must render that as a missing required key and not as
"no domain". Collapsing the two is mutation **M6**.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from core.canonical import sha256_of, write_canonical
from services.checkpoints.paths import qualified_name
from services.checkpoints.writer import WriteResult
from tests.support import db as support
from tests.support.checkpoints import saved_checkpoint, write_chain
from tests.support.test_adapter_doubles import (
    FakePeftModel,
    adapter_config,
    checkpoint_record,
    hub_record,
    optimizer_block,
)
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import save as adapter_save

BASE_MODEL = "LiquidAI/LFM2-1.2B"
HUB_REPO = "LiquidAI/LFM2.5-1.2B-Instruct"

__all__ = [
    "BASE_MODEL",
    "HUB_REPO",
    "KEY_ABSENT",
    "adapter",
    "branch",
    "chain",
    "checkpoint_without_domain_key",
    "cycle_of_two",
    "indexed",
    "merged_checkpoint",
    "root_checkpoint",
    "self_parented_checkpoint",
    "session_for",
    "unresolvable_parent_checkpoint",
]


# ====================================================================== checkpoints (F5)


def root_checkpoint(
    root: Path,
    *,
    run_id: str = "run-root",
    step: int = 100,
    domain: str | None = "support",
    **overrides: Any,
) -> WriteResult:
    """A checkpoint with ``parent: null`` - an **assertion** that this is a root.

    Not the same artifact as one whose parent does not resolve. The tree must render the
    two differently (``root`` vs ``truncated``, F5 FR-005.39 / its E-11), and conflating
    them is mutation **M9**.
    """
    return saved_checkpoint(
        root,
        run_id=run_id,
        step=step,
        payload=f"tensors-{run_id}-{step}".encode(),
        domain=domain,
        parent=None,
        base_model=BASE_MODEL,
        **overrides,
    )


def chain(
    root: Path,
    *,
    run_id: str = "run-chain",
    steps: Sequence[int] = (100, 200, 300),
) -> list[WriteResult]:
    """A run's successive saves, each one's ``parent`` the previous one's ``name``.

    Through F5's :func:`tests.support.checkpoints.write_chain`, which calls the real writer
    in sequence - so the chain the tree walks is the chain the writer produces.
    """
    return write_chain(root, run_id=run_id, steps=tuple(steps), base_model=BASE_MODEL)


def branch(root: Path, *, parent: str, run_id: str = "run-branch", step: int = 400) -> WriteResult:
    """A second child of one parent: a branch (R-02.23). Its ``run_id`` is a new one."""
    return saved_checkpoint(
        root,
        run_id=run_id,
        step=step,
        payload=f"tensors-{run_id}-{step}".encode(),
        parent=parent,
        base_model=BASE_MODEL,
    )


def unresolvable_parent_checkpoint(
    root: Path,
    *,
    run_id: str = "run-truncated",
    step: int = 700,
    parent: str = f"{BASE_MODEL}:run-gone:step-1",
) -> WriteResult:
    """A checkpoint whose named parent is in no index and on no disk: **truncated**.

    The record stops here. The tree names the parent it could not find rather than
    presenting the chain as rooted.
    """
    return saved_checkpoint(
        root,
        run_id=run_id,
        step=step,
        payload=b"tensors-truncated",
        parent=parent,
        base_model=BASE_MODEL,
    )


def self_parented_checkpoint(
    root: Path, *, run_id: str = "run-self", step: int = 800
) -> WriteResult:
    """A checkpoint naming **itself** as its parent: a broken edge, node still visible.

    The name is computed with the production
    :func:`services.checkpoints.paths.qualified_name` rather than spelled out, because a
    self-parent fixture whose two names merely *look* the same proves nothing about the
    comparison the resolver performs.
    """
    own_name = qualified_name(BASE_MODEL, run_id, step)
    return saved_checkpoint(
        root,
        run_id=run_id,
        step=step,
        payload=b"tensors-self",
        parent=own_name,
        base_model=BASE_MODEL,
    )


def cycle_of_two(
    root: Path, *, run_id: str = "run-cycle", steps: tuple[int, int] = (10, 20)
) -> tuple[WriteResult, WriteResult]:
    """``a -> b -> a``: a cycle longer than a self-parent.

    :func:`services.tree_edges.resolve_parent` cannot see this - it catches only a
    self-edge - so it is the case that proves ``compose``'s visited set is doing the work
    (FR-009.30). Both nodes stay visible; the edges are named broken.
    """
    first, second = steps
    name_a = qualified_name(BASE_MODEL, run_id, first)
    name_b = qualified_name(BASE_MODEL, run_id, second)
    a = saved_checkpoint(
        root,
        run_id=run_id,
        step=first,
        payload=b"tensors-cycle-a",
        parent=name_b,
        base_model=BASE_MODEL,
    )
    b = saved_checkpoint(
        root,
        run_id=run_id,
        step=second,
        payload=b"tensors-cycle-b",
        parent=name_a,
        base_model=BASE_MODEL,
    )
    return a, b


def checkpoint_without_domain_key(
    root: Path, *, run_id: str = "run-nonconforming", step: int = 5
) -> WriteResult:
    """A **nonconforming** checkpoint manifest: the required ``domain`` key is absent.

    Written through the real writer, then the key is removed from the published file. See
    the module docstring for why this one cannot come from the writer alone. The returned
    :class:`WriteResult` carries the manifest **as it now is on disk**, so a test comparing
    against it is comparing against the artifact rather than against the writer's input.
    """
    published = saved_checkpoint(
        root,
        run_id=run_id,
        step=step,
        payload=b"tensors-nonconforming",
        domain="to-be-removed",
        base_model=BASE_MODEL,
    )
    path = published.directory / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["domain"]
    write_canonical(path, document)
    return WriteResult(directory=published.directory, manifest=document)


class _Sentinel:
    """A named marker. ``None`` cannot serve here: it is one of the three real values."""

    def __init__(self, label: str) -> None:
        self._label = label

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self._label


#: Pass as ``manifest_sha256`` to omit the key entirely. ``None`` means *present and null*,
#: which is a **conforming** manifest; this means a missing required key, which is not.
KEY_ABSENT = _Sentinel("KEY_ABSENT")
#: The default: compute the adapter's real manifest hash, so the edge is **verified**.
_DEFAULT_HASH = _Sentinel("_DEFAULT_HASH")


def merged_checkpoint(
    root: Path,
    *,
    base: WriteResult,
    adapter_document: dict[str, Any],
    run_id: str = "run-merged",
    step: int = 900,
    manifest_sha256: Any = _DEFAULT_HASH,
) -> WriteResult:
    """A merged checkpoint: ``parent`` to its base **and** ``merged_adapter`` to the adapter.

    ``manifest_sha256`` has **three** legal settings, and they are three different
    artifacts (E-12, E-12a):

    * omitted (the default) -> the adapter's real manifest hash: the edge is **verified**;
    * ``None`` -> the key present and ``null``: **conforming**, the edge deliberately
      unverified. Never a defect;
    * :data:`KEY_ABSENT` -> the key removed **from the published file**, because
      ``manifest_sha256`` is ``required`` and F5's validating builder refuses a manifest
      without it. That refusal is correct and is the reason this artifact can only be
      produced by editing the file afterwards - the same hand as
      :func:`checkpoint_without_domain_key`, for the same reason.

    ``None`` is **not** the default, and that is deliberate: a default of ``None`` would
    make the "omitted" and the "present and null" calls produce the same artifact, so the
    verified case would silently be tested against an unverified one. (Which is exactly
    what happened on the first draft of this fixture, caught by its own test.)

    A single "no hash" fixture would serve two of those three and a test built on it would
    pass against an implementation that conflated them - which is the conflation
    FR-009.17 rule 3 forbids for ``domain``.
    """
    block: dict[str, Any] = {
        "name": adapter_document["name"],
        "step": adapter_document["step"],
    }
    if manifest_sha256 is None:
        block["manifest_sha256"] = None
    elif isinstance(manifest_sha256, str):
        block["manifest_sha256"] = manifest_sha256
    else:
        block["manifest_sha256"] = sha256_of(adapter_document)

    published = saved_checkpoint(
        root,
        run_id=run_id,
        step=step,
        payload=b"tensors-merged",
        parent=base.name,
        base_model=BASE_MODEL,
        merged_adapter=block,
    )
    if manifest_sha256 is not KEY_ABSENT:
        return published

    path = published.directory / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["merged_adapter"]["manifest_sha256"]
    write_canonical(path, document)
    return WriteResult(directory=published.directory, manifest=document)


# ========================================================================= adapters (F8)


def adapter(
    root: Path,
    *,
    base: WriteResult | None = None,
    run_id: str = "run-adapter",
    step: int = 50,
    rank: int = 8,
    alpha: float = 16.0,
) -> dict[str, Any]:
    """One adapter through **F8's real writer** - layout, digest, validation and all.

    ``base`` is a saved checkpoint, in which case the ``base`` block carries that
    checkpoint's ``name`` and its **manifest hash**; with no ``base`` the adapter is bound
    to a Hub repo id and ``manifest_sha256`` is ``null``, which the schema's own
    description says is what distinguishes the two cases.

    Returns the published manifest document, which is what :func:`adapter_save.save`
    returns after adding ``weights_sha256`` **from the file as written**.
    """
    if base is None:
        block = adapter_manifest.base_block(hub_record(HUB_REPO))
    else:
        block = adapter_manifest.base_block(
            checkpoint_record(checkpoint_name=base.name),
            manifest_sha256=sha256_of(base.manifest),
        )
    document = adapter_manifest.build(
        run_id=run_id,
        step=step,
        base=block,
        adapter=adapter_config(rank=rank, alpha=alpha, target_modules=("q_proj", "v_proj")),
        optimizer=optimizer_block(),
    )
    return adapter_save.save(
        FakePeftModel(rank=rank, alpha=alpha, target_modules=("q_proj", "v_proj")),
        document,
        adapter_paths.adapter_dir(run_id, step, root=root),
    )


# ============================================================================ the index


def indexed(engine: Engine, root: Path, *, run_ids: Sequence[str] = ()) -> None:
    """Index everything under ``root`` **from the manifests**, then insert the run rows.

    ``run_ids`` are the runs that exist as ``mf_runs`` rows. Every ``run_id`` **left out**
    is the FR-009.28 case - a valid manifest whose run has no row, which ``reindex``
    indexes (F5's OQ-4) and which the tree must render with its run named not indexed
    rather than drop. Defaulting this to "all of them" would make that case unreachable.
    """
    from models.run import Run
    from services.reindex import reindex
    from tests.support.jobs import insert_run

    with support.session_for(engine) as session:
        reindex(session, root=root)
        for run_id in run_ids:
            # Skipped when it is already there. ``reindex`` is idempotent and a test that
            # seeds, adds one artifact and re-indexes is the ordinary shape; an insert
            # error from the second call would be a fixture failure wearing a constraint
            # violation, which says nothing about what the test was asserting.
            if session.get(Run, run_id) is None:
                insert_run(session, run_id)
        session.commit()


#: Re-exported, by binding rather than by wrapping, so a tree test needs one import for its
#: fixtures and its session - and so the context manager stays a context manager.
session_for = support.session_for
