"""Hashing what a run started from  (R-02.47, ADR-005; FR-001.11).

Two identities, both SHA-256 through ``core.canonical`` - the codebase's one hashing path,
because *two canonicalisers are two answers* (ADR-005):

* :func:`weights_sha256` - the content hash of the weights miForge loaded;
* :func:`config_identity` - the identity of the architecture config a creation came from.

## Streamed, never materialised

A full parameter set is gigabytes. Hashing it by loading the tensors would double peak
memory on the card that R-02.42's budget is already counting to the megabyte, and it would
do so *at acquisition*, which is the single moment the budget is tightest. So the hash runs
over the **files**, through :func:`core.canonical.sha256_of_file`, which reads in 1 MiB
chunks.

## Why a hash over a mapping rather than a concatenation

Many models ship as shards. Concatenating shard bytes in directory order makes the digest
depend on how the filesystem listed them; hashing a *sorted mapping of relative name to
per-file digest* makes it depend only on the content and the names. It also means a
renamed shard changes the digest, which is correct: ``model-00001-of-00002.safetensors``
renamed is a different model directory even if the bytes are the same.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.canonical import sha256_of, sha256_of_file
from core.errors import refuse


def weights_sha256(weight_files: Sequence[Path], *, relative_to: Path | None = None) -> str:
    """The content hash of a model's weights (R-02.47).

    Args:
        weight_files: every weight shard. Order does not matter - the mapping is sorted.
        relative_to: render names relative to this directory, so the same model under two
            different mount points hashes the same. Without it an absolute path would enter
            the digest and FR-001.4's *"a second run against the same repo and revision
            records an identical starting hash"* would fail the moment the Hub cache moved.

    Raises:
        Refusal: ``model_weights_absent`` when there are no files. An empty digest would
            be a perfectly stable hash of nothing, recorded as though it described weights.
    """
    if not weight_files:
        raise refuse(
            "model_weights_absent",
            "No weight files were given, so there is nothing to hash. A run's starting "
            "hash describes the weights it started from; an empty one would be a stable "
            "hash of nothing recorded as though it described a model.",
        )

    per_file: dict[str, str] = {}
    for path in sorted(weight_files):
        name = path.name if relative_to is None else str(path.relative_to(relative_to))
        per_file[name] = sha256_of_file(path)
    return sha256_of({"weights": per_file})


def config_identity(config: Mapping[str, Any]) -> str:
    """The canonical identity of an architecture config (R-02.47's *"config identity"*).

    This is what a creation's ``base_model`` is built from, so it has to be stable under
    key reordering and whitespace - which is exactly what ``canonical_bytes`` guarantees
    and is why this does not hash the config *file*: two byte-different files describing
    the same architecture are the same lineage root.
    """
    return sha256_of(dict(config))


def created_weights_sha256(*, config: Mapping[str, Any], seed: int | None) -> str:
    """The starting hash for a ``create`` run (R-02.47 for the creation case).

    **The gap this closes, and the gap it does not.** R-02.47 requires the content hash of
    the weights a run started from for *both* acquisition modes, and no requirement in
    BRD-02 records a seed - so for a random initialisation the hash identified *which*
    initialisation happened without letting anyone reproduce it. The operator's decision of
    2026-10-14 records a seed, which makes ``(config, seed)`` a reproducible description of
    the initial parameters: the same config and the same seed produce the same weights for
    a given library version.

    **``seed=None`` is still recorded, and still irreproducible.** Following
    :attr:`providers.base.GenerationParams.seed`'s precedent in this codebase - *null is an
    assertion that no seed was set* - the digest then covers the config alone, and the
    record says so rather than implying a reproducibility it does not have. The honest
    failure mode is a hash that cannot be reproduced; the dishonest one is a hash that
    looks reproducible and is not.

    This is **not** a hash of the realised tensors. Hashing those would mean serialising a
    freshly-initialised parameter set to disk purely to describe it, at the moment the
    budget is tightest, to learn something ``(config, seed)`` already determines.
    """
    return sha256_of(
        {
            "created_from": {"config": dict(config), "seed": seed},
        }
    )
