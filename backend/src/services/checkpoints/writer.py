"""The standard Hugging Face directory, written atomically  (FR-005.2, FR-005.11, 19-23).

**Tokenizer files at the ROOT, never in a ``tokenizer/`` subdirectory.** This is not a
detail: as the contract was first written a checkpoint could not be loaded by
``transformers``, by vLLM, by TGI or by miLLM's own loader, and it broke R-02.21 and
acceptance criterion 2 before any code existed. ``save_pretrained`` already does the right
thing, so :func:`assert_file_set` is what stops a later "tidy" from undoing it.

``manifest.json`` is the **only** non-standard file in the directory and is purely
additive: a consumer that ignores it still loads the model. Nothing else goes in - and in
particular **no optimizer or trainer state**, which lives under its own root (R-02.15, F1
FR-001.30); a checkpoint directory that gains optimizer tensors stops being a standard
Hugging Face directory and the output contract is gone.

## The order is the requirement, and it is not stated here

``fsync`` every file, ``fsync`` the temp directory, ``rename``, swap ``latest``. That lives
in :func:`core.atomic_dir.atomic_dir` - the backend's only place a directory is renamed
into place - and this module is a **caller**. The order is reproduced in no comment here on
purpose: a second statement of it is a second thing to keep in step.

What this module does own is the order *inside* the block: weights, then config and
tokenizer, then the hash **of the written file**, then ``manifest.json`` **last**. Last so
a manifest never describes files that are not there yet, and the hash after the write
because the hash is of the bytes on disk rather than of the tensors in memory.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.atomic_dir import TMP_PREFIX, atomic_dir
from core.canonical import sha256_of_file, write_canonical
from core.errors import refuse
from services.checkpoints import manifest as manifest_module
from services.checkpoints.manifest import MANIFEST_NAME, ManifestFields

log = logging.getLogger("miforge.checkpoints.writer")

__all__ = [
    "WEIGHTS_NAME",
    "WriteResult",
    "assert_file_set",
    "candidate_weights_sha256",
    "verify_weights",
    "write",
]

#: The weights file. ``safe_serialization=True``, so ``model.safetensors`` and never
#: ``pytorch_model.bin`` (R-02.51, ADR-002) - a pickle is not a portable artifact and a
#: consumer is entitled to refuse to unpickle one.
WEIGHTS_NAME = "model.safetensors"

#: What ``save_pretrained`` must have produced for the directory to be loadable at all.
REQUIRED_FILES: frozenset[str] = frozenset({WEIGHTS_NAME, "config.json", MANIFEST_NAME})

#: Every filename a standard ``save_pretrained`` pair may leave in a model directory, plus
#: ``manifest.json``. **An allowlist, so the file-set assertion is an equality** rather than
#: "the three I remembered are present": anything else present is refused by name.
#:
#: Enumerated rather than pattern-matched for everything because the point is to notice a
#: file nobody expected. The two patterns below cover the only names whose *text* varies -
#: shard files and their index - and they vary in a shape the library fixes.
ALLOWED_FILES: frozenset[str] = frozenset(
    {
        WEIGHTS_NAME,
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        MANIFEST_NAME,
        # The tokenizer's own output, AT THE ROOT. Which of these appear depends on the
        # tokenizer; all of them belong to `tokenizer.save_pretrained`.
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "spiece.model",
        "chat_template.jinja",
        "preprocessor_config.json",
    }
)

#: Files whose presence has a *specific* diagnosis worth giving. Trainer state inside a
#: checkpoint is the one that breaks the output contract while every test of the save path
#: still passes, so it is named rather than reported as "an unexpected file".
TRAINER_STATE_FILES: frozenset[str] = frozenset(
    {"optimizer.pt", "scheduler.pt", "rng.pt", "trainer_state.json"}
)


@dataclass(frozen=True)
class WriteResult:
    """A published checkpoint directory and the manifest that describes it."""

    directory: Path
    manifest: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def weights_sha256(self) -> str:
        return str(self.manifest["weights_sha256"])


def _is_shard(name: str) -> bool:
    """``model-00001-of-00002.safetensors``, the sharded form ``save_pretrained`` emits."""
    return name.startswith("model-") and name.endswith(".safetensors")


def assert_file_set(directory: Path) -> list[str]:
    """Assert the directory is a standard HF model directory and **nothing more**.

    An **equality**, not a subset: the required files are present, and every other entry is
    one ``save_pretrained`` is allowed to have produced. R-02.51's contract is a file *set*,
    so a subset assertion would pass against a directory carrying an extra miForge file that
    makes the artifact non-standard - and against optimizer state, which breaks criterion 23
    without breaking anything the save path would notice.

    Returns:
        The sorted file set, so a caller can log what it published.

    Raises:
        Refusal: ``checkpoint_contains_subdirectory`` for any nested directory - which is
            how a ``tokenizer/`` subdirectory would arrive; ``checkpoint_trainer_state_
            inside`` for optimizer state; ``checkpoint_file_set_unexpected`` otherwise.
    """
    names: list[str] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_dir():
            raise refuse(
                "checkpoint_contains_subdirectory",
                f"{entry.name!r} is a subdirectory inside a checkpoint. A checkpoint is a "
                f"flat standard Hugging Face directory with the tokenizer's files AT THE "
                f"ROOT: as first specified, a 'tokenizer/' subdirectory could not be loaded "
                f"by transformers, vLLM, TGI or miLLM's own loader.",
                entry=entry.name,
                directory=str(directory),
            )
        names.append(entry.name)

    missing = sorted(REQUIRED_FILES - set(names))
    if missing:
        raise refuse(
            "checkpoint_file_set_incomplete",
            f"The checkpoint directory is missing {missing}. Nothing is published that a "
            f"standard loader could not read.",
            missing=missing,
            present=names,
        )

    state = sorted(set(names) & TRAINER_STATE_FILES)
    if state:
        raise refuse(
            "checkpoint_trainer_state_inside",
            f"{state} are trainer-state files and they are inside a checkpoint directory. "
            f"Optimizer state lives under its own root: a checkpoint that gains "
            f"optimizer tensors stops being a standard Hugging Face directory and can no "
            f"longer be loaded by transformers, vLLM, TGI or miLLM's loader.",
            files=state,
        )

    unexpected = sorted(n for n in names if n not in ALLOWED_FILES and not _is_shard(n))
    if unexpected:
        raise refuse(
            "checkpoint_file_set_unexpected",
            f"{unexpected} are in a checkpoint directory and are not part of a standard "
            f"Hugging Face model directory. manifest.json is the ONE non-standard file a "
            f"checkpoint carries and it is purely additive; a second one makes the artifact "
            f"something only miForge understands.",
            unexpected=unexpected,
            allowed=sorted(ALLOWED_FILES),
        )
    return names


def _populate(
    model: Any,
    tokenizer: Any,
    staging: Path,
    fields: ManifestFields,
) -> dict[str, Any]:
    """Fill ``staging`` with the artifact and return the manifest written into it.

    Shared by the publishing path and by :func:`candidate_weights_sha256` so there is one
    definition of *what a checkpoint contains*. The publishing path adds the rename; this
    function never renames anything.
    """
    # Weights and `config.json`. `safe_serialization=True` is the contract, not a default to
    # inherit: a `pytorch_model.bin` is a pickle, and R-02.51 promises an artifact a
    # consumer can load without executing it.
    model.save_pretrained(staging, safe_serialization=True)
    # The tokenizer's own files, AT THE ROOT. `save_pretrained` on the same directory is
    # what puts them there; a `tokenizer/` subdirectory is what `assert_file_set` refuses.
    #
    # THE LAST LINE, not the gate. `None` here was `AttributeError: 'NoneType' object has
    # no attribute 'save_pretrained'` until 2026-10-16 - a traceback at the first save
    # point of every `create` run, hours in, having produced nothing. The *fix* for that is
    # the tokenizer reference on `AcquisitionSpec` (see `training/acquisition/scratch.py`),
    # which refuses at submit; this is the refusal of last resort, with its own code so the
    # two cannot be confused. If it ever fires, miForge reached a save with no tokenizer for
    # a run that was admitted - which is a defect here, not an operator's mistake.
    if tokenizer is None:
        raise refuse(
            "checkpoint_tokenizer_missing",
            "This save was handed no tokenizer, so the checkpoint would be a directory "
            "that transformers, vLLM, TGI and miLLM's loader all refuse. "
            "Nothing is published. A `create` run names its tokenizer "
            "on the model source (`tokenizer_path` or `tokenizer_repo`) and is refused at "
            "submit without one; reaching this line means that gate was bypassed.",
            directory=str(staging),
        )
    tokenizer.save_pretrained(staging)

    weights = staging / WEIGHTS_NAME
    if not weights.is_file():
        raise refuse(
            "checkpoint_weights_not_written",
            f"save_pretrained produced no {WEIGHTS_NAME} in {staging}. Nothing is published "
            f"and no manifest is written: a manifest whose weights_sha256 described a file "
            f"that is not there would be worse than a failed save.",
            directory=str(staging),
            present=sorted(p.name for p in staging.iterdir()),
        )

    # Streamed, through the project's one hashing path (ADR-005, foundation 2.6). Of the
    # bytes ON DISK, after the write, because that is what a consumer will verify - and
    # streamed because a 2.4 GB read into memory to hash it would double the peak footprint
    # on a card already budgeted to the megabyte.
    sha = sha256_of_file(weights)

    doc = manifest_module.build(weights_sha256=sha, **fields.as_kwargs())

    # LAST, after the weights and their hash, so a manifest never describes files that are
    # not there yet. `write_canonical` is the one serialiser (ADR-005): the bytes that are
    # hashed and the bytes that are written come from one function, or they are two answers.
    write_canonical(staging / MANIFEST_NAME, doc)
    assert_file_set(staging)
    return doc


def write(
    model: Any,
    tokenizer: Any,
    *,
    dest: Path,
    fields: ManifestFields,
) -> WriteResult:
    """Publish one checkpoint at ``dest``, atomically. Returns what was written.

    The destination is **absent or complete**: the whole body runs inside
    :func:`core.atomic_dir.atomic_dir`, so any failure removes the staging directory and
    leaves the previous checkpoint and ``latest`` untouched (E-1, E-4, AC8).

    Raises:
        Refusal: ``artifact_already_exists`` from the primitive when ``dest`` is present -
            the resave rule in :mod:`services.checkpoints.service` is what decides what to
            do about that, because the decision needs the candidate's hash and this function
            is past the point of deciding.
    """
    with atomic_dir(dest) as staging:
        doc = _populate(model, tokenizer, staging, fields)

    log.info(
        "checkpoint written",
        extra={
            "code": "checkpoint_written",
            "run_id": fields.run_id,
            "step": fields.step,
            "path": str(dest),
            # `checkpoint`, not `name`: `name` is a reserved LogRecord attribute and
            # logging raises KeyError on an `extra` that carries one.
            "checkpoint": doc["name"],
            "weights_sha256": doc["weights_sha256"],
        },
    )
    return WriteResult(directory=dest, manifest=doc)


def candidate_weights_sha256(
    model: Any,
    tokenizer: Any,
    *,
    parent: Path,
    fields: ManifestFields,
) -> str:
    """Serialise into a throwaway staging directory and return the weights hash.

    **Why a full write to answer a question.** The resave rule (FR-005.6) needs to know
    whether the checkpoint that would be written matches the one already on disk, and the
    hash of ``model.safetensors`` is not knowable without writing it: it is a hash of
    serialised bytes, not of the tensors in memory. The alternatives are both worse - guess
    (a fabricated answer, ADR-018) or remove the old directory and rename over it, which
    reopens exactly the window the atomic write closes.

    So the cost is paid and nothing is published: the staging directory is dot-prefixed, so
    no walk sees it, and it is removed in a ``finally``.
    """
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f"{TMP_PREFIX}{uuid4().hex}"
    staging.mkdir()
    try:
        return str(_populate(model, tokenizer, staging, fields)["weights_sha256"])
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def verify_weights(directory: Path, expected: str) -> str:
    """Recompute ``weights_sha256`` and compare. Returns the computed digest.

    **Never repairs a manifest to match the bytes it found** (FR-005.27). The manifest is
    the record of what was written and the bytes are what is being questioned; rewriting
    the record to agree with the thing under suspicion would destroy the only evidence.

    Raises:
        Refusal: ``checkpoint_not_found`` when there are no weights to hash;
            ``checkpoint_hash_mismatch`` naming both digests.
    """
    weights = directory / WEIGHTS_NAME
    if not weights.is_file():
        raise refuse(
            "checkpoint_not_found",
            f"No {WEIGHTS_NAME} in {directory}, so there is nothing to verify. The "
            f"directory is reported unusable rather than treated as an empty checkpoint.",
            directory=str(directory),
        )
    actual = sha256_of_file(weights)
    if actual != expected:
        raise refuse(
            "checkpoint_hash_mismatch",
            f"{WEIGHTS_NAME} in {directory} hashes to {actual}, but its manifest records "
            f"{expected}. Refusing without changing anything: the manifest is not repaired "
            f"to match the bytes it found. Restore the checkpoint from a copy, or continue "
            f"from an earlier step.",
            directory=str(directory),
            expected=expected,
            actual=actual,
        )
    return actual


def contains_secret_shaped_values(doc: Mapping[str, Any]) -> list[str]:
    """Values in a manifest that look like a path, a host, a URL or a credential.

    Used by the privacy assertion (AC-005.14) rather than by the writer, and it scans the
    **written** document rather than inspecting the builder, because the question is what
    ends up in the artifact. A manifest that named the machine it was written on would stop
    being portable the moment it was copied, and ADR-020 forbids a secret in a manifest at
    all.
    """
    import re

    suspicious: list[str] = []
    pattern = re.compile(
        r"(^/|^[a-zA-Z]:\\|://|\b(?:token|secret|password|apikey|api_key)\b)",
        re.IGNORECASE,
    )

    def scan(value: Any, where: str) -> None:
        if isinstance(value, Mapping):
            for key, inner in value.items():
                if pattern.search(str(key)):
                    suspicious.append(f"{where}.{key}: key name")
                scan(inner, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, inner in enumerate(value):
                scan(inner, f"{where}[{index}]")
        elif isinstance(value, str) and pattern.search(value):
            suspicious.append(f"{where}: {value}")

    for key, value in doc.items():
        scan(value, key)
    return suspicious
