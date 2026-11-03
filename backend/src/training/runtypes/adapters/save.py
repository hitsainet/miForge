"""Publish one adapter directory, atomically  (FR-008.26 - FR-008.31; R-02.44, ADR-015).

## F8 is the third **caller** of the atomicity primitive, not the third implementation

``core/atomic_dir.py`` is the one place a directory is renamed into place - F5's checkpoint
writer, F1's trainer-state writer and this module all call it, *"because two implementations
of this guarantee are two answers."* An F8-local temp-dir dance would be the third, and a
test asserts this package contains no ``os.rename``, ``os.replace`` or ``fsync`` call. What
F8 owns inside the staging directory is exactly three things: ``peft``'s
``save_pretrained``, the streamed digest of the file **as written**, and validation. When
the bytes become visible is F5's.

## Three files, and two of them must agree

An adapter directory holds ``adapter_model.safetensors``, ``adapter_config.json`` (PEFT's
own, which ``PeftConfig.from_pretrained`` **reads**) and ``adapter_manifest.json``.

``r`` and ``lora_alpha`` in the PEFT config must match ``rank`` and ``alpha`` in the
manifest, and :func:`assert_config_agrees` is why. **miLLM's Feature 033 loader cross-checks
those two fields and refuses on a mismatch**, so a writer that derived the manifest's
numbers independently of what it handed ``save_pretrained`` would produce a directory that
validates against both schemas, loads under plain ``peft``, and is refused by miLLM -
failing acceptance criterion 15 for a reason **neither schema can see**. Both come from one
:class:`~training.runtypes.adapters.config.AdapterConfigIn`; the check is defence against a
later edit that splits them.

## Validation sits before the hand-back, never after

Once ``atomic_dir`` renames, the directory is published. A manifest that does not conform
must fail the write (FR-008.28), so :func:`training.runtypes.adapters.schema.validate` runs
inside the ``with`` block - and raising inside it discards the staging directory, leaving
``latest`` where it was.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.atomic_dir import atomic_dir
from core.canonical import sha256_of_file, write_canonical
from core.errors import refuse
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import schema as adapter_schema

log = logging.getLogger(__name__)

#: ``adapter_config.json``'s names for the two numbers that must agree with the manifest's.
#: PEFT's spelling, not miForge's, and that is the whole reason a cross-check is needed.
PEFT_RANK_KEY = "r"
PEFT_ALPHA_KEY = "lora_alpha"


def assert_config_agrees(directory: Path, manifest_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Assert ``adapter_config.json`` and the manifest describe the **same** adapter.

    Returns the parsed PEFT config, so a caller can log what was published.

    Raises:
        Refusal: ``adapter_config_missing`` when ``peft`` wrote no configuration - the
            directory would be unloadable by ``PeftConfig.from_pretrained``, which is the
            whole reason the third file exists;
            ``adapter_config_manifest_disagree`` when ``r``/``lora_alpha`` do not match
            ``rank``/``alpha``. **Refused, not reconciled**: the scale is alpha over rank,
            so a disagreement silently changes the served model, and nothing downstream can
            tell which of the two files is right.
    """
    config_path = directory / adapter_paths.PEFT_CONFIG_NAME
    if not config_path.is_file():
        raise refuse(
            "adapter_config_missing",
            f"{adapter_paths.PEFT_CONFIG_NAME} is not in the adapter directory. "
            f"PeftConfig.from_pretrained reads that file, so without it the directory is "
            f"not loadable by standard tooling - the same shape as a "
            f"tokenizer in a subdirectory. Nothing is published.",
            directory=str(directory),
            present=sorted(p.name for p in directory.iterdir()),
        )

    try:
        peft_config = json.loads(config_path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise refuse(
            "adapter_config_missing",
            f"{adapter_paths.PEFT_CONFIG_NAME} could not be parsed: {exc}",
            directory=str(directory),
        ) from exc

    block = manifest_doc.get("adapter") or {}
    disagreements = {
        name: {"adapter_config": peft_config.get(peft_key), "adapter_manifest": block.get(key)}
        for name, peft_key, key in (
            ("rank", PEFT_RANK_KEY, "rank"),
            ("alpha", PEFT_ALPHA_KEY, "alpha"),
        )
        if peft_config.get(peft_key) != block.get(key)
    }
    if disagreements:
        raise refuse(
            "adapter_config_manifest_disagree",
            "adapter_config.json and adapter_manifest.json describe different adapters. "
            "miLLM's loader cross-checks these two fields and refuses on a mismatch, so "
            "this directory would validate against both schemas, load under plain peft, "
            "and be refused where it matters. Both numbers must come from one source; "
            "nothing is published.",
            directory=str(directory),
            disagreements=disagreements,
        )
    return dict(peft_config)


def save(
    adapter_model: Any,
    manifest_doc: dict[str, Any],
    dest: Path,
) -> dict[str, Any]:
    """Write the adapter at ``dest`` and return the manifest **as published**.

    Args:
        adapter_model: anything with ``save_pretrained``. Typed loosely on purpose: a
            ``peft``-wrapped model in production, and a stub in the tests that cover
            atomicity, the digest, the agreement check and validation - all of which must
            be exercisable with no card and no ML stack (ADR-023).
        manifest_doc: from :func:`training.runtypes.adapters.manifest.build`, **without**
            ``weights_sha256``. This function adds it.
        dest: the adapter's directory. Must not already exist.

    Raises:
        Refusal: ``adapter_weights_not_written`` when ``save_pretrained`` produced no
            tensors; ``adapter_config_missing`` / ``adapter_config_manifest_disagree`` from
            :func:`assert_config_agrees`; ``adapter_manifest_invalid`` from validation;
            ``artifact_already_exists`` / ``checkpoint_root_not_atomic`` from the primitive.
    """
    published: dict[str, Any] = {}

    with atomic_dir(dest) as staging:
        adapter_model.save_pretrained(staging)

        weights = staging / adapter_paths.WEIGHTS_NAME
        if not weights.is_file():
            raise refuse(
                "adapter_weights_not_written",
                f"save_pretrained produced no {adapter_paths.WEIGHTS_NAME} in {staging}. "
                f"Nothing is published and no manifest is written: a manifest whose "
                f"weights_sha256 described a file that is not there would be worse than a "
                f"failed save.",
                directory=str(staging),
                present=sorted(p.name for p in staging.iterdir()),
            )

        # Of the bytes ON DISK, after the write, because that is what a consumer verifies
        # (BRD-01 R-01.27) - and streamed, because materialising a second copy competes
        # with the very budget being guarded, on the card that is the constraint.
        manifest_doc["weights_sha256"] = sha256_of_file(weights)

        peft_config = assert_config_agrees(staging, manifest_doc)

        # BEFORE the hand-back. Raising here discards the staging directory and leaves
        # `latest` pointing where it was.
        adapter_schema.validate(manifest_doc)

        write_canonical(staging / adapter_paths.MANIFEST_NAME, manifest_doc)
        published = dict(manifest_doc)

    log.info(
        "adapter saved",
        extra={
            "code": "adapter_saved",
            "run_id": published["run_id"],
            "step": published["step"],
            # `adapter`, not `name`: `name` is a reserved LogRecord attribute and logging
            # raises KeyError on an `extra` that carries one.
            "adapter": published["name"],
            "path": str(dest),
            "weights_sha256": published["weights_sha256"],
            "base": published["base"]["name"],
            "rank": peft_config.get(PEFT_RANK_KEY),
            "validated": True,
        },
    )
    return published
