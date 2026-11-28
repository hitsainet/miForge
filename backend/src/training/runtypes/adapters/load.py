"""Read an adapter manifest, and refuse a scaling rule this release does not implement.

## Validate first, then check the value

:func:`load_manifest` validates against the vendored schema and *then* checks ``scaling``
against :func:`training.runtypes.adapters.schema.known_scaling`.

**As of 2026-10-13 that second check is defence in depth, not the enforcement, and this is
worth reading before deciding it is dead code.** The enum refuses ``"rslora"``, every object
in the schema now sets ``additionalProperties: false`` so an invented key is refused too,
and because validation runs first a v1 reader never reaches the check for a v2 document -
``schema_version``'s ``const`` rejects that earlier. The line is kept for exactly two
futures and **no present gap**: a reader that is ever handed a document it did not validate,
and a v2 schema vendored beside this one under which ``schema_version: "2"`` validates and
the enum may admit a rule this release does not implement. Guessing the scale changes the
served model's behaviour silently (ADR-018), so the cheap line stays - but **do not justify
it by claiming the schema misses the present case.**

## The ``.get`` default is the schema's own

``scaling`` is **not** in ``AdapterConfig.required`` (``rank``, ``alpha`` and
``target_modules`` are) and the schema declares ``"default": "alpha_over_r"``, so an absent
key means alpha-over-rank *by contract*. Reading that default is not the fabricated value
ADR-018 forbids; an unrecognised **value** is a different thing and is refused. Stated
because "not alpha-over-rank by default" reads as forbidding this line, and someone would
otherwise turn an absent key into a refusal the contract does not authorise.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.errors import refuse
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import schema as adapter_schema


def read_manifest_document(path: Path) -> dict[str, Any]:
    """Parse one ``adapter_manifest.json``, refusing by name on anything unreadable."""
    if not path.is_file():
        raise refuse(
            "adapter_manifest_missing",
            f"{path} does not exist, so there is no adapter to read. A manifest is not "
            f"reconstructed from a directory listing: the base, the rank and the digest "
            f"are in the document and nowhere else.",
            path=str(path),
        )
    try:
        parsed = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise refuse(
            "adapter_manifest_invalid",
            f"{path} could not be parsed as JSON: {exc}",
            path=str(path),
        ) from exc
    if not isinstance(parsed, Mapping):
        raise refuse(
            "adapter_manifest_invalid",
            f"{path} parses to {type(parsed).__name__}, not an object.",
            path=str(path),
        )
    return dict(parsed)


def assert_scaling_known(doc: Mapping[str, Any]) -> str:
    """Return the manifest's scaling rule, or refuse it **by name**.

    Raises:
        Refusal: ``adapter_scaling_unrecognised``. Refused and never guessed: the scale is
            what multiplies the adapter's contribution, so guessing it changes what the
            model answers with nothing saying so.
    """
    block = doc.get("adapter") or {}
    scaling = block.get("scaling", adapter_schema.scaling_default())
    known = adapter_schema.known_scaling()
    if scaling not in known:
        raise refuse(
            "adapter_scaling_unrecognised",
            f"This release does not implement the scaling rule {scaling!r}, so the "
            f"adapter is refused rather than applied with a guessed scale. Guessing it "
            f"changes what the model answers with nothing saying so.",
            scaling=scaling,
            known=sorted(known),
        )
    return str(scaling)


def load_manifest(path: Path) -> dict[str, Any]:
    """Read, validate, and refuse an unknown scaling rule. The read path's one entry point.

    Raises:
        Refusal: ``adapter_manifest_missing``, ``adapter_manifest_invalid`` (which covers a
            missing or malformed ``weights_sha256`` - the schema requires it, so E-9 is
            caught here rather than by a second rule), ``adapter_scaling_unrecognised``.
    """
    doc = read_manifest_document(path)
    adapter_schema.validate(doc)
    assert_scaling_known(doc)
    return doc


def load_for(run_id: str, step: int, *, root: Path | None = None) -> dict[str, Any]:
    """The manifest of one run's adapter at one step."""
    directory = adapter_paths.adapter_dir(run_id, step, root=root)
    return load_manifest(directory / adapter_paths.MANIFEST_NAME)


def assert_base_matches(doc: Mapping[str, Any], held_base_name: str) -> None:
    """Refuse an adapter offered against a base other than the one it names (E-10, E-13).

    The identity checked is ``base.name``; the **shape** a consumer checks is
    ``adapter.target_modules`` against the resident base, which the schema's own description
    names as the field for it and which BRD-01 R-01.27 makes miLLM's obligation. An adapter
    applied to a different base produces undefined behaviour rather than a worse answer,
    which is why this refuses instead of warning.
    """
    named = (doc.get("base") or {}).get("name")
    if named != held_base_name:
        raise refuse(
            "adapter_base_mismatch",
            f"This adapter was trained against {named!r} and the base in hand is "
            f"{held_base_name!r}. An adapter applied to a different base produces "
            f"undefined behaviour rather than a worse answer, so it is refused rather than "
            f"applied with a warning.",
            adapter=doc.get("name"),
            adapter_base=named,
            held_base=held_base_name,
        )
