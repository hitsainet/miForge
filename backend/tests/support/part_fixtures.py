"""Part fixtures that round-trip through the real writer and reader  (F10; ADR-016).

**The valid manifest is miDataworks'**, taken byte-for-byte from their own fixture builder -
see ``fixtures/midataworks_export/PROVENANCE.md``. A hand-written one would agree with a
buggy validator, because both would encode the same misreading of the schema.

Everything here derives from that document by **recorded** perturbations:

* :func:`miforge_set_manifest` swaps ``target`` for one miForge consumes and recomputes the
  split digests against bytes that actually exist. Their committed fixture carries ``"a" *
  64`` placeholders, and a digest that cannot match real bytes would make the *tampered
  byte refuses* test - the one that matters most - impossible to write.
* :func:`write_export` writes a real export directory: the manifest under its **instance**
  filename ``midataworks-dataset-version.json`` (no ``-v1``) beside one file per split, which
  is the layout miDataworks' export service produces.
* :func:`perturbations` generates the invalid variants **from the schema's own ``required``
  lists**, so a new required field upstream becomes a new case without anyone editing this
  file.

Nothing here builds a part row by hand. A part's assertions come from importing through the
real ingest path and reading back through the real index - *round trip or it did not happen.*
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "midataworks_export"
MANIFEST_FIXTURE = FIXTURE_DIR / "midataworks-dataset-version.json"


def real_manifest() -> dict[str, Any]:
    """miDataworks' own document, parsed. A deep copy per caller, so a test may break it."""
    return json.loads(MANIFEST_FIXTURE.read_text("utf-8"))


def real_manifest_bytes() -> bytes:
    """The fixture's bytes exactly as committed - for the contract round-trip assertion."""
    return MANIFEST_FIXTURE.read_bytes()


def miforge_set_manifest(
    *,
    miforge_set_kind: str = "prompt_set",
    reference: tuple[str, str, str] | None = ("prompt-set", "support-tickets", "3"),
    split_bodies: dict[str, bytes] | None = None,
    held_out: dict[str, bool] | None = None,
    flat_paths: bool = True,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """A ``miforge_set`` version derived from the real document, plus its split bytes.

    Args:
        miforge_set_kind: One of the contract's five set kinds.
        reference: ``(type, name, version)``, or **``None`` for a conformant version
            carrying no contract reference at all** - which is the case E-21 is about and
            which must import with its three reference columns null.
        split_bodies: ``name -> bytes`` to write. Defaults to one line per declared row so
            the bytes differ between splits and a digest cannot match the wrong file.
        held_out: ``name -> bool`` overriding the fixture's flags, for the reader's rule.
        flat_paths: Rewrite ``splits[].path`` to its basename, which is **what a real
            export carries**. Verified in their ``services/exports/export_service.py``:
            the manifest it writes sets ``"path": data_path.name`` and the export record's
            ``files[].path`` is ``path.name`` too, matching their single-segment
            ``GET /exports/{id}/files/{name}`` route. The committed fixture came from their
            *publishing* builder, whose paths are the **Hub repository** layout
            (``data/train.parquet``) - a different thing, and one the HTTP acquisition path
            could not fetch because a nested name does not match that route. Pass
            ``False`` to keep the repository layout, which is what a cloned Hub repo looks
            like and which ``from_dir`` must also handle.

    Returns:
        ``(manifest, split_bodies)``. The manifest's ``sha256``, ``bytes`` and ``rows`` are
        computed from ``split_bodies``, so the digests are real.
    """
    doc = real_manifest()
    doc["target"] = {
        "kind": "miforge_set",
        "dataset_target_type": None,
        "trl_type": None,
        "trl_version": None,
        "detector_role": None,
        "miforge_set_kind": miforge_set_kind,
        "contract_reference": (
            None
            if reference is None
            else {"type": reference[0], "name": reference[1], "version": reference[2]}
        ),
    }

    bodies: dict[str, bytes] = {}
    for split in doc["content"]["splits"]:
        name = str(split["name"])
        if flat_paths:
            split["path"] = str(split["path"]).rsplit("/", 1)[-1]
        if split_bodies is not None and name in split_bodies:
            body = split_bodies[name]
        else:
            body = b"".join(f'{{"prompt": "{name}-{i}"}}\n'.encode() for i in range(3))
        bodies[name] = body
        split["sha256"] = hashlib.sha256(body).hexdigest()
        split["bytes"] = len(body)
        split["rows"] = body.count(b"\n")
        if held_out is not None and name in held_out:
            split["held_out"] = held_out[name]
    return doc, bodies


def config_manifest(
    *, kind: str = "selector", body: bytes = b'{"threshold": 0.5}'
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """A ``selector_config`` / ``grader_config`` version derived from the real document.

    ``content`` is replaced wholesale because the contract's ``content`` is a **discriminated
    union** on ``content_kind``: a config version has no ``splits`` and a rows version has no
    ``config``, so there is no half-way document to derive.
    """
    doc = real_manifest()
    doc["target"] = {
        "kind": f"{kind}_config",
        "dataset_target_type": None,
        "trl_type": None,
        "trl_version": None,
        "detector_role": None,
        "miforge_set_kind": None,
        "contract_reference": {"type": kind, "name": f"{kind}-strict", "version": "2"},
    }
    doc["content"] = {
        "content_kind": "config",
        "config": {
            "kind": kind,
            "name": f"{kind}-strict",
            "number": 2,
            "plugin": "rule",
            "body_format": "json",
            "sha256": hashlib.sha256(body).hexdigest(),
            "path": f"config/{kind}.json",
        },
    }
    return doc, {f"config/{kind}.json": body}


def write_export(directory: Path, manifest: dict[str, Any], bodies: dict[str, bytes]) -> Path:
    """Write an export directory in miDataworks' layout. Returns the directory.

    The manifest filename is the **instance** name - ``midataworks-dataset-version.json``,
    no ``-v1`` - verified in their ``services/publishing/manifest_builder.py``
    (``MANIFEST_FILE``). Hard-coding the schema's filename instead fails only at
    integration time, with a real export in front of you.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        target = directory / _path_for(manifest, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    (directory / "midataworks-dataset-version.json").write_bytes(
        json.dumps(manifest, indent=2).encode("utf-8")
    )
    return directory


def _path_for(manifest: dict[str, Any], name: str) -> str:
    """The relative path the manifest gives for a split or config body named ``name``."""
    content = manifest["content"]
    if content["content_kind"] == "rows":
        for split in content["splits"]:
            if split["name"] == name:
                return str(split["path"])
    elif content["content_kind"] == "config":
        return str(content["config"]["path"])
    return name


def required_pointers(schema: dict[str, Any]) -> Iterator[tuple[str, list[str]]]:
    """Every ``(definition, required field)`` pair in the vendored schema.

    Drives the perturbation cases from the **schema's own** ``required`` lists, so a field
    that becomes required upstream produces a new test case automatically. A hand-written
    list of fields to delete is a third description of the contract and drifts from both.
    """
    yield "<root>", list(schema.get("required", []))
    for name, definition in schema.get("$defs", {}).items():
        if isinstance(definition, dict) and definition.get("required"):
            yield name, list(definition["required"])


def without(document: dict[str, Any], *path: str) -> dict[str, Any]:
    """A deep copy of ``document`` with the key at ``path`` removed.

    One perturbation per variant, which is the whole discipline: a fixture that differs from
    the valid one in three ways cannot tell you which of the three the refusal was about.
    """
    out = copy.deepcopy(document)
    node: Any = out
    for key in path[:-1]:
        node = node[key]
    node.pop(path[-1], None)
    return out


def tamper_split(directory: Path, manifest: dict[str, Any], name: str) -> None:
    """Change one byte of a split file, leaving the manifest's digest as it was.

    The shape of a real corruption: the manifest still says what miDataworks published and
    the bytes no longer match it. Appending rather than rewriting, so the failure is a
    digest mismatch and not a short read.
    """
    path = directory / _path_for(manifest, name)
    path.write_bytes(path.read_bytes() + b"x")
