"""The WSP contracts and their validators.

Loaded from ``docs/schemas/`` (``MIFORGE_SCHEMA_DIR`` in the image).

Owned: ``miforge-wsp-{bundle,patch,run}-v1.json``. Vendored (phase 2):
``miworkbench-wsp-scoring-v1.json``.
A missing schema is refused (``schema_not_vendored``), never replaced by an empty one, which would
make
every validation pass vacuously. Validation errors are refused with the caller's code and the first
error's path, so a refusal names the field.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from core.errors import refuse

_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas"

BUNDLE = "miforge-wsp-bundle-v1.json"
PATCH = "miforge-wsp-patch-v1.json"
RUN = "miforge-wsp-run-v1.json"


def schema_dir() -> Path:
    return Path(os.environ.get("MIFORGE_SCHEMA_DIR", str(_DEFAULT_SCHEMA_DIR)))


@cache
def _validator(name: str, directory: str) -> Draft202012Validator:
    path = Path(directory) / name
    if not path.is_file():
        raise refuse(
            "schema_not_vendored",
            f"The contract {name} is not at {path}; set MIFORGE_SCHEMA_DIR to the directory "
            "holding it.",
            schema=name,
        )
    return Draft202012Validator(
        json.loads(path.read_text(encoding="utf-8")), format_checker=FormatChecker()
    )


def errors(name: str, document: Any) -> list[str]:
    """Every validation error, as ``path: message``, sorted for a stable report."""
    validator = _validator(name, str(schema_dir()))
    return sorted(
        f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
        for e in validator.iter_errors(document)
    )


def validate(name: str, document: Any, *, code: str) -> None:
    found = errors(name, document)
    if found:
        raise refuse(
            code, f"The document does not conform to {name}: {found[0]}", errors=found[:20]
        )


def validate_bundle(document: Any) -> None:
    validate(BUNDLE, document, code="wsp_bundle_invalid")


def validate_patch(document: Any) -> None:
    validate(PATCH, document, code="wsp_patch_malformed")


def validate_run(document: Any) -> None:
    validate(RUN, document, code="wsp_result_invalid")


@cache
def _def_validator(name: str, definition: str, directory: str) -> Draft202012Validator:
    root = _validator(name, directory).schema
    assert isinstance(root, dict)
    return Draft202012Validator(
        {"$defs": root["$defs"], "$ref": f"#/$defs/{definition}"}, format_checker=FormatChecker()
    )


def validate_frozen_request(document: Any) -> None:
    """Validate a ``FrozenRequest`` (the run schema's ``$defs``); refused ``validation_failed``."""
    validator = _def_validator(RUN, "FrozenRequest", str(schema_dir()))
    found = sorted(
        f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
        for e in validator.iter_errors(document)
    )
    if found:
        raise refuse(
            "validation_failed", f"The run request is not valid: {found[0]}", errors=found[:20]
        )


def submitted_check_errors(document: Any) -> list[str]:
    """Every way ``document`` is not a ``SubmittedCheck`` (the run schema's ``$defs``; FU-2)."""
    validator = _def_validator(RUN, "SubmittedCheck", str(schema_dir()))
    return sorted(
        f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
        for e in validator.iter_errors(document)
    )
