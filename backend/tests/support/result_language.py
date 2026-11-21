"""Validate a value against the VENDORED result-language contract (017 FR-017.2).

Every shape test reads ``docs/schemas/mistudio-result-language-v1.json`` through here, so a test
can never agree with the code by holding its own copy of the shape. ``validate("approval", v)``
checks ``v`` against ``#/$defs/approval`` with the file's own ``$defs`` in scope.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED = REPO_ROOT / "docs" / "schemas" / "mistudio-result-language-v1.json"


@cache
def contract() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(VENDORED.read_text(encoding="utf-8"))
    return document


def validate(definition: str, value: Any) -> None:
    """Raise ``jsonschema.ValidationError`` unless ``value`` is a valid ``$defs.<definition>``."""
    import jsonschema

    document = contract()
    schema = {"$defs": document["$defs"], "$ref": f"#/$defs/{definition}"}
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    ).validate(value)


def status_enum() -> list[str]:
    return list(contract()["$defs"]["approval_status"]["enum"])


def error_code_pattern() -> str:
    return str(contract()["$defs"]["error_code"]["pattern"])
