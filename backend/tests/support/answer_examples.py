"""A minimal valid answer for any contract route, built from the contract itself (021 FR-021.14).

The MCP harness's recording client answers each call with :func:`example_answer` of the route the
call reaches, instead of a body that no route sends: so every fake answer is one the route can give,
and ``check_fake_answer`` holds every hand-written fake to the same rule. The example fills every
``required`` property with the simplest value its schema accepts (a const, the first enum value, a
sample for each pattern the schemas use, an empty map, the fewest items), resolving ``$ref`` through
the contract and the documents it references. ``tests/unit/support/test_answer_examples.py`` checks
that the example of every route validates.
"""

from __future__ import annotations

from typing import Any

from tests.support import answer_validator

#: A value for each ``pattern`` the contract and its documents use.
PATTERN_SAMPLES = {
    "^[0-9a-f]{64}$": "0" * 64,
    "^[0-9a-f]{32}$": "0" * 32,
    "^[a-z0-9][a-z0-9-]{0,63}$": "example",
    "^[A-Za-z0-9_.\\-]{1,128}$": "example",
    "^[A-Za-z0-9_.\\-]{1,256}$": "example",
    "^wsb_[0-9a-z]{6,40}$": "wsb_example",
    "^(parent|candidate|control|control:[0-9]+|baseline:[a-z_]+)$": "candidate",
    "^wspc_[0-9a-f]{12}$": "wspc_000000000000",
    "^wsp_[0-9a-f]{12}$": "wsp_000000000000",
}
DATE_TIME = "2026-11-26T00:00:00+00:00"
PENDING = "#/$defs/ApprovalPending"


class _Builder:
    def __init__(self) -> None:
        check = answer_validator.answer_check()
        self.registry = check.registry
        self.base = check.contract["$id"]

    def build(self, schema: Any, resolver: Any, depth: int = 0) -> Any:
        if depth > 60 or not isinstance(schema, dict):
            return None
        if "const" in schema:
            return schema["const"]
        if "enum" in schema:
            return schema["enum"][0]
        if "$ref" in schema:
            resolved = resolver.lookup(schema["$ref"])
            return self.build(resolved.contents, resolved.resolver, depth + 1)
        for key in ("anyOf", "oneOf"):
            if key in schema:
                members = [m for m in schema[key] if not _is_null(m)]
                return self.build(members[0], resolver, depth + 1) if members else None
        if "allOf" in schema:
            out: Any = None
            for member in schema["allOf"]:
                if "if" in member and set(member) <= {"if", "then", "else"}:
                    continue  # a conditional branch: the example takes the unconditioned shape
                part = self.build(member, resolver, depth + 1)
                out = {**(out or {}), **part} if isinstance(part, dict) else part
            own = {k: v for k, v in schema.items() if k != "allOf"}
            if own.get("properties") or own.get("type"):
                base = self.build(own, resolver, depth + 1)
                if isinstance(base, dict) and isinstance(out, dict):
                    return {**base, **out}
                return base if out is None else out
            return out
        kind = schema.get("type")
        if isinstance(kind, list):
            kind = next((k for k in kind if k != "null"), "null")
        if kind is None and "properties" in schema:
            kind = "object"
        if kind == "object":
            props = schema.get("properties") or {}
            return {
                name: self.build(props[name], resolver, depth + 1)
                for name in schema.get("required", [])
                if name in props
            }
        if kind == "array":
            count = int(schema.get("minItems", 0))
            return [
                self.build(schema.get("items") or {}, resolver, depth + 1) for _ in range(count)
            ]
        if kind == "string":
            if "pattern" in schema:
                return PATTERN_SAMPLES[schema["pattern"]]
            if schema.get("format") == "date-time":
                return DATE_TIME
            return "x" * max(1, int(schema.get("minLength", 1)))
        if kind in ("integer", "number"):
            low = schema.get("minimum", schema.get("exclusiveMinimum", 0))
            value = max(0, low) + (1 if "exclusiveMinimum" in schema else 0)
            high = schema.get("exclusiveMaximum", schema.get("maximum"))
            if high is not None and value >= high and kind == "number":
                value = (max(0, low) + high) / 2
            return int(value) if kind == "integer" else float(value)
        if kind == "boolean":
            return False
        return None


def _is_null(schema: Any) -> bool:
    return isinstance(schema, dict) and schema.get("type") == "null"


def example_answer(key: str, status: str | None = None) -> Any:
    """A minimal valid answer of contract route ``key`` at ``status`` (default: its first own
    2xx answer, not the gate's pending approval)."""
    from tests.support.answer_validator import _pointer

    contract = answer_validator.answer_check().contract
    entry = contract["responses"][key]
    statuses = [s for s in sorted(entry) if s[:1] == "2"]
    chosen = status or statuses[0]
    builder = _Builder()
    resolver = builder.registry.resolver(base_uri=builder.base)
    schema = {"$ref": builder.base + "#" + _pointer("responses", key, chosen)}
    target = resolver.lookup(schema["$ref"])
    content = target.contents
    while set(content) <= {"$ref", "description", "title"} and "$ref" in content:
        target = target.resolver.lookup(content["$ref"])  # to the union a PendingOr<Own> holds
        content = target.contents
    members = [m for m in content.get("anyOf", [content]) if m.get("$ref") != PENDING] or [content]
    return builder.build(members[0] if len(members) == 1 else {"anyOf": members}, target.resolver)


def example_of(definition: str) -> Any:
    """A minimal valid instance of the contract's ``$defs`` entry ``definition``."""
    builder = _Builder()
    resolver = builder.registry.resolver(base_uri=builder.base)
    target = resolver.lookup(builder.base + "#/$defs/" + definition)
    return builder.build(target.contents, target.resolver)
