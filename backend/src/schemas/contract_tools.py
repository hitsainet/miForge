# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/schemas/contract_tools.py @ 6b50a8f4
# Mode: adapt (docs/REUSE.md). Kept: every function and the classifier's row names, byte for byte in
# behaviour, so miForge's, miDataworks' and miServer's classifiers agree on shared pairs. Changed:
# the docstrings cite miForge 021's requirements; nothing else.
"""Pure transforms behind the published answer contract (021, FTID §3.1). STANDARD LIBRARY ONLY.

``answers_contract.py`` builds ``docs/schemas/miforge-answers-v1.json`` from the live OpenAPI
document with these functions; ``backend/scripts/generate_output_schemas.py`` uses the reference and
reachability helpers; the test validator (``tests/support/answer_validator.py``) derives its strict
form with ``strict_form``; and the classifier decides whether a new contract is additive
(FR-021.19).

The design follows miServer feature 044 (``miserver/api/schemas/contract_tools.py``) by way of
miDataworks 023 (MF7, MF8; FTDD FT1): the ``required`` rule, the open and strict forms, and the
classifier's row names are the same, so one reader serves every app's contract and the cross-repo
parity test can hold the classifiers to the same answer on shared pairs. The code imports nothing
from a sibling (ADR-001); it imports only the standard library (``tests/contract/
test_contract_tools.py`` holds that), so the parity test can load it in any interpreter.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

#: JSON Schema's key for "what other keys an object may carry".
OPEN_KEY = "additionalProperties"
#: A field some answers leave out (``answers_common.may_be_absent()``); never ``required`` here.
MAY_BE_ABSENT = "x-may-be-absent"
#: A field whose value is a published ``docs/schemas`` document (``answers_common.external_ref()``).
EXTERNAL = "x-external-ref"
#: A field the served model types loosely whose shape another answer model declares: the contract
#: references that model (``#/$defs/<name>``) instead.
MODEL_REF = "x-model-ref"
#: Beside ``EXTERNAL``: the document's top-level properties this answer leaves out.
EXTERNAL_WITHOUT = "x-external-without"
#: A stored value that may predate its model (an approval's ``result``, miServer TD14): the contract
#: keeps its types and drops every ``required`` below it (``answers_common.history_form()``).
HISTORY_FORM = "x-history-form"
#: The suffix of a definition's history copy (not one Pydantic produces, so it cannot collide).
HISTORY_SUFFIX = "__history"
#: A JSON answer's other media types, each with the request that selects it (021 FT4).
MEDIA_ALTERNATIVES = "x-media-alternatives"
#: Where the OpenAPI document keeps its models, and where the contract keeps them.
OPENAPI_REF = "#/components/schemas/"
DEFS_REF = "#/$defs/"
#: The two top-level keys the digest leaves out; the writer sets both (FR-021.19).
UNDIGESTED = ("revision", "content_sha256")


def render(doc: dict[str, Any]) -> str:
    """The committed byte form: sorted keys, two-space indent, UTF-8 text, a trailing newline."""
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def walk(node: Any) -> Iterator[dict[str, Any]]:
    """Every dict at or below ``node``, depth first."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


def rewrite_refs(node: Any, prefix: str = OPENAPI_REF) -> Any:
    """A copy of ``node`` whose ``<prefix>X`` references read ``#/$defs/X``."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str) and value.startswith(prefix):
                out[key] = DEFS_REF + value[len(prefix) :]
            else:
                out[key] = rewrite_refs(value, prefix)
        return out
    if isinstance(node, list):
        return [rewrite_refs(item, prefix) for item in node]
    return node


def refs(node: Any) -> set[str]:
    """The names of every ``#/$defs/<name>`` reference at or below ``node``."""
    return {
        s["$ref"][len(DEFS_REF) :]
        for s in walk(node)
        if isinstance(s.get("$ref"), str) and s["$ref"].startswith(DEFS_REF)
    }


def reachable_defs(roots: list[Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Only the definitions a ``#/$defs/`` reference reaches from ``roots``, sorted by name
    (FR-021.22). A reference to a definition ``defs`` lacks raises ``KeyError``: a dangling
    reference is a generator bug, never something to drop quietly."""
    seen: dict[str, Any] = {}
    todo = sorted(refs(roots))
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen[name] = defs[name]
        todo.extend(sorted(refs(defs[name]) - set(seen)))
    return dict(sorted(seen.items()))


def always_sent_required(node: Any) -> None:
    """In place: every object schema with ``properties`` requires each property not marked
    ``x-may-be-absent``, and the marker is removed. "Required" means "always sent" (MF10); an object
    whose list comes out empty carries no ``required``."""
    for schema in list(walk(node)):
        props = schema.get("properties")
        if not isinstance(props, dict):
            continue
        required = sorted(
            name
            for name, prop in props.items()
            if not (isinstance(prop, dict) and prop.pop(MAY_BE_ABSENT, False))
        )
        if required:
            schema["required"] = required
        else:
            schema.pop("required", None)


def open_objects(node: Any) -> Any:
    """In place: delete every ``additionalProperties: false``, so extra keys never break a client
    (D3). A schema-valued ``additionalProperties`` (a map) stays."""
    for schema in walk(node):
        if schema.get(OPEN_KEY) is False:
            del schema[OPEN_KEY]
    return node


def strict_form(node: Any) -> Any:
    """A COPY in which every object with named ``properties`` refuses keys it does not declare
    (MF11). A map (only ``additionalProperties``) still accepts any key, and a model that declares
    its extras (``additionalProperties: true``) keeps them. The members of ``allOf`` and the
    branches of ``if``/``then``/``else`` are views of one object, so they are not closed
    themselves; the objects nested inside them are."""
    out = copy.deepcopy(node)

    def close(value: Any, view: bool) -> None:
        if isinstance(value, dict):
            if not view and isinstance(value.get("properties"), dict) and OPEN_KEY not in value:
                value[OPEN_KEY] = False
            for key, sub in value.items():
                if key == "properties" and isinstance(sub, dict):
                    for prop in sub.values():
                        close(prop, False)
                elif key == "allOf" and isinstance(sub, list):
                    for member in sub:
                        close(member, True)
                else:
                    close(sub, key in ("if", "then", "else"))
        elif isinstance(value, list):
            for item in value:
                close(item, False)

    close(out, False)
    return out


def deref(schema: Any, defs: dict[str, Any]) -> Any:
    """Follow ``#/$defs/`` references until a schema that is not one (an unknown name is ``{}``)."""
    hops = 0
    while (
        isinstance(schema, dict)
        and isinstance(schema.get("$ref"), str)
        and schema["$ref"].startswith(DEFS_REF)
        and hops <= 50
    ):
        schema = defs.get(schema["$ref"][len(DEFS_REF) :], {})
        hops += 1
    return schema


def _is_null(schema: Any, defs: dict[str, Any]) -> bool:
    resolved = deref(schema, defs)
    return isinstance(resolved, dict) and resolved.get("type") == "null"


def is_typed(
    schema: Any,
    defs: dict[str, Any],
    external: frozenset[str] | set[str] = frozenset(),
    _depth: int = 0,
) -> bool:
    """FR-021.4. Typed: an object with named properties; an array of typed items; a ``oneOf`` or
    ``anyOf`` whose non-null members are all typed; a ``$ref`` to a document listed in
    ``external``. Not typed: ``{}``, ``Any``, an object without properties (``dict[str, Any]`` or
    any other map at the top), an array of untyped items, anything else."""
    if _depth > 40 or not isinstance(schema, dict) or not schema:
        return False
    ref = schema.get("$ref")
    if isinstance(ref, str) and not ref.startswith(DEFS_REF):
        return document_id(ref) in external
    schema = deref(schema, defs)
    if not isinstance(schema, dict) or not schema:
        return False
    ref = schema.get("$ref")
    if isinstance(ref, str) and not ref.startswith(DEFS_REF):
        return document_id(ref) in external
    for key in ("anyOf", "oneOf"):
        if key in schema:
            members = [m for m in schema[key] if not _is_null(m, defs)]
            return bool(members) and all(is_typed(m, defs, external, _depth + 1) for m in members)
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        return True
    if schema.get("type") == "array":
        return is_typed(schema.get("items") or {}, defs, external, _depth + 1)
    return False


def document_id(ref: str) -> str:
    """The document a reference names: ``ref`` without its fragment (``…/x-v1.json#/$defs/A``
    names ``…/x-v1.json``)."""
    return ref.split("#", 1)[0]


def history_form(node: Any, defs: dict[str, Any]) -> Any:
    """miServer TD14: a copy of ``node`` whose every reference points at a ``<Name>__history`` copy
    of its definition, made (in ``defs``) with no ``required`` at any depth. The live definitions
    keep theirs. A reference to a published document is left as it is."""
    out = copy.deepcopy(node)

    def strip(schema: Any) -> None:
        for s in walk(schema):
            if isinstance(s.get("required"), list):  # not a property that is named "required"
                del s["required"]
            ref = s.get("$ref")
            if isinstance(ref, str) and ref.startswith(DEFS_REF):
                name = ref[len(DEFS_REF) :]
                if not name.endswith(HISTORY_SUFFIX):
                    target = name + HISTORY_SUFFIX
                    s["$ref"] = DEFS_REF + target
                    if target not in defs:
                        defs[target] = copy.deepcopy(defs[name])
                        strip(defs[target])

    strip(out)
    return out


def members(schema: Any, defs: dict[str, Any]) -> list[Any]:
    """The non-null alternatives of a schema: itself, or its ``anyOf``/``oneOf`` members,
    flattened through references."""
    resolved = deref(schema, defs)
    for key in ("anyOf", "oneOf"):
        if isinstance(resolved, dict) and key in resolved:
            return [m for member in resolved[key] for m in members(member, defs)]
    return [] if _is_null(resolved, defs) else [schema]


def content_digest(doc: dict[str, Any]) -> str:
    """FR-021.19: SHA-256 of the rendered document with ``revision`` and ``content_sha256`` left
    out, so the pin recomputes it anywhere, the mirror included."""
    body = {k: v for k, v in doc.items() if k not in UNDIGESTED}
    return hashlib.sha256(render(body).encode("utf-8")).hexdigest()


# -- the classifier (FR-021.19; miServer FR-044.16's table: D6 plus SH12) -------------------------

#: The rows that are additive. Every other row is breaking, and so is a change no row names.
ADDITIVE_ROWS = frozenset(
    {
        "new route",
        "new status code",
        "new optional field",
        "new always-sent field",
        "descriptive",
        "vendored schema revised (additive)",
    }
)
#: The fail-closed row.
UNLISTED = "unlisted change"
#: Keys that validate nothing; a change to one is descriptive.
_DESCRIPTIVE = ("description", "title", "$comment")
#: Keys whose change retypes a value.
_RETYPING = ("type", "format")
#: Top-level keys derived from others, never compared on their own.
_DERIVED_TOP = ("coverage", *UNDIGESTED)


@dataclass(frozen=True)
class Change:
    """One difference between two contract documents, and the row of the table it falls under."""

    path: str
    row: str

    @property
    def breaking(self) -> bool:
        return self.row not in ADDITIVE_ROWS

    def __str__(self) -> str:
        return f"{self.path}: {self.row}"


#: ``resolve($id, sha256)`` -> the document with that ``$id`` whose bytes hash to that sha256, or
#: None.
Resolver = Callable[[str, str], "dict[str, Any] | None"]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


class _Classify:
    """One comparison of an old and a new document; ``changes`` collects what it finds."""

    def __init__(self, old: dict[str, Any], new: dict[str, Any], resolve: Resolver | None) -> None:
        self.old_defs: dict[str, Any] = old.get("$defs") or {}
        self.new_defs: dict[str, Any] = new.get("$defs") or {}
        self.resolve = resolve
        self.changes: list[Change] = []
        self.compared: set[str] = set()

    def add(self, path: str, row: str) -> None:
        self.changes.append(Change(path, row))

    # definitions ---------------------------------------------------------------------------------

    def definition(self, name: str) -> None:
        """Compare definition ``name`` once, old against new."""
        if name in self.compared:
            return
        self.compared.add(name)
        if name not in self.new_defs:
            self.add(f"$defs.{name}", "removed")
        elif name in self.old_defs:
            self.schema(self.old_defs[name], self.new_defs[name], f"$defs.{name}")

    def follow(self, node: Any) -> None:
        """Equal text can still reference a definition whose content changed: compare those."""
        for name in sorted(refs(node)):
            self.definition(name)

    # schemas -------------------------------------------------------------------------------------

    def schema(self, old: Any, new: Any, path: str) -> None:
        if old == new:
            self.follow(new)
            return
        if not isinstance(old, dict) or not isinstance(new, dict):
            self.add(path, UNLISTED)
            return
        old_ref, new_ref = old.get("$ref"), new.get("$ref")
        if old_ref != new_ref:
            self.add(f"{path}.$ref", "retyped")
        elif isinstance(old_ref, str) and old_ref.startswith(DEFS_REF):
            self.definition(old_ref[len(DEFS_REF) :])
        for key in sorted(set(old) | set(new)):
            if key == "$ref":
                continue
            a, b = old.get(key), new.get(key)
            sub = f"{path}.{key}"
            if a == b:
                self.follow(b)
            elif key in _DESCRIPTIVE:
                self.add(sub, "descriptive")
            elif key in _RETYPING:
                self.add(sub, "retyped")
            elif key == "properties":
                self.properties(a or {}, b or {}, set(new.get("required") or ()), sub)
            elif key == "required":
                self.required(
                    set(a or ()),
                    set(b or ()),
                    old.get("properties") or {},
                    new.get("properties") or {},
                    path,
                )
            elif key == "enum":
                widened = {_canonical(v) for v in b or ()} - {_canonical(v) for v in a or ()}
                self.add(sub, "enum widened" if widened else UNLISTED)
            elif key == "const":
                self.add(sub, "enum widened")
            elif self._descend(key, a, b):
                self.schema(a, b, sub)
            elif (
                key in ("anyOf", "oneOf", "allOf")
                and isinstance(a, list)
                and isinstance(b, list)
                and len(a) == len(b)
            ):
                for i, (x, y) in enumerate(zip(a, b, strict=True)):
                    self.schema(x, y, f"{sub}[{i}]")
            else:
                self.add(sub, UNLISTED)

    @staticmethod
    def _descend(key: str, a: Any, b: Any) -> bool:
        """Keys whose two values are compared as schemas of their own."""
        if key == "items":
            return True
        both = isinstance(a, dict) and isinstance(b, dict)
        return both and key in (OPEN_KEY, "if", "then", "else")

    def properties(
        self, a: dict[str, Any], b: dict[str, Any], new_required: set[str], path: str
    ) -> None:
        for name in sorted(set(a) | set(b)):
            sub = f"{path}.{name}"
            if name not in b:
                self.add(sub, "removed")
            elif name not in a:
                row = "new always-sent field" if name in new_required else "new optional field"
                self.add(sub, row)
                self.follow(b[name])
            else:
                self.schema(a[name], b[name], sub)

    def required(
        self,
        a: set[str],
        b: set[str],
        old_props: dict[str, Any],
        new_props: dict[str, Any],
        path: str,
    ) -> None:
        # A name new to ``properties`` is classified there; here only an existing field changes.
        for name in sorted(b - a):
            if name in old_props:
                self.add(f"{path}.required.{name}", "made required")
        for name in sorted(a - b):
            if name in new_props:
                self.add(f"{path}.required.{name}", "sometimes absent")

    # the document -------------------------------------------------------------------------------

    def routes(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        old_r, new_r = old.get("responses") or {}, new.get("responses") or {}
        old_x, new_x = old.get("excluded") or {}, new.get("excluded") or {}
        for key in sorted(set(old_r) | set(new_r) | set(old_x) | set(new_x)):
            was = "responses" if key in old_r else "excluded" if key in old_x else None
            now = "responses" if key in new_r else "excluded" if key in new_x else None
            if was and now and was != now:
                self.add(f"{now}.{key}", "moved between responses and excluded")
            elif was is None:
                self.add(f"{now}.{key}", "new route")
            elif now is None:
                self.add(f"{was}.{key}", "removed")
            elif now == "excluded":
                if old_x[key] != new_x[key]:
                    self.add(f"excluded.{key}", UNLISTED)
            else:
                self.statuses(old_r[key], new_r[key], f"responses.{key}")

    def statuses(self, a: dict[str, Any], b: dict[str, Any], path: str) -> None:
        for status in sorted(set(a) | set(b)):
            sub = f"{path}.{status}"
            if not status[:1].isdigit():
                # ``x-approval-action`` and any other annotation of the entry.
                if a.get(status) != b.get(status):
                    self.add(sub, UNLISTED)
            elif status not in b:
                self.add(sub, "removed")
            elif status not in a:
                self.add(sub, "new status code")
                self.follow(b[status])
            else:
                self.schema(a[status], b[status], sub)

    def external(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        """A referenced document removed is breaking; a new one is descriptive (its ``$ref`` site
        is classified where it appears); a moved file is unlisted. The same ``$id`` and file with a
        new sha256 is classified by comparing the two documents themselves: all additive is one
        ``vendored schema revised (additive)`` change, anything else is breaking. A document that
        cannot be read (no resolver, or no bytes with that sha256) fails closed."""
        for sid in sorted(set(a) | set(b)):
            path = f"external.{sid}"
            if sid not in b:
                self.add(path, "removed")
            elif sid not in a:
                self.add(path, "descriptive")
            elif a[sid] != b[sid]:
                if a[sid].get("file") != b[sid].get("file"):
                    self.add(f"{path}.file", UNLISTED)
                    continue
                old_doc = self.resolve(sid, a[sid]["sha256"]) if self.resolve else None
                new_doc = self.resolve(sid, b[sid]["sha256"]) if self.resolve else None
                if old_doc is None or new_doc is None:
                    self.add(f"{path}.sha256", UNLISTED)
                    continue
                inner = _Classify(old_doc, new_doc, None)
                inner.schema(_without_defs(old_doc), _without_defs(new_doc), path)
                inner.sweep()
                bad = [c for c in inner.changes if c.breaking]
                if bad:
                    self.changes.extend(bad)
                else:
                    self.add(f"{path}.sha256", "vendored schema revised (additive)")

    def errors(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        for name in sorted(set(a) | set(b)):
            if name not in b:
                self.add(f"errors.{name}", "removed")
            elif name not in a:
                self.add(f"errors.{name}", UNLISTED)
            else:
                self.schema(a[name], b[name], f"errors.{name}")

    def sweep(self) -> None:
        """Compare every definition both sides hold that no path above reached."""
        for name in sorted(set(self.old_defs) & set(self.new_defs)):
            self.definition(name)

    def top(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        self.routes(old, new)
        for key in sorted(set(old) | set(new)):
            if key in ("responses", "excluded", "$defs", *_DERIVED_TOP):
                continue
            a, b = old.get(key), new.get(key)
            if a == b:
                if key == "errors":
                    self.follow(b)
            elif key in _DESCRIPTIVE:
                self.add(key, "descriptive")
            elif key == "external" and isinstance(a, dict) and isinstance(b, dict):
                self.external(a, b)
            elif key == "errors" and isinstance(a, dict) and isinstance(b, dict):
                self.errors(a, b)
            else:
                self.add(key, UNLISTED)
        self.sweep()
        for name in sorted(set(self.new_defs) - set(self.old_defs)):
            self.add(f"$defs.{name}", "descriptive")
        for name in sorted(set(self.old_defs) - set(self.new_defs)):
            if not any(c.path == f"$defs.{name}" for c in self.changes):
                self.add(f"$defs.{name}", "removed")


def _without_defs(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "$defs"}


def classify(
    old: dict[str, Any], new: dict[str, Any], resolve: Resolver | None = None
) -> list[Change]:
    """Every difference between two contract documents, each mapped onto a row of the table. A
    difference no row names is ``unlisted change``, which is breaking (fail closed). A rename is a
    removal plus an addition. ``resolve`` reads a referenced document by ``$id`` and sha256, for an
    ``external`` entry whose bytes changed."""
    run = _Classify(old, new, resolve)
    run.top(old, new)
    return run.changes


def breaking(changes: list[Change]) -> list[Change]:
    """The breaking subset of ``changes``."""
    return [c for c in changes if c.breaking]


def git_resolver(root: Any, files: dict[str, str]) -> Resolver:
    """A ``Resolver`` over a git checkout: ``files`` maps each ``$id`` to its path under ``root``;
    the bytes are looked for in the working tree, then in every commit that touched the file, by
    sha256. It runs the ``git`` command (standard library only)."""
    import subprocess
    from pathlib import Path

    def resolve(sid: str, sha: str) -> dict[str, Any] | None:
        rel = files.get(sid)
        if rel is None:
            return None
        here = Path(root) / rel
        if here.is_file():
            raw = here.read_bytes()
            if hashlib.sha256(raw).hexdigest() == sha:
                found: dict[str, Any] = json.loads(raw)
                return found
        try:
            revs = subprocess.run(
                ["git", "rev-list", "--all", "--", rel],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout.split()
        except (OSError, subprocess.SubprocessError):
            return None
        for rev in revs:
            shown = subprocess.run(
                ["git", "show", f"{rev}:{rel}"],
                cwd=root,
                capture_output=True,
                timeout=60,
            )
            if shown.returncode == 0 and hashlib.sha256(shown.stdout).hexdigest() == sha:
                parsed: dict[str, Any] = json.loads(shown.stdout)
                return parsed
        return None

    return resolve
