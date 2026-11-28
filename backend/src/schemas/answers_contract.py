# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/schemas/answers_contract.py @ 6b50a8f4
# Mode: adapt (docs/REUSE.md). Kept: the layout, the transforms and their order, the writer's
# refusal, the digest and the staleness messages. Changed: miForge's names and command; the routes
# come from api.route_table.mounted_api_routes(create_app()) (FastAPI 0.159's RouteContext has no
# include_in_schema, so the declared route's is read); x-approval-escalation is copied beside
# x-approval-action; the `attachment:` exception is kept but unused (021 FT4 types document bytes
# as $refs instead).
"""The published answer contract, DERIVED from the live route table (021; BRD-MIFORGE-SHAPES-001).

``docs/schemas/miforge-answers-v1.json`` (``miforge.answer-contract/v1``) describes every answer
miForge serves, in miServer 044's layout (MF7):

- ``responses``: ``"<METHOD> <path>"`` -> each declared 2xx status -> that answer's JSON body
  schema, and ``x-approval-action`` (with ``x-approval-escalation`` where set) on a gated route;
- ``excluded``: each route with no JSON body, with the media type its declaration names and why (a
  ``204`` is ``no body``);
- ``errors``: ``app``, the one error envelope (``core/errors.py``);
- ``external``: each published ``docs/schemas`` document an answer references, with its file and
  sha256 (MF15);
- ``coverage``: the counts (FR-021.16); ``$defs``: only the definitions something reaches
  (FR-021.22);
- ``revision`` and ``content_sha256``, which only the writer sets (FR-021.19).

It is generated from ``create_app().openapi()``, never written by hand, and pinned by
``tests/contract/test_answers_contract_sync.py``. Write and check it with the output-schema
generator (IC-2)::

    PYTHONPATH=backend/src:backend python backend/scripts/generate_output_schemas.py --write
    PYTHONPATH=backend/src:backend python backend/scripts/generate_output_schemas.py --check

The writer (``write``) refuses a breaking change (FR-021.19): that needs ``miforge-answers-v2.json``
beside this file. Nothing at run time reads the contract (FT1).

Transforms, in order (FTDD §6.2): references rewritten to ``#/$defs/``; ``external_ref`` fields
replaced by a reference to their document; every always-sent field ``required`` (MF10); no
``additionalProperties: false``; external references checked against ``docs/schemas``; only
reachable definitions kept; coverage counted; the digest computed by the writer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import contract_tools as tools

CONTRACT_FILENAME = "miforge-answers-v1.json"
#: ``backend/src/schemas/answers_contract.py`` -> the repository root is three levels up.
#: The repository's ``docs/schemas`` (the writer's default target and the pin's file). The documents
#: the contract references are read through :func:`schema_dir`, which honours
#: ``MIFORGE_SCHEMA_DIR`` as every other site does (``tests/unit/test_schema_dir_override.py``).
_SCHEMA_DIR_ENV = "MIFORGE_SCHEMA_DIR"
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas"
CONTRACT_PATH = SCHEMAS_DIR / CONTRACT_FILENAME


def schema_dir() -> Path:
    """The published-contract directory, honouring ``MIFORGE_SCHEMA_DIR``."""
    return Path(os.environ.get(_SCHEMA_DIR_ENV, str(SCHEMAS_DIR)))


ID = "https://hitsai.net/schemas/" + CONTRACT_FILENAME
KIND = "miforge.answer-contract/v1"
NEXT_MAJOR = "miforge-answers-v2.json"
REGENERATE = (
    "PYTHONPATH=backend/src:backend python backend/scripts/generate_output_schemas.py --write"
)

_JSON = "application/json"
#: The one non-2xx status a route declares a body for: health's 503 (``api/health.py``).
_ALSO_DECLARED = ("503",)
#: The statuses of an entry that answer no body.
NO_BODY = "x-no-body"
#: What an approval's stored ``result`` is when its route answered a bare ``Response`` (a 204):
#: ``services/approval_service.py:_result_of``.
_STATUS_ONLY_RESULT = {
    "type": "object",
    "properties": {"status_code": {"type": "integer"}},
    "required": ["status_code"],
}
_PENDING = "#/$defs/ApprovalPending"
_ATTACHMENT = "attachment:"
_DEFAULT_DESCRIPTION = "Successful Response"
_EXCLUDE_FLAGS = (
    "response_model_exclude_none",
    "response_model_exclude_unset",
    "response_model_exclude_defaults",
)
#: The one exception to "required means always sent", stated in the file (miServer TD14).
HISTORY_EXCEPTION = (
    "The one exception to 'required means always sent': an approval's stored `result` is checked "
    "in its HISTORY FORM. Each variant keeps its model's properties and types but has no "
    "`required` at any depth, because a stored row can predate a field. The variant is chosen by "
    "`action`; an action that is no longer listed leaves `result` unconstrained."
)
_DESCRIPTION = (
    "GENERATED from miForge's registered routes (their response declarations, through FastAPI's "
    "OpenAPI document); do not edit by hand. Regenerate with `" + REGENERATE + "`; the writer "
    "refuses a breaking change, which needs " + NEXT_MAJOR + " beside this file. Keys of "
    "`responses` are '<METHOD> <path>'; each value maps a 2xx status to that answer's JSON body "
    "schema (health also declares its 503), `x-no-body` lists a status that answers no body "
    "beside them, and `x-media-alternatives` names each other media type a JSON route can answer "
    "and the request that selects it. `required` lists every field a route always sends; extra "
    "keys are allowed. `excluded` lists each route with no JSON body, with its media type and "
    "why. `coverage` counts "
    "the `responses` entries whose every body is typed (`typed`) or not (`untyped`), the body "
    "schemas with more than one alternative (`unions`), the entries that reference a published "
    "document (`external`), and the `excluded` routes. " + HISTORY_EXCEPTION
)


class BreakingAnswerChange(Exception):  # noqa: N818 - the name the suite's contract writers share
    """The writer met a change the classifier calls breaking (or one no row names)."""

    def __init__(self, changes: list[tools.Change]) -> None:
        self.changes = changes
        lines = "\n".join(f"  {c}" for c in changes)
        super().__init__(
            f"{len(changes)} breaking change(s) to {CONTRACT_FILENAME}:\n{lines}\n"
            f"a breaking change needs {NEXT_MAJOR} beside v1 (D6, MF8)"
        )


def _external_files() -> dict[str, tuple[str, str]]:
    """Every published document schema under ``docs/schemas`` by its ``$id``: (file, sha256)."""
    out: dict[str, tuple[str, str]] = {}
    for path in sorted(schema_dir().glob("*.json")):
        if path.name == CONTRACT_FILENAME:
            continue
        raw = path.read_bytes()
        try:
            doc = json.loads(raw)
        except ValueError:
            continue
        sid = doc.get("$id") if isinstance(doc, dict) else None
        if isinstance(sid, str):
            out[sid] = (f"docs/schemas/{path.name}", hashlib.sha256(raw).hexdigest())
    return out


def _error_defs(components: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``errors.app`` and the definitions it needs, from Pydantic's serialization schema. No route
    declares the envelope, so the served components do not move; a name that collides with a
    served component fails closed."""
    from pydantic.json_schema import models_json_schema

    from .answers_common import ErrorEnvelope

    _, schema = models_json_schema(
        [(ErrorEnvelope, "serialization")], ref_template=tools.OPENAPI_REF + "{model}"
    )
    extra: dict[str, Any] = schema.get("$defs", {})
    clash = sorted(name for name in extra if name in components and components[name] != extra[name])
    if clash:
        raise ValueError(f"error-envelope definitions clash with served components: {clash}")
    return {"app": {"$ref": tools.OPENAPI_REF + "ErrorEnvelope"}}, extra


def _reason(response: dict[str, Any], media: str, content: dict[str, Any]) -> str:
    reason = content.get("x-reason")
    if isinstance(reason, str) and reason:
        return reason
    text = response.get("description")
    if isinstance(text, str) and text and text != _DEFAULT_DESCRIPTION:
        return text
    return f"the route answers {media}, not JSON"


def _resolve_external_markers(node: Any) -> None:
    """In place: a schema carrying ``x-external-ref`` becomes a ``$ref`` to that document's ``$id``
    (its description kept). With ``x-external-without``, it becomes the document's top-level
    object without those properties: each other property a ``$ref`` to the document's own, and
    only what the document requires marked always sent."""
    for schema in list(tools.walk(node)):
        model = schema.pop(tools.MODEL_REF, None)
        if model is not None:
            description = schema.get("description")
            schema.clear()
            schema["$ref"] = tools.DEFS_REF + model
            if description:
                schema["description"] = description
            continue
        sid = schema.pop(tools.EXTERNAL, None)
        if sid is None:
            continue
        without = schema.pop(tools.EXTERNAL_WITHOUT, None)
        description = schema.get("description")
        schema.clear()
        if without:
            schema.update(_document_without(sid, set(without)))
        else:
            schema["$ref"] = sid
        if description:
            schema["description"] = description


def _document_without(sid: str, without: set[str]) -> dict[str, Any]:
    """The top level of the published document ``sid`` without the properties ``without``."""
    file = _external_files().get(sid)
    if file is None:
        raise ValueError(f"unknown external $ref {sid!r}: no docs/schemas/*.json has that $id")
    doc = json.loads((schema_dir() / Path(file[0]).name).read_text(encoding="utf-8"))
    props = doc.get("properties")
    if doc.get("type") != "object" or not isinstance(props, dict) or not without <= set(props):
        raise ValueError(f"{sid} has no top-level properties {sorted(without)} to leave out")
    required = set(doc.get("required") or ())
    out: dict[str, Any] = {"type": "object", "properties": {}}
    for name in sorted(set(props) - without):
        prop: dict[str, Any] = {"$ref": f"{sid}#/properties/{name}"}
        if name not in required:
            prop[tools.MAY_BE_ABSENT] = True
        out["properties"][name] = prop
    return out


#: What FastAPI merges into a declared ``responses`` model from a route's inferred
#: ``dict[str, Any]`` response model (MF9 as amended 2026-11-26: typing never changes a serializer).
#: The same rule as miDataworks 023's ``answers_contract._documented`` (``feat/023-typed-answers``).
_INFERRED_DICT = {"type": "object", "additionalProperties": True}


def _documented(schema: dict[str, Any]) -> dict[str, Any]:
    """The declared answer of a documented-only route. Such a route keeps its inferred
    ``dict[str, Any]`` response model, so its served bytes are unchanged; FastAPI then merges that
    model's ``type``, ``additionalProperties`` and ``title`` beside the declared ``$ref`` or union.
    Those siblings describe the run-time serializer, not the answer: they are dropped here."""
    declared = "$ref" in schema or "anyOf" in schema or "oneOf" in schema
    merged = all(schema.get(k) == v for k, v in _INFERRED_DICT.items())
    # An endpoint annotated ``-> Any`` infers ``Any``: FastAPI merges only its generated title.
    any_merged = set(schema) - {"$ref", "anyOf", "oneOf", "description"} == {"title"}
    if declared and (merged or any_merged):
        return {k: v for k, v in schema.items() if k not in (*_INFERRED_DICT, "title")}
    return schema


def _operation_entry(
    key: str, operation: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The ``responses`` entry, or the ``excluded`` entry, of one operation."""
    entry: dict[str, Any] = {}
    other: dict[str, str] = {}
    alternatives: dict[str, str] = {}
    attachment: str | None = None
    no_body: list[str] = []
    for status, response in sorted(operation.get("responses", {}).items()):
        if not (status.startswith("2") or status in _ALSO_DECLARED):
            continue
        content = response.get("content") or {}
        if not content:
            no_body.append(status)
        alternatives.update(response.get(tools.MEDIA_ALTERNATIVES) or {})
        for media, body in sorted(content.items()):
            reason = body.get("x-reason")
            if media == _JSON and isinstance(reason, str) and reason.startswith(_ATTACHMENT):
                attachment = reason
            elif media == _JSON:
                entry[status] = _documented(tools.rewrite_refs(body.get("schema", {})))
            else:
                other[media] = _reason(response, media, body)
    if entry:
        if attachment or set(other) != set(alternatives):
            raise ValueError(
                f"{key} declares a JSON body beside a non-JSON answer that names no request "
                f"selecting it ({sorted(set(other) ^ set(alternatives))}); declare it with "
                "answers_common.media_alternatives, or one answer per route"
            )
        if alternatives:
            entry[tools.MEDIA_ALTERNATIVES] = dict(sorted(alternatives.items()))
        # A 2xx with no body beside a JSON answer (an operator's 204 where an agent's call answers
        # the pending 202): listed by status under ``x-no-body``, with no schema (miDataworks 023).
        if no_body:
            entry[NO_BODY] = sorted(no_body)
        return dict(sorted(entry.items())), None
    if attachment is not None:
        return None, {"media_type": _JSON, "reason": attachment}
    if other:
        media = sorted(other)[0]
        return None, {"media_type": media, "reason": other[media]}
    if no_body:
        return None, {"media_type": None, "reason": "no body"}
    raise ValueError(f"{key} declares no 2xx answer at all")


def build(openapi: dict[str, Any], routes: Iterable[Any]) -> dict[str, Any]:
    """The contract body (no ``revision``, no ``content_sha256``: the writer sets both) of the route
    contexts ``routes`` (``api.route_table.mounted_api_routes(app)``), read from their OpenAPI
    document ``openapi``. Routes are read from the route table, not the document, so each one's own
    flags are checked and a route the document lacks fails closed."""
    from fastapi.routing import APIRoute

    routes = list(routes)
    components: dict[str, Any] = copy.deepcopy(openapi.get("components", {}).get("schemas", {}))
    responses: dict[str, Any] = {}
    excluded: dict[str, Any] = {}
    for ctx in routes:
        route = ctx.original_route
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        path = str(ctx.path_format)
        flags = [f for f in _EXCLUDE_FLAGS if getattr(route, f, False)]
        if flags:
            raise ValueError(f"{path} sets {flags}: 'always sent' would be false")
        for method in sorted(ctx.methods or ()):
            key = f"{method} {path}"
            operation = openapi["paths"][path][method.lower()]
            entry, exclusion = _operation_entry(key, operation)
            if entry is not None:
                for mark in ("x-approval-action", "x-approval-escalation"):
                    value = operation.get(mark)
                    if value:
                        entry[mark] = value
                responses[key] = entry
            else:
                excluded[key] = exclusion

    errors, error_defs = _error_defs(components)
    defs = tools.rewrite_refs({**components, **error_defs})
    errors = tools.rewrite_refs(errors)
    _resolve_external_markers([responses, defs])
    tools.always_sent_required(defs)
    tools.always_sent_required(responses)
    _history_result(defs, _own_answers(routes, responses, defs))
    tools.open_objects(defs)
    tools.open_objects(responses)

    kept = tools.reachable_defs([responses, errors], defs)
    known = _external_files()
    external: dict[str, Any] = {}
    for schema in tools.walk([responses, kept]):
        ref = schema.get("$ref")
        if not isinstance(ref, str) or not _is_external(ref):
            continue
        sid = tools.document_id(ref)
        if sid not in known:
            raise ValueError(f"unknown external $ref {ref!r}: no docs/schemas/*.json has that $id")
        for sibling in ("type", tools.OPEN_KEY, "title"):
            schema.pop(sibling, None)
        file, sha = known[sid]
        external[sid] = {"file": file, "sha256": sha}
    referencing = sum(
        1
        for entry in responses.values()
        if any(
            _is_external(s.get("$ref"))
            for s in tools.walk([entry, tools.reachable_defs([entry], kept)])
        )
    )

    ext = frozenset(external)
    typed = sum(1 for entry in responses.values() if _entry_typed(entry, kept, ext))
    unions = sum(
        1
        for entry in responses.values()
        for status, schema in entry.items()
        if status[:1].isdigit() and len(tools.members(schema, kept)) > 1
    )
    coverage = {
        "typed": typed,
        "unions": unions,
        "external": referencing,
        "excluded": len(excluded),
        "untyped": len(responses) - typed,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ID,
        "kind": KIND,
        "title": "miForge answers",
        "description": _DESCRIPTION,
        "responses": dict(sorted(responses.items())),
        "excluded": dict(sorted(excluded.items())),
        "errors": errors,
        "external": dict(sorted(external.items())),
        "coverage": coverage,
        "$defs": kept,
    }


def _own_answers(
    routes: Iterable[Any], responses: dict[str, Any], defs: dict[str, Any]
) -> dict[str, list[Any]]:
    """Each approval action -> the own answers (not the pending approval) of the gated routes that
    file it, at every status they declare, read from the contract entries just built. A route whose
    own answer is a bare ``Response`` (a 204) stores ``{"status_code"}``. The escalation action
    files the same request, so it gets the same answers."""
    from fastapi.routing import APIRoute

    out: dict[str, list[Any]] = {}
    for ctx in routes:
        route = ctx.original_route
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        actions = [
            a
            for a in (
                getattr(ctx.endpoint, "__mf_approval_action__", None),
                getattr(ctx.endpoint, "__mf_approval_escalation__", None),
            )
            if a
        ]
        if not actions:
            continue
        for method in sorted(ctx.methods or ()):
            key = f"{method} {ctx.path_format}"
            entry = responses[key]
            own = [_STATUS_ONLY_RESULT] if entry.get(NO_BODY) else []
            for status, schema in entry.items():
                if status[:1].isdigit():
                    own += [m for m in tools.members(schema, defs) if m.get("$ref") != _PENDING]
            if not own or not _pending_of(entry.get("202"), defs):
                raise ValueError(
                    f"{key}: a gated route must declare its own answer and the pending approval "
                    "at 202 (answers_common.gated_responses)"
                )
            for action in actions:
                out.setdefault(action, []).extend(own)
    return out


def _pending_of(schema: Any, defs: dict[str, Any]) -> bool:
    return any(m.get("$ref") == _PENDING for m in tools.members(schema or {}, defs))


def _history_result(defs: dict[str, Any], own: dict[str, list[Any]]) -> None:
    """miServer TD8 / TD14: an approval's ``result``, tagged by ``action``. For each action,
    ``result`` is one of the history forms of that action's routes' own answers, or null. An
    action with no branch leaves ``result`` unconstrained."""
    for name, body in list(defs.items()):
        props = body.get("properties") if isinstance(body, dict) else None
        if not isinstance(props, dict):
            continue
        marked = [
            p
            for p, sub in props.items()
            if isinstance(sub, dict) and sub.pop(tools.HISTORY_FORM, False)
        ]
        if not marked:
            continue
        if "action" not in props or marked != ["result"]:
            raise ValueError(
                f"{name}: the history form is defined for an approval's `result` beside its "
                "`action`"
            )
        props["result"] = {k: v for k, v in props["result"].items() if k == "description"}
        branches = []
        for action in sorted(own):
            variants: list[Any] = []
            for schema in own[action]:
                variant = tools.history_form(schema, defs)
                if variant not in variants:
                    variants.append(variant)
            variants.append({"type": "null"})
            branches.append(
                {
                    "if": {"properties": {"action": {"const": action}}, "required": ["action"]},
                    "then": {"properties": {"result": {"anyOf": variants}}},
                }
            )
        body["allOf"] = [*body.get("allOf", []), *branches]
        _redacted_in_history(defs)
        body["description"] = (
            body.get("description", "").rstrip() + "\n\n" + HISTORY_EXCEPTION
        ).strip()


def _redacted_in_history(defs: dict[str, Any]) -> None:
    """In place: in every history copy (an approval's stored ``result``), a secret-named property
    is the redaction marker, because every approval answer redacts it on the way out
    (``core.secret_keys``; ``services/approval_service.py:as_payload``)."""
    from core.secret_keys import REDACTED, is_secret_key

    for name, body in defs.items():
        if not name.endswith(tools.HISTORY_SUFFIX):
            continue
        for schema in tools.walk(body):
            props = schema.get("properties")
            if not isinstance(props, dict):
                continue
            for prop in [p for p in props if is_secret_key(p)]:
                props[prop] = {
                    "const": REDACTED,
                    "description": "Redacted in every approval answer (core.secret_keys).",
                }


def _is_external(ref: Any) -> bool:
    return isinstance(ref, str) and ref.startswith("https://")


def _entry_typed(entry: dict[str, Any], defs: dict[str, Any], external: frozenset[str]) -> bool:
    return all(tools.is_typed(s, defs, external) for k, s in entry.items() if k[:1].isdigit())


def untyped_routes(doc: dict[str, Any]) -> list[str]:
    """The ``responses`` keys with a body that is not typed (FR-021.4), sorted."""
    defs = doc.get("$defs") or {}
    ext = frozenset(doc.get("external") or {})
    return sorted(k for k, e in doc["responses"].items() if not _entry_typed(e, defs, ext))


def answers_contract() -> dict[str, Any]:
    """The contract body of a freshly built app (``api.app.create_app``)."""
    from api.app import create_app
    from api.route_table import mounted_api_routes

    app = create_app()
    return build(app.openapi(), mounted_api_routes(app))


def body_of(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` without the two keys the writer sets."""
    return {k: v for k, v in doc.items() if k not in tools.UNDIGESTED}


def finished(body: dict[str, Any], revision: int) -> dict[str, Any]:
    """``body`` with its ``revision`` and ``content_sha256`` (FR-021.19)."""
    doc = dict(body_of(body))
    doc["revision"] = revision
    doc["content_sha256"] = tools.content_digest(doc)
    return doc


def resolver(*docs: dict[str, Any]) -> tools.Resolver:
    """Reads each document the contracts' ``external`` names, by sha256, from this checkout's
    working tree and git history (the re-vendor row)."""
    files = {sid: row["file"] for doc in docs for sid, row in (doc.get("external") or {}).items()}
    return tools.git_resolver(SCHEMAS_DIR.parents[1], files)


def render(doc: dict[str, Any]) -> str:
    """The committed byte form (``contract_tools.render``)."""
    return tools.render(doc)


def write(body: dict[str, Any] | None = None, path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Write ``path`` from ``body`` (default: this process's contract), refusing a breaking change.

    Equal bodies: nothing is written. All changes additive: ``revision`` + 1. No committed file:
    revision 1. Any breaking change: ``BreakingAnswerChange``, and the committed bytes stay as
    they were."""
    new = answers_contract() if body is None else body_of(body)
    if path.is_file():
        old: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if body_of(old) == new and not problems(old):
            return old
        bad = tools.breaking(tools.classify(body_of(old), new, resolver(old, new)))
        if bad:
            raise BreakingAnswerChange(bad)
        doc = finished(new, int(old.get("revision", 0)) + 1)
    else:
        doc = finished(new, 1)
    path.write_text(render(doc), encoding="utf-8")
    return doc


def problems(doc: dict[str, Any]) -> list[str]:
    """What is wrong with a committed contract on its own: a hand edit leaves ``content_sha256``
    stale, because only the writer sets it (FR-021.19)."""
    out = []
    revision = doc.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        out.append(f"revision {revision!r} is not a positive integer")
    if doc.get("content_sha256") != tools.content_digest(doc):
        out.append(
            "content_sha256 does not match the content: was the file edited by hand? "
            f"Regenerate it with `{REGENERATE}`"
        )
    return out


def stale(committed_text: str, body: dict[str, Any]) -> list[str]:
    """Why the committed contract text is not the generated one, or [] when it is."""
    if not committed_text:
        return [f"{CONTRACT_PATH} is missing: run `{REGENERATE}`"]
    old = json.loads(committed_text)
    out = problems(old)
    if body_of(old) != body:
        out.append(f"{CONTRACT_PATH} is stale against the live routes: run `{REGENERATE}`")
    elif committed_text != render(finished(body, int(old.get("revision", 0)))):
        out.append(f"{CONTRACT_PATH} is not in the writer's byte form: run `{REGENERATE}`")
    return out
