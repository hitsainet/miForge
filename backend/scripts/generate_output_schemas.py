"""Generate ``backend/src/mcp_server/output_schemas.json`` from the answer contract (021 FR-021.17,
FR-021.18; FTDD §6.6, FTID §7.1; it replaced 017's permissive wrapping, FR-017.19, .20).

For each live MCP tool, its ledger route (``tests/support/mcp_ledger.py``), then
(:func:`output_transform`):

1. the route's ``responses`` entry in ``docs/schemas/miforge-answers-v1.json``: every declared
   2xx answer as an alternative (``anyOf``), the gate's pending approval taken out (it is a root
   branch), and ``{status_code}`` for a status that answers no body (``mcp_server/client.py``
   answers an empty body so);
2. the tool's overlay, if it has one (``backend/scripts/output_overlays.py``, FT6): keys withheld,
   one variant kept, or a hand schema for a tool that makes no call;
3. a non-object answer under ``result``, where ``shape_results`` puts it in ``structuredContent``
   (A30); no top-level ``next`` (A30);
4. the root ``{"type": "object", "anyOf": [answer..., Unavailable, ApprovalPending if the route is
   gated], "$defs": …}``; ``Unavailable`` is the vendored result language's;
5. the published documents the answer references embedded under ``$defs`` (``<file stem>`` and
   ``<file stem>.<definition>``), because an MCP client resolves no ``https://`` reference;
6. generated ``title`` and every ``description`` key stripped: the contract keeps the descriptions
   (operator, 2026-11-26, FT7 option (b), amending Q-T7; ``tools/list`` stays under the ceiling).

Nothing closes an object: an answer may carry keys its schema does not name (D3). The MCP pod
imports no backend module, so this runs in tests and by hand; the server reads only the generated
JSON. ``--check`` regenerates both files and fails on any difference.

    PYTHONPATH=backend/src:backend python backend/scripts/generate_output_schemas.py [--check]

The same command writes and checks the published answer contract first (IC-2), through its writer,
which refuses a breaking change (FR-021.19).
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "backend" / "src" / "mcp_server" / "output_schemas.json"
RESULT_LANGUAGE = REPO / "docs" / "schemas" / "mistudio-result-language-v1.json"

DEFS = "#/$defs/"
PENDING = DEFS + "ApprovalPending"
UNAVAILABLE = "Unavailable"
#: Marks the object the generator wraps a non-object answer in (``{"result": …}``, A30), so a rule
#: reading the schema can tell it from a route's own object whose only property is ``result``.
WRAPPED = "x-mcp-wrapped"
#: What the MCP client answers for an empty body (``mcp_server/client.py``).
STATUS_ONLY = {
    "type": "object",
    "properties": {"status_code": {"type": "integer"}},
    "required": ["status_code"],
    "description": "The route answered no body; the MCP client reports its status.",
}
#: Keys a schema position never carries in an output schema: generated titles, and (operator,
#: 2026-11-26, FT7 option (b), amending Q-T7) descriptions, which the contract keeps; tool
#: descriptions and ``forge_howto`` carry the guidance, and ``tools/list`` stays under the ceiling.
_DROPPED = ("title", "description")
#: Keys whose value maps NAMES to schemas, not schema keywords.
_NAMED = ("properties", "$defs", "patternProperties", "dependentSchemas")


def _strip(node: Any) -> Any:
    """A copy without ``title`` and ``description`` keys; property and definition names (a property
    may be NAMED ``description``) are kept."""
    if isinstance(node, list):
        return [_strip(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED:
            continue
        if key in _NAMED and isinstance(value, dict):
            out[key] = {name: _strip(sub) for name, sub in value.items()}
        else:
            out[key] = _strip(value)
    return out


class _Documents:
    """The published documents an answer references, embedded under ``$defs``: a document as
    ``<stem>`` (its root, without ``$defs``) and its definitions as ``<stem>.<name>``."""

    def __init__(self, external: dict[str, Any]) -> None:
        self.files = {sid: REPO / row["file"] for sid, row in external.items()}
        self.loaded: dict[str, dict[str, Any]] = {}

    @staticmethod
    def stem(sid: str) -> str:
        return sid.rsplit("/", 1)[-1].removesuffix(".json")

    def local(self, ref: str, base: str | None = None) -> str:
        """``ref`` (absolute, or relative to the document ``base``) as a reference into the
        output schema's ``$defs``."""
        absolute = urljoin(base, ref) if base and not ref.startswith("https://") else ref
        sid, _, fragment = absolute.partition("#")
        if sid not in self.files:
            raise ValueError(f"{ref!r}: no published document has the $id {sid!r}")
        stem = self.stem(sid)
        if fragment.startswith("/$defs/"):
            name, _, rest = fragment[len("/$defs/") :].partition("/")
            return f"{DEFS}{stem}.{name}" + (f"/{rest}" if rest else "")
        return f"{DEFS}{stem}" + fragment

    def document(self, sid: str) -> dict[str, Any]:
        if sid not in self.loaded:
            self.loaded[sid] = json.loads(self.files[sid].read_text(encoding="utf-8"))
        return self.loaded[sid]

    def rewrite(self, node: Any, base: str) -> Any:
        """A copy of a document's schema whose references point into the output's ``$defs``."""
        if isinstance(node, list):
            return [self.rewrite(item, base) for item in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                target = value if not value.startswith("#") else base + value
                out[key] = self.local(target, base)
            else:
                out[key] = self.rewrite(value, base)
        return out

    def definition(self, name: str) -> dict[str, Any]:
        """The embedded definition ``<stem>`` or ``<stem>.<name>``."""
        stem, _, inner = name.partition(".")
        sid = next(s for s in self.files if self.stem(s) == stem)
        doc = self.document(sid)
        if inner:
            body = doc["$defs"][inner]
        else:
            body = {k: v for k, v in doc.items() if k not in ("$defs", "$id", "$schema")}
        rewritten: dict[str, Any] = self.rewrite(body, sid)
        return rewritten


def _localise(node: Any, documents: _Documents) -> Any:
    """A copy of a contract schema whose ``https://`` references point at embedded documents."""
    if isinstance(node, list):
        return [_localise(item, documents) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "$ref" and isinstance(value, str) and value.startswith("https://"):
            out[key] = documents.local(value)
        else:
            out[key] = _localise(value, documents)
    return out


def _names(node: Any) -> set[str]:
    """The ``$defs`` names ``node`` references (the first segment after ``#/$defs/``)."""
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            ref = current.get("$ref")
            if isinstance(ref, str) and ref.startswith(DEFS):
                found.add(ref[len(DEFS) :].split("/", 1)[0])
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return found


def _deref(schema: Any, defs: dict[str, Any]) -> Any:
    hops = 0
    while isinstance(schema, dict) and str(schema.get("$ref", "")).startswith(DEFS) and hops < 50:
        name = schema["$ref"][len(DEFS) :]
        if "/" in name:
            break
        schema = defs.get(name, {})
        hops += 1
    return schema


def _members(schema: Any, defs: dict[str, Any]) -> list[Any]:
    resolved = _deref(schema, defs)
    for key in ("anyOf", "oneOf"):
        if isinstance(resolved, dict) and key in resolved:
            return [m for member in resolved[key] for m in _members(member, defs)]
    return [schema]


def _answers(
    entry: dict[str, Any], defs: dict[str, Any], keep: tuple[str, ...] | None
) -> list[Any]:
    """The route's own answers (the pending approval taken out), one per alternative."""
    out: list[Any] = []
    for status in sorted(k for k in entry if k[:1] == "2"):
        if keep is not None and status not in keep:
            continue
        for member in _members(entry[status], defs):
            if member.get("$ref") != PENDING and member not in out:
                out.append(member)
    if entry.get("x-no-body") and (keep is None or set(keep) & set(entry["x-no-body"])):
        out.append(STATUS_ONLY)
    return out


def _is_object(schema: Any, defs: dict[str, Any]) -> bool:
    resolved = _deref(schema, defs)
    return isinstance(resolved, dict) and (
        resolved.get("type") == "object" or "properties" in resolved or "$ref" in resolved
    )


def _without(schema: dict[str, Any], defs: dict[str, Any], pointer: str, keys: tuple[str, ...]):
    """A copy of ``schema`` (inlined along ``pointer``) with ``keys`` removed at ``pointer``."""
    root = copy.deepcopy(_deref(schema, defs))
    node: Any = root
    parts = [p for p in pointer.split("/") if p]
    for index, part in enumerate(parts):
        node[part] = copy.deepcopy(_deref(node[part], defs))
        node = node[part]
        del index
    for key in keys:
        node.get("properties", {}).pop(key, None)
        if key in node.get("required", []):
            node["required"] = [k for k in node["required"] if k != key]
    return root


def _withhold_anywhere(schema: dict[str, Any], keys: tuple[str, ...]) -> None:
    """In place, over one tool's self-contained schema: ``keys`` leave every ``properties`` and
    ``required`` at any depth (the tool removes them from its answer wherever they are)."""
    stack: list[Any] = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                for key in keys:
                    props.pop(key, None)
                if isinstance(node.get("required"), list):
                    node["required"] = [k for k in node["required"] if k not in keys]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _top_without_next(schema: Any, defs: dict[str, Any]) -> Any:
    resolved = _deref(schema, defs)
    if isinstance(resolved, dict) and "next" in (resolved.get("properties") or {}):
        inlined = copy.deepcopy(resolved)
        inlined["properties"].pop("next")
        inlined["required"] = [k for k in inlined.get("required", []) if k != "next"]
        return inlined
    return schema


def output_transform(
    contract: dict[str, Any],
    calls: dict[str, tuple[str, str]],
    overlays: dict[str, Any],
    result_language: dict[str, Any],
    gated: set[str],
) -> dict[str, dict[str, Any]]:
    """Each tool's output schema. ``calls``: tool -> (METHOD, concrete path) from the ledger;
    ``gated``: the tools whose route the gate holds. A tool with neither a route nor a hand
    overlay is a generator failure (no tool is generic)."""
    from tests.support.mcp_ledger import full, template_matches

    defs = contract["$defs"]
    documents = _Documents(contract.get("external") or {})
    unavailable = {
        k: v
        for k, v in result_language["$defs"]["unavailable"].items()
        if k != "additionalProperties"
    }
    out: dict[str, dict[str, Any]] = {}
    for tool in sorted(set(calls) | set(overlays)):
        overlay = overlays.get(tool)
        if overlay is not None and overlay.hand is not None:
            answers: list[Any] = [overlay.hand]
            is_gated = False
        else:
            if tool not in calls:
                raise ValueError(f"{tool}: no ledger route and no hand schema")
            method, path = calls[tool]
            keys = [
                k
                for k in contract["responses"]
                if k.split(" ", 1)[0] == method and template_matches(k.split(" ", 1)[1], full(path))
            ]
            # A literal segment wins over a parameter (``/approvals/actions`` is not an id).
            fewest = min((k.count("{") for k in keys), default=0)
            keys = [k for k in keys if k.count("{") == fewest]
            if len(keys) != 1:
                raise ValueError(f"{tool}: {method} {path} matches {keys} in the contract")
            entry = contract["responses"][keys[0]]
            answers = _answers(entry, defs, overlay.variant if overlay else None)
            if not answers:
                raise ValueError(f"{tool}: {keys[0]} declares no answer of its own")
            if overlay is not None and overlay.withheld:
                for pointer, withheld in overlay.withheld.items():
                    answers = [_without(a, defs, pointer, withheld) for a in answers]
            is_gated = tool in gated
        shaped = []
        for answer in answers:
            if _is_object(answer, defs):
                shaped.append(_top_without_next(answer, defs))
            else:
                shaped.append(
                    {
                        "type": "object",
                        "properties": {"result": answer},
                        "required": ["result"],
                        WRAPPED: True,
                    }
                )
        branches = [*shaped, {"$ref": DEFS + UNAVAILABLE}]
        if is_gated:
            branches.append({"$ref": PENDING})
        root: dict[str, Any] = {"type": "object", "anyOf": branches}
        local = _localise(root, documents)
        embedded: dict[str, Any] = {}
        pending = sorted(_names(local))
        while pending:
            name = pending.pop()
            if name in embedded:
                continue
            if name == UNAVAILABLE:
                body = unavailable
            elif name in defs:
                body = _localise(defs[name], documents)
            else:
                body = documents.definition(name)
            embedded[name] = body
            pending.extend(sorted(_names(body) - set(embedded)))
        if embedded:
            local["$defs"] = dict(sorted(embedded.items()))
        if overlay is not None and overlay.withheld_anywhere:
            _withhold_anywhere(local, overlay.withheld_anywhere)
        out[tool] = _strip(local)
    return out


def _is_generic(schema: dict[str, Any]) -> bool:
    """A tool whose own answer (a root branch other than Unavailable and the pending approval)
    types nothing, by FR-021.4's rule (``schemas.contract_tools.is_typed``, the contract's own,
    review round 5); a wrapped non-object answer is judged by what it wraps."""
    from schemas import contract_tools

    defs = schema.get("$defs", {})
    for branch in schema.get("anyOf", []):
        if branch.get("$ref") in (DEFS + UNAVAILABLE, PENDING):
            continue
        target = _deref(branch, defs)
        if isinstance(target, dict) and target.get(WRAPPED):
            branch = target["properties"]["result"]
        if not contract_tools.is_typed(branch, defs):
            return True
    return False


def generate(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """``{"typed": {tool: schema}, "untyped": {}}`` for the live registry."""
    from tests.support.mcp_ledger import EXPECTED_CALLS, GATED_CLASSES

    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server
    from schemas import answers_contract

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from output_overlays import OVERLAYS

    if contract is None:
        contract = json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8"))
    mcp, _ = build_server(MCPSettings(auth_token="schema-generation"))
    tools = sorted(t.name for t in asyncio.run(mcp.list_tools()))
    calls = {t: (EXPECTED_CALLS[t][0], EXPECTED_CALLS[t][1]) for t in tools if t in EXPECTED_CALLS}
    gated = {t for t in calls if EXPECTED_CALLS[t][4] in GATED_CLASSES}
    missing = sorted(set(tools) - set(calls) - set(OVERLAYS))
    if missing:
        raise ValueError(f"tools with no ledger route and no overlay: {missing}")
    unknown = sorted(set(OVERLAYS) - set(tools))
    if unknown:
        raise ValueError(f"overlays for tools that are not registered: {unknown}")
    result_language = json.loads(RESULT_LANGUAGE.read_text(encoding="utf-8"))
    typed = output_transform(contract, calls, OVERLAYS, result_language, gated)
    ordered = {t: typed[t] for t in tools}
    # FR-021.16: the file states its coverage (test_the_file_states_its_coverage_and_it_recounts).
    generic = sum(1 for schema in ordered.values() if _is_generic(schema))
    coverage = {"tools": len(tools), "typed": len(ordered), "generic": generic, "untyped": 0}
    return {"coverage": coverage, "typed": ordered, "untyped": {}}


def render(contract: dict[str, Any] | None = None) -> str:
    return json.dumps(generate(contract), indent=1, sort_keys=True) + "\n"


def check_contract() -> list[str]:
    """Why the committed answer contract is not the generated one, or [] (021 FR-021.2)."""
    from schemas import answers_contract

    path = answers_contract.CONTRACT_PATH
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return answers_contract.stale(text, answers_contract.answers_contract())


def main(argv: list[str]) -> int:
    """``--check``: exit 1 if either generated file differs (021 IC-2). Otherwise write both: the
    answer contract first, through its writer, which refuses a breaking change (FR-021.19)."""
    from schemas import answers_contract

    if "--check" in argv:
        problems = check_contract()
        text = render(answers_contract.answers_contract())
        if OUT.read_text(encoding="utf-8") != text:
            problems.append(f"{OUT} is stale: regenerate it")
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    written = answers_contract.write()
    OUT.write_text(render(written), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
