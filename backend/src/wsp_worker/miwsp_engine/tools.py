"""The editor's tools, per operation and style (feature 018; FR-018.5, FR-018.6; FTDD §6.3).

The set is ``weightspace``'s (``ws:agent.py:153-233``):

- every session: ``list_tensors``, ``describe_tensor``, ``read_tensor`` (at most
  :data:`READ_LIMIT` values), ``python`` (unless the request switches it off), ``submit`` and one
  ``forward_<name>`` per requested observation tool;
- ``repair`` and ``create`` add ``write_tensor``, ``diff``, ``checkpoint``, ``restore`` and
  ``revert``; ``inspect`` gets none of them, and its Python view of the weights is read-only;
- ``program`` style is the inspect set plus ``test_program`` and ``submit_program``;
- with paging on (2026-11-21), every style adds ``recall`` (``paging.Pager.recall``).

Arguments are validated against each tool's schema before it runs (:func:`validate`, the subset
of JSON Schema miServer enforces). Output is cut to :data:`MAX_TOOL_CHARS`. A forward tool is
charged only after its function RETURNS: a call that raises costs nothing (``ws:agent.py:256-266``),
and when the budget is spent the call answers ``weightspace``'s exhaustion text and runs nothing.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from wsp_worker.miwsp_engine import prompts

MAX_TOOL_CHARS = 12_000
READ_LIMIT = 4_096
EDIT_TOOLS = ("write_tensor", "diff", "checkpoint", "restore", "revert")
READ_TOOLS = ("list_tensors", "describe_tensor", "read_tensor")
PROGRAM_TOOLS = ("test_program", "submit_program")
#: The JSON Schema keywords the engine's own validator (and miServer's constrained decoder)
#: enforces; every tool schema must stay inside them (a unit test walks each one).
SUBSET = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "const",
        "anyOf",
        "enum",
        "items",
        "minimum",
        "maximum",
    }
)
ANNOTATIONS = frozenset({"description", "title"})


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the editor sees it. ``cost`` is set for forward tools only."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    cost: int | None = None


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def closed(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """An observation tool's parameters as a closed object (a copy; the catalogue is untouched)."""
    return {**dict(parameters), "additionalProperties": False}


# ----------------------------------------------------------------------------- validation

_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
}


def validate(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Every violation of ``schema`` (the :data:`SUBSET`) by ``value``; empty when it conforms."""
    errors: list[str] = []
    if "anyOf" in schema:
        options = [validate(value, s, path) for s in schema["anyOf"]]
        if all(options):
            return [f"{path}: matches none of {len(options)} alternatives"]
    kind = schema.get("type")
    if kind is not None:
        kinds = kind if isinstance(kind, list) else [kind]
        if not any(_TYPES[k](value) for k in kinds):
            return [f"{path}: expected {'/'.join(kinds)}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{path}: missing {name!r}")
        for name, item in value.items():
            if name in properties:
                errors += validate(item, properties[name], f"{path}.{name}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected {name!r}")
    return errors


def keywords_outside_subset(schema: Any, path: str = "$") -> list[str]:
    """Every keyword of ``schema`` outside :data:`SUBSET` (and the annotations)."""
    if not isinstance(schema, Mapping):
        return []
    bad: list[str] = []
    for key, value in schema.items():
        if key in ANNOTATIONS:
            continue
        if key not in SUBSET:
            bad.append(f"{path}.{key}")
        elif key == "properties":
            for name, sub in value.items():
                bad += keywords_outside_subset(sub, f"{path}.properties.{name}")
        elif key == "items":
            bad += keywords_outside_subset(value, f"{path}.items")
        elif key == "anyOf":
            for i, sub in enumerate(value):
                bad += keywords_outside_subset(sub, f"{path}.anyOf[{i}]")
    return bad


def truncate(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + prompts.TRUNCATED.format(n=f"{len(text) - limit:,}")


def render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=_jsonable)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


# ----------------------------------------------------------------------------- formatting


def fmt_num(x: float, precision: int = 4) -> str:
    s = f"{x:.{precision}g}"
    return "0" if s in ("0", "-0") else s


def fmt_array(arr: np.ndarray, precision: int = 4) -> str:
    """One line per row (last axis), leading axes labelled (``weightspace``'s layout)."""
    a = np.asarray(arr)
    if a.ndim == 0:
        return fmt_num(float(a), precision)
    if a.ndim == 1:
        return " ".join(fmt_num(v, precision) for v in a.tolist())
    flat = a.reshape(-1, a.shape[-1])
    rows = []
    for idx, row in zip(np.ndindex(*a.shape[:-1]), flat.tolist(), strict=True):
        label = ",".join(str(i) for i in idx)
        rows.append(f"[{label}] " + " ".join(fmt_num(v, precision) for v in row))
    return "\n".join(rows)


_SLICE_PART = re.compile(r"^\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*)?)?$")


def parse_index(index: str | None) -> tuple[Any, ...]:
    """numpy-style index text such as ``"0:8, :, 3"``, parsed without ``eval``."""
    if index is None or not str(index).strip():
        return ()
    text = str(index).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    out: list[Any] = []
    for part in text.split(","):
        part = part.strip()
        if part in ("", ":", "..."):
            out.append(Ellipsis if part == "..." else slice(None))
            continue
        m = _SLICE_PART.match(part)
        if not m:
            raise ValueError(f"bad index component {part!r}")
        if ":" not in part:
            out.append(int(m.group(1)))
        else:
            a, b, c = (int(g) if g is not None else None for g in m.groups())
            out.append(slice(a, b, c))
    return tuple(out)


def describe(name: str, values: np.ndarray, dtype: str) -> str:
    """Shape, dtype, moments, quantiles, and row/column norms and singular values for a matrix."""
    a = np.asarray(values, dtype=np.float64)
    flat = a.reshape(-1)
    finite = flat[np.isfinite(flat)]
    lines = [f"{name} shape={list(a.shape)} dtype={dtype} numel={flat.size:,}"]
    if finite.size:
        q = np.quantile(finite, [0.01, 0.25, 0.5, 0.75, 0.99])
        lines.append(
            f"mean={fmt_num(float(finite.mean()))} std={fmt_num(float(finite.std()))} "
            f"min={fmt_num(float(finite.min()))} max={fmt_num(float(finite.max()))} "
            f"abs_mean={fmt_num(float(np.abs(finite).mean()))} "
            f"frac_zero={fmt_num(float(np.mean(finite == 0)))}"
        )
        lines.append("q01 q25 q50 q75 q99: " + " ".join(fmt_num(float(v)) for v in q))
    if flat.size != finite.size:
        lines.append(f"non-finite entries: {flat.size - finite.size:,}")
    if a.ndim == 2 and finite.size == flat.size and min(a.shape) > 0:
        rows = np.linalg.norm(a, axis=1)
        cols = np.linalg.norm(a, axis=0)
        lines.append(f"row norms: {fmt_array(rows[:16])}" + (" ..." if rows.size > 16 else ""))
        lines.append(f"col norms: {fmt_array(cols[:16])}" + (" ..." if cols.size > 16 else ""))
        sv = np.linalg.svd(a, compute_uv=False)
        lines.append(f"singular values: {fmt_array(sv[:16])}" + (" ..." if sv.size > 16 else ""))
    return "\n".join(lines)


def list_tensors(
    names: list[str],
    shapes: Mapping[str, tuple[int, ...]],
    dtypes: Mapping[str, str],
    pattern: str | None = None,
) -> str:
    chosen = [
        n for n in names if pattern is None or pattern in n or fnmatch.fnmatchcase(n, pattern)
    ]
    lines = [
        f"{n}  {list(shapes[n])}  {dtypes[n]}  {int(np.prod(shapes[n], dtype=np.int64)):,}"
        for n in chosen
    ]
    total = sum(int(np.prod(shapes[n], dtype=np.int64)) for n in chosen)
    return "\n".join([*lines, f"{len(chosen)} tensors, {total:,} weights"])


# ----------------------------------------------------------------------------- the specs

STRING = {"type": "string"}


def read_specs(
    read_tensor: Callable[..., Any],
    describe_tensor: Callable[..., Any],
    list_fn: Callable[..., Any],
) -> list[ToolSpec]:
    return [
        ToolSpec(
            "list_tensors",
            "List weight tensors (name, shape, dtype, numel). Optional glob/substring filter.",
            _object({"pattern": STRING}, []),
            list_fn,
        ),
        ToolSpec(
            "describe_tensor",
            "Statistics of one tensor: quantiles, row/col norms, singular values.",
            _object({"name": STRING}, ["name"]),
            describe_tensor,
        ),
        ToolSpec(
            "read_tensor",
            "Read raw values of a tensor or a numpy-style slice of it, e.g. index='0:4, :16'. "
            f"At most {READ_LIMIT} values per call.",
            _object(
                {
                    "name": STRING,
                    "index": STRING,
                    "precision": {"type": "integer", "minimum": 1, "maximum": 17},
                },
                ["name"],
            ),
            read_tensor,
        ),
    ]


def python_spec(fn: Callable[..., Any], *, can_edit: bool) -> ToolSpec:
    names = "np, R (read-only weights), notes (a list you append to; it survives a context reset)"
    if can_edit:
        names += ", W (writable weights: edit in place or assign W[name] = array)"
    return ToolSpec(
        "python",
        f"Run Python in a persistent namespace with {names}. The last expression is printed. Use "
        "it for static analysis of weights and for writing weights. It cannot run the network.",
        _object({"code": STRING}, ["code"]),
        fn,
    )


def edit_specs(
    write: Callable[..., Any],
    diff: Callable[..., Any],
    checkpoint: Callable[..., Any],
    restore: Callable[..., Any],
    revert: Callable[..., Any],
) -> list[ToolSpec]:
    return [
        ToolSpec(
            "write_tensor",
            "Overwrite a tensor or a slice of it with values (nested list or scalar).",
            _object(
                {
                    "name": STRING,
                    "values": {"anyOf": [{"type": "number"}, {"type": "array"}]},
                    "index": STRING,
                },
                ["name", "values"],
            ),
            write,
        ),
        ToolSpec(
            "diff",
            "Summarise all edits made so far versus the original weights.",
            _object({}, []),
            diff,
        ),
        ToolSpec(
            "checkpoint",
            "Save the current weights under a label.",
            _object({"label": STRING}, ["label"]),
            checkpoint,
        ),
        ToolSpec(
            "restore",
            "Restore weights saved under a label.",
            _object({"label": STRING}, ["label"]),
            restore,
        ),
        ToolSpec(
            "revert",
            "Revert one tensor (or all, if name is omitted) to the original weights.",
            _object({"name": STRING}, []),
            revert,
        ),
    ]


def submit_spec(fn: Callable[..., Any]) -> ToolSpec:
    return ToolSpec(
        "submit",
        "Finish: give the final answer (and for edits, a summary of what you changed and why).",
        _object(
            {"answer": STRING, "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            ["answer"],
        ),
        fn,
    )


def recall_spec(fn: Callable[..., Any]) -> ToolSpec:
    """``recall(page)``: a paged-out result again (paging, 2026-11-21). No forward-pass cost."""
    return ToolSpec("recall", prompts.RECALL_DESCRIPTION, _object({"page": STRING}, ["page"]), fn)


def forward_spec(
    name: str, description: str, parameters: Mapping[str, Any], cost: int, fn: Callable[..., Any]
) -> ToolSpec:
    return ToolSpec(
        f"forward_{name}",
        f"[forward pass, costs {cost}] {description}",
        closed(parameters),
        fn,
        cost,
    )


class ForwardBudget:
    """The forward-pass budget: a call is charged only after it returns (``ws:agent.py:256``)."""

    def __init__(self, budget: int) -> None:
        self.budget = int(budget)
        self.calls: list[dict[str, Any]] = []

    @property
    def used(self) -> int:
        return sum(int(c["cost"]) for c in self.calls)

    def charge(self, name: str, cost: int, fn: Callable[..., Any], args: dict[str, Any]) -> str:
        remaining = self.budget - self.used
        if cost > remaining:
            return prompts.FORWARD_EXHAUSTED.format(remaining=remaining, cost=cost)
        out = fn(**args)  # raises -> nothing appended, nothing charged
        text = out if isinstance(out, str) else json.dumps(out, indent=1, default=_jsonable)
        self.calls.append({"tool": name, "arguments": args, "cost": cost})
        return text + prompts.FORWARD_LEFT.format(left=remaining - cost, budget=self.budget)


class Toolbox:
    """Name -> spec, and :meth:`call`: validate, run, render, truncate. Never raises."""

    def __init__(self, specs: list[ToolSpec], forward: ForwardBudget) -> None:
        self.specs = {s.name: s for s in specs}
        self.forward = forward

    def names(self) -> list[str]:
        return list(self.specs)

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        spec = self.specs.get(name)
        if spec is None:
            out = f"unknown tool {name!r}; available: {', '.join(self.specs)}"
        else:
            errors = validate(dict(args), spec.parameters)
            if errors:
                out = f"invalid arguments for {name}: " + "; ".join(errors[:5])
            else:
                try:
                    if spec.cost is not None:
                        out = self.forward.charge(name, spec.cost, spec.fn, dict(args))
                    else:
                        out = render(spec.fn(**args))
                except Exception as exc:  # the editor's own failure, reported to it as text
                    out = f"{type(exc).__name__}: {exc}"
        return truncate(out)
