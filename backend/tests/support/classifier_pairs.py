# Origin: miDataworks (Onegaishimas/miDataworks)
#   backend/tests/support/classifier_pairs.py @ 6b50a8f4
# Mode: adapt (docs/REUSE.md). Kept: every function, so both apps read the shared pairs alike.
# Changed: the docstring's paths, and the pairs file sits in tests/support/fixtures/ (a byte copy of
# miDataworks' tests/fixtures/answer_classifier_pairs.json at the same commit, the shared pairs).
"""The classifier's shared before-and-after pairs (021 FTASKS 1.7). STANDARD LIBRARY ONLY.

``tests/support/fixtures/answer_classifier_pairs.json`` holds one old document (``base``) and, per
pair, the edits that make the new one. This module applies them and runs a classifier over every
pair. It is loaded by file path in another interpreter too (the cross-repo parity run in
``tests/contract/test_answers_contract_consumers.py``), so it imports nothing outside the
standard library and nothing from this repository.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

PAIRS_PATH = Path(__file__).resolve().parent / "fixtures" / "answer_classifier_pairs.json"


def load(path: Path | str = PAIRS_PATH) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return loaded


def _parent(doc: Any, path: list[Any]) -> tuple[Any, Any]:
    node = doc
    for key in path[:-1]:
        node = node[key]
    return node, path[-1]


def apply(doc: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    """A copy of ``doc`` with ``ops`` applied (set, delete, append, remove_value)."""
    out = copy.deepcopy(doc)
    for op in ops:
        if op["op"] == "set":
            node, key = _parent(out, op["path"])
            node[key] = copy.deepcopy(op["value"])
        elif op["op"] == "delete":
            node, key = _parent(out, op["path"])
            del node[key]
        elif op["op"] == "append":
            node, key = _parent(out, op["path"])
            node[key].append(copy.deepcopy(op["value"]))
        elif op["op"] == "remove_value":
            node, key = _parent(out, op["path"])
            node[key].remove(op["value"])
        else:
            raise ValueError(f"unknown op {op['op']!r}")
    return out


def run(classify: Any, pairs: dict[str, Any]) -> dict[str, list[list[Any]]]:
    """Per pair id, the sorted distinct ``[row, breaking]`` the classifier ``classify(old, new,
    resolve)`` reports. A change's ``breaking`` is read from the change itself."""
    documents = pairs.get("documents", {})

    def resolve(_sid: str, sha: str) -> dict[str, Any] | None:
        doc = documents.get(sha)
        return copy.deepcopy(doc) if doc is not None else None

    out: dict[str, list[list[Any]]] = {}
    for pair in pairs["pairs"]:
        new = apply(pairs["base"], pair["ops"])
        changes = classify(copy.deepcopy(pairs["base"]), new, resolve)
        seen = sorted({(c.row, bool(c.breaking)) for c in changes})
        out[pair["id"]] = [[row, brk] for row, brk in seen]
    return out


def main(classifier_file: str, pairs_file: str) -> None:
    """Load ``classifier_file`` (a ``contract_tools.py``) by path and print ``run`` as JSON."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_classifier_under_test", classifier_file)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {classifier_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module by name
    spec.loader.exec_module(module)
    print(json.dumps(run(module.classify, load(pairs_file)), sort_keys=True))


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])
