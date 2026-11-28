"""miForge's env-name table equals census's independent copy (FR-015.5; CF-3, CF-25 m13).

``core/legacy_names.py`` writes the three pairs as literals; census writes its own
``ENV_RENAMES["miForge"]`` and emits it as the ``env`` section of ``docs/legacy-identifiers.md``.
Two authorities written independently and compared is the point: a table LOADED from the export
would compare the export with itself, so the module is also checked, by AST, to read no file.

``LEGACY_IDENTIFIERS_EXPORT`` (an absolute path) points the loader at a scratch copy, which is how
controls N23/N23b edit "the export" without touching the committed file.
"""

from __future__ import annotations

import ast
import inspect

from core import legacy_names
from tests.support import legacy_identifiers


def _ours() -> tuple[dict[str, str], dict[str, str]]:
    return dict(legacy_names.LEGACY_ENV_ALIASES), dict(legacy_names.RETIRED_ENV_NAMES)


def test_env_pairs_equal_export() -> None:
    export = legacy_identifiers.load()
    transition = {r["old"]: r["new"] for r in export.env if r["status"] == "transition"}
    retired = {r["old"]: r["new"] for r in export.env if r["status"] == "retired"}
    aliases, ours_retired = _ours()
    assert export.env, "the export carries no env records; miForge reads three pairs"
    assert aliases == transition, (
        f"core/legacy_names.py LEGACY_ENV_ALIASES {aliases} != the export's transition env "
        f"records {transition} (census ENV_RENAMES['miForge'])"
    )
    assert (
        ours_retired == retired
    ), f"RETIRED_ENV_NAMES {ours_retired} != the export's retired env records {retired}"
    fields = {e.env for e in legacy_names.ENV_NAMES}
    assert set(transition.values()) | set(retired.values()) == fields


def test_host_section_empty() -> None:
    """miForge serves no old host name, so it records no host use (CF-14 does not apply)."""
    assert legacy_identifiers.load().host == []


#: Calls or names whose presence means the module reads a file or a resource.
_FILE_READERS = {"open", "Path", "json", "tomllib", "importlib", "resources", "read_text"}


def test_legacy_names_module_reads_no_file() -> None:
    tree = ast.parse(inspect.getsource(legacy_names))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FILE_READERS:
            found.append(f"{node.lineno}: name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _FILE_READERS:
            found.append(f"{node.lineno}: attribute {node.attr}")
        elif isinstance(node, ast.Import | ast.ImportFrom):
            modules = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if module.split(".")[0] in _FILE_READERS or module in {"pathlib"}:
                    found.append(f"{node.lineno}: import {module}")
    assert (
        not found
    ), "core/legacy_names.py must hold its tables as literals, never read them: " + str(found)
    assigned = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        if isinstance(target, ast.Name)
    }
    names = assigned["ENV_NAMES"]
    assert isinstance(names, ast.Tuple) and names.elts, "ENV_NAMES must be a literal tuple"
    for element in names.elts:
        assert isinstance(element, ast.Call)
        assert all(isinstance(a, ast.Constant) for a in element.args), ast.dump(element)
