"""The brand guard, backend half (FR-015.10, FR-015.22; FTID §3.8; CF-13, CF-15, CF-20 to CF-23).

Scans every surface a reader meets for miLLM, miStudio and miSuite: Python string constants
(not docstrings), the live MCP registry, the manual and public markdown, ``.env.example``, and
YAML keys and values under ``gitops/`` and in ``docker-compose.yml``. Until the export header's
``closed`` reaches P4 it REPORTS (prints every finding and passes); from then it is ENFORCED in
the private repository and still reports on the public mirror (suite R23, ``brand_scan.mode``).
The planted-input tests run it enforced from day one, so it is proven able to fail.
``MIFORGE_BRAND_GUARD_ENFORCE=1`` enforces locally before the flip (IPD-8).
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.support import brand_scan as bs
from tests.support import legacy_identifiers as li

PLANTS = json.loads(
    (Path(__file__).resolve().parents[1] / "support" / "fixtures" / "brand_plants.json").read_text(
        "utf-8"
    )
)

#: ``<private>`` in the shared plants stands for the private organisation (see li.PRIVATE).
PLANTS["repo_names"] = [n.replace("<private>", li.PRIVATE) for n in PLANTS["repo_names"]]

SURFACES: dict[str, list[str]] = {
    "python": ["backend/src/**/*.py"],
    "markdown": ["manual/docs/**", "docs/mcp-contract.md", "docs/schemas/README.md", "README.md"],
    "manual_config": ["manual/docusaurus.config.ts", "manual/sidebars.ts"],
    "env_example": [".env.example"],
    "yaml": ["gitops/**", "docker-compose.yml"],
}
ROOTS: dict[str, Path] = dict.fromkeys(SURFACES, bs.REPO_ROOT)


def surface_items(name: str) -> list[bs.Item]:
    root = ROOTS[name]
    files = bs.discover(SURFACES[name], root)
    items: list[bs.Item] = []
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text("utf-8")
        if name == "python":
            items += bs.python_items(text, rel)
        elif name == "markdown":
            items += bs.text_items(text, rel, name, strip_html_comments=rel.endswith(".md"))
        elif name in ("manual_config", "env_example"):
            items += bs.text_items(text, rel, name)
        elif name == "yaml" and rel.endswith((".yaml", ".yml")):
            for index, doc in enumerate(yaml.safe_load_all(text)):
                items += bs.yaml_items(doc, f"{rel}#{index}")
                if isinstance(doc, dict) and doc.get("kind") in ("Kustomization", "Component"):
                    for patch in doc.get("patches") or []:
                        if isinstance(patch, dict) and isinstance(patch.get("patch"), str):
                            for body in yaml.safe_load_all(patch["patch"]):
                                items += bs.yaml_items(body, f"{rel}#{index} patch")
    return items


def _report(name: str, findings: list[bs.Finding], mode: str | None = None) -> None:
    mode = bs.mode() if mode is None else mode
    lines = sorted(f.line() for f in findings)
    print(f"\n[brand guard: {name}] {len(lines)} finding(s) ({mode} mode)")
    for line in lines:
        print(f"  {line}")
    if mode == "enforced":
        assert not lines, f"{name}: old brand names on a guarded surface:\n" + "\n".join(lines)


def _surface_test(name: str) -> None:
    items = surface_items(name)
    assert items, f"the {name} surface yielded no items"
    _report(name, bs.scan(items, bs.default_rules()))


def test_backend_ast_surface() -> None:
    _surface_test("python")


def test_markdown_surfaces() -> None:
    _surface_test("markdown")
    _surface_test("manual_config")


def test_env_example_surface() -> None:
    _surface_test("env_example")


def test_yaml_surfaces() -> None:
    _surface_test("yaml")


def test_mcp_registry_surface() -> None:
    items, _count = bs.build_registry()
    _report("mcp", bs.scan(items, bs.default_rules()))


def test_surface_discovery_nonempty() -> None:
    for name in SURFACES:
        assert bs.discover(SURFACES[name], ROOTS[name]), f"{name}: discovery found no file"
        assert surface_items(name), f"{name}: discovery found files but no items"


def test_registry_count_matches() -> None:
    import asyncio

    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    tools = len(asyncio.run(mcp.list_tools()))
    items, counted = bs.build_registry()
    assert counted == tools and counted >= 80, (counted, tools)
    assert (
        sum(1 for i in items if i.where.startswith("mcp tool ") and " schema" not in i.where)
        == tools
    )
    assert any(i.where == "mcp instructions" and i.text for i in items), "no server instructions"


# -------------------------------------------------------------------- enforced on plants


def _scratch_tool_items(docstring: str) -> list[bs.Item]:
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))

    async def forge_planted() -> str:
        return "x"

    forge_planted.__doc__ = docstring
    mcp.tool()(forge_planted)
    items, _ = bs.registry_items(mcp)
    return [i for i in items if "forge_planted" in i.where]


def _plant(kind: str) -> list[bs.Item]:
    if kind == "mcp_docstring":
        return _scratch_tool_items(PLANTS["mcp_docstring"])
    if kind == "refusal":
        source = f'raise refuse("x_failed", "{PLANTS["refusal"]}")\n'
        return bs.python_items(source, "planted.py")
    if kind == "gitops":
        doc = yaml.safe_load(f"data:\n  MISERVER_BASE_URL: {PLANTS['gitops_value']}\n")
        return bs.yaml_items(doc, "planted.yaml")
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["mcp_docstring", "refusal", "gitops"])
def test_enforced_mode_fails_on_plants(kind: str) -> None:
    items = _plant(kind)
    assert items, f"the {kind} plant produced no item"
    findings = bs.scan(items, bs.default_rules())
    assert findings, f"the guard did not see the planted {kind}"


def test_deleted_literal_row_is_finding() -> None:
    """``"millm_base_url"`` in code is allowed only because DR-39's record says so."""
    planted = bs.python_items('key = "millm_base_url"\n', "planted.py")
    export = li.load()
    assert not bs.scan(planted, bs.default_rules(export)), "DR-39 no longer masks its own key"
    without = li.Export(
        path=export.path,
        header=export.header,
        entries=[e for e in export.entries if e["id"] != "DR-39"],
        env=export.env,
        host=export.host,
        history_tools=export.history_tools,
        repo=export.repo,
        guards=export.guards,
    )
    assert bs.scan(planted, bs.default_rules(without)), "a literal with no record must be a finding"


@pytest.mark.parametrize(
    "name", PLANTS["repo_names"], ids=lambda n: n.replace(li.PRIVATE, "private")
)
def test_repo_allowance_only_from_transition_records(name: str) -> None:
    export = li.load()
    stand_in = li.withheld_key("repo", name)
    if export.withholds(stand_in):
        pytest.skip(f"{stand_in} withheld from the public variant")
    statuses = {r["old"]: r["status"] for r in export.repo}
    # CF-13: `transition` until P7, `retired` from P7 on; the committed export decides which.
    assert statuses.get(name) in {"transition", "retired"}, f"{name}: {statuses.get(name)}"
    item = [bs.Item("text", "planted.md:1", f"see github.com/{name}")]
    transition = _with_repo_status(export, "transition")
    assert not bs.scan(item, bs.default_rules(transition)), f"{name} is allowed while transition"
    retired = _with_repo_status(export, "retired")
    assert bs.scan(item, bs.default_rules(retired)), "a retired repo name must be a finding"
    assert bool(bs.scan(item, bs.default_rules(export))) == (
        statuses[name] == "retired"
    ), f"{name} is {statuses[name]} in the committed export, and the guard disagrees"
    # No hand-kept repo list in the scanner: the allowance comes only from the export.
    constants = [
        n.value
        for n in ast.walk(ast.parse(inspect.getsource(bs)))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert not [c for c in constants if f"{li.PRIVATE}/" in c or "hitsainet/" in c]


def _with_repo_status(export: li.Export, status: str) -> li.Export:
    """``export`` with every ``repo`` record set to ``status``."""
    return li.Export(
        path=export.path,
        header=export.header,
        entries=export.entries,
        env=export.env,
        host=export.host,
        history_tools=export.history_tools,
        repo=[{**r, "status": status} for r in export.repo],
        guards=export.guards,
    )


def _with_phase(phase: str) -> li.Export:
    text = li.export_path().read_text("utf-8")
    return li.parse(li.with_header(text, phase=phase))


def _rules_of(item: list[bs.Finding]) -> list[tuple[str, ...]]:
    return [f.rules for f in item]


def test_suite_phrase() -> None:
    """``-k suite_phrase`` (S001 FR-S001.69, CF-22, CF-23): what P5 allows and what it adds."""
    phrases = PLANTS["suite_phrase"]
    if li.load().withholds("G-STUDIO-PATH"):
        pytest.skip("G-STUDIO-PATH withheld from the public variant")

    def findings(phase: str, text: str) -> list[bs.Finding]:
        export = _with_phase(phase)
        return bs.scan([bs.Item("text", "planted.md:1", text)], bs.default_rules(export, phase))

    for allowed in phrases["allowed_from_p5"]:
        assert not findings("P5", allowed), f"P5 must allow {allowed!r}"
        assert findings("P4", allowed), f"P4 must still deny {allowed!r}"
    for denied in phrases["always_denied"]:
        assert findings("P5", denied) and findings("P4", denied), denied
    misuite = phrases["misuite"]
    at_p5 = findings("P5", misuite)
    at_p4 = findings("P4", misuite)
    assert at_p5 and "G-MISUITE" in at_p5[0].rules, _rules_of(at_p5)
    assert at_p4 and at_p4[0].rules == ("token:misuite",), _rules_of(at_p4)


def test_export_never_counts(tmp_path: Path) -> None:
    """CF-20: a root-level and a nested census export are never scanned; other docs are."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel in (
        "docs/legacy-identifiers.md",
        "a/docs/legacy-identifiers.public.md",
        "docs/other.md",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("miLLM\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    assert bs.discover(["docs/**", "a/**"], tmp_path) == ["docs/other.md"]
    # One positive pathspec sharing the `docs/` prefix: the shape where `**/docs/...` alone
    # failed to exclude the root-level file on git 2.43.
    assert bs.discover(["docs/**"], tmp_path) == ["docs/other.md"]
    assert bs.discover(["a/**"], tmp_path) == []
    # In the real tree: adding the committed export's text to the scanned docs changes nothing,
    # because discovery never returns it (control N56 records both counts).
    real = bs.discover(["docs/**"])
    assert "docs/legacy-identifiers.md" not in real
    assert "docs/legacy-identifiers.public.md" not in real


def test_enforced_keys_to_closed() -> None:
    export = li.load()
    assert bs.configured_enforced() == bs.expected_enforced(export), (
        f"brand-guard.json enforced={bs.configured_enforced()} but the export says closed="
        f"{export.closed!r} (CF-21: enforced exactly when closed >= P4)"
    )
    text = li.export_path().read_text("utf-8")
    early = li.parse(li.with_header(text, phase="P4", closed="P3"))
    late = li.parse(li.with_header(text, phase="P4", closed="P4"))
    assert bs.expected_enforced(early) is False
    assert bs.expected_enforced(late) is True


def test_report_mode_prints_and_passes(capsys: pytest.CaptureFixture[str]) -> None:
    _report("planted", bs.scan(_plant("refusal"), bs.default_rules()), mode="report")
    assert "[brand guard: planted] 1 finding(s) (report mode)" in capsys.readouterr().out


# ------------------------------------------------- where the guard enforces (suite R23)


def _tree(root: Path, shape: str, *, enforced: bool) -> Path:
    """A git tree of ``shape`` (see ``mirror_snapshot.make_repo``) with a ``brand-guard.json``."""
    from tests.support.mirror_snapshot import make_repo

    make_repo(root, shape)
    (root / bs.CONFIG).write_text(json.dumps({"enforced": enforced}), "utf-8")
    return root


def test_mode_enforces_privately_and_reports_on_the_mirror(tmp_path: Path) -> None:
    """Suite R23 (operator, 2026-11-11): from P4 the guard is ENFORCED in the private
    repository and REPORTS on the public mirror, whose export withholds the private repository
    names and the old host names, so a finding there cannot be judged. The mirror is the sync's
    commit shape (one parentless commit with both trailers), never the absence of a file: any
    other tree - no repository at all included - enforces."""
    from tests.support.mirror_snapshot import EXPECTED

    assert sum(EXPECTED.values()) == 1 and len(EXPECTED) >= 6, EXPECTED
    for shape, is_mirror in EXPECTED.items():
        on = _tree(tmp_path / f"on-{shape}", shape, enforced=True)
        off = _tree(tmp_path / f"off-{shape}", shape, enforced=False)
        want = "report" if is_mirror else "enforced"
        assert bs.mode(on, environ={}) == want, f"{shape}: enforced=true must give {want}"
        assert bs.mode(off, environ={}) == "report", f"{shape}: enforced=false reports"
        # IPD-8: the local proof before the flip (015 12.9) enforces whatever the file says.
        assert bs.mode(off, environ={bs.ENFORCE_ENV: "1"}) == "enforced", shape
        assert bs.mode(off, environ={bs.ENFORCE_ENV: "0"}) == "report", shape


def test_the_same_finding_fails_privately_and_is_reported_on_the_mirror(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mode reaches the assertion: one planted finding is a failure in a private-shaped
    tree and a printed line in a mirror-shaped one."""
    findings = bs.scan(_plant("refusal"), bs.default_rules())
    assert len(findings) == 1
    private = _tree(tmp_path / "private", "private", enforced=True)
    mirror = _tree(tmp_path / "mirror", "mirror", enforced=True)
    with pytest.raises(AssertionError, match="old brand names on a guarded surface"):
        _report("planted", findings, bs.mode(private, environ={}))
    capsys.readouterr()
    _report("planted", findings, bs.mode(mirror, environ={}))
    assert "[brand guard: planted] 1 finding(s) (report mode)" in capsys.readouterr().out


def test_the_surface_tests_use_the_trees_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wiring: each surface test fails when the tree's mode is enforced and it has a finding
    (a surface test that printed and passed in every mode would make the flip a no-op)."""
    finding = bs.scan(_plant("refusal"), bs.default_rules())
    monkeypatch.setattr(bs, "scan", lambda items, rules: finding)
    surface_tests = (
        test_backend_ast_surface,
        test_markdown_surfaces,
        test_env_example_surface,
        test_yaml_surfaces,
        test_mcp_registry_surface,
    )
    monkeypatch.setattr(bs, "mode", lambda *a, **k: "enforced")
    for surface_test in surface_tests:
        with pytest.raises(AssertionError, match="old brand names on a guarded surface"):
            surface_test()
    monkeypatch.setattr(bs, "mode", lambda *a, **k: "report")
    for surface_test in surface_tests:
        surface_test()


def test_env_alias_needs_both_authorities() -> None:
    """Allowance (c): an old variable is allowed only while the code table has it as a live
    alias AND its ``env`` record is ``transition``. After P7 the record is ``retired``, so a
    fallback reinstated in ``core/legacy_names.py`` alone (control N65) allows nothing."""
    export = li.load()
    old = "MILLM_BASE_URL"
    item = [bs.Item("text", "planted.md:1", f"set {old} (old name)")]

    def with_env_status(status: str) -> li.Export:
        return li.Export(
            path=export.path,
            header=export.header,
            entries=export.entries,
            env=[{**r, "status": status} for r in export.env],
            host=export.host,
            history_tools=export.history_tools,
            repo=export.repo,
            guards=export.guards,
        )

    alias = frozenset({old})
    rules = bs.rules_for(with_env_status("transition"), legacy_aliases=alias, retired=frozenset())
    assert not bs.scan(item, rules), "a live alias with a transition record is allowed"
    rules = bs.rules_for(with_env_status("retired"), legacy_aliases=alias, retired=frozenset())
    assert bs.scan(item, rules), "a live alias against a retired record must be a finding"
    rules = bs.rules_for(
        with_env_status("transition"), legacy_aliases=frozenset(), retired=frozenset()
    )
    assert bs.scan(item, rules), "a transition record with no live alias must be a finding"
