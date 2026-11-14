"""The committed census export parses, and the parser refuses everything outside the format.

FR-015.1/.2 (CF-1, CF-2, CF-15, CF-16, CF-21, CF-24, CF-25). The refusal cases run on synthetic
copies of the committed file, each changed in one way; the real file is never edited here.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from tests.support import legacy_identifiers as li
from tests.support.mirror_snapshot import mirror_snapshot

REPO_ROOT = li.REPO_ROOT
SYNC = REPO_ROOT / ".github" / "workflows" / "sync-to-clean.yml"


def _text() -> str:
    return li.export_path().read_text("utf-8")


def test_block_parses() -> None:
    export = li.load()
    assert export.header["format"] == 1
    assert "closed" in export.header, "CF-21: the header separates phase from last closed"
    assert export.header["repo"] == "miForge"
    assert export.entries, "the export lists no register entry; miForge carries ten"
    for entry in export.entries:
        assert set(entry) == set(li.SECTION_KEYS["entry"]), entry["id"]
        assert "locations" not in entry


def test_literals_match_pattern() -> None:
    for entry in li.load().entries:
        for literal in entry["literals"]:
            assert re.search(entry["pattern"], literal), (entry["id"], literal)


def _records(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in li._block(text)]


def _rewrite(text: str, index: int, change: dict[str, object], drop: str | None = None) -> str:
    """The export with record ``index`` (0 = header) changed; everything else verbatim."""
    lines = text.split("\n")
    seen = -1
    for number, line in enumerate(lines):
        if line.startswith("{"):
            seen += 1
            if seen == index:
                record = json.loads(line)
                record.update(change)
                if drop:
                    record.pop(drop)
                lines[number] = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                break
    return "\n".join(lines)


def _first(text: str, section: str) -> int:
    for index, record in enumerate(_records(text)):
        if record.get("section") == section:
            return index
    raise AssertionError(f"the committed export has no {section} record to mutate")


def _host_line(text: str) -> str:
    """The committed export carries no host record, so the host cases add one."""
    host = {"section": "host", "old": "k8s-old.example", "new": "k8s-new.example"}
    # `observed` PRESENT but not a boolean, so the CF-24 check itself must refuse it; a record
    # lacking the key is refused by the generic missing-key check (control N22f).
    line = json.dumps({**host, "status": "transition", "observed": "yes"}, separators=(",", ":"))
    return _append(text, line)


def _append(text: str, line: str) -> str:
    """``line`` added as the block's last record (before the CLOSING fence, not the opening)."""
    head, tail = text.rsplit("\n```", 1)
    return f"{head}\n{line}\n```{tail}"


def _case(name: str) -> tuple[str, str | None]:
    """(synthetic export text, expected refusal fragment or None for "accepted")."""
    text = _text()
    entry = _first(text, "entry")
    guard = _first(text, "guard")
    if name == "regex_flag":
        return _rewrite(text, entry, {"flags": ["i"]}), "flags"
    if name == "locations":
        # The dedicated CF-1 refusal, not the generic unknown-key one (control N22b found the
        # generic check answering for it, so a bare "locations" fragment proved nothing).
        return _rewrite(text, entry, {"locations": ["README.md:1"]}), "never enter the export"
    if name == "modifier_group":
        return _rewrite(text, entry, {"pattern_js": "(?i:millm)"}), "inline flag"
    if name == "guard_effect":
        return _rewrite(text, guard, {"effect": "warn"}), "effect"
    if name == "no_closed":
        return _rewrite(text, 0, {}, drop="closed"), "closed"
    if name == "host_observed":
        return _host_line(text), "boolean 'observed'"
    if name == "lookaround_is_legal":
        return _rewrite(text, entry, {"pattern_js": "millm(?![-\\w])", "literals": []}), None
    if name == "anchor_A":
        return _rewrite(text, entry, {"pattern": "\\Amillm", "literals": []}), "\\A"
    if name == "anchor_Z":
        return _rewrite(text, entry, {"pattern_js": "millm\\Z", "literals": []}), "\\Z"
    if name == "named_group":
        return _rewrite(text, entry, {"pattern": "(?P<x>millm)", "literals": []}), "(?P"
    if name == "format":
        return _rewrite(text, 0, {"format": 2}), "format"
    if name == "unknown_section":
        return _append(text, '{"section":"bogus"}'), "unknown section"
    if name == "unknown_key":
        return _rewrite(text, entry, {"surprise": 1}), "unknown keys"
    if name == "missing_key":
        return _rewrite(text, entry, {}, drop="exercise"), "missing keys"
    if name == "bad_status":
        return _rewrite(text, entry, {"status": "deferred"}), "status"
    if name == "js_flags":
        return _rewrite(text, entry, {"js_flags": "g"}), "js_flags"
    if name == "literal_miss":
        return _rewrite(text, entry, {"literals": ["nothing-like-it"]}), "does not match"
    if name == "guard_phase":
        return _rewrite(text, guard, {"from_phase": "P9"}), "phase"
    raise AssertionError(name)


CASES = [
    "regex_flag",
    "locations",
    "modifier_group",
    "guard_effect",
    "no_closed",
    "host_observed",
    "lookaround_is_legal",
    "anchor_A",
    "anchor_Z",
    "named_group",
    "format",
    "unknown_section",
    "unknown_key",
    "missing_key",
    "bad_status",
    "js_flags",
    "literal_miss",
    "guard_phase",
]


@pytest.mark.parametrize("name", CASES)
def test_parser_refusals(name: str) -> None:
    text, refusal = _case(name)
    assert text != _text(), f"{name}: the synthetic copy is unchanged; the case tests nothing"
    if refusal is None:
        li.parse(text)
        return
    with pytest.raises(li.ExportError) as refused:
        li.parse(text)
    assert refusal in str(refused.value), str(refused.value)


def test_forbidden_list_unread_is_accepted_only_on_the_public_variant() -> None:
    """``census.py --public-unchecked`` adds ``"forbidden_list":"unread"`` to the public header;
    ``--public`` (once the list is read) omits it. Both parse; nothing else does."""
    # Start from a header WITHOUT the key, whichever variant the loader found.
    text = _rewrite(_text(), 0, {"forbidden_list": None}, drop="forbidden_list")
    public = _rewrite(text, 0, {"variant": "public", "forbidden_list": "unread"})
    assert li.parse(public).header["forbidden_list"] == "unread"
    assert "forbidden_list" not in li.parse(_rewrite(text, 0, {"variant": "public"})).header
    with pytest.raises(li.ExportError, match="forbidden_list"):
        li.parse(_rewrite(text, 0, {"variant": "public", "forbidden_list": "read"}))
    with pytest.raises(li.ExportError, match="public variant"):
        li.parse(_rewrite(text, 0, {"variant": "full", "forbidden_list": "unread"}))
    committed = REPO_ROOT / li.PUBLIC
    if committed.is_file():
        header = li.parse(committed.read_text("utf-8")).header
        assert header.get("forbidden_list", "unread") == "unread"


def test_public_equals_full_minus_withheld() -> None:
    """The public variant is the full one minus the records it names in ``withheld``."""
    full_path, public_path = REPO_ROOT / li.FULL, REPO_ROOT / li.PUBLIC
    if not full_path.is_file() and mirror_snapshot(REPO_ROOT):
        pytest.skip("MIRROR: the full export is private and stripped from the public mirror")
    assert full_path.is_file(), f"{li.FULL} is missing from a tree that is not the mirror snapshot"
    full = li.parse(full_path.read_text("utf-8"))
    public = li.parse(public_path.read_text("utf-8"))
    assert full.variant == "full" and public.variant == "public"
    assert full.withheld == []
    withheld = set(public.withheld)
    for attr in ("register_hash", "phase", "closed", "repo"):
        assert full.header[attr] == public.header[attr], attr

    def key(record: dict[str, object]) -> str:
        return li.record_key(record)  # type: ignore[arg-type]

    def records(export: li.Export) -> list[dict[str, object]]:
        out = []
        for group in (export.entries, export.env, export.host, export.history_tools):
            out.extend(group)
        out.extend(export.repo)
        out.extend(export.guards)
        return out

    expected = []
    for record in records(full):
        if key(record) in withheld:
            continue
        flags = record.get("flags")
        if isinstance(flags, list) and "public_withheld" in flags:
            continue
        expected.append(record)

    assert not variant_mismatches(records(public), expected)


def variant_mismatches(
    public: list[dict[str, object]], expected: list[dict[str, object]]
) -> list[str]:
    """Entry literals are per variant: the public file keeps only those the mirror itself
    carries (census 09b4737), so they are compared as a subset, everything else exactly."""

    def without_literals(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{k: v for k, v in r.items() if k != "literals"} for r in rows]

    if without_literals(public) != without_literals(expected):
        return ["the records differ beyond their literals"]
    return [
        str(mine)
        for mine, theirs in zip(public, expected, strict=True)
        if not set(mine.get("literals") or []) <= set(theirs.get("literals") or [])  # type: ignore[arg-type]
    ]


def test_variant_comparison_self_test() -> None:
    """Review round 2: deleting the subset loop stayed green, because the real exports agree.
    A public literal the full export lacks, or a changed field, must be reported."""
    full = [{"id": "DR-1", "literals": ["a", "b"]}, {"id": "DR-2", "literals": []}]
    assert variant_mismatches([{"id": "DR-1", "literals": ["a"]}, {"id": "DR-2"}], full) == []
    assert variant_mismatches([{"id": "DR-1", "literals": ["a", "z"]}, full[1]], full)
    assert variant_mismatches([{"id": "DR-1", "literals": ["a"], "x": 1}, full[1]], full)


def _kept_docs_names(workflow: str) -> set[str]:
    """The names the sync's ``find docs/ ... ! -name X ... -exec rm`` keeps."""
    for line in workflow.splitlines():
        if line.strip().startswith("find docs/") and "-exec rm" in line:
            words = shlex.split(line.strip().split("-exec", 1)[0])
            return {
                words[i + 2] for i, w in enumerate(words) if w == "!" and words[i + 1] == "-name"
            }
    raise AssertionError("sync-to-clean.yml has no `find docs/ ... -exec rm` line")


def test_sync_keeps_public_strips_full() -> None:
    """The mirror carries the public export (its Backend Tests read it) and never the full one."""
    kept = _kept_docs_names(SYNC.read_text("utf-8"))
    assert "schemas" in kept, "the parse found the wrong line"
    assert li.PUBLIC.name in kept, "sync-to-clean.yml strips legacy-identifiers.public.md"
    assert li.FULL.name not in kept, "sync-to-clean.yml publishes the full legacy-identifiers.md"


def test_export_override_reads_the_scratch_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The override the controls use really replaces the committed file."""
    copy = tmp_path / "export.md"
    copy.write_text(_rewrite(_text(), 0, {"closed": "P3", "phase": "P4"}))
    monkeypatch.setenv(li.OVERRIDE_ENV, str(copy))
    export = li.load()
    assert (export.phase, export.closed) == ("P4", "P3")


def test_withheld_key_is_the_census_stand_in() -> None:
    """``<section>:sha256:`` + 16 hex of the SHA-256 of the old name (census ``ef57482``)."""
    assert li.withheld_key("repo", f"{li.PRIVATE}/miLLM") == "repo:sha256:7c12ca6b1b211fe8"
    assert li.record_key({"section": "entry", "id": "DR-29"}) == "DR-29"
    assert li.record_key({"section": "repo", "old": f"{li.PRIVATE}/miStudio"}) == (
        "repo:sha256:6ca49c4ed3b92ac9"
    )


def test_active_guards_follow_from_and_until() -> None:
    export = li.load()
    if export.withholds("G-STUDIO-PATH"):
        pytest.skip("G-STUDIO-PATH withheld from the public variant")
    at_p4 = {g["id"] for g in li.active_guards(export, "P4")}
    at_p5 = {g["id"] for g in li.active_guards(export, "P5")}
    assert "G-STUDIO" in at_p4 and "G-MISUITE" not in at_p4
    assert {"G-MISUITE", "G-STUDIO-PATH"} <= at_p5
