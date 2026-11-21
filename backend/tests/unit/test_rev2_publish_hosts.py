"""Review round 2 of 015 P0: the ``internal-host`` exception admits ``docs-<app>.hitsai.net``
only when that IS the whole host - not when it is the first labels of a longer internal name.

Round 1's lookahead ended in ``\\b``, which holds before ``.`` and ``-``, so
``docs-x.hitsai.net.local``, ``docs-x.hitsai.net.hitsai.local`` and
``docs-x.hitsai.net-k8s.hitsai.local`` were admitted. The same cases are checked with Node's
``RegExp``, which is what ``manual/scripts/check-published.mjs`` scans the built site with.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.unit.test_manual_publish_allowlist import RULES_PATH, _compiled, scan

REFUSED = [
    "docs-miforge.hitsai.net.local",
    "docs-miforge.hitsai.net.hitsai.local",
    "docs-miforge.hitsai.net.internal",
    "docs-miforge.hitsai.net-k8s.hitsai.local",
    "docs-miforge.hitsai.network.local",
    "see https://docs-miforge.hitsai.net.lan/x",
    "docs-k8s-miforge.hitsai.net",
]
ADMITTED = [
    "docs-miforge.hitsai.net",
    "the manual at https://docs-miworkbench.hitsai.net/advanced/x",
    "see docs-miforge.hitsai.net.",
    "docs-miforge.hitsai.net:443",
    "DOCS-MIFORGE.HITSAI.NET",
]


def _caught(text: str) -> bool:
    rules = [r for r in _compiled() if r[0] == "internal-host"]
    assert rules, "no internal-host rule"
    return bool(scan([("planted.md", text)], rules))


@pytest.mark.parametrize("planted", REFUSED)
def test_a_longer_host_starting_with_the_docs_name_is_refused(planted: str) -> None:
    assert _caught(planted), f"internal-host admitted {planted!r}"


@pytest.mark.parametrize("planted", ADMITTED)
def test_the_whole_public_manual_host_is_admitted(planted: str) -> None:
    assert not _caught(planted), f"internal-host refused {planted!r}"


def test_node_reads_the_rule_the_same_way() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the Manual CI job runs check-published.mjs")
    script = (
        "const fs=require('fs');"
        "const r=JSON.parse(fs.readFileSync(process.argv[1],'utf8')).forbidden"
        ".find(x=>x.id==='internal-host');"
        "const re=new RegExp(r.pattern,'g'+(r.flags||''));"
        "const out={};for(const t of JSON.parse(process.argv[2])){re.lastIndex=0;"
        "out[t]=re.exec(t)!==null}console.log(JSON.stringify(out));"
    )
    texts = REFUSED + ADMITTED
    done = subprocess.run(
        [node, "-e", script, str(RULES_PATH), json.dumps(texts)],
        capture_output=True,
        text=True,
        check=True,
    )
    caught = json.loads(done.stdout)
    assert [t for t in REFUSED if not caught[t]] == [], "Node admitted a refused host"
    assert [t for t in ADMITTED if caught[t]] == [], "Node refused the public manual host"
